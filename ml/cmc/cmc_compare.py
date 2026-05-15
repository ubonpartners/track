"""Offline CMC comparison tool.

For each JAAD clip:
  1. Decode RGB frames with OpenCV.
  2. Per consecutive frame pair compute:
        (a) Our CMC: feed both frames to upyc.c_motion_tracker, read
            cmc_transform_t (tx, ty, p, q) — normalised image fraction.
        (b) Reference CMC (OpenCV): ORB keypoints + descriptor match +
            RANSAC partial affine (similarity: rotation + uniform scale +
            translation). This is the standard CMC baseline used by ECCV
            CMC ablations and competition trackers.
  3. Decompose both transforms onto a common parameterisation:
            tx, ty       — translation (image fraction)
            scale - 1    — uniform scale residual
            rot_deg      — rotation in degrees
     Our 4-DOF model (p, q, tx, ty) is the displacement form:
         dx = p·x - q·y + tx
         dy = q·x + p·y + ty
     which equals the *residual* of an exact similarity x'=(1-p)x+qy-tx.
     So scale - 1 = -p (small-angle) and rot_rad ≈ -q.
  4. Emit per-frame diffs + per-clip aggregate diagnostics.

Outputs:
  /tmp/cmc_compare/<clip>.csv      one row per frame
  /tmp/cmc_compare/summary.csv     one row per clip
  /tmp/cmc_compare/summary.json    full structured summary

Usage:
  python -m ml.cmc.cmc_compare                # JAAD val
  python -m ml.cmc.cmc_compare --clips video_0021 video_0072
  python -m ml.cmc.cmc_compare --max-clips 5  # quick sanity sweep
"""
import argparse, csv, json, math, os, sys, time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ubon_pycstuff.ubon_pycstuff as upyc


OUT_DIR = Path('/tmp/cmc_compare')
PROD_YAML = '/mldata/config/track/trackers/uc_v11.yaml'

# Down-scale for the OpenCV reference. ORB+RANSAC on full 1080p is slow;
# the global-motion estimate is scale-invariant once normalised.
REF_MAX_DIM = 480


# ----------------------------------------------------------------------
# Our CMC: pump frames through c_motion_tracker
# ----------------------------------------------------------------------
def make_motion_tracker(yaml_path: str):
    # The C side reads only the `motiontrack:` block; we just point it at
    # the prod yaml so the same params used in eval are exercised.
    return upyc.c_motion_tracker(yaml_path)


def ours_cmc(mt, rgb_uint8_hwc):
    """Push one RGB frame and read the resulting CMC transform.

    The full per-frame protocol is add_frame → set_roi(get_roi) →
    get_cmc_transform. The set_roi step is what commits the new
    reference image inside the tracker — without it `mt->ref` stays
    NULL and `motion_track_get_cmc_transform` early-returns the no-op
    translation-median fallback every frame. This is the same pattern
    utrack.c uses on the live path.
    """
    img = upyc.c_image.from_numpy(rgb_uint8_hwc)
    mt.add_frame(img)
    mt.set_roi(mt.get_roi())
    out = mt.get_cmc_transform()
    # Binding returns (tx, ty, p, q[, alpha]) — accept both for forward compat.
    if len(out) >= 5:
        tx, ty, p, q, _alpha = out
    else:
        tx, ty, p, q = out
    return dict(tx=float(tx), ty=float(ty), p=float(p), q=float(q))


# ----------------------------------------------------------------------
# OpenCV reference: ORB + RANSAC partial affine (similarity)
# ----------------------------------------------------------------------
ORB = cv2.ORB_create(nfeatures=2000, fastThreshold=15)


def opencv_cmc(prev_gray, cur_gray):
    """Estimate a similarity transform (rotation + uniform scale + tx,ty)
    from prev → cur using ORB+BFMatcher+RANSAC. Returns the same dict
    schema as `ours_cmc` (normalised image fraction)."""
    H, W = cur_gray.shape
    kp0, des0 = ORB.detectAndCompute(prev_gray, None)
    kp1, des1 = ORB.detectAndCompute(cur_gray, None)
    if des0 is None or des1 is None or len(des0) < 8 or len(des1) < 8:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des0, des1)
    if len(matches) < 10:
        return None
    pts0 = np.float32([kp0[m.queryIdx].pt for m in matches])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in matches])
    # Partial affine = uniform scale + rotation + translation = similarity
    A, inliers = cv2.estimateAffinePartial2D(
        pts0, pts1, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=2000, confidence=0.99)
    if A is None:
        return None
    # OpenCV partial-affine convention: A = [[a, -b, tx], [b, a, ty]] with
    # a = s·cos(θ), b = s·sin(θ). cur_pt ≈ A · prev_pt. Extract `b` from
    # row 1, col 0 (not row 0, col 1 — that's -b!).
    a     = float(A[0, 0])
    b     = float(A[1, 0])
    tx_px = float(A[0, 2])
    ty_px = float(A[1, 2])
    scale = math.hypot(a, b)
    rot = math.atan2(b, a)  # radians
    # Normalise translation to image fraction. Note our convention:
    #   ours.tx > 0 ⟺ objects appeared to move LEFT
    # OpenCV's tx_px is the per-pixel translation of a *prev* point INTO
    # the *cur* frame: cur = a*prev + ... + tx_px. So positive tx_px means
    # objects moved RIGHT in the image, i.e. camera panned LEFT. To match
    # our sign convention we negate.
    tx = -tx_px / W
    ty = -ty_px / H
    # Map OpenCV's forward similarity into the C tracker's
    # backward-flow convention (which is what NVOF feeds the IRLS).
    # Empirically the C tracker's p, q are the negation of the forward
    # (s·cos θ − 1, s·sin θ) — i.e. fits to "where this current pixel
    # came from in the previous frame". Putting OpenCV into the same
    # convention is just  p ≡ 1 − s·cos θ,  q ≡ −s·sin θ.
    p = 1.0 - scale * math.cos(rot)
    q = -scale * math.sin(rot)
    n_inliers = int(inliers.sum()) if inliers is not None else 0
    return dict(tx=tx, ty=ty, p=p, q=q, scale=scale,
                rot_rad=rot, n_matches=len(matches), n_inliers=n_inliers)


# ----------------------------------------------------------------------
# Decomposition helper for reporting
# ----------------------------------------------------------------------
def decompose(d):
    """(tx, ty, p, q) → (tx, ty, scale-1, rot_deg) in the C tracker's
    backward-flow convention. The compensation matrix
    [[1-p, q], [-q, 1-p]] matches s·[cos θ, -sin θ; sin θ, cos θ]
    where s and θ are the *backward* similarity (current → previous).
    So |scale_residual| equals |s_forward − 1| up to sign.
    """
    if d is None:
        return None
    p, q = d['p'], d['q']
    scale = math.hypot(1.0 - p, q)
    rot_deg = math.degrees(math.atan2(-q, 1.0 - p))
    return dict(tx=d['tx'], ty=d['ty'],
                scale_residual=scale - 1.0, rot_deg=rot_deg)


# ----------------------------------------------------------------------
# Per-clip runner
# ----------------------------------------------------------------------
def run_clip(clip_name: str, yaml_path: str, video_path: str,
             out_csv: Path, frame_cap: int | None = None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # OpenCV reference works on a downsampled grayscale.
    s = REF_MAX_DIM / max(W, H)
    Wref = max(1, int(round(W * s)))
    Href = max(1, int(round(H * s)))

    mt = make_motion_tracker(yaml_path)

    rows = []
    prev_gray = None
    n = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        n += 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ours = ours_cmc(mt, rgb)
        # Down-scale once, share between prev and ref calls
        small = cv2.resize(bgr, (Wref, Href), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            ref = opencv_cmc(prev_gray, gray)
        else:
            ref = None
        prev_gray = gray

        odec = decompose(ours)
        rdec = decompose(ref) if ref else None
        row = dict(
            clip=clip_name, frame=n,
            ours_tx=odec['tx'], ours_ty=odec['ty'],
            ours_scale_res=odec['scale_residual'], ours_rot_deg=odec['rot_deg'],
            ref_tx=(rdec['tx'] if rdec else None),
            ref_ty=(rdec['ty'] if rdec else None),
            ref_scale_res=(rdec['scale_residual'] if rdec else None),
            ref_rot_deg=(rdec['rot_deg'] if rdec else None),
            ref_n_matches=(ref['n_matches'] if ref else None),
            ref_n_inliers=(ref['n_inliers'] if ref else None),
        )
        rows.append(row)
        if frame_cap and n >= frame_cap:
            break
    cap.release()

    # Skip the first frame in our records — both ours and ref are
    # warmup placeholders (ours_cmc has no `ref` to diff against, ref
    # has no prev). Drop frame 1 from CSV.
    if rows:
        rows = rows[1:]

    with open(out_csv, 'w', newline='') as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # Aggregate diagnostics on rows that have a reference
    valid = [r for r in rows
             if r['ref_tx'] is not None and r['ref_n_inliers'] is not None
             and r['ref_n_inliers'] >= 15]
    if not valid:
        return dict(clip=clip_name, n_frames=n, n_valid=0)

    def col(rs, key):
        return np.array([r[key] for r in rs], dtype=np.float64)

    ours_t = np.stack([col(valid, 'ours_tx'), col(valid, 'ours_ty')], axis=1)
    ref_t  = np.stack([col(valid, 'ref_tx'),  col(valid, 'ref_ty')],  axis=1)
    err_t  = ours_t - ref_t
    err_mag = np.linalg.norm(err_t, axis=1)
    ref_mag = np.linalg.norm(ref_t, axis=1)
    ours_mag = np.linalg.norm(ours_t, axis=1)

    err_scale = col(valid, 'ours_scale_res') - col(valid, 'ref_scale_res')
    err_rot   = col(valid, 'ours_rot_deg')   - col(valid, 'ref_rot_deg')

    # Per-frame correlation across the clip
    def corr(a, b):
        if len(a) < 3: return float('nan')
        sa, sb = a.std(), b.std()
        if sa < 1e-9 or sb < 1e-9: return float('nan')
        return float(np.corrcoef(a, b)[0, 1])

    return dict(
        clip=clip_name,
        n_frames=n, n_valid=len(valid),
        # Magnitude stats
        ref_tx_p50=float(np.median(np.abs(ref_t[:, 0]))),
        ref_ty_p50=float(np.median(np.abs(ref_t[:, 1]))),
        ref_tx_p95=float(np.quantile(np.abs(ref_t[:, 0]), 0.95)),
        ref_ty_p95=float(np.quantile(np.abs(ref_t[:, 1]), 0.95)),
        ours_tx_p50=float(np.median(np.abs(ours_t[:, 0]))),
        ours_ty_p50=float(np.median(np.abs(ours_t[:, 1]))),
        # Error stats
        err_translation_median=float(np.median(err_mag)),
        err_translation_p95=float(np.quantile(err_mag, 0.95)),
        err_translation_max=float(err_mag.max()),
        err_scale_median=float(np.median(np.abs(err_scale))),
        err_rot_deg_median=float(np.median(np.abs(err_rot))),
        # Correlations (do we agree on direction even if magnitude differs?)
        corr_tx=corr(col(valid, 'ours_tx'), col(valid, 'ref_tx')),
        corr_ty=corr(col(valid, 'ours_ty'), col(valid, 'ref_ty')),
        corr_scale=corr(col(valid, 'ours_scale_res'),
                        col(valid, 'ref_scale_res')),
        corr_rot=corr(col(valid, 'ours_rot_deg'), col(valid, 'ref_rot_deg')),
        # How big is the error relative to the reference motion?
        rel_err_translation=float(np.median(err_mag) /
                                   max(1e-9, np.median(ref_mag))),
    )


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clips', nargs='*',
                   help='clip names (default: JAAD default val split)')
    p.add_argument('--split-file',
                   default='/tmp/jaad_splits/jaad_default_val.txt',
                   help='one clip-name per line (used when --clips omitted)')
    p.add_argument('--video-template',
                   default='/mldata/tracking/jaad/video/{clip}.mp4',
                   help='format string mapping clip name → mp4 path')
    p.add_argument('--out-dir', default=str(OUT_DIR),
                   help='where to write csv + summary outputs')
    p.add_argument('--max-clips', type=int, default=None)
    p.add_argument('--frame-cap', type=int, default=None,
                   help='cap frames per clip (debug)')
    p.add_argument('--yaml', default=PROD_YAML,
                   help='tracker yaml (only motiontrack: block is read)')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.clips:
        clip_names = args.clips
    else:
        clip_names = open(args.split_file).read().split()
    if args.max_clips:
        clip_names = clip_names[:args.max_clips]

    summaries = []
    print(f'{"clip":15s}  {"frames":>6}  {"valid":>5}  '
          f'{"err_t_med":>9}  {"rel_err_t":>9}  '
          f'{"corr_tx":>7}  {"corr_ty":>7}  '
          f'{"corr_scl":>8}  {"corr_rot":>8}', flush=True)
    for clip in clip_names:
        video = args.video_template.format(clip=clip)
        if not os.path.isfile(video):
            print(f'  [warn] missing video: {video}')
            continue
        t0 = time.time()
        s = run_clip(clip, args.yaml, video, out_dir / f'{clip}.csv',
                     frame_cap=args.frame_cap)
        dt = time.time() - t0
        if s is None or s.get('n_valid', 0) == 0:
            print(f'{clip:15s}  {(s or {}).get("n_frames", 0):>6}  '
                  f'{0:>5}  -no valid frames-  ({dt:.0f}s)')
            continue
        print(
            f'{clip:15s}  {s["n_frames"]:>6}  {s["n_valid"]:>5}  '
            f'{s["err_translation_median"]:>9.5f}  '
            f'{s["rel_err_translation"]:>9.3f}  '
            f'{s["corr_tx"]:>7.3f}  {s["corr_ty"]:>7.3f}  '
            f'{s["corr_scale"]:>8.3f}  {s["corr_rot"]:>8.3f}  '
            f'({dt:.0f}s)', flush=True)
        summaries.append(s)

    # Persist summary
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summaries, f, indent=2)
    if summaries:
        with open(out_dir / 'summary.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            w.writeheader()
            w.writerows(summaries)

    if not summaries:
        return
    # Quick aggregate readout
    def pct(arr, q): return float(np.quantile(arr, q))
    err_t = np.array([s['err_translation_median'] for s in summaries])
    rel = np.array([s['rel_err_translation'] for s in summaries])
    corr_tx = np.array([s['corr_tx'] for s in summaries])
    corr_ty = np.array([s['corr_ty'] for s in summaries])
    corr_scl = np.array([s['corr_scale'] for s in summaries])
    corr_rot = np.array([s['corr_rot'] for s in summaries])
    print()
    print(f'=== aggregate across {len(summaries)} clips ===')
    print(f'  median translation error (image fraction): '
          f'p50={np.median(err_t):.5f}  p95={pct(err_t, 0.95):.5f}')
    print(f'  median relative error / reference magnitude: '
          f'p50={np.median(rel):.3f}  p95={pct(rel, 0.95):.3f}')
    print(f'  per-clip pearson r(ours,ref): '
          f'tx mean={np.nanmean(corr_tx):.3f}  '
          f'ty mean={np.nanmean(corr_ty):.3f}  '
          f'scale mean={np.nanmean(corr_scl):.3f}  '
          f'rot mean={np.nanmean(corr_rot):.3f}')
    print(f'  per-clip CSVs + summary written under {OUT_DIR}')


if __name__ == '__main__':
    main()
