"""Per-frame diagnostic for one clip: what's driving the systematic
underestimate of camera motion?

For each frame we record:
  ours_cmc           — 4-DOF fit: tx, ty, p, q
  raw OF stats       — median/p25/p75 of the dense NVOF flow that feeds
                        the fit. If raw OF median tracks the OpenCV
                        reference but the CMC fit doesn't, the bug is
                        in the fit (Huber weighting, ref-blending).
                        If raw OF underestimates too, NVOF resolution
                        / preset is the culprit.
  motion-mask ROI    — `motion_track_get_roi()` area. If it stays small
                        on a high-motion frame, the motion-mask noise
                        floor is over-thresholding.
  ref_cmc            — OpenCV ORB+RANSAC similarity

Outputs:
  /tmp/cmc_dig/<clip>.csv         per-frame rows
  /tmp/cmc_dig/<clip>_summary.json
  Per-segment correlation/median-ratio bandyards on stdout.

Also runs the same clip at a *higher* motion_track input resolution
(max_width/max_height = 480) to test whether NVOF precision is the
limiting factor.

Usage:
  python -m ml.cmc.cmc_dig --clip video_0102
  python -m ml.cmc.cmc_dig --clip MOT17-13 \
      --video /mldata/tracking/mot/video/MOT17-13.mp4
"""
import argparse, csv, json, math, os, sys, time, tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.cmc.cmc_compare import opencv_cmc, decompose
import ubon_pycstuff.ubon_pycstuff as upyc


OUT_DIR = Path('/tmp/cmc_dig')
PROD_YAML = '/mldata/config/track/trackers/uc_v11.yaml'
# REF_MAX_DIM controls the downsample for the OpenCV reference. ORB at
# 480 was fine for direction agreement but biased the magnitude estimate
# low; bumping to 960 makes the reference a near-ceiling-precision yard
# stick at the cost of ~3x ORB time.
REF_MAX_DIM = int(os.environ.get('CMC_REF_MAX_DIM', '960'))


def make_yaml(base_path, max_dim):
    """Clone the prod yaml but override motiontrack.max_width/height."""
    cfg = yaml.safe_load(open(base_path))
    cfg['motiontrack']['max_width']  = max_dim
    cfg['motiontrack']['max_height'] = max_dim
    fd, path = tempfile.mkstemp(suffix='.yaml', prefix='cmc_dig_')
    os.close(fd)
    yaml.safe_dump(cfg, open(path, 'w'))
    return path


def step(mt, rgb):
    img = upyc.c_image.from_numpy(rgb)
    mt.add_frame(img)
    roi = mt.get_roi()
    mt.set_roi(roi)
    out = mt.get_cmc_transform()
    # Binding now returns (tx, ty, p, q, alpha) post-2026-05-14.
    if len(out) >= 5:
        tx, ty, p, q, _alpha = out
    else:
        tx, ty, p, q = out
    # OF stats (in normalised image-fraction units thanks to the binding)
    of = mt.get_of_results()  # (gh, gw, 2)
    return dict(
        ours_tx=float(tx), ours_ty=float(ty),
        ours_p=float(p),   ours_q=float(q),
        roi_x0=roi[0], roi_y0=roi[1], roi_x1=roi[2], roi_y1=roi[3],
        roi_area=max(0.0, roi[2]-roi[0]) * max(0.0, roi[3]-roi[1]),
        of_dx_med=float(np.median(of[:, :, 0])),
        of_dy_med=float(np.median(of[:, :, 1])),
        of_dx_p25=float(np.quantile(of[:, :, 0], 0.25)),
        of_dx_p75=float(np.quantile(of[:, :, 0], 0.75)),
        of_dy_p25=float(np.quantile(of[:, :, 1], 0.25)),
        of_dy_p75=float(np.quantile(of[:, :, 1], 0.75)),
        of_dx_mean=float(np.mean(of[:, :, 0])),
        of_dy_mean=float(np.mean(of[:, :, 1])),
        of_dx_max=float(np.max(np.abs(of[:, :, 0]))),
        of_dy_max=float(np.max(np.abs(of[:, :, 1]))),
        of_gh=int(of.shape[0]), of_gw=int(of.shape[1]),
    )


def run(clip, video_path, yaml_path, frame_cap=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'cannot open {video_path}')
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = REF_MAX_DIM / max(W, H)
    Wref, Href = max(1, int(round(W * s))), max(1, int(round(H * s)))

    mt = upyc.c_motion_tracker(yaml_path)
    rows = []
    prev_gray = None
    n = 0
    while True:
        ok, bgr = cap.read()
        if not ok: break
        n += 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ours = step(mt, rgb)
        small = cv2.resize(bgr, (Wref, Href), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        ref = opencv_cmc(prev_gray, gray) if prev_gray is not None else None
        prev_gray = gray

        ours_dec = decompose(dict(tx=ours['ours_tx'], ty=ours['ours_ty'],
                                  p=ours['ours_p'],  q=ours['ours_q']))
        ref_dec = decompose(ref) if ref else None
        row = dict(clip=clip, frame=n,
                   ours_tx=ours_dec['tx'], ours_ty=ours_dec['ty'],
                   ours_scale_res=ours_dec['scale_residual'],
                   ours_rot_deg=ours_dec['rot_deg'],
                   ref_tx=(ref_dec['tx'] if ref_dec else None),
                   ref_ty=(ref_dec['ty'] if ref_dec else None),
                   ref_scale_res=(ref_dec['scale_residual'] if ref_dec else None),
                   ref_rot_deg=(ref_dec['rot_deg'] if ref_dec else None),
                   ref_n_inliers=(ref['n_inliers'] if ref else None),
                   **{k: v for k, v in ours.items()
                      if k not in ('ours_tx', 'ours_ty', 'ours_p', 'ours_q')})
        rows.append(row)
        if frame_cap and n >= frame_cap:
            break
    cap.release()
    return rows[1:], (W, H), (Wref, Href)


def summarise(label, rows):
    valid = [r for r in rows if r['ref_tx'] is not None
             and (r['ref_n_inliers'] or 0) >= 15]
    print(f'\n=== {label}: {len(rows)} frames, {len(valid)} valid ===')
    if not valid: return
    def col(k): return np.array([r[k] for r in valid], dtype=np.float64)
    def corr(a, b):
        if a.std() < 1e-9 or b.std() < 1e-9: return float('nan')
        return float(np.corrcoef(a, b)[0, 1])

    ours_t = np.stack([col('ours_tx'), col('ours_ty')], axis=1)
    ref_t  = np.stack([col('ref_tx'),  col('ref_ty')],  axis=1)
    of_t   = np.stack([col('of_dx_med'), col('of_dy_med')], axis=1)
    of_mean = np.stack([col('of_dx_mean'), col('of_dy_mean')], axis=1)

    def magstats(x, name):
        m = np.linalg.norm(x, axis=1)
        return (f'  {name:14s}  median|·|={np.median(m):.5f}  '
                f'p95|·|={np.quantile(m, 0.95):.5f}')

    print(magstats(ref_t,   'ref translation'))
    print(magstats(ours_t,  'our 4-DOF tr.'))
    print(magstats(of_t,    'OF dx/dy median'))
    print(magstats(of_mean, 'OF dx/dy mean'))

    err_t = np.linalg.norm(ours_t - ref_t, axis=1)
    rel = err_t / np.maximum(np.linalg.norm(ref_t, axis=1), 1e-9)
    print(f'  err |ours-ref| p50={np.median(err_t):.5f}  '
          f'rel-err p50={np.median(rel):.3f}')

    # Per-axis magnitude ratio at the 80th percentile of reference motion
    # (avoid the near-zero regime where ratios are unstable)
    def ratio_at_motion(o, r, q=0.5):
        thr = np.quantile(np.abs(r), q)
        mask = np.abs(r) >= thr
        if mask.sum() < 5: return float('nan'), thr
        return float(np.median(np.abs(o[mask]) / np.maximum(np.abs(r[mask]), 1e-9))), float(thr)

    for name, o, r in [
        ('tx',        col('ours_tx'),        col('ref_tx')),
        ('ty',        col('ours_ty'),        col('ref_ty')),
        ('OF_dy_med', col('of_dy_med'),      col('ref_ty')),
        ('OF_dy_mean',col('of_dy_mean'),     col('ref_ty')),
        ('scale_res', col('ours_scale_res'), col('ref_scale_res')),
        ('rot_deg',   col('ours_rot_deg'),   col('ref_rot_deg')),
    ]:
        r_med, thr = ratio_at_motion(o, r)
        print(f'  ratio |our_{name}|/|ref| (where |ref|≥p50={thr:.5f}): '
              f'{r_med:.3f}')

    print(f'  corr(ours, ref): '
          f'tx={corr(col("ours_tx"), col("ref_tx")):.3f}  '
          f'ty={corr(col("ours_ty"), col("ref_ty")):.3f}  '
          f'scale={corr(col("ours_scale_res"), col("ref_scale_res")):.3f}  '
          f'rot={corr(col("ours_rot_deg"), col("ref_rot_deg")):.3f}')
    print(f'  corr(OF_dy_mean, ref_ty)={corr(col("of_dy_mean"), col("ref_ty")):.3f}')
    print(f'  corr(OF_dy_med,  ref_ty)={corr(col("of_dy_med"),  col("ref_ty")):.3f}')

    print(f'  motion-mask ROI area: median={np.median(col("roi_area")):.3f}  '
          f'min={col("roi_area").min():.3f}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clip', required=True)
    p.add_argument('--video', help='path to .mp4 (default: jaad/<clip>.mp4)')
    p.add_argument('--frame-cap', type=int, default=None)
    p.add_argument('--max-dims', nargs='*', type=int, default=[320, 480],
                   help='motion_track.max_width/_height variants to test')
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    video = args.video or f'/mldata/tracking/jaad/video/{args.clip}.mp4'
    if not os.path.isfile(video):
        sys.exit(f'no such video: {video}')

    results = {}
    for max_dim in args.max_dims:
        ypath = make_yaml(PROD_YAML, max_dim)
        t0 = time.time()
        rows, (W, H), (Wref, Href) = run(args.clip, video, ypath,
                                          frame_cap=args.frame_cap)
        dt = time.time() - t0
        label = f'max_dim={max_dim}'
        # Write per-frame csv
        out_csv = OUT_DIR / f'{args.clip}_md{max_dim}.csv'
        with open(out_csv, 'w', newline='') as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
        print(f'\n[{label}] source={W}x{H} ref={Wref}x{Href} '
              f'OF grid={rows[0]["of_gh"]}x{rows[0]["of_gw"]} '
              f'({len(rows)} frames in {dt:.0f}s)  → {out_csv}')
        summarise(label, rows)
        results[label] = rows


if __name__ == '__main__':
    main()
