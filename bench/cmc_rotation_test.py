"""Isolate whether the tx-over / ty-under bias is model-side (aspect /
numerics) or data-side (NVOF / scene asymmetry).

Run modes:
  A: native 1920x1080 input (baseline — alpha != 1)
  B: 1080x1080 center crop  (alpha = 1, no aspect math kicks in)
  C: 1080x1080 center crop, rotated 90° CW (swaps scene x/y)

For each: compute our CMC and OpenCV-ORB+RANSAC reference, report the
tx/ty ratios. Interpretation:

  asymmetry direction:
    A: ty under,  tx over   (observed)
    B: if persists → not the leftover aspect math; numerics or NVOF
    B: if disappears → still an aspect-math issue
    C: if asymmetry follows data → swaps to ty over,  tx under
    C: if asymmetry follows model → stays at ty under,  tx over
"""
import argparse, math, os, sys, time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.cmc_compare import opencv_cmc, decompose
import ubon_pycstuff.ubon_pycstuff as upyc


PROD_YAML = '/mldata/config/track/trackers/uc_v11.yaml'
REF_MAX_DIM = int(os.environ.get('CMC_REF_MAX_DIM', '960'))


def square_crop(bgr):
    H, W = bgr.shape[:2]
    s = min(H, W)
    y0 = (H - s) // 2
    x0 = (W - s) // 2
    return bgr[y0:y0+s, x0:x0+s]


def transform_frame(bgr, mode):
    if mode == 'A':
        return bgr
    cropped = square_crop(bgr)
    if mode == 'B':
        return cropped
    if mode == 'C':
        return cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
    raise ValueError(mode)


def step_ours(mt, rgb):
    img = upyc.c_image.from_numpy(rgb)
    mt.add_frame(img)
    mt.set_roi(mt.get_roi())
    out = mt.get_cmc_transform()
    tx, ty, p, q = float(out[0]), float(out[1]), float(out[2]), float(out[3])
    return dict(tx=tx, ty=ty, p=p, q=q)


def run_mode(video_path, mode, frame_cap=None):
    print(f'\n=== mode {mode} ===')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(video_path)
    mt = upyc.c_motion_tracker(PROD_YAML)

    rows = []
    prev_gray = None
    n = 0
    while True:
        ok, bgr = cap.read()
        if not ok: break
        n += 1
        bgr_t = transform_frame(bgr, mode)
        rgb = cv2.cvtColor(bgr_t, cv2.COLOR_BGR2RGB)
        H, W = bgr_t.shape[:2]
        ours = step_ours(mt, rgb)
        # OpenCV reference works on a downsampled grayscale
        s = REF_MAX_DIM / max(W, H)
        Wref, Href = max(1, int(round(W * s))), max(1, int(round(H * s)))
        small = cv2.resize(bgr_t, (Wref, Href), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        ref = opencv_cmc(prev_gray, gray) if prev_gray is not None else None
        prev_gray = gray

        if ref and ref['n_inliers'] >= 15:
            ours_dec = decompose(ours)
            ref_dec = decompose(ref)
            rows.append((ours_dec, ref_dec, ref['n_inliers']))
        if frame_cap and n >= frame_cap:
            break
    cap.release()
    if not rows:
        return None

    ours_t = np.array([(r[0]['tx'], r[0]['ty']) for r in rows])
    ref_t  = np.array([(r[1]['tx'], r[1]['ty']) for r in rows])
    ours_s = np.array([r[0]['scale_residual'] for r in rows])
    ref_s  = np.array([r[1]['scale_residual'] for r in rows])
    ours_r = np.array([r[0]['rot_deg'] for r in rows])
    ref_r  = np.array([r[1]['rot_deg'] for r in rows])

    def ratio_at(o, r, q=0.5):
        thr = np.quantile(np.abs(r), q)
        m = np.abs(r) >= thr
        if m.sum() < 5: return float('nan'), thr
        return float(np.median(np.abs(o[m]) / np.maximum(np.abs(r[m]), 1e-9))), float(thr)

    def corr(a, b):
        if a.std() < 1e-9 or b.std() < 1e-9: return float('nan')
        return float(np.corrcoef(a, b)[0, 1])

    print(f'  n_valid    = {len(rows)} of {n}')
    print(f'  shape      = {H}x{W}  (W>H means landscape)')
    print(f'  ref translation median |·| = ({np.median(np.abs(ref_t[:,0])):.4f}, {np.median(np.abs(ref_t[:,1])):.4f})')
    print(f'  our translation median |·| = ({np.median(np.abs(ours_t[:,0])):.4f}, {np.median(np.abs(ours_t[:,1])):.4f})')
    r_tx, thr_tx = ratio_at(ours_t[:,0], ref_t[:,0])
    r_ty, thr_ty = ratio_at(ours_t[:,1], ref_t[:,1])
    r_sc, _      = ratio_at(ours_s, ref_s)
    r_ro, _      = ratio_at(ours_r, ref_r)
    print(f'  ratio |our_tx|/|ref|       = {r_tx:.3f}     (|ref|≥p50={thr_tx:.4f})')
    print(f'  ratio |our_ty|/|ref|       = {r_ty:.3f}     (|ref|≥p50={thr_ty:.4f})')
    print(f'  ratio |scale_res|          = {r_sc:.3f}')
    print(f'  ratio |rot_deg|            = {r_ro:.3f}')
    print(f'  corr(tx, ty, scale, rot)   = ({corr(ours_t[:,0],ref_t[:,0]):.3f}, '
          f'{corr(ours_t[:,1],ref_t[:,1]):.3f}, '
          f'{corr(ours_s, ref_s):.3f}, {corr(ours_r, ref_r):.3f})')
    return dict(mode=mode, r_tx=r_tx, r_ty=r_ty, r_sc=r_sc, r_ro=r_ro,
                W=W, H=H)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', default='/mldata/tracking/mot/video/MOT17-13.mp4')
    p.add_argument('--frame-cap', type=int, default=400,
                   help='cap frames to keep run fast (MOT17-13 is ~750)')
    args = p.parse_args()

    rs = []
    for mode in ('A', 'B', 'C'):
        r = run_mode(args.video, mode, args.frame_cap)
        if r: rs.append(r)

    print('\n=== summary ===')
    print(f'{"mode":4s}  {"shape":>11s}  {"r_tx":>6s}  {"r_ty":>6s}  {"r_scale":>7s}  {"r_rot":>7s}')
    for r in rs:
        print(f'  {r["mode"]:2s}  {r["W"]}x{r["H"]:<5}  {r["r_tx"]:6.3f}  {r["r_ty"]:6.3f}  '
              f'{r["r_sc"]:7.3f}  {r["r_ro"]:7.3f}')

    print('\nInterpretation:')
    print('  A->B: if tx>1 / ty<1 PERSISTS in B (square, α=1) → not aspect-math; numerics or NVOF')
    print('  B->C: if pattern FLIPS (ty>1 / tx<1) → bias follows the SCENE/DATA')
    print('  B->C: if pattern STAYS (tx>1 / ty<1) → bias follows the MODEL/numerics')


if __name__ == '__main__':
    main()
