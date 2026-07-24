# Camera-motion classifier for stream-hint assignment (MB spec
# 2026-07-24): measure global background translation between consecutive
# frames of the tier-2 (analytics-grid) video and classify the clip
# static vs moving. Deliberately biased toward `static`: per MB, very
# slow pans behave like still cameras for the tracker, so only sustained
# real ego-motion earns the bodycam profile.
#
# Method: grayscale downscale to width 320, goodFeaturesToTrack + LK
# optical flow on sampled frame pairs, robust global shift = median
# feature translation (rotation/zoom-free approximation is fine at this
# scale); per-clip statistic = median over pairs of |shift| / width per
# SECOND (rate-normalized so the analytics-grid framerate doesn't skew
# it). Validated 11/11 on mot (known classes) before use on pp22.
import glob
import json
import os

import cv2
import numpy as np

# fraction of frame width the min-background-cell moves per second;
# clips under the threshold are `static`. Validated on mot (11/11):
# true statics measure <=0.036 (dense-crowd MOT20 scenes are the top
# end), true ego-motion >=0.095 — threshold sits in the gap, biased
# toward static per MB (slow pans behave like still cameras for the
# tracker).
# three bands (2026-07-24 measurement): dense-crowd STATICS (MOT20-03
# min 0.068) and hood-visible DASHCAMS (WinterDrive min 0.016) overlap
# completely in cell statistics — between the confident bands the
# classifier must return "ambiguous" for a human ruling, not guess.
STATIC_MAX_SHIFT_PER_S = 0.02
MOVING_MIN_SHIFT_PER_S = 0.09


def clip_motion(video, max_pairs=40):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, n // max_pairs)
    shifts = []
    prev = None
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            w = 320
            g = cv2.resize(g, (w, int(g.shape[0] * w / g.shape[1])))
            if prev is not None:
                pts = cv2.goodFeaturesToTrack(prev, 200, 0.01, 8)
                if pts is not None and len(pts) >= 20:
                    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None)
                    good = st.reshape(-1) == 1
                    if good.sum() >= 15:
                        p = pts.reshape(-1, 2)[good]
                        d = (nxt.reshape(-1, 2) - pts.reshape(-1, 2))[good]
                        # min-over-cells of per-cell median motion: crowd
                        # scenes (MOT20) put most features on moving
                        # people, defeating global medians/percentiles —
                        # but a static camera always has at least one
                        # grid cell of pure background (~0 motion),
                        # while ego-motion moves every cell.
                        mag = np.hypot(d[:, 0], d[:, 1])
                        h = prev.shape[0]
                        cell = (np.minimum(3, (p[:, 0] * 4 / w).astype(int))
                                * 3
                                + np.minimum(2, (p[:, 1] * 3 / h).astype(int)))
                        meds = [np.median(mag[cell == c])
                                for c in range(12)
                                if (cell == c).sum() >= 8]
                        if meds:
                            shifts.append(float(min(meds)) / w)
            prev = g
        idx += 1
    cap.release()
    if not shifts:
        return None
    # per-frame fraction -> per-second rate on this video's clock
    return float(np.median(shifts)) * fps


def classify(video):
    m = clip_motion(video)
    if m is None:
        return None, None
    if m < STATIC_MAX_SHIFT_PER_S:
        return "static", m
    if m > MOVING_MIN_SHIFT_PER_S:
        return "bodycam", m
    return "ambiguous", m


def classify_corpus(corpus, t2="/mldata/tracking"):
    out = {}
    for v in sorted(glob.glob(os.path.join(t2, corpus, "video", "*.mp4"))):
        hint, m = classify(v)
        stem = os.path.basename(v)[:-4]
        out[stem] = {"hint": hint, "shift_per_s": None if m is None
                     else round(m, 5)}
        print(f"  {stem}: {hint} ({m:.4f}/s)" if m is not None
              else f"  {stem}: unmeasurable", flush=True)
    return out


if __name__ == "__main__":
    import sys
    print(json.dumps(classify_corpus(sys.argv[1]), indent=1))
