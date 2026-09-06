# Match-level gap diagnostic for the cadence test (evidence, not
# conjecture): for every gap crossing in a bursty variant, classify what
# happened to each GT identity that spans the gap — ID kept / switched /
# missing at re-entry — and measure the GT box displacement across the
# gap. Run on (GT annotation, saved test trackset) pairs produced by
# track.py --track --save-trackset.
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for src.*
import src.trackset as trackset


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u > 0 else 0.0


def match_at(gt, ts, t, min_iou=0.4):
    """GT-id -> (test-id, iou) at time t, greedy by IoU."""
    gto = [o for o in (gt.objects_at_time(t) or [])
           if gt.metadata["classes"][o.cl] == "person"]
    tso = [o for o in (ts.objects_at_time(t) or [])
           if ts.metadata["classes"][o.cl] == "person"]
    if not gto or not tso:
        return {}, gto
    from src.track_test import permissive_iou_matrix
    M = permissive_iou_matrix([g.box for g in gto], [s.box for s in tso])
    out = {}
    used_g, used_s = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(-M, axis=None), M.shape))[0]
    for gi, si in order:
        v = M[gi, si]
        if v < min_iou:
            break
        if gi in used_g or si in used_s:
            continue
        used_g.add(gi); used_s.add(si)
        out[gto[gi].track_id] = (tso[si].track_id, float(v))
    return out, gto


def diagnose(gt_path, trk_path, gap_thresh=0.3):
    gt = trackset.TrackSet(gt_path)
    ts = trackset.TrackSet(trk_path)
    times = sorted(f["frame_time"] for f in gt.frames)
    gaps = [(times[i], times[i+1]) for i in range(len(times)-1)
            if times[i+1] - times[i] > gap_thresh]
    stats = {"kept": 0, "switched": 0, "miss_reentry": 0,
             "recovered_late": 0, "disp": [], "disp_kept": [],
             "disp_lost": []}
    for t0, t1 in gaps:
        pre, gto0 = match_at(gt, ts, t0)
        post, gto1 = match_at(gt, ts, t1)
        g0 = {o.track_id: o for o in (gt.objects_at_time(t0) or [])}
        g1 = {o.track_id: o for o in (gt.objects_at_time(t1) or [])}
        for gid in set(g0) & set(g1):          # identity spans the gap
            b0, b1 = g0[gid].box, g1[gid].box
            c_disp = float(np.hypot((b0[0]+b0[2])/2 - (b1[0]+b1[2])/2,
                                    (b0[1]+b0[3])/2 - (b1[1]+b1[3])/2))
            # displacement in units of own box width (re-associability)
            rel = c_disp / max(1e-6, (b0[2]-b0[0]))
            if gid not in pre:
                continue                       # wasn't tracked pre-gap
            stats["disp"].append(rel)
            if gid in post and post[gid][0] == pre[gid][0]:
                stats["kept"] += 1
                stats["disp_kept"].append(rel)
            elif gid in post:
                stats["switched"] += 1
                stats["disp_lost"].append(rel)
            else:
                # missing at re-entry; late recovery within 3 frames?
                later = [t for t in times if t > t1][:3]
                rec = any(gid in match_at(gt, ts, t)[0] for t in later)
                stats["miss_reentry"] += 1
                stats["recovered_late"] += rec
                stats["disp_lost"].append(rel)
    return len(gaps), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+",
                    help="gt_annotation.json:saved.trk per clip")
    ap.add_argument("--gap-thresh", type=float, default=0.3)
    a = ap.parse_args()
    tot = {"kept": 0, "switched": 0, "miss_reentry": 0, "recovered_late": 0}
    d_all, d_kept, d_lost = [], [], []
    for pair in a.pairs:
        gtp, tkp = pair.split(":")
        n, s = diagnose(gtp, tkp, a.gap_thresh)
        print(f"{gtp.split('/')[-1][:-5]:24s} gaps={n:3d} kept={s['kept']:3d} "
              f"switched={s['switched']:3d} miss@reentry={s['miss_reentry']:3d} "
              f"(late-recovered {s['recovered_late']})")
        for k in tot:
            tot[k] += s[k]
        d_all += s["disp"]; d_kept += s["disp_kept"]; d_lost += s["disp_lost"]
    print(f"\nTOTAL: kept={tot['kept']} switched={tot['switched']} "
          f"miss@reentry={tot['miss_reentry']} "
          f"(of which late-recovered {tot['recovered_late']})")
    if d_kept and d_lost:
        print(f"GT displacement across gap (box-widths): "
              f"kept median {np.median(d_kept):.2f}  "
              f"broken median {np.median(d_lost):.2f}  "
              f"all p90 {np.percentile(d_all, 90):.2f}")


if __name__ == "__main__":
    main()
