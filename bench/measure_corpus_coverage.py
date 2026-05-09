"""Measure whether the state corpus has enough signal for the head to
learn 'long unmatched run on an FP track means low p_TP'.

For each track, compute:
- total length (frames)
- fraction of frames matched (i.e., gt_id_now != -1)
- longest unmatched-run length

Bin tracks by track-level is_TP and report distributions. If FP tracks
are too short OR if their unmatched runs are short, the head won't have
enough examples to learn the temporal pattern.
"""
from __future__ import annotations
import argparse
import numpy as np

from bench.train_state_head_gru import group_rows_by_track


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    args = p.parse_args()

    rec = np.load(args.corpus, allow_pickle=False)["records"]
    print(f"corpus rows: {len(rec)}")

    groups = group_rows_by_track(rec)
    print(f"tracks: {len(groups)}")

    # Per-track stats
    is_TP = []
    lens = []
    matched_frac = []
    unmatched_run_max = []
    for rows in groups:
        # Track is TP if any row has gt_id_now != -1
        tp = bool(np.any(rec["gt_id_now"][rows] != -1))
        is_TP.append(tp)
        lens.append(len(rows))
        matched = (rec["gt_id_now"][rows] != -1).astype(np.int32)
        matched_frac.append(matched.mean())
        # Longest unmatched run
        max_run = 0
        cur = 0
        for m in matched:
            if m == 0:
                cur += 1
                if cur > max_run: max_run = cur
            else:
                cur = 0
        unmatched_run_max.append(max_run)

    is_TP = np.array(is_TP)
    lens = np.array(lens)
    matched_frac = np.array(matched_frac)
    unmatched_run_max = np.array(unmatched_run_max)

    print(f"\noverall: TP={is_TP.sum()} ({is_TP.mean()*100:.1f}%)  "
          f"FP={(~is_TP).sum()} ({(~is_TP).mean()*100:.1f}%)")

    for label, mask in [("TP tracks", is_TP), ("FP tracks", ~is_TP)]:
        L = lens[mask]
        F = matched_frac[mask]
        U = unmatched_run_max[mask]
        print(f"\n=== {label} (n={mask.sum()}) ===")
        print(f"  length:        median={np.median(L):.0f}  mean={L.mean():.1f}  "
              f"p90={np.percentile(L, 90):.0f}  max={L.max()}")
        print(f"  matched_frac:  median={np.median(F):.2f}  mean={F.mean():.2f}  "
              f"p90={np.percentile(F, 90):.2f}")
        print(f"  longest unmatched run:  median={np.median(U):.0f}  mean={U.mean():.1f}  "
              f"p90={np.percentile(U, 90):.0f}  max={U.max()}")

        # FP tracks with long unmatched runs are the diagnostic signal
        if not label.startswith("TP"):
            for thr in [5, 10, 20, 40]:
                n = int((U >= thr).sum())
                pct = n / max(1, mask.sum()) * 100
                print(f"  FP tracks with unmatched_run >= {thr}: {n} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
