"""Determine the right new_track_thr from val data.

For each track in the corpus, the first-frame detection confidence is
the value the C tracker gates against `new_track_thr` at creation. So
we can simulate the effect of any NTT without rerunning the tracker:

  - Tracks with first_conf > NTT would be created
  - Tracks with first_conf <= NTT would not

Sweep NTT and report, at each threshold:
  - TP track recall  = fraction of TP tracks (any-frame GT-aligned) we'd still create
  - FP tracks created absolute count
  - Track-level precision  = TP_created / (TP_created + FP_created)

The knee of the curve — where precision climbs fast but recall is still
near 1 — is the principled NTT choice.

Run:
    python -m bench.measure_ntt_curve \\
        --corpus bench/data/state_corpus_v15ld_val.npz
"""
from __future__ import annotations
import argparse
import numpy as np

from bench.train_state_head_gru import group_rows_by_track


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.77,0.85")
    args = p.parse_args()

    rec = np.load(args.corpus, allow_pickle=False)["records"]
    print(f"corpus rows: {len(rec)}", flush=True)

    groups = group_rows_by_track(rec)
    print(f"tracks: {len(groups)}", flush=True)

    # Per-track first-frame det_conf and is_TP (any GT alignment).
    first_conf = np.zeros(len(groups), dtype=np.float32)
    is_TP     = np.zeros(len(groups), dtype=np.bool_)
    for i, rows in enumerate(groups):
        # rows are sorted by frame_idx within the track; first row is creation.
        first_conf[i] = rec["det_conf"][rows[0]]
        is_TP[i] = bool(np.any(rec["gt_id_now"][rows] != -1))

    n_TP = int(is_TP.sum())
    n_FP = int((~is_TP).sum())
    print(f"\ntotal: {n_TP} TP tracks, {n_FP} FP tracks  (TP rate={n_TP/(n_TP+n_FP):.1%})")

    # Distribution of first_conf, by class
    print("\nfirst_conf distribution:")
    print(f"  TP: median={np.median(first_conf[is_TP]):.3f}  "
          f"p10={np.percentile(first_conf[is_TP], 10):.3f}  "
          f"p25={np.percentile(first_conf[is_TP], 25):.3f}  "
          f"min={first_conf[is_TP].min():.3f}")
    print(f"  FP: median={np.median(first_conf[~is_TP]):.3f}  "
          f"p75={np.percentile(first_conf[~is_TP], 75):.3f}  "
          f"p90={np.percentile(first_conf[~is_TP], 90):.3f}  "
          f"max={first_conf[~is_TP].max():.3f}")

    # Sweep
    thrs = [float(x) for x in args.thresholds.split(",")]
    print(f"\n{'NTT':>6}  {'TP_kept':>8}  {'TP_recall':>9}  "
          f"{'FP_kept':>8}  {'FP_drop':>8}  {'precision':>9}  {'F1':>6}")
    for thr in thrs:
        keep = first_conf > thr
        tp_kept = int((keep & is_TP).sum())
        fp_kept = int((keep & ~is_TP).sum())
        recall = tp_kept / max(1, n_TP)
        prec = tp_kept / max(1, tp_kept + fp_kept)
        fp_drop = (n_FP - fp_kept) / max(1, n_FP)
        f1 = 2 * prec * recall / max(1e-6, prec + recall)
        print(f"  {thr:.2f}  {tp_kept:8d}  {recall:9.3f}  "
              f"{fp_kept:8d}  {fp_drop:8.3f}  {prec:9.3f}  {f1:6.3f}")


if __name__ == "__main__":
    main()
