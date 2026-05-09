"""Capture a (sequence, track_id) from a corpus into a regression-test
trace `.npz` file. After capturing, edit the embedded `meta_json` to add
the `asserts` list — `capture` writes a stub list with the most common
shapes so editing is mostly filling in numbers.

Run:
    python -m bench.trace_library.capture \\
        --corpus bench/data/state_corpus_v15ld_val.npz \\
        --sequence INof_FD_OutFD_Light_FFcam_001 --track-id 42 \\
        --trace-id mu_tp_collapse_occlusion \\
        --issue mu_TP_collapse_during_occlusion \\
        --description "Real TP track, rows 23-26 occluded; head should keep μ_TP > 1.0" \\
        --out bench/trace_library/traces/mu_tp_collapse_occlusion.npz
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os

import numpy as np

from bench.train_state_head_gru import build_input_matrix_no_state, group_rows_by_track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--track-id", type=int, required=True)
    ap.add_argument("--trace-id", required=True,
                    help="short slug used as the trace's filename")
    ap.add_argument("--issue", required=True,
                    help="machine tag of the failure mode tested")
    ap.add_argument("--description", required=True)
    ap.add_argument("--discovered",
                    default=dt.date.today().isoformat())
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rec = np.load(args.corpus, allow_pickle=False)["records"]
    seq = rec["sequence"].astype(str)
    tid = rec["track_id"]

    # Use the same grouping the trainer/runner uses. group_rows_by_track
    # dedups by frame_idx (the corpus emits multiple state-pass rows per
    # frame; the head only sees ONE per frame at runtime). Without this,
    # a "len=149" track captured naively comes out as 170+ rows.
    groups = group_rows_by_track(rec)
    matching = [g for g in groups
                if seq[g[0]] == args.sequence and int(tid[g[0]]) == args.track_id]
    if not matching:
        raise SystemExit(
            f"no group for sequence={args.sequence!r} track_id={args.track_id} "
            f"in {args.corpus}")
    if len(matching) > 1:
        # Multiple track-segments share the (seq, tid) pair (track_id reuse
        # after a kill). Pick the longest segment by default; warn so the
        # caller knows.
        sizes = [len(g) for g in matching]
        idx = int(np.argmax(sizes))
        print(f"  WARNING: {len(matching)} segments share (seq, tid); "
              f"picking the longest (len={sizes[idx]} of {sizes})")
        rows = matching[idx]
    else:
        rows = matching[0]
    n = len(rows)

    inputs   = build_input_matrix_no_state(rec)[rows]
    matched  = (rec["matched"][rows] != 0).astype(bool)
    gt_id    = rec["gt_id_now"][rows].astype(np.int64)
    frame_ix = rec["frame_idx"][rows].astype(np.int64)

    meta = {
        "trace_id":          args.trace_id,
        "issue":              args.issue,
        "description":        args.description,
        "discovered":         args.discovered,
        "source_corpus":      os.path.basename(args.corpus),
        "source_sequence":    args.sequence,
        "source_track_id":    int(args.track_id),
        "n_rows":             int(n),
        "matched_count":      int(matched.sum()),
        "gt_aligned_count":   int((gt_id != -1).sum()),
        # Stub asserts — caller is expected to edit this list. The five
        # shapes below are the ones that have actually mattered so far;
        # delete the ones that don't apply, fill in the ones that do.
        "asserts": [
            {"rows": "all",      "field": "p_TP",            "op": ">=", "value": 0.5,
             "tag": "track-level p_TP should never collapse"},
            {"rows": [0],        "field": "cost_rule_state", "op": "==", "value": "UNCONFIRMED",
             "tag": "first-row state"},
            {"rows": "all",      "field": "mu_TP",           "op": ">=", "value": 0.5,
             "tag": "μ_TP should not collapse during gaps"},
            {"rows": "all",      "field": "cost_rule_state", "op": "==", "value": "TRACKED",
             "tag": "real TP should stay TRACKED throughout"},
            {"rows": "all",      "field": "p_TP",            "op": "<=", "value": 0.2,
             "tag": "(use for FP traces) head should suppress"},
        ],
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out,
             inputs=inputs.astype(np.float32),
             matched=matched,
             gt_id_now=gt_id,
             frame_idx=frame_ix,
             meta_json=json.dumps(meta, indent=2))
    print(f"captured {n} rows from {args.sequence} track_id={args.track_id} -> {args.out}")
    print(f"  matched: {int(matched.sum())}/{n}  GT-aligned: {int((gt_id != -1).sum())}/{n}")
    print(f"  edit meta_json inside the .npz to refine `asserts`")


if __name__ == "__main__":
    main()
