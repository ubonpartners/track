"""
Aggregate per-sequence pair-log NPZs into split-level training datasets.

The pair_logger analysis module writes one NPZ per sequence under
<output_root>/pair_log/<seq>.npz. This script globs those files for the
chosen split (train/val/test, taken from the analysis YAML's dataset
block), concatenates the records, and writes a single
pairs_<split>.npz under track/bench/data/.

Also computes per-feature mean/std on the train split and writes
feature_norm.json — used by the trainer in Phase 2.

Usage:
    python -m bench.build_pair_dataset \\
        --pair-log-dir runs/track_analysis/pair_log_v1/pair_log \\
        --analysis-yaml bench/pair_log_config.yaml \\
        --split train \\
        --out bench/data/pairs_train.npz
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import stuff

from src.analysis.pair_log_schema import (
    PAIR_LOG_DTYPE,
    PAIR_LOG_FEATURE_NAMES,
    PAIR_LOG_VERSION,
    record_size_bytes,
)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "unnamed"


def _scenes_for_split(analysis_yaml: str, split: str) -> List[str]:
    cfg = stuff.load_dictionary(analysis_yaml)
    dataset = cfg.get("dataset") or cfg.get("datasets") or {}
    if isinstance(dataset, list):
        items = []
        for item in dataset:
            if isinstance(item, dict):
                items.append((item.get("name", ""), item))
    else:
        items = list(dataset.items())
    out: List[str] = []
    for name, entry in items:
        if not isinstance(entry, dict):
            continue
        s = str(entry.get("split", "")).lower().strip()
        if split == "all" or s == split:
            out.append(str(name))
    return out


def _features_matrix(records: np.ndarray) -> np.ndarray:
    """Stack the configured feature columns into an (N, F) float32 matrix."""
    n = int(records.shape[0])
    f = len(PAIR_LOG_FEATURE_NAMES)
    out = np.empty((n, f), dtype=np.float32)
    for j, name in enumerate(PAIR_LOG_FEATURE_NAMES):
        col = records[name].astype(np.float32, copy=False)
        out[:, j] = col
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="bench.build_pair_dataset")
    ap.add_argument("--pair-log-dir", required=True, help="dir holding <seq>.npz files")
    ap.add_argument("--analysis-yaml", required=True, help="YAML with dataset+splits")
    ap.add_argument("--split", required=True, choices=["train", "val", "test", "all"])
    ap.add_argument("--out", required=True, help="output pairs_<split>.npz path")
    ap.add_argument(
        "--norm-out",
        default=None,
        help="(train split only) path for feature mean/std JSON; "
             "default: <out_dir>/feature_norm.json",
    )
    ap.add_argument("--allow-missing-scenes", action="store_true",
                    help="Don't error if some split scenes lack pair-log NPZs")
    ap.add_argument("--comment", required=True,
                    help="Free-form note recorded in the output .npz's "
                         "_meta. Required so future readers can recover "
                         "what this dataset is.")
    ap.add_argument("--delta-filter", type=float, default=None,
                    help="Cheap delta filter — mirrors the C-side runtime "
                         "filter (utrack.match_cheap_filter_delta). Two "
                         "gates applied together per row:\n"
                         "  (a) per-event: drop rows whose pre_thr_score is "
                         "more than this much below the (frame_time, "
                         "det_index) group's top score.\n"
                         "  (b) per-row threshold-aware: drop rows whose "
                         "pre_thr_score is more than this much below the "
                         "pass's match_thr (looked up from pass_id + "
                         "tracker yaml referenced by --analysis-yaml).\n"
                         "Focuses training on pairs the matcher could "
                         "plausibly accept and/or pick. None (default) = "
                         "no filter; e.g. 0.5 = keep top + within 0.5 of "
                         "it AND within 0.5 of match_thr.")
    args = ap.parse_args()

    from bench._pipeline_checks import (
        assert_file_exists, assert_dir_has_files,
    )
    assert_file_exists(args.analysis_yaml, "--analysis-yaml")
    # Pair-log dir must hold at least 30 npz — anything lower means
    # the previous stage failed silently. The full pair-log dataset
    # is ~178 clips so 30 is a very generous floor.
    assert_dir_has_files(args.pair_log_dir, ext=".npz", min_count=30,
                         label="--pair-log-dir")

    scenes = _scenes_for_split(args.analysis_yaml, args.split)
    if not scenes:
        raise SystemExit(
            f"No scenes labeled split={args.split!r} in {args.analysis_yaml}"
        )

    # Per-pass match_thr lookup (mirrors utrack.c defaults; pair-log pass_id
    # values map to these). Used by the threshold-aware delta filter so the
    # training distribution matches the C runtime when
    # utrack.match_cheap_filter_delta > 0.
    match_thr_by_pass = (0.66, 0.225, 0.022)
    if args.delta_filter is not None:
        # Honour overrides from the analysis YAML + (transitively) its
        # referenced tracker_config, so this filter sees the same
        # thresholds the C runtime actually used during pair-log gen.
        try:
            ana = stuff.load_dictionary(args.analysis_yaml)
            tcfg_path = ana.get("tracker_config")
            ovr = (ana.get("tracker_config_overrides") or {}).get("utrack") or {}
            ut_yaml = {}
            if tcfg_path and os.path.isfile(tcfg_path):
                base = stuff.load_dictionary(tcfg_path) or {}
                ut_yaml = base.get("utrack") or {}
            for k in ("match_thr_initial", "match_thr_high", "match_thr_low"):
                if k in ovr:
                    ut_yaml[k] = ovr[k]
            if all(k in ut_yaml for k in ("match_thr_initial", "match_thr_high", "match_thr_low")):
                match_thr_by_pass = (
                    float(ut_yaml["match_thr_initial"]),
                    float(ut_yaml["match_thr_high"]),
                    float(ut_yaml["match_thr_low"]),
                )
        except Exception as e:
            print(f"  [warn] match_thr lookup from analysis yaml failed: {e};"
                  f" using defaults {match_thr_by_pass}")
        print(f"  delta-filter: per-pass match_thr = {match_thr_by_pass}")

    record_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []
    scene_id_chunks: List[np.ndarray] = []
    scene_id_to_name: Dict[int, str] = {}

    found = 0
    for seq_name in sorted(scenes):
        seq_safe = _safe_name(seq_name)
        path = os.path.join(args.pair_log_dir, f"{seq_safe}.npz")
        if not os.path.isfile(path):
            if args.allow_missing_scenes:
                print(f"  [skip] missing: {path}")
                continue
            raise SystemExit(
                f"Missing pair-log NPZ for scene {seq_name!r} at {path}. "
                f"Re-run the pair_logger analysis or pass --allow-missing-scenes."
            )
        with np.load(path, allow_pickle=True) as z:
            schema_version = int(z["schema_version"])
            if schema_version != PAIR_LOG_VERSION:
                raise SystemExit(
                    f"Schema version mismatch in {path}: file={schema_version} "
                    f"current={PAIR_LOG_VERSION}. Regenerate the pair log."
                )
            recs = z["records"]
            labs = z["labels"]
        if recs.dtype != PAIR_LOG_DTYPE:
            # Could happen if PAIR_LOG_FEATURE_NAMES changed between runs
            # but the on-disk records still encode the current dtype.
            try:
                recs = recs.astype(PAIR_LOG_DTYPE, copy=False)
            except Exception as e:
                raise SystemExit(f"Dtype mismatch in {path}: {e}")

        # Cheap delta filter — mirrors the C-side
        # utrack.match_cheap_filter_delta runtime gate. Two filters
        # applied jointly per row:
        #   (a) per-event: drop rows >delta below the (frame_time,
        #       det_index) group's top pre_thr_score (out-of-contention
        #       for the matcher's ranking)
        #   (b) per-row: drop rows >delta below the pass's match_thr
        #       (so far below the threshold that no plausible NN
        #       residual could rescue them)
        # Order matches the C runtime: (b) is the early-reject, applied
        # first; (a) operates on the remaining rows so the per-event
        # top reflects only "potentially acceptable" candidates.
        n_pre_filter = int(recs.shape[0])
        pos_pre_filter = int(labs.sum())
        if args.delta_filter is not None and n_pre_filter > 0:
            d = float(args.delta_filter)
            pass_id = recs["pass_id"]
            # Vectorised per-pass match_thr lookup (pass_id is uint8 in
            # the schema, clipped to range).
            thr_arr = np.asarray(match_thr_by_pass, dtype=np.float64)
            pid_idx = np.clip(pass_id.astype(np.int32), 0, len(thr_arr) - 1)
            row_thr = thr_arr[pid_idx]
            pre = recs["pre_thr_score"].astype(np.float64)

            keep_thr = pre >= (row_thr - d)   # gate (b)

            recs_b = recs[keep_thr]; labs_b = labs[keep_thr]

            if recs_b.shape[0] > 0:
                ft = recs_b["frame_time"]; di = recs_b["det_index"]
                order = np.lexsort((di, ft))
                recs_s = recs_b[order]; labs_s = labs_b[order]
                # Per-row row_thr after sorting (needed for near-thr rescue)
                pid_s = np.clip(recs_s["pass_id"].astype(np.int32), 0, len(thr_arr) - 1)
                row_thr_s = thr_arr[pid_s]
                pre_s_all = recs_s["pre_thr_score"].astype(np.float64)
                near_thr  = np.abs(pre_s_all - row_thr_s) < d
                ft_s = recs_s["frame_time"]; di_s = recs_s["det_index"]
                same = (ft_s[1:] == ft_s[:-1]) & (di_s[1:] == di_s[:-1])
                starts = np.concatenate(([0], np.where(~same)[0] + 1, [len(recs_s)]))
                keep_top = np.zeros(len(recs_s), dtype=bool)
                for k in range(len(starts) - 1):
                    s, e = starts[k], starts[k+1]
                    pre_se = pre_s_all[s:e]
                    top = pre_se.max()
                    # Gate (a) per-event: keep if within delta of top.
                    # Gate (a-bypass): also keep if near match_thr — NN
                    # can flip accept/reject for those (mirrors the C
                    # runtime near_thr exception in utrack_match.c).
                    keep_top[s:e] = (pre_se >= (top - d)) | near_thr[s:e]
                recs = recs_s[keep_top]
                labs = labs_s[keep_top]
            else:
                recs = recs_b
                labs = labs_b

        scene_id = hash(seq_name) & 0xFFFFFFFF
        scene_id_to_name[scene_id] = seq_name
        record_chunks.append(recs)
        label_chunks.append(labs)
        scene_id_chunks.append(np.full(int(recs.shape[0]), scene_id, dtype=np.uint32))
        found += 1
        if args.delta_filter is not None:
            n_post = int(recs.shape[0])
            pos_post = int(labs.sum())
            kept_pct = 100.0 * n_post / max(n_pre_filter, 1)
            pos_rate_pre = 100.0 * pos_pre_filter / max(n_pre_filter, 1)
            pos_rate_post = 100.0 * pos_post / max(n_post, 1)
            print(f"  [keep] {seq_name:50s} n={n_post:8d}/{n_pre_filter:<8d} "
                  f"({kept_pct:5.1f}%)  pos%: {pos_rate_pre:4.1f}→{pos_rate_post:4.1f}")
        else:
            print(f"  [keep] {seq_name:50s} n={int(recs.shape[0]):8d} pos={int(labs.sum()):6d}")

    if found == 0:
        raise SystemExit("No pair-log NPZs found; nothing to build.")

    records = (np.concatenate(record_chunks) if record_chunks
               else np.zeros(0, dtype=PAIR_LOG_DTYPE))
    labels = (np.concatenate(label_chunks) if label_chunks
              else np.zeros(0, dtype=np.uint8))
    scene_ids = (np.concatenate(scene_id_chunks) if scene_id_chunks
                 else np.zeros(0, dtype=np.uint32))

    # Stable sort by (scene_id, frame_time, track_id, det_x0) so re-runs
    # produce byte-identical files (Phase-1 gate).
    order = np.lexsort((records["det_x0"], records["track_id"],
                         records["frame_time"], scene_ids))
    records = records[order]
    labels = labels[order]
    scene_ids = scene_ids[order]

    features = _features_matrix(records)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    tmp = args.out + ".building.npz"
    from bench._artefact_meta import make_pt_meta, save_npz_with_meta
    meta = make_pt_meta(
        artefact_kind="pair_dataset",
        args=args,
        hparams={
            "split": args.split,
            "n_pairs": int(records.shape[0]),
            "n_pos": int(labels.sum()),
            "n_scenes": int(found),
            "schema_version": int(PAIR_LOG_VERSION),
            "feature_names": list(PAIR_LOG_FEATURE_NAMES),
        },
        dataset_info={
            "pair_log_dir": args.pair_log_dir,
            "analysis_yaml": args.analysis_yaml,
        },
        comment=args.comment,
    )
    save_npz_with_meta(
        tmp, meta,
        records=records,
        features=features,
        labels=labels,
        scene_ids=scene_ids,
        feature_names=np.array(PAIR_LOG_FEATURE_NAMES, dtype=object),
        scene_id_to_name=np.array(
            sorted(scene_id_to_name.items()), dtype=object
        ),
        schema_version=np.int64(PAIR_LOG_VERSION),
    )
    os.replace(tmp, args.out)

    n = int(records.shape[0])
    pos = int(labels.sum())
    print(f"\nWrote {args.out}: {n} pairs from {found} scenes, "
          f"pos={pos} ({100.0 * pos / max(1, n):.2f}%)")

    # Train-split feature stats for the trainer to use.
    if args.split == "train":
        norm_out = args.norm_out or os.path.join(out_dir, "feature_norm.json")
        if features.shape[0] == 0:
            print(f"(train split is empty; skipping {norm_out})")
        else:
            mean = features.mean(axis=0)
            std = features.std(axis=0)
            std[std < 1e-6] = 1.0
            payload = {
                "schema_version": PAIR_LOG_VERSION,
                "feature_names": list(PAIR_LOG_FEATURE_NAMES),
                "mean": mean.tolist(),
                "std": std.tolist(),
                "n_pairs": n,
            }
            with open(norm_out, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Wrote {norm_out}")


if __name__ == "__main__":
    main()
