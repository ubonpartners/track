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
    args = ap.parse_args()

    scenes = _scenes_for_split(args.analysis_yaml, args.split)
    if not scenes:
        raise SystemExit(
            f"No scenes labeled split={args.split!r} in {args.analysis_yaml}"
        )

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
        scene_id = hash(seq_name) & 0xFFFFFFFF
        scene_id_to_name[scene_id] = seq_name
        record_chunks.append(recs)
        label_chunks.append(labs)
        scene_id_chunks.append(np.full(int(recs.shape[0]), scene_id, dtype=np.uint32))
        found += 1
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
