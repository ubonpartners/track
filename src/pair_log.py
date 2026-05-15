"""Pair-log emission: run the C tracker on a dataset and label per-pair
trace records against GT, writing one ``<seq>.npz`` per clip for the NN
training pipeline.

History: this file replaces the old ``src/analysis/`` package, which once
supported a Python-side variant-comparison framework with seven evaluator
modules (detection, optical_flow, kalman, reid, match_cost, roi_skip,
pair_logger). After the C runtime became canonical, every live analysis
yaml enabled ``pair_logger`` only — the other six modules were dormant
and the variant framework was no-op. This module preserves the surviving
pieces in one place:

  * ``run_track_analysis(config)`` — orchestrates per-clip parallel
    tracker runs with UBTRK2 caching, then invokes the pair-logger
    evaluator on each clip's trace.
  * ``evaluate_pair_logger(sequence_ctx, module_params)`` — labels each
    (track, det) pair via best-IoU GT alignment and writes the NPZ.
  * Small IoU / GT-objects helpers re-used by ``ml/data_prep/build_state_corpus.py``.

Public API is exposed at module scope; the older variant-keyed summary
shape and ``compare.{json,md}`` outputs are gone.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import time
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import stuff

import src.trackset as ts
from src.pair_log_schema import (
    PAIR_LOG_DTYPE,
    PAIR_LOG_FEATURE_NAMES,
    PAIR_LOG_MAGIC,
    PAIR_LOG_VERSION,
    decode_records,
    record_size_bytes,
)


# ----------------------------------------------------------------------
# Path safety + JSON IO helpers
# ----------------------------------------------------------------------
def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "unnamed"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    try:
        as_float = float(value)
        return as_float if math.isfinite(as_float) else None
    except Exception:
        return str(value)


def _write_json(path: str, data: Dict[str, Any]) -> None:
    stuff.makedir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, indent=2, sort_keys=True)


def _write_text(path: str, text: str) -> None:
    stuff.makedir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ----------------------------------------------------------------------
# Dataset yaml normalisation
# ----------------------------------------------------------------------
def _normalise_dataset(dataset_cfg: Any) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if isinstance(dataset_cfg, dict):
        for key, value in dataset_cfg.items():
            if isinstance(value, str):
                entries.append({"name": key, "trackset": value})
            elif isinstance(value, dict):
                item = dict(value)
                item.setdefault("name", key)
                if "trackset" not in item and "path" in item:
                    item["trackset"] = item["path"]
                if "trackset" not in item and "ds_path" in item:
                    item["trackset"] = item["ds_path"]
                entries.append(item)
    elif isinstance(dataset_cfg, list):
        for i, item in enumerate(dataset_cfg):
            if isinstance(item, str):
                entries.append({"name": stuff.name_from_file(item), "trackset": item})
            elif isinstance(item, dict):
                out = dict(item)
                if "trackset" not in out and "path" in out:
                    out["trackset"] = out["path"]
                if "trackset" not in out and "ds_path" in out:
                    out["trackset"] = out["ds_path"]
                if "name" not in out:
                    out["name"] = out.get("trackset", f"seq_{i}")
                entries.append(out)
    for entry in entries:
        if "trackset" not in entry and "path" in entry:
            entry["trackset"] = entry["path"]
        if "trackset" not in entry and "ds_path" in entry:
            entry["trackset"] = entry["ds_path"]
        if "trackset" not in entry:
            raise ValueError(f"Dataset entry missing trackset: {entry}")
        if "name" not in entry:
            entry["name"] = stuff.name_from_file(entry["trackset"])
    return entries


def _filter_dataset_by_split(
    dataset: List[Dict[str, Any]], split_mode: str
) -> List[Dict[str, Any]]:
    split_mode = str(split_mode or "train").lower().strip()
    if split_mode not in {"train", "val", "test", "all"}:
        raise ValueError("dataset_split must be one of: train, val, test, all")
    if split_mode == "all":
        return dataset

    entries_with_split = 0
    filtered: List[Dict[str, Any]] = []
    for entry in dataset:
        split = entry.get("split")
        if split is None:
            continue
        entries_with_split += 1
        if str(split).lower() == split_mode:
            filtered.append(entry)

    # If no entries are labeled with split, keep old behavior and use all.
    if entries_with_split == 0:
        return dataset
    return filtered


# ----------------------------------------------------------------------
# IoU + GT helpers — kept here because ml/data_prep/build_state_corpus.py imports
# them (best_iou_match, gt_objects_at_time_class) and evaluate_pair_logger
# uses them too.
# ----------------------------------------------------------------------
def _to_float_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        out = float(v)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _det_box(det: Dict[str, Any]) -> Optional[List[float]]:
    box = det.get("box")
    if not isinstance(box, list) or len(box) != 4:
        return None
    vals = [_to_float_or_none(v) for v in box]
    if any(v is None for v in vals):
        return None
    return [float(v) for v in vals]


def _class_name(class_names: Sequence[str], class_id: Any) -> Optional[str]:
    try:
        idx = int(class_id)
    except Exception:
        return None
    if idx < 0 or idx >= len(class_names):
        return None
    return str(class_names[idx])


def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    return float(stuff.box_iou(list(a), list(b)))


def best_iou_match(
    box: Sequence[float], gt_objects: Sequence[Any]
) -> Tuple[Optional[Any], float]:
    """Find the GT object with highest IoU against `box`. Returns (gt, iou)."""
    best = None
    best_iou = 0.0
    for g in gt_objects:
        iou = _box_iou(box, g.box)
        if iou > best_iou:
            best_iou = iou
            best = g
    return best, best_iou


def _cache_map(sequence_ctx: Dict[str, Any], name: str) -> Dict[Any, Any]:
    cache = sequence_ctx.get("_analysis_cache")
    if not isinstance(cache, dict):
        cache = {}
        sequence_ctx["_analysis_cache"] = cache
    slot = cache.get(name)
    if not isinstance(slot, dict):
        slot = {}
        cache[name] = slot
    return slot


def _gt_objects_at_time(sequence_ctx: Dict[str, Any], t: float) -> List[Any]:
    cache = _cache_map(sequence_ctx, "gt_objects_at_time")
    key = float(t)
    if key not in cache:
        gt = sequence_ctx["gt_trackset"]
        cache[key] = gt.objects_at_time(key) or []
    return cache[key]


def gt_objects_at_time_class(
    sequence_ctx: Dict[str, Any], t: float, class_name: str
) -> List[Any]:
    """Return GT objects at time `t` filtered by class name. Cached per sequence."""
    cache = _cache_map(sequence_ctx, "gt_objects_at_time_class")
    key = (float(t), str(class_name))
    if key not in cache:
        gt = sequence_ctx["gt_trackset"]
        gt_classes = gt.metadata.get("classes", [])
        cache[key] = [
            g
            for g in _gt_objects_at_time(sequence_ctx, float(t))
            if _class_name(gt_classes, g.cl) == class_name
        ]
    return cache[key]


def _run_objects_for_frame_class(
    sequence_ctx: Dict[str, Any],
    frame_idx: int,
    frame: Dict[str, Any],
    class_name: str,
) -> List[Tuple[Any, Dict[str, Any], List[float]]]:
    cache = _cache_map(sequence_ctx, "run_objects_for_frame_class")
    key = (int(frame_idx), str(class_name))
    if key in cache:
        return cache[key]
    out: List[Tuple[Any, Dict[str, Any], List[float]]] = []
    run_classes = sequence_ctx["run_trackset"].metadata.get("classes", [])
    frame_objects = frame.get("objects")
    if isinstance(frame_objects, dict):
        for track_id, obj in frame_objects.items():
            if not isinstance(obj, dict):
                continue
            if _class_name(run_classes, obj.get("class")) != class_name:
                continue
            box = _det_box(obj)
            if box is None:
                continue
            out.append((track_id, obj, box))
    cache[key] = out
    return out


# ----------------------------------------------------------------------
# pair_logger evaluator
# ----------------------------------------------------------------------
def evaluate_pair_logger(
    sequence_ctx: Dict[str, Any], module_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Label one sequence's per-frame match-cost trace against GT.

    Reads ``frame["debug"]["match_cost_trace"]`` (populated by the C side
    when ``utrack.debug_match_cost_trace`` is true), labels each
    (track, det) pair via best-IoU GT alignment on both sides, and writes
    a per-sequence NPZ to ``module_params["output_dir"]``.

    Required module_params:
        output_dir: str  — where to write <seq>.npz
    Optional:
        seed_match_iou:    float = 0.5  — track-side GT IoU threshold
        det_match_iou:     float = 0.5  — det-side GT IoU threshold
        class_name:        str   = "person"
        require_any_trace: bool  = True — fail loudly if no frame has a
                                          trace (catches forgotten flag)
    """
    output_dir = module_params.get("output_dir")
    if not output_dir:
        raise ValueError("pair_logger: module_params.output_dir is required")
    seed_match_iou = float(module_params.get("seed_match_iou", 0.5))
    det_match_iou = float(module_params.get("det_match_iou", 0.5))
    # honest-fp-train-adopt-v2: PRECISE+DOSED phantom-negatives. Only a
    # phantom this real track could plausibly ABSORB (det overlaps the
    # track's predicted box by > min_track_iou) is the gaming vector;
    # v1's blunt "every real-track × any far det" over-dosed and cost
    # MOTA. subsample (stride) keeps negatives from swamping positives.
    phantom_neg_min_track_iou = float(
        module_params.get("phantom_neg_min_track_iou", 0.1))
    _pn_sub = float(module_params.get("phantom_neg_subsample", 1.0))
    phantom_neg_stride = max(1, int(round(1.0 / _pn_sub))) if _pn_sub > 0 else 1
    class_name = str(module_params.get("class_name", "person"))
    require_any_trace = bool(module_params.get("require_any_trace", True))

    run = sequence_ctx["run_trackset"]
    frames = run.frames

    # Walk frames in order, building a per-track GT-id history along the
    # way. The GT id of a track is whatever GT object its tracked box
    # most recently matched above seed_match_iou. Tracks that never
    # match GT this run never end up in track_gt_history; their pairs
    # get dropped.
    track_gt_history: Dict[Any, int] = {}

    record_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []

    n_frames_with_trace = 0
    n_frames_with_records = 0
    n_dropped_no_gt_track = 0
    n_dropped_no_gt_det = 0
    n_phantom_neg = 0   # honest-fp-train-adopt-v2: kept phantom-negs
    n_phantom_neg_seen = 0      # passed precise gate (pre-subsample)
    n_phantom_neg_dropped = 0   # dropped by subsample (dose)
    n_pos = 0
    n_neg = 0

    for i, frame in enumerate(frames):
        t = float(frame.get("frame_time", 0.0))
        gt_curr = gt_objects_at_time_class(sequence_ctx, t, class_name)

        # Update track→GT history from this frame's tracked outputs.
        # Only covers TRACKED (emitted) tracks; UNCONFIRMED/LOST refresh
        # happens below from the pair-trace records.
        if frame.get("objects"):
            for track_id, _obj, box in _run_objects_for_frame_class(
                sequence_ctx, i, frame, class_name
            ):
                best_gt, best_iou = best_iou_match(box, gt_curr)
                if best_gt is not None and best_iou >= seed_match_iou:
                    track_gt_history[track_id] = int(best_gt.track_id)

        # Pull this frame's pair trace, if any.
        debug = frame.get("debug") or {}
        trace_blob = debug.get("match_cost_trace")
        if not isinstance(trace_blob, dict):
            continue

        n_frames_with_trace += 1

        magic = int(trace_blob.get("magic", 0))
        version = int(trace_blob.get("version", 0))
        if magic != PAIR_LOG_MAGIC:
            raise ValueError(
                f"pair_logger: trace magic mismatch in frame {i}: "
                f"got 0x{magic:08x}, expected 0x{PAIR_LOG_MAGIC:08x}"
            )
        if version != PAIR_LOG_VERSION:
            raise ValueError(
                f"pair_logger: trace version mismatch in frame {i}: "
                f"got {version}, expected {PAIR_LOG_VERSION}"
            )

        record_size = int(trace_blob.get("record_size", 0))
        if record_size != record_size_bytes():
            raise ValueError(
                f"pair_logger: trace record_size {record_size} != "
                f"schema {record_size_bytes()} (frame {i})"
            )

        n_records = int(trace_blob.get("n_records", 0))
        data = trace_blob.get("data") or b""
        if n_records == 0 or len(data) == 0:
            continue
        records = decode_records(data, n_records)
        n_frames_with_records += 1

        # Refresh track_gt_history from this frame's pair-trace track
        # boxes BEFORE labelling. Covers tracks the per-frame
        # objects-update missed (UNCONFIRMED/LOST never emit objects)
        # and overwrites stale cache entries when a track has drifted to
        # a different GT identity. Walks distinct track_ids only — most
        # frames have many records per track.
        seen_tids: Dict[int, Tuple[float, float, float, float]] = {}
        for r_idx in range(int(records.shape[0])):
            rec = records[r_idx]
            tid = int(rec["track_id"])
            if tid not in seen_tids:
                seen_tids[tid] = (
                    float(rec["track_x0"]),
                    float(rec["track_y0"]),
                    float(rec["track_x1"]),
                    float(rec["track_y1"]),
                )
        for tid, track_box in seen_tids.items():
            best_gt, best_iou = best_iou_match(list(track_box), gt_curr)
            if best_gt is not None and best_iou >= seed_match_iou:
                track_gt_history[tid] = int(best_gt.track_id)

        # Vectorised labelling: build numpy arrays of track and det
        # boxes for the whole frame, then loop only to do per-pair GT
        # lookups (which are O(|gt|) and cheap in practice).
        kept_indices: List[int] = []
        kept_labels: List[int] = []
        for r_idx in range(int(records.shape[0])):
            rec = records[r_idx]

            track_id_int: int = int(rec["track_id"])
            track_gt_id = track_gt_history.get(track_id_int)
            if track_gt_id is None:
                n_dropped_no_gt_track += 1
                continue

            det_box = [
                float(rec["det_x0"]),
                float(rec["det_y0"]),
                float(rec["det_x1"]),
                float(rec["det_y1"]),
            ]
            det_best_gt, det_best_iou = best_iou_match(det_box, gt_curr)
            if det_best_gt is None or det_best_iou < det_match_iou:
                # honest-fp-train-adopt-v2 (20260515): the track here is
                # GT-aligned (real). A detection overlapping NO real
                # object (IoU==0 with every GT = the resolved honest
                # ruler's phantom) that THIS track could plausibly
                # ABSORB (overlaps the track's predicted box) is exactly
                # the gaming vector — emit it as a NEGATIVE so the
                # match-NN learns not to absorb it (training aligned with
                # the de-gamed fitness). PRECISE: require track∩det IoU >
                # phantom_neg_min_track_iou (v1 made *every* far det a
                # negative → over-rejection → MOTA −0.020). DOSE: keep
                # 1/stride of them so negatives don't swamp positives.
                # Gray zone 0<det_IoU<det_match_iou stays DROPPED (loose-
                # but-near-real; not what honest fitness penalises).
                if det_best_gt is None or det_best_iou <= 0.0:
                    track_box = [
                        float(rec["track_x0"]), float(rec["track_y0"]),
                        float(rec["track_x1"]), float(rec["track_y1"]),
                    ]
                    if _box_iou(track_box, det_box) > phantom_neg_min_track_iou:
                        n_phantom_neg_seen += 1
                        if n_phantom_neg_seen % phantom_neg_stride == 0:
                            kept_indices.append(r_idx)
                            kept_labels.append(0)
                            n_neg += 1
                            n_phantom_neg += 1
                        else:
                            n_phantom_neg_dropped += 1
                    else:
                        n_dropped_no_gt_det += 1
                else:
                    n_dropped_no_gt_det += 1
                continue
            det_gt_id = int(det_best_gt.track_id)

            label = 1 if det_gt_id == track_gt_id else 0
            kept_indices.append(r_idx)
            kept_labels.append(label)
            if label:
                n_pos += 1
            else:
                n_neg += 1

        if kept_indices:
            record_chunks.append(records[np.asarray(kept_indices, dtype=np.int64)])
            label_chunks.append(np.asarray(kept_labels, dtype=np.uint8))

    if require_any_trace and n_frames_with_trace == 0:
        raise RuntimeError(
            "pair_logger: no frame in this sequence carried a "
            "match_cost_trace debug payload. Check that the tracker "
            "config has `utrack.debug_match_cost_trace: true` and that "
            "the run was generated with that config (force_regen: true "
            "may be needed to invalidate the UBTRK2 cache)."
        )

    # Sort records deterministically by (frame_idx, track_id, ...) so
    # repeated runs produce byte-identical NPZ files (Phase-1 gate).
    if record_chunks:
        records_arr = np.concatenate(record_chunks)
        labels_arr = np.concatenate(label_chunks)
        sort_key = np.lexsort((
            records_arr["det_x0"],
            records_arr["track_id"],
            records_arr["pass_id"],
            records_arr["frame_time"],
        ))
        records_arr = records_arr[sort_key]
        labels_arr = labels_arr[sort_key]
    else:
        records_arr = np.zeros(0, dtype=PAIR_LOG_DTYPE)
        labels_arr = np.zeros(0, dtype=np.uint8)

    seq_name = str(sequence_ctx.get("sequence_name", "unknown"))
    seq_safe = _safe_name(seq_name)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{seq_safe}.npz")
    tmp_path = out_path + ".tmp"
    np.savez(
        tmp_path,
        records=records_arr,
        labels=labels_arr,
        feature_names=np.array(PAIR_LOG_FEATURE_NAMES, dtype=object),
        schema_version=np.int64(PAIR_LOG_VERSION),
        schema_magic=np.uint32(PAIR_LOG_MAGIC),
    )
    # np.savez writes "<tmp>.npz" if path lacks the suffix; we passed
    # tmp+.tmp so the actual file is tmp+.tmp.npz. Normalise.
    real_tmp = tmp_path if os.path.exists(tmp_path) else tmp_path + ".npz"
    os.replace(real_tmp, out_path)

    n_total = int(records_arr.shape[0])
    return {
        "n_pairs": n_total,
        "n_positives": n_pos,
        "n_negatives": n_neg,
        "positive_rate": (float(n_pos) / float(n_total)) if n_total > 0 else None,
        "n_dropped_no_gt_track": n_dropped_no_gt_track,
        "n_dropped_no_gt_det": n_dropped_no_gt_det,
        "n_phantom_neg": n_phantom_neg,
        "n_phantom_neg_seen": n_phantom_neg_seen,
        "n_phantom_neg_dropped": n_phantom_neg_dropped,
        "n_frames_with_trace": n_frames_with_trace,
        "n_frames_with_records": n_frames_with_records,
        "n_frames_total": len(frames),
        "output_npz": out_path,
    }


# ----------------------------------------------------------------------
# Workers (multi-process via stuff.mp_workqueue_run)
# ----------------------------------------------------------------------
def _generate_worker(
    task: Dict[str, Any], mpwq_context: Optional[Dict[str, Any]], mpwq_progress_fn
) -> Dict[str, Any]:
    """Run the C tracker on one clip + cache its UBTRK2 output."""
    start = time.time()
    sequence_name = task["sequence_name"]
    try:
        mpwq_progress_fn(mpwq_context, desc=f"gen {sequence_name}", total=1)
        ubtrk2_path = task["ubtrk2_path"]
        if os.path.isfile(ubtrk2_path) and not task["force_regen"]:
            mpwq_progress_fn(mpwq_context, update=1)
            return {
                "ok": True,
                "sequence": sequence_name,
                "ubtrk2_path": ubtrk2_path,
                "status": "cached",
                "elapsed_sec": time.time() - start,
            }

        gt = ts.TrackSet(task["trackset_path"])
        out = ts.TrackSet()
        # Resolve tracker config: optionally deep-merge a small overrides
        # dict on top of the loaded YAML so analysis configs can flip
        # narrowly-scoped tracker flags without duplicating the full
        # tracker YAML. import_create accepts a dict in place of a path.
        tracker_config_arg: Any = task["tracker_config"]
        overrides = task.get("tracker_config_overrides") or {}
        if overrides:
            base = stuff.load_dictionary(task["tracker_config"])

            def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
                for k, v in src.items():
                    if isinstance(v, dict) and isinstance(dst.get(k), dict):
                        _deep_update(dst[k], v)
                    else:
                        dst[k] = v
            _deep_update(base, overrides)
            tracker_config_arg = base
        out.import_create(
            gt,
            config_file=tracker_config_arg,
            track_min_interval=float(task["min_interval"]),
            debug_enable=bool(task["debug_enable"]),
            start_time=float(task["start_time"]),
            end_time=float(task["end_time"]),
            display=f"analysis:{sequence_name}",
            mpwq_context=mpwq_context,
            mpwq_progress_fn=mpwq_progress_fn,
        )
        stuff.makedir(os.path.dirname(ubtrk2_path))
        out.export_track_file(ubtrk2_path)
        mpwq_progress_fn(mpwq_context, update=1)
        return {
            "ok": True,
            "sequence": sequence_name,
            "ubtrk2_path": ubtrk2_path,
            "status": "generated",
            "num_frames": len(out.frames),
            "elapsed_sec": time.time() - start,
        }
    except Exception:
        return {
            "ok": False,
            "sequence": sequence_name,
            "status": "failed",
            "error": traceback.format_exc(),
            "elapsed_sec": time.time() - start,
        }


def _analysis_worker(
    task: Dict[str, Any], mpwq_context: Optional[Dict[str, Any]], mpwq_progress_fn
) -> Dict[str, Any]:
    """Run the pair-logger evaluator on one cached UBTRK2 trace."""
    start = time.time()
    sequence_name = task["sequence_name"]
    try:
        mpwq_progress_fn(mpwq_context, desc=f"an {sequence_name}", total=1)
        # Defer ndarray materialization until the evaluator actually
        # touches a payload. Avoids eager decode of large UBTRK2 debug/reid
        # blobs.
        run_trackset = ts.TrackSet(
            task["ubtrk2_path"],
            decode_payloads=False,
            analysis_mode=True,
        )
        gt_trackset = ts.TrackSet(task["trackset_path"])
        sequence_ctx = {
            "sequence_name": sequence_name,
            "run_trackset": run_trackset,
            "gt_trackset": gt_trackset,
        }
        metrics = evaluate_pair_logger(sequence_ctx, task["module_params"])
        mpwq_progress_fn(mpwq_context, update=1)
        return {
            "ok": True,
            "sequence": sequence_name,
            "trackset_path": task["trackset_path"],
            "ubtrk2_path": task["ubtrk2_path"],
            "metrics": metrics,
            "elapsed_sec": time.time() - start,
        }
    except Exception:
        return {
            "ok": False,
            "sequence": sequence_name,
            "status": "failed",
            "error": traceback.format_exc(),
            "elapsed_sec": time.time() - start,
        }


# ----------------------------------------------------------------------
# Summary + markdown
# ----------------------------------------------------------------------
def _build_summary(
    config: Dict[str, Any],
    generation_results: List[Dict[str, Any]],
    analysis_results: List[Dict[str, Any]],
    started_at: float,
) -> Dict[str, Any]:
    ok_analysis = [r for r in analysis_results if r.get("ok")]
    fail_analysis = [r for r in analysis_results if not r.get("ok")]

    per_seq_metrics: Dict[str, Dict[str, Any]] = {}
    totals: Dict[str, int] = defaultdict(int)
    for r in ok_analysis:
        seq_name = r.get("sequence", "unknown")
        m = r.get("metrics", {})
        per_seq_metrics[seq_name] = m
        for k in ("n_pairs", "n_positives", "n_negatives",
                  "n_dropped_no_gt_track", "n_dropped_no_gt_det",
                  "n_phantom_neg", "n_phantom_neg_seen",
                  "n_phantom_neg_dropped",
                  "n_frames_with_trace", "n_frames_with_records",
                  "n_frames_total"):
            try:
                totals[k] += int(m.get(k, 0) or 0)
            except Exception:
                pass

    total_pairs = int(totals.get("n_pairs", 0))
    total_pos = int(totals.get("n_positives", 0))
    aggregates = {
        "n_sequences_ok": len(ok_analysis),
        "n_sequences_failed": len(fail_analysis),
        "n_pairs": total_pairs,
        "n_positives": total_pos,
        "n_negatives": int(totals.get("n_negatives", 0)),
        "positive_rate": (total_pos / total_pairs) if total_pairs > 0 else None,
        "n_dropped_no_gt_track": int(totals.get("n_dropped_no_gt_track", 0)),
        "n_dropped_no_gt_det": int(totals.get("n_dropped_no_gt_det", 0)),
        "n_phantom_neg": int(totals.get("n_phantom_neg", 0)),
        "n_phantom_neg_seen": int(totals.get("n_phantom_neg_seen", 0)),
        "n_phantom_neg_dropped": int(totals.get("n_phantom_neg_dropped", 0)),
    }

    generation_ok = [g for g in generation_results if g.get("ok")]
    generation_failed = [g for g in generation_results if not g.get("ok")]
    generation_cached = [g for g in generation_ok if g.get("status") == "cached"]
    generation_generated = [g for g in generation_ok if g.get("status") == "generated"]

    return {
        "analysis_name": config.get("analysis_name", "track_analysis"),
        "dataset_split": str(config.get("dataset_split", "train")).lower().strip(),
        "sequences": sorted(per_seq_metrics.keys()),
        "aggregates": aggregates,
        "per_sequence_metrics": per_seq_metrics,
        "generation": {
            "total_sequences": len(generation_results),
            "generated": len(generation_generated),
            "cached": len(generation_cached),
            "failed": len(generation_failed),
            "details": generation_results,
        },
        "analysis_failures": fail_analysis,
        "elapsed_sec": time.time() - started_at,
    }


def _summary_markdown(summary: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# Track analysis summary: {summary['analysis_name']}")
    lines.append("")
    lines.append(f"- dataset_split: {summary.get('dataset_split', 'train')}")
    lines.append(f"- elapsed_sec: {summary.get('elapsed_sec', 0):.1f}")
    lines.append("")
    lines.append("## Aggregate metrics")
    lines.append("")
    for k, v in summary["aggregates"].items():
        if isinstance(v, float):
            lines.append(f"- {k}: {v:.6f}")
        else:
            lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Generation")
    lines.append("")
    lines.append(f"- total_sequences: {summary['generation']['total_sequences']}")
    lines.append(f"- generated: {summary['generation']['generated']}")
    lines.append(f"- cached: {summary['generation']['cached']}")
    lines.append(f"- failed: {summary['generation']['failed']}")
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------
def run_track_analysis(config: Dict[str, Any]) -> Dict[str, Any]:
    """Two-phase per-clip pipeline: (1) run the C tracker on each dataset
    entry with UBTRK2 caching, (2) invoke ``evaluate_pair_logger`` on
    each cached trace to emit a ``<seq>.npz``.

    Returns a dict with ``output_root``, ``summary_json``, ``summary_md``,
    and the in-memory ``summary`` + raw worker result lists. Earlier
    versions of this code also wrote ``compare.{json,md}`` for the
    now-removed variant-comparison framework; those are gone.
    """
    started_at = time.time()
    analysis_name = str(config.get("analysis_name", "track_analysis"))
    output_root = str(
        config.get(
            "output_root",
            os.path.join(os.getcwd(), "runs", "track_analysis", _safe_name(analysis_name)),
        )
    )
    ubtrk2_dir = os.path.join(output_root, "ubtrk2")
    metrics_dir = os.path.join(output_root, "metrics")
    for p in [output_root, ubtrk2_dir, metrics_dir]:
        stuff.makedir(p)

    dataset_cfg = config.get("dataset")
    if dataset_cfg is None:
        dataset_cfg = config.get("datasets", [])
    dataset = _normalise_dataset(dataset_cfg)
    dataset_split = str(config.get("dataset_split", "train")).lower().strip()
    dataset = _filter_dataset_by_split(dataset, dataset_split)
    if len(dataset) == 0:
        raise ValueError(
            f"Config dataset is empty after applying dataset_split={dataset_split!r}"
        )

    # Single supported module after the 2026-05-15 cull. If the yaml asks
    # for anything else, fail loudly — it's almost certainly a copy-paste
    # from an obsolete config.
    enabled = config.get("metrics_enabled")
    if enabled is None:
        enabled = ["pair_logger"]
    if isinstance(enabled, dict):
        enabled = [k for k, v in enabled.items() if bool(v)]
    if isinstance(enabled, list):
        enabled = [str(x) for x in enabled]
    else:
        raise ValueError(f"Unsupported metrics_enabled type: {type(enabled)}")
    unknown = [m for m in enabled if m != "pair_logger"]
    if unknown:
        raise ValueError(
            f"Only 'pair_logger' is supported now (the other evaluator "
            f"modules were dead code, removed 2026-05-15). Got: {unknown}"
        )

    workers = stuff.resolve_num_workers(config.get("workers", config.get("num_workers", 1)))
    if workers < 0:
        workers = 0

    tracker_config = config.get("tracker_config", config.get("config"))
    if tracker_config is None:
        raise ValueError("Missing tracker config; expected `tracker_config` or `config` in yaml")
    if not os.path.isfile(tracker_config):
        raise FileNotFoundError(
            f"tracker_config not found: {tracker_config!r}. "
            f"Pair-log generation needs the live tracker YAML to load."
        )

    # Optional shallow overrides merged on top of the loaded tracker
    # YAML at generate time. Lets the analysis YAML flip narrowly-scoped
    # flags (e.g. `utrack.debug_match_cost_trace: true` for pair logging)
    # without duplicating the entire tracker config.
    tracker_config_overrides = dict(config.get("tracker_config_overrides") or {})

    # Reject unknown / dead utrack keys at YAML-load time. We do this
    # before launching N workers because a typo in a pair-log config
    # otherwise silently runs ALL clips with the wrong tracker behavior
    # and we only notice when the corpus comes back wrong.
    try:
        from ml.util._pipeline_checks import (
            validate_utrack_overrides, assert_tracker_bins_present,
        )
    except Exception:
        # ml/ may be absent in some downstream installs. Skip silently.
        validate_utrack_overrides = None
        assert_tracker_bins_present = None
    if validate_utrack_overrides is not None:
        _ut_overrides = tracker_config_overrides.get("utrack") or {}
        # nn_state activeness: override sets it, OR the base tracker yaml does.
        _nn_active = False
        if "nn_state_path" in _ut_overrides:
            _nn_active = bool(_ut_overrides.get("nn_state_path"))
        else:
            try:
                import yaml as _yaml
                _base = _yaml.safe_load(open(tracker_config))
                _nn_active = bool(((_base or {}).get("utrack") or {})
                                  .get("nn_state_path"))
            except Exception:
                pass
        validate_utrack_overrides(
            # Filter ignored bookkeeping keys (path fields & nn_lambda
            # are accepted regardless of NN-mode).
            {k: v for k, v in _ut_overrides.items()
             if k not in ("nn_path", "nn_state_path", "nn_lambda")},
            nn_state_active=_nn_active,
            label="tracker_config_overrides.utrack",
        )
        assert_tracker_bins_present(
            nn_path=_ut_overrides.get("nn_path"),
            nn_state_path=_ut_overrides.get("nn_state_path"),
            detector_engine=(((tracker_config_overrides.get("inference_config")
                               or {}).get("detection")) or {}).get("trt"),
            label="tracker_config_overrides",
        )

    # Phase 1: tracker runs (one per clip, cached by UBTRK2 file presence)
    generation_tasks = []
    for entry in dataset:
        seq_name = str(entry["name"])
        seq_safe = _safe_name(seq_name)
        generation_tasks.append(
            {
                "sequence_name": seq_name,
                "trackset_path": entry["trackset"],
                "ubtrk2_path": os.path.join(ubtrk2_dir, f"{seq_safe}.ubtrk2"),
                "tracker_config": tracker_config,
                "tracker_config_overrides": copy.deepcopy(tracker_config_overrides),
                "min_interval": float(config.get("min_interval", 0.2)),
                "debug_enable": bool(config.get("debug_enable", True)),
                "force_regen": bool(config.get("force_regen", False)),
                "start_time": float(entry.get("start_time", 0.0)),
                "end_time": float(entry.get("end_time", 100000.0)),
            }
        )

    generation_results = stuff.mp_workqueue_run(
        generation_tasks,
        _generate_worker,
        num_workers=workers,
        desc="track-analysis generate",
    )
    generation_ok_by_seq = {r["sequence"]: r for r in generation_results if r.get("ok")}

    # Phase 2: pair-logger evaluator
    module_params_global = dict(config.get("module_params") or {})
    pair_logger_params = dict(module_params_global.get("pair_logger") or {})
    analysis_tasks = []
    for entry in dataset:
        seq_name = str(entry["name"])
        gen = generation_ok_by_seq.get(seq_name)
        if gen is None:
            continue
        analysis_tasks.append(
            {
                "sequence_name": seq_name,
                "trackset_path": entry["trackset"],
                "ubtrk2_path": gen["ubtrk2_path"],
                "module_params": pair_logger_params,
            }
        )

    analysis_results = stuff.mp_workqueue_run(
        analysis_tasks,
        _analysis_worker,
        num_workers=workers,
        desc="track-analysis pair-log",
    )

    summary = _build_summary(
        config=config,
        generation_results=generation_results,
        analysis_results=analysis_results,
        started_at=started_at,
    )

    summary_json_path = os.path.join(output_root, "summary.json")
    summary_md_path = os.path.join(output_root, "summary.md")
    _write_json(summary_json_path, summary)
    _write_text(summary_md_path, _summary_markdown(summary))

    return {
        "analysis_name": summary["analysis_name"],
        "output_root": output_root,
        "summary_json": summary_json_path,
        "summary_md": summary_md_path,
        "generation_results": generation_results,
        "analysis_results": analysis_results,
        "summary": summary,
    }
