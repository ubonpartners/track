from __future__ import annotations

import copy
import json
import math
import os
import re
import time
import traceback
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, List, Optional

import stuff

import src.trackset as ts
from src.analysis.modules import list_registered_modules, run_module
from src.analysis.variants import create_variant

_VARIANT_INDEPENDENT_MODULES = {"detection", "kalman", "roi_skip", "pair_logger"}


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


def _normalise_enabled_modules(metrics_enabled: Any) -> List[str]:
    if metrics_enabled is None:
        return ["detection", "optical_flow", "kalman", "reid", "match_cost", "roi_skip"]
    if isinstance(metrics_enabled, list):
        return [str(x) for x in metrics_enabled]
    if isinstance(metrics_enabled, dict):
        return [str(k) for k, v in metrics_enabled.items() if bool(v)]
    raise ValueError(f"Unsupported metrics_enabled type: {type(metrics_enabled)}")


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


def _normalise_variants(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    variants = config.get("variants")
    if variants is None:
        variants = [
            {"name": "baseline", "impl": "baseline"},
            {"name": "of_proto_v1", "impl": "of_proto_v1"},
        ]
    if not isinstance(variants, list) or len(variants) == 0:
        raise ValueError("variants must be a non-empty list")

    out = []
    has_baseline_name = False
    seen = set()
    for item in variants:
        if not isinstance(item, dict):
            raise ValueError(f"Each variant must be dict, got {type(item)}")
        variant = dict(item)
        variant.setdefault("impl", "baseline")
        variant.setdefault("name", variant["impl"])
        name = str(variant["name"])
        if name in seen:
            raise ValueError(f"Duplicate variant name {name!r}")
        seen.add(name)
        if name == "baseline":
            has_baseline_name = True
        out.append(variant)

    if not has_baseline_name:
        out.insert(0, {"name": "baseline", "impl": "baseline"})
    return out


def _flatten_metrics(module_results: Dict[str, Dict[str, Any]]) -> Dict[str, Optional[float]]:
    flat: Dict[str, Optional[float]] = {}
    for module_name, module_data in module_results.items():
        metrics = module_data.get("metrics", {})
        for metric_key, value in metrics.items():
            flat[f"{module_name}.{metric_key}"] = value
    return flat


def _summary_markdown(summary: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# Track analysis summary: {summary['analysis_name']}")
    lines.append("")
    lines.append(f"- dataset_split: {summary.get('dataset_split', 'train')}")
    lines.append("")
    lines.append("## Weighted metrics by variant")
    lines.append("")
    for variant, metric_map in summary["weighted_mean_metrics_by_variant"].items():
        lines.append(f"### {variant}")
        if len(metric_map) == 0:
            lines.append("- no metrics")
            lines.append("")
            continue
        for k in sorted(metric_map.keys()):
            v = metric_map[k]
            if v is None:
                lines.append(f"- {k}: n/a")
            else:
                lines.append(f"- {k}: {v:.6f}")
        lines.append("")

    lines.append("## Generation")
    lines.append("")
    lines.append(f"- total_sequences: {summary['generation']['total_sequences']}")
    lines.append(f"- generated: {summary['generation']['generated']}")
    lines.append(f"- cached: {summary['generation']['cached']}")
    lines.append(f"- failed: {summary['generation']['failed']}")
    lines.append("")
    return "\n".join(lines)


def _compare_markdown(compare: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# Variant comparison (baseline: {compare['baseline']})")
    lines.append("")
    for variant, metric_map in compare.get("metric_deltas", {}).items():
        lines.append(f"## {variant}")
        if len(metric_map) == 0:
            lines.append("- no comparable metrics")
            lines.append("")
            continue
        for metric, d in sorted(metric_map.items()):
            abs_delta = d.get("abs_delta")
            rel_delta = d.get("rel_delta")
            b = d.get("baseline")
            v = d.get("variant")
            lines.append(
                f"- {metric}: baseline={b:.6f} variant={v:.6f} "
                f"abs_delta={abs_delta:+.6f} rel_delta={rel_delta:+.4f}"
            )
        lines.append("")

    if compare.get("experiments"):
        lines.append("## Experiment gates")
        lines.append("")
        for gate in compare["experiments"]:
            status = "PASS" if gate.get("pass") else "FAIL"
            lines.append(f"- {status}: {gate.get('name', 'unnamed')} ({gate.get('reason', '')})")
        lines.append("")
    return "\n".join(lines)


def _build_summary(
    config: Dict[str, Any],
    generation_results: List[Dict[str, Any]],
    analysis_results: List[Dict[str, Any]],
    variant_specs: List[Dict[str, Any]],
    started_at: float,
) -> Dict[str, Any]:
    weighted_sum: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    weighted_count: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    unweighted_vals: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    per_seq: Dict[str, Dict[str, Dict[str, Optional[float]]]] = defaultdict(dict)

    ok_analysis = [r for r in analysis_results if r.get("ok")]
    fail_analysis = [r for r in analysis_results if not r.get("ok")]

    for r in ok_analysis:
        variant = r["variant"]
        seq_name = r["sequence"]
        module_results = r.get("modules", {})
        per_seq[variant][seq_name] = _flatten_metrics(module_results)
        for module_name, module_data in module_results.items():
            metrics = module_data.get("metrics", {})
            metric_counts = module_data.get("metric_counts", {})
            for mk, mv in metrics.items():
                if mv is None:
                    continue
                key = f"{module_name}.{mk}"
                mcount = int(metric_counts.get(mk, 0))
                if mcount > 0:
                    weighted_sum[variant][key] += float(mv) * mcount
                    weighted_count[variant][key] += mcount
                else:
                    unweighted_vals[variant][key].append(float(mv))

    weighted_means: Dict[str, Dict[str, Optional[float]]] = {}
    variant_names = [str(v["name"]) for v in variant_specs]
    for variant in variant_names:
        metrics = set(weighted_sum[variant].keys()) | set(unweighted_vals[variant].keys())
        weighted_means[variant] = {}
        for key in sorted(metrics):
            if weighted_count[variant].get(key, 0) > 0:
                weighted_means[variant][key] = (
                    weighted_sum[variant][key] / weighted_count[variant][key]
                )
            elif len(unweighted_vals[variant].get(key, [])) > 0:
                weighted_means[variant][key] = mean(unweighted_vals[variant][key])
            else:
                weighted_means[variant][key] = None

    generation_ok = [g for g in generation_results if g.get("ok")]
    generation_failed = [g for g in generation_results if not g.get("ok")]
    generation_cached = [g for g in generation_ok if g.get("status") == "cached"]
    generation_generated = [g for g in generation_ok if g.get("status") == "generated"]

    return {
        "analysis_name": config.get("analysis_name", "track_analysis"),
        "dataset_split": str(config.get("dataset_split", "train")).lower().strip(),
        "sequences": sorted(list({r.get("sequence") for r in ok_analysis if "sequence" in r})),
        "variants": variant_names,
        "weighted_mean_metrics_by_variant": weighted_means,
        "per_sequence_metrics_by_variant": per_seq,
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


def _build_compare(
    summary: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    weighted = summary["weighted_mean_metrics_by_variant"]
    variants = summary["variants"]
    baseline_name = str(config.get("baseline_variant", "baseline"))
    if baseline_name not in weighted:
        baseline_name = variants[0]
    baseline_metrics = weighted.get(baseline_name, {})

    metric_deltas: Dict[str, Dict[str, Dict[str, float]]] = {}
    for variant in variants:
        if variant == baseline_name:
            continue
        variant_metrics = weighted.get(variant, {})
        delta_map: Dict[str, Dict[str, float]] = {}
        for metric in sorted(set(baseline_metrics.keys()) | set(variant_metrics.keys())):
            b = baseline_metrics.get(metric)
            v = variant_metrics.get(metric)
            if b is None or v is None:
                continue
            abs_delta = float(v - b)
            rel_delta = float(abs_delta / (abs(b) + 1e-9))
            delta_map[metric] = {
                "baseline": float(b),
                "variant": float(v),
                "abs_delta": abs_delta,
                "rel_delta": rel_delta,
            }
        metric_deltas[variant] = delta_map

    experiments = []
    all_pass = True
    for gate in config.get("experiments", []) or []:
        gname = str(gate.get("name", "gate"))
        variant = str(gate.get("variant", ""))
        metric = str(gate.get("metric", ""))
        entry: Dict[str, Any] = {"name": gname, "variant": variant, "metric": metric}
        delta_entry = metric_deltas.get(variant, {}).get(metric)
        if delta_entry is None:
            entry["pass"] = False
            entry["reason"] = "missing metric delta"
            experiments.append(entry)
            all_pass = False
            continue
        passed = True
        abs_delta = float(delta_entry["abs_delta"])
        rel_delta = float(delta_entry["rel_delta"])
        if "min_abs_delta" in gate and abs_delta < float(gate["min_abs_delta"]):
            passed = False
        if "max_abs_delta" in gate and abs_delta > float(gate["max_abs_delta"]):
            passed = False
        if "min_rel_delta" in gate and rel_delta < float(gate["min_rel_delta"]):
            passed = False
        if "max_rel_delta" in gate and rel_delta > float(gate["max_rel_delta"]):
            passed = False
        entry["pass"] = passed
        entry["delta"] = delta_entry
        entry["reason"] = "ok" if passed else "threshold check failed"
        experiments.append(entry)
        all_pass = all_pass and passed

    primary_metric = (
        config.get("report", {}).get("primary_metric", "optical_flow.mean_iou")
    )
    ranking = []
    for variant in variants:
        v = weighted.get(variant, {}).get(primary_metric)
        if v is None:
            continue
        ranking.append({"variant": variant, "primary_metric": primary_metric, "value": float(v)})
    ranking.sort(key=lambda x: x["value"], reverse=True)

    return {
        "baseline": baseline_name,
        "compared_variants": [v for v in variants if v != baseline_name],
        "metric_deltas": metric_deltas,
        "experiments": experiments,
        "passed_all_experiments": all_pass,
        "ranking": ranking,
    }


def generate_worker(
    task: Dict[str, Any], mpwq_context: Optional[Dict[str, Any]], mpwq_progress_fn
) -> Dict[str, Any]:
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


def analysis_worker(
    task: Dict[str, Any], mpwq_context: Optional[Dict[str, Any]], mpwq_progress_fn
) -> Dict[str, Any]:
    start = time.time()
    sequence_name = task["sequence_name"]
    variant_entries = list(task.get("variants") or [])
    try:
        total_steps = max(1, len(variant_entries))
        mpwq_progress_fn(mpwq_context, desc=f"an {sequence_name}", total=total_steps)
        # Defer ndarray materialization until modules actually need a payload.
        # This avoids expensive eager decode of large UBTRK2 debug/reid blobs.
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
        shared_module_cache: Dict[str, Dict[str, Any]] = {}
        results: List[Dict[str, Any]] = []
        for variant_entry in variant_entries:
            variant_spec = variant_entry["variant_spec"]
            variant_name = str(variant_spec.get("name", variant_spec.get("impl", "variant")))
            try:
                variant_impl = create_variant(variant_spec)
                module_results: Dict[str, Dict[str, Any]] = {}
                for module_name in variant_entry["enabled_modules"]:
                    if module_name in _VARIANT_INDEPENDENT_MODULES:
                        if module_name not in shared_module_cache:
                            shared_module_cache[module_name] = run_module(
                                module_name=module_name,
                                sequence_ctx=sequence_ctx,
                                variant_impl=variant_impl,
                                module_params=variant_entry["module_params_by_name"].get(module_name, {}),
                            )
                        module_results[module_name] = copy.deepcopy(shared_module_cache[module_name])
                        continue
                    module_results[module_name] = run_module(
                        module_name=module_name,
                        sequence_ctx=sequence_ctx,
                        variant_impl=variant_impl,
                        module_params=variant_entry["module_params_by_name"].get(module_name, {}),
                    )

                result = {
                    "ok": True,
                    "sequence": sequence_name,
                    "trackset_path": task["trackset_path"],
                    "ubtrk2_path": task["ubtrk2_path"],
                    "variant": variant_impl.name,
                    "impl": variant_impl.impl,
                    "modules": module_results,
                    "elapsed_sec": time.time() - start,
                }
                _write_json(variant_entry["output_json_path"], result)
                result["output_json_path"] = variant_entry["output_json_path"]
            except Exception:
                result = {
                    "ok": False,
                    "sequence": sequence_name,
                    "variant": variant_name,
                    "status": "failed",
                    "error": traceback.format_exc(),
                    "elapsed_sec": time.time() - start,
                }
            results.append(result)
            mpwq_progress_fn(mpwq_context, update=1)

        return {
            "ok": any(r.get("ok") for r in results),
            "sequence": sequence_name,
            "results": results,
            "elapsed_sec": time.time() - start,
        }
    except Exception:
        trace = traceback.format_exc()
        fail_results = []
        for variant_entry in variant_entries:
            variant_spec = variant_entry.get("variant_spec", {})
            variant_name = str(variant_spec.get("name", variant_spec.get("impl", "variant")))
            fail_results.append(
                {
                    "ok": False,
                    "sequence": sequence_name,
                    "variant": variant_name,
                    "status": "failed",
                    "error": trace,
                    "elapsed_sec": time.time() - start,
                }
            )
        return {
            "ok": False,
            "sequence": sequence_name,
            "status": "failed",
            "error": trace,
            "elapsed_sec": time.time() - start,
            "results": fail_results,
        }


def run_track_analysis(config: Dict[str, Any]) -> Dict[str, Any]:
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
    variants = _normalise_variants(config)

    known_modules = set(list_registered_modules())
    enabled_modules_global = _normalise_enabled_modules(config.get("metrics_enabled"))
    unknown = [m for m in enabled_modules_global if m not in known_modules]
    if unknown:
        raise ValueError(f"Unknown modules in metrics_enabled: {unknown}")

    workers = int(config.get("workers", config.get("num_workers", 1)))
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
    # flags (e.g. `utrack.debug_match_cost_trace: true` for Phase-1 pair
    # logging) without duplicating the entire tracker config.
    tracker_config_overrides = dict(config.get("tracker_config_overrides") or {})

    # Reject unknown / dead utrack keys at YAML-load time. We do this
    # before launching N workers because a typo in a pair-log config
    # otherwise silently runs ALL 178 clips with the wrong tracker
    # behavior and we only notice when the corpus comes back wrong.
    # The known-keys set is the source-of-truth list extracted from
    # `yaml_base["…"]` calls in utrack.c; see bench/_pipeline_checks.py.
    try:
        from bench._pipeline_checks import (
            validate_utrack_overrides, assert_tracker_bins_present,
        )
    except Exception:
        # bench/ may be absent in some downstream installs of this
        # engine. Skip silently in that case rather than crash.
        validate_utrack_overrides = None
        assert_tracker_bins_present = None
    if validate_utrack_overrides is not None:
        _ut_overrides = tracker_config_overrides.get("utrack") or {}
        # Determine nn_state activeness: an override sets it, OR the
        # base tracker yaml does. Read the base yaml to find out.
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
        # NN bin existence: only check the override fields, since the
        # base yaml is read by the C tracker which has its own checks.
        assert_tracker_bins_present(
            nn_path=_ut_overrides.get("nn_path"),
            nn_state_path=_ut_overrides.get("nn_state_path"),
            detector_engine=(((tracker_config_overrides.get("inference_config")
                               or {}).get("detection")) or {}).get("trt"),
            label="tracker_config_overrides",
        )

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
        generate_worker,
        num_workers=workers,
        desc="track-analysis generate",
    )
    generation_ok = [r for r in generation_results if r.get("ok")]
    generation_ok_by_seq = {r["sequence"]: r for r in generation_ok}

    module_params_global = dict(config.get("module_params") or {})
    analysis_tasks = []
    for entry in dataset:
        seq_name = str(entry["name"])
        gen = generation_ok_by_seq.get(seq_name)
        if gen is None:
            continue
        variant_entries = []
        for variant in variants:
            variant_name = str(variant["name"])
            variant_enabled = variant.get("enabled_modules")
            if variant_enabled is None:
                enabled_modules = list(enabled_modules_global)
            else:
                if isinstance(variant_enabled, list):
                    variant_enabled_list = [str(x) for x in variant_enabled]
                elif isinstance(variant_enabled, dict):
                    variant_enabled_list = [
                        str(k) for k, v in variant_enabled.items() if bool(v)
                    ]
                else:
                    raise ValueError(
                        f"enabled_modules for variant {variant_name} must be list/dict"
                    )
                enabled_modules = [m for m in enabled_modules_global if m in variant_enabled_list]

            module_params_by_name: Dict[str, Dict[str, Any]] = {}
            variant_module_params = dict(variant.get("module_params") or {})
            for module_name in enabled_modules:
                g = dict(module_params_global.get(module_name) or {})
                v = dict(variant_module_params.get(module_name) or {})
                g.update(v)
                module_params_by_name[module_name] = g

            seq_safe = _safe_name(seq_name)
            variant_safe = _safe_name(variant_name)
            variant_entries.append(
                {
                    "variant_spec": variant,
                    "enabled_modules": enabled_modules,
                    "module_params_by_name": module_params_by_name,
                    "output_json_path": os.path.join(metrics_dir, f"{seq_safe}__{variant_safe}.json"),
                }
            )
        if len(variant_entries) > 0:
            analysis_tasks.append(
                {
                    "sequence_name": seq_name,
                    "trackset_path": entry["trackset"],
                    "ubtrk2_path": gen["ubtrk2_path"],
                    "variants": variant_entries,
                }
            )

    analysis_worker_results = stuff.mp_workqueue_run(
        analysis_tasks,
        analysis_worker,
        num_workers=workers,
        desc="track-analysis metrics",
    )
    analysis_results: List[Dict[str, Any]] = []
    for worker_result in analysis_worker_results:
        worker_entries = worker_result.get("results")
        if isinstance(worker_entries, list):
            analysis_results.extend(worker_entries)
        else:
            analysis_results.append(worker_result)

    summary = _build_summary(
        config=config,
        generation_results=generation_results,
        analysis_results=analysis_results,
        variant_specs=variants,
        started_at=started_at,
    )
    compare = _build_compare(summary=summary, config=config)

    summary_json_path = os.path.join(output_root, "summary.json")
    compare_json_path = os.path.join(output_root, "compare.json")
    summary_md_path = os.path.join(output_root, "summary.md")
    compare_md_path = os.path.join(output_root, "compare.md")
    _write_json(summary_json_path, summary)
    _write_json(compare_json_path, compare)
    _write_text(summary_md_path, _summary_markdown(summary))
    _write_text(compare_md_path, _compare_markdown(compare))

    return {
        "analysis_name": summary["analysis_name"],
        "output_root": output_root,
        "summary_json": summary_json_path,
        "compare_json": compare_json_path,
        "summary_md": summary_md_path,
        "compare_md": compare_md_path,
        "generation_results": generation_results,
        "analysis_results": analysis_results,
        "summary": summary,
        "compare": compare,
    }
