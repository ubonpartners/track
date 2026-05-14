"""Pipeline defensive-check helpers.

Every long-running pipeline step (pair-log gen, build_pair_dataset,
build_state_corpus, train_phase3, train_state_head, eval_head_fitness)
imports from this module and runs validators at its entry point so we
fail loud at the boundary instead of producing silently-wrong artefacts.

Design rule: every validator either succeeds silently or raises
`PipelineCheckError` with a precise message that names the bad input.
No warning-only paths — they get ignored.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Set


class PipelineCheckError(RuntimeError):
    """Raised when a pipeline pre/post-condition fails.

    Tools should let this propagate (caller sees a clear stderr trace)
    rather than catching and continuing — silent recovery is the bug
    these checks exist to prevent.
    """


# ---------------------------------------------------------------------------
# utrack yaml keys
# ---------------------------------------------------------------------------

# Source of truth: keys actually read by utrack.c
#   grep -oE 'yaml_base\["[a-zA-Z_0-9]+"\]' utrack.c | sort -u
# (note: regex MUST include 0-9 — earlier version omitted digits and silently
#  dropped kf_d2_* keys.)
# tools/check_utrack_keys_in_sync.py re-runs this grep and fails CI on drift.
KNOWN_UTRACK_KEYS: Set[str] = {
    "bayes_c_FP_frame", "bayes_c_FP_track", "bayes_c_MOTA",
    "bayes_debug", "bayes_match_rate_TP",
    "debug_match_cost_trace",
    "delete_dup_iou",
    "fuse_scores",
    "immediate_confirm_thr",       # dead under nn_state; see DEAD_KEYS_NN
    "in_roi_min_overlap",          # unified-deletion ROI gate threshold
    "kf_cmc_enabled",
    "kf_d2_enabled", "kf_d2_weight",
    "kf_fps_scale", "kf_warmup", "kf_weight",
    "match_cheap_filter_delta",     # skips match NN on out-of-contention pairs
    "match_clip_to_roi",
    "match_thr_high", "match_thr_initial", "match_thr_low",
    "max_consecutive_misses",      # unified-deletion miss-count threshold
    "max_tracks_per_cell",
    "min_confirm_observations",     # dead under nn_state; see DEAD_KEYS_NN
    "new_track_thr",
    "nn_lambda", "nn_path", "nn_state_path",
    "no_prethr_fusion",
    "person_det_thr_single_frame",
    "reid_z_clip", "reid_zscore_ema_alpha",
    "reid_zscore_enabled", "reid_zscore_min_pairs",
    "roi_expand_ratio",
    "simple",
    "sim_weight", "sim_weight_zscore",
    "subdiou_warped",
    "track_buffer_seconds",
    "track_high_thr", "track_initial_thr", "track_low_thr",
    "vbox_expand",
}

# Keys mentioned in docs / production yamls but NOT read anywhere in the
# C source. They've been silently no-op for at least the entire NN era.
# Listed explicitly so the validator's error message can name the dead
# knob — "pose_conf" alone in an unknown-keys reject is unhelpful, but
# "pose_conf: documented in UTRACK_REVIEW.md but not read by C; remove
# it from your YAML" is actionable.
ORPHAN_KEYS: Set[str] = {
    "pose_conf",  # UTRACK_REVIEW.md:456 mentions param_pose_conf for
                  # pose-kp confidence weighting, but no yaml_get_*
                  # call exists. Either never implemented or removed
                  # without doc + yaml cleanup. Drop from yamls.
}

# Keys whose `else` branch is guarded by `!ut->nn_state` in utrack.c.
# When nn_state_path is non-empty, supplying these is a no-op and almost
# always indicates a stale or copy-pasted config. We REJECT them rather
# than silently ignore — surfacing dead config is the whole point.
DEAD_KEYS_NN: Set[str] = {
    "immediate_confirm_thr",  # utrack.c:1254 — `!ut->nn_state && conf>thr`
    "min_confirm_observations",  # utrack.c:1173 — same guard pattern
}


def validate_utrack_overrides(
    overrides: Optional[Mapping],
    *,
    nn_state_active: bool,
    label: str = "tracker_config_overrides.utrack",
) -> None:
    """Reject typos, deprecated keys, and dead-under-NN keys.

    `overrides` is the dict from tracker_config_overrides.utrack in a
    pair-log or eval YAML. `nn_state_active` should be True iff
    nn_state_path is non-empty (either set in this dict or in the base
    tracker yaml — caller is responsible for resolving that).
    """
    if overrides is None:
        return
    if not isinstance(overrides, Mapping):
        raise PipelineCheckError(
            f"{label}: expected mapping, got {type(overrides).__name__}"
        )

    # Orphan keys — listed but never read. Specific message so the
    # user knows the value has been silently ignored, not just typo'd.
    orphan = [k for k in overrides.keys() if k in ORPHAN_KEYS]
    if orphan:
        raise PipelineCheckError(
            f"{label}: orphan utrack keys {orphan!r} are NOT read by "
            f"any C source. They've been silent no-ops; remove them "
            f"from your YAML."
        )

    unknown = [k for k in overrides.keys()
               if k not in KNOWN_UTRACK_KEYS and k not in ORPHAN_KEYS]
    if unknown:
        raise PipelineCheckError(
            f"{label}: unknown utrack keys {unknown!r}. "
            f"Likely a typo or a deprecated knob. "
            f"Allowed keys (from utrack.c): {sorted(KNOWN_UTRACK_KEYS)}"
        )

    if nn_state_active:
        dead_present = [k for k in overrides.keys() if k in DEAD_KEYS_NN]
        if dead_present:
            raise PipelineCheckError(
                f"{label}: {dead_present!r} are dead under nn_state mode "
                f"(guarded by `!ut->nn_state` in utrack.c). Remove them — "
                f"silently carrying dead config drifts recipes over time."
            )


# ---------------------------------------------------------------------------
# File / directory existence
# ---------------------------------------------------------------------------

def assert_file_exists(path: Optional[str], label: str) -> None:
    if not path:
        raise PipelineCheckError(f"{label}: path is empty/None")
    if not os.path.isfile(path):
        raise PipelineCheckError(f"{label}: file not found at {path!r}")


def assert_dir_exists(path: Optional[str], label: str) -> None:
    if not path:
        raise PipelineCheckError(f"{label}: path is empty/None")
    if not os.path.isdir(path):
        raise PipelineCheckError(f"{label}: dir not found at {path!r}")


def assert_dir_has_files(
    path: str,
    *,
    ext: str,
    min_count: int,
    label: str,
) -> int:
    """Asserts `path` contains at least `min_count` files with extension `ext`.
    Returns the actual count for the caller to log.
    """
    assert_dir_exists(path, label)
    files = list(Path(path).glob(f"*{ext}"))
    if len(files) < min_count:
        raise PipelineCheckError(
            f"{label}: expected ≥{min_count} {ext!r} files in {path!r}, "
            f"found {len(files)}. The previous stage probably failed silently."
        )
    return len(files)


# ---------------------------------------------------------------------------
# Corpus / dataset invariants
# ---------------------------------------------------------------------------

def assert_row_count(
    n: int,
    *,
    min_rows: int,
    max_rows: Optional[int] = None,
    label: str,
) -> None:
    if n < min_rows:
        raise PipelineCheckError(
            f"{label}: row count {n} below floor {min_rows}. "
            f"Earlier stage likely dropped most data silently."
        )
    if max_rows is not None and n > max_rows:
        raise PipelineCheckError(
            f"{label}: row count {n} above ceiling {max_rows}. "
            f"Something unexpectedly multiplied the dataset."
        )


def assert_split_scenes(
    actual_scenes: Sequence[str],
    expected_scenes: Iterable[str],
    *,
    label: str,
) -> None:
    """Asserts the actual scene set matches the expected set exactly.

    Use to verify the pair-log / state-corpus loader read every split
    scene declared in the YAML (no silent drop because a file was
    missing on disk).
    """
    actual = set(actual_scenes)
    expected = set(expected_scenes)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise PipelineCheckError(
            f"{label}: scene set mismatch. "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )


# ---------------------------------------------------------------------------
# Tracker-bin sanity (nn_path / nn_state_path / detector engine)
# ---------------------------------------------------------------------------

def assert_tracker_bins_present(
    *,
    nn_path: Optional[str],
    nn_state_path: Optional[str],
    detector_engine: Optional[str],
    label: str = "tracker_config",
) -> None:
    """Verifies the NN .bin and detector engine files exist BEFORE the
    C tracker tries to load them. Failing here gives a clean Python
    traceback at the pipeline boundary instead of an opaque C abort
    deep inside `track_stream_create`.
    """
    if nn_path is not None and nn_path != "":
        assert_file_exists(nn_path, f"{label}.utrack.nn_path")
    if nn_state_path is not None and nn_state_path != "":
        assert_file_exists(nn_state_path, f"{label}.utrack.nn_state_path")
    if detector_engine is not None and detector_engine != "":
        assert_file_exists(
            detector_engine, f"{label}.inference_config.detection.trt"
        )


# ---------------------------------------------------------------------------
# Recipe provenance — log everything at pipeline start
# ---------------------------------------------------------------------------

def print_pipeline_banner(
    *,
    stage: str,
    inputs: Mapping[str, object],
    outputs: Mapping[str, object],
) -> None:
    """Emits a single, human-readable banner naming exactly what this
    stage reads and writes, so logs are self-describing.

    Use at every pipeline step so a future archaeologist can rebuild
    the recipe from logs alone.
    """
    print(f"=== {stage} ===")
    print("inputs:")
    for k, v in inputs.items():
        print(f"  {k}: {v}")
    print("outputs:")
    for k, v in outputs.items():
        print(f"  {k}: {v}")
