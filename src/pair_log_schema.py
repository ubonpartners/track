"""
Schema for the per-frame (track, det) pair feature trace.

This is the Python mirror of
ubon_cstuff/src/track/utrack/utrack_pair_trace.h.  Adding/removing fields
or changing their order requires the matching change there, and the
round-trip parity test in track/ml/round_trip_schema_test.py must
pass before merging either side.

Design notes:
- The C side packs records with `#pragma pack(push, 4)` so that uint64
  track_id is 4-byte aligned. Numpy's `align=False` matches this.
- All floats are float32. Integer types match the C declaration.
- `__pad__` is the trailing 1-byte padding the C struct carries to keep
  the record size a multiple of 4 bytes.

When loading bytes that came over the bindings:
    arr = np.frombuffer(data, dtype=PAIR_LOG_DTYPE)
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


PAIR_LOG_VERSION = 3
PAIR_LOG_MAGIC = 0x55545054  # 'UTPT' big-endian, matches C #define


# (numpy dtype, name) pairs in the order the C struct lays them out.
_FIELDS: List[Tuple[str, np.dtype]] = [
    ("track_id",         np.dtype("<u8")),
    ("det_index",        np.dtype("<u4")),
    ("scene_density",    np.dtype("<u4")),
    ("frame_time",       np.dtype("<f4")),
    ("time_since_det",   np.dtype("<f4")),
    ("observations",     np.dtype("<u4")),
    ("num_missed",       np.dtype("<u4")),
    ("of_score",         np.dtype("<f4")),
    ("kf_score",         np.dtype("<f4")),
    ("kf_d2",            np.dtype("<f4")),
    ("reid_cos_raw",     np.dtype("<f4")),
    ("reid_z",           np.dtype("<f4")),
    ("sim_term",         np.dtype("<f4")),
    ("ocm_cos",          np.dtype("<f4")),
    ("track_speed",      np.dtype("<f4")),
    ("h_ratio",          np.dtype("<f4")),
    ("a_ratio",          np.dtype("<f4")),
    ("conf_delta",       np.dtype("<f4")),
    ("track_x0",         np.dtype("<f4")),
    ("track_y0",         np.dtype("<f4")),
    ("track_x1",         np.dtype("<f4")),
    ("track_y1",         np.dtype("<f4")),
    ("det_x0",           np.dtype("<f4")),
    ("det_y0",           np.dtype("<f4")),
    ("det_x1",           np.dtype("<f4")),
    ("det_y1",           np.dtype("<f4")),
    ("size_ratio",       np.dtype("<f4")),
    ("det_conf",         np.dtype("<f4")),
    ("prev_det_conf",    np.dtype("<f4")),
    ("pose_kp_visible",  np.dtype("<u4")),
    # v2 face/subbox features + v3 OF-warped subbox DIoU. Order matches
    # utrack_pair_trace.h.
    ("det_subbox_conf",  np.dtype("<f4")),
    ("track_subbox_conf",np.dtype("<f4")),
    ("subbox_iou",       np.dtype("<f4")),
    ("subbox_diou_warped", np.dtype("<f4")),
    ("det_fiqa_score",   np.dtype("<f4")),
    ("pre_thr_score",    np.dtype("<f4")),
    ("match_cost_score", np.dtype("<f4")),
    ("pass_id",          np.dtype("u1")),
    ("reid_stats_valid", np.dtype("u1")),
    ("was_matched",      np.dtype("u1")),
    ("__pad__",          np.dtype("u1")),
]


PAIR_LOG_DTYPE = np.dtype([(name, dt) for name, dt in _FIELDS])

# v2 dtype — same field order as v3 minus `subbox_diou_warped`. Used by
# decode_records to read pair-log corpora generated before the 2026-05-11
# schema bump (record size 152 bytes). Decoded rows are upcast to the v3
# dtype with subbox_diou_warped=0.0, so downstream consumers always see a
# v3 record. Drop this fallback once all live corpora are regenerated.
_FIELDS_V2 = [(n, dt) for n, dt in _FIELDS if n != "subbox_diou_warped"]
PAIR_LOG_DTYPE_V2 = np.dtype(_FIELDS_V2)


# Names of the float32 features used as model inputs. Excludes identifiers,
# pass_id, label, etc. — those are metadata, not features. Order matters:
# this is the column ordering for the (n, F) feature matrix the trainer
# will consume.
PAIR_LOG_FEATURE_NAMES: List[str] = [
    "of_score",
    "kf_score",
    "kf_d2",
    "reid_cos_raw",
    "reid_z",
    "sim_term",
    "ocm_cos",
    "track_speed",
    "h_ratio",
    "a_ratio",
    "conf_delta",
    "size_ratio",
    "det_conf",
    "prev_det_conf",
    "time_since_det",
    # v2 face/subbox
    "det_subbox_conf",
    "track_subbox_conf",
    "subbox_iou",
    "subbox_diou_warped",
    "det_fiqa_score",
    # Integer-typed inputs come through as float32 after a log transform
    # in the trainer; we keep the raw integer in the record and let the
    # trainer compute log1p on the fly.
    "observations",
    "num_missed",
    "pose_kp_visible",
    "scene_density",
]


def record_size_bytes() -> int:
    """Return the on-the-wire size of one pair record."""
    return PAIR_LOG_DTYPE.itemsize


def decode_records(data: bytes, n_records: int) -> np.ndarray:
    """Validate length and decode raw bytes into a structured array.

    Accepts either current (v3) or legacy (v2) records, distinguished by
    byte length. v2 records are upcast to the v3 dtype with the new
    `subbox_diou_warped` column zeroed — letting state-corpus builds
    consume corpora generated before the schema bump. Drop the v2
    fallback once all live pair-log corpora are regenerated.
    """
    n = int(n_records)
    v3_size = n * PAIR_LOG_DTYPE.itemsize
    v2_size = n * PAIR_LOG_DTYPE_V2.itemsize
    if len(data) == v3_size:
        return np.frombuffer(data, dtype=PAIR_LOG_DTYPE)
    if len(data) == v2_size:
        v2 = np.frombuffer(data, dtype=PAIR_LOG_DTYPE_V2)
        out = np.zeros((n,), dtype=PAIR_LOG_DTYPE)
        for name in PAIR_LOG_DTYPE_V2.names:
            out[name] = v2[name]
        # subbox_diou_warped already 0 from np.zeros.
        return out
    raise ValueError(
        f"pair-trace size mismatch: got {len(data)} bytes for "
        f"{n} records; expected {v3_size} (v3) or {v2_size} (v2). "
        f"Likely a schema mismatch between C and Python — re-check "
        f"utrack_pair_trace.h vs pair_log_schema.py."
    )
