"""
Build per-(track, frame, prior_state) examples for the unified state-
transition head (UTRACK_NN.md follow-on).

Three binary classifiers share a backbone:
  promote: UNCONFIRMED → TRACKED      (output gate / FP-suppression decision)
  demote:  TRACKED     → LOST         (start-of-occlusion decision)
  drop:    UNCONFIRMED|LOST → REMOVED (phantom / unrecoverable decision)

Each example's loss masks the heads that don't fire for its prior_state.

Inputs (per example):
  prior_state one-hot[3] (UNCONFIRMED/TRACKED/LOST)
  matched_this_frame (0/1)
  log1p(observations), log1p(num_missed)
  time_since_detect (seconds)
  log1p(scene_density)
  det_conf (0 if not matched)
  prev_det_conf (carried even when not matched)
  phase3_pair_score (0 if not matched)
  e_track[16] — replayed via the trained Phase 3 f_obs subnet when
                --phase3-model is supplied (otherwise zeros).

The e_track replay uses the same offline EMA update as the Phase 3
trainer: e_track ← (1-α)·e_track + α·f_obs(z(obs)) on each matched
frame, with α and (mean, std) from the checkpoint.  The e_track stored
in each example is the post-update value (i.e. what a state head running
*after* match resolution would see).

Labels via H-second GT lookahead (default H=3.0s):
  promote_label = 1 if track's GT-id history is consistent (≥2 distinct frames
                  aligned to the same GT id within the next H seconds)
  demote_label  = 1 if track's GT id stops being matched within the next
                  match_iou window for the next H seconds
  drop_label    = 1 if track will never re-align with any GT within H seconds

Usage:
  python -m bench.build_state_corpus \\
      --pair-log-dir runs/track_analysis/pair_log_v4_nopose \\
      --gt-config bench/pair_log_config_v3_p2off.yaml \\
      --out bench/data/state_corpus

Splits follow the dataset_split tags in the gt-config (train/val/test).
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

import src.trackset as ts
from src.analysis.modules import _best_iou_match, _gt_objects_at_time_class
from src.analysis.pair_log_schema import (
    PAIR_LOG_DTYPE,
    PAIR_LOG_MAGIC,
    PAIR_LOG_VERSION,
    decode_records,
    record_size_bytes,
)


STATE_UNCONFIRMED = 0
STATE_TRACKED = 1
STATE_LOST = 2
STATE_REMOVED = 3

# Match the C-side defaults so the offline replay is faithful to production.
DEFAULT_K_MIN = 2
DEFAULT_TRACK_BUFFER_SEC = 2.0
DEFAULT_MAX_MISSED = 10

# e_track dim. Must match the trained Phase 3 e_dim AND the state-corpus
# EXAMPLE_DTYPE field width below.
E_TRACK_DIM = 16


# --- f_obs subnet replay (numpy, no torch dep at extract time) --------------

class FObsReplay:
    """Tiny numpy replay of the Phase 3 f_obs MLP (13 → tower_hidden → 16).

    Loads the relevant weights, biases, normalisation stats and EMA α from a
    Phase 3 checkpoint (.pt). On each matched frame we feed the 13-dim obs
    feature row through z-score → linear → ReLU → linear and EMA-update the
    per-track accumulator. Pure numpy keeps this dependency-light and ~10x
    faster than torch on CPU for a handful of features at a time.
    """

    OBS_FEATURE_NAMES_V1 = (
        "reid_cos_raw", "reid_z", "kf_d2", "of_score", "kf_score", "sim_term",
        "ocm_cos", "track_speed", "log_observations", "log_num_missed",
        "pose_kp_visible", "det_conf", "prev_det_conf",
    )
    OBS_FEATURE_NAMES_V2 = OBS_FEATURE_NAMES_V1 + (
        "det_subbox_conf", "track_subbox_conf", "det_fiqa_score",
    )

    def __init__(self, ckpt_path: str):
        import torch  # local import — only when --phase3-model is supplied
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.alpha = float(ckpt["alpha"])
        self.e_dim = int(ckpt["e_dim"])
        self.obs_in = int(ckpt["obs_in"])
        if self.e_dim != E_TRACK_DIM:
            raise ValueError(
                f"phase3 checkpoint e_dim={self.e_dim} doesn't match "
                f"corpus E_TRACK_DIM={E_TRACK_DIM}; bump the dtype "
                f"width or retrain to match"
            )
        names = tuple(ckpt["obs_feature_names"])
        if names == self.OBS_FEATURE_NAMES_V1:
            self.with_face = False
        elif names == self.OBS_FEATURE_NAMES_V2:
            self.with_face = True
        else:
            raise ValueError(
                f"phase3 obs_feature_names not recognised; expected v1 "
                f"({self.OBS_FEATURE_NAMES_V1}) or v2 "
                f"({self.OBS_FEATURE_NAMES_V2}); got {names}"
            )
        self.mean = np.asarray(ckpt["obs_mean"], dtype=np.float32)
        self.std  = np.asarray(ckpt["obs_std"],  dtype=np.float32)
        sd = ckpt["state_dict"]
        self.W1 = sd["f_obs.net.0.weight"].numpy().astype(np.float32)  # (H, obs_in)
        self.b1 = sd["f_obs.net.0.bias"].numpy().astype(np.float32)    # (H,)
        self.W2 = sd["f_obs.net.2.weight"].numpy().astype(np.float32)  # (E, H)
        self.b2 = sd["f_obs.net.2.bias"].numpy().astype(np.float32)    # (E,)

    def build_obs_row(self, rec) -> np.ndarray:
        """Build the obs feature row from one pair-trace record. Width
        depends on whether the loaded ckpt is v1 (13) or v2 (16)."""
        row = np.empty(self.obs_in, dtype=np.float32)
        row[0]  = float(rec["reid_cos_raw"])
        row[1]  = float(rec["reid_z"])
        row[2]  = float(rec["kf_d2"])
        row[3]  = float(rec["of_score"])
        row[4]  = float(rec["kf_score"])
        row[5]  = float(rec["sim_term"])
        row[6]  = float(rec["ocm_cos"])
        row[7]  = float(rec["track_speed"])
        row[8]  = float(np.log1p(float(rec["observations"])))
        row[9]  = float(np.log1p(float(rec["num_missed"])))
        row[10] = float(rec["pose_kp_visible"])
        row[11] = float(rec["det_conf"])
        row[12] = float(rec["prev_det_conf"])
        if self.with_face:
            # Pair-trace records have these fields after the v2 schema bump.
            # Older v1 .npz files won't have them and would raise KeyError.
            row[13] = float(rec["det_subbox_conf"])
            row[14] = float(rec["track_subbox_conf"])
            row[15] = float(rec["det_fiqa_score"])
        np.nan_to_num(row, copy=False)
        return row

    def f_obs(self, raw_row: np.ndarray) -> np.ndarray:
        """Forward pass: z-score → Linear → ReLU → Linear. Returns (e_dim,)."""
        x = (raw_row - self.mean) / self.std
        h = self.W1 @ x + self.b1
        np.maximum(h, 0.0, out=h)
        return self.W2 @ h + self.b2

    def update(self, e_track: np.ndarray, raw_row: np.ndarray,
               seen: bool) -> Tuple[np.ndarray, bool]:
        """EMA-update e_track for one matched frame.

        First match sets e_track = f_obs(obs); subsequent matches blend.
        Mirrors `compute_etrack_for_pairs` in train_phase3.py.
        Returns (new_e_track, new_seen).
        """
        f = self.f_obs(raw_row)
        if not seen:
            return f.astype(np.float32, copy=False), True
        return ((1.0 - self.alpha) * e_track + self.alpha * f).astype(np.float32), True


# --- per-example record schema -----------------------------------------------

EXAMPLE_DTYPE = np.dtype([
    # identifiers
    ("sequence",         "U64"),
    ("track_id",         "<u8"),
    ("frame_idx",        "<i4"),
    ("frame_time",       "<f4"),

    # prior_state at decision time (end-of-frame, after rule replay).
    # 0=UNCONFIRMED, 1=TRACKED, 2=LOST.
    ("prior_state",      "u1"),

    # which heads are *valid* for this example (mask in loss)
    ("valid_promote",    "u1"),
    ("valid_demote",     "u1"),
    ("valid_drop",       "u1"),

    # features
    ("matched",          "u1"),
    ("observations",     "<u4"),
    ("num_missed",       "<u4"),
    ("time_since_det",   "<f4"),
    ("scene_density",    "<u4"),
    ("det_conf",         "<f4"),
    ("prev_det_conf",    "<f4"),
    ("phase3_pair_score","<f4"),
    # 2026-05-05: extra spatial / detection-quality context. The state
    # head was originally blind to these (it only saw e_track's lossy
    # 16-dim compression of pose/motion). Adding them lets it
    # distinguish "object leaving frame" (drop is correct) from
    # "object briefly occluded mid-frame" (drop is wrong). Coords are
    # normalised [0,1] in both pair-trace and .ubtrk2.
    ("near_edge",        "<f4"),  # min distance to image edge in [0, 0.5]
    ("det_w",            "<f4"),  # detection box width in [0, 1]
    ("det_h",            "<f4"),  # detection box height in [0, 1]
    ("log_aspect",       "<f4"),  # log(w / max(eps, h)), clamped to [-3, 3]
    ("log_pose_kp",      "<f4"),  # log1p(pose_kp_visible)
    # e_track[16] from Phase 3 f_obs replay — zeros if --phase3-model not given
    ("e_track",          "<f4", (16,)),

    # labels (only meaningful where valid_<x>=1)
    ("promote_label",    "u1"),
    ("demote_label",     "u1"),
    ("drop_label",       "u1"),

    # Per-(track, frame) GT alignment from track_gt_history. -1 = no GT
    # alignment that frame (i.e., either no GT visible or the run-track box
    # didn't IoU-match any GT). Drives the simulator's per-frame TP/FP score.
    ("gt_id_now",        "<i8"),

    # Sample weight for the loss. 1.0 for natural-trajectory examples;
    # < 1.0 for synthetic / augmented examples (e.g. fake-start delay).
    # The trainer multiplies BCE by this; existing code that read this
    # dtype without the field stays compatible because the field defaults
    # to 1.0 on construction.
    ("weight",           "<f4"),
])


# --- spatial / detection-quality feature helpers ----------------------------

def _box_features(x0: float, y0: float, x1: float, y1: float,
                  pose_kp_visible: int) -> Tuple[float, float, float, float, float]:
    """Return (near_edge, w, h, log_aspect, log_pose_kp) for a box in
    normalised [0,1] coords. Order matches EXAMPLE_DTYPE."""
    w = max(0.0, x1 - x0)
    h = max(0.0, y1 - y0)
    near_edge = max(0.0, min(min(x0, y0), min(1.0 - x1, 1.0 - y1)))
    aspect = w / max(1e-6, h)
    log_aspect = float(np.clip(np.log(max(1e-3, aspect)), -3.0, 3.0))
    log_pose_kp = float(np.log1p(max(0, int(pose_kp_visible))))
    return float(near_edge), float(w), float(h), log_aspect, log_pose_kp


# --- per-track state during the offline replay -------------------------------

@dataclass
class TrackHist:
    """Replay state for a single track-id over its lifetime."""
    track_id: int
    state: int
    observations: int
    num_missed: int
    last_detect_time: float
    last_box: Tuple[float, float, float, float]
    prev_det_conf: float
    # Per-frame GT-id history (frame_idx → gt_track_id or -1 for "no GT alignment").
    gt_history: Dict[int, int]
    # Frames at which the track existed and was alive (state ≠ REMOVED).
    alive_frames: List[int]
    # Frames at which the track had matched_this_frame=1.
    match_frames: List[int]
    # f_obs EMA accumulator (zeros until first matched obs is seen).
    e_track: np.ndarray = None  # type: ignore[assignment]
    e_seen: bool = False
    # Last seen pose keypoint count, carried over for absent-track
    # examples that have no pair-trace record this frame.
    last_pose_kp: int = 0


# --- corpus extraction for one sequence -------------------------------------

def extract_sequence(
    sequence_name: str,
    ubtrk2_path: str,
    gt_trackset_path: str,
    *,
    k_min: int,
    buffer_sec: float,
    max_missed: int,
    h_lookahead_sec: float,
    seed_match_iou: float = 0.5,
    class_name: str = "person",
    f_obs_replay: Optional["FObsReplay"] = None,
) -> np.ndarray:
    """Walk one sequence and emit per-(track, frame, prior_state) examples.

    Algorithm:
      1. Walk frames in time order. For each, decode pair-trace records and
         the .ubtrk2 output objects.
      2. Maintain per-track replay state: state, observations, num_missed,
         last_detect_time, prev_det_conf.
      3. After processing each frame's matches:
           - For UNCONFIRMED/TRACKED/LOST tracks at this frame, emit one
             example with prior_state = state-at-start-of-frame and the
             features that would have been visible to a head running at
             end-of-frame (post-match update). The label heads for that
             prior_state are marked valid; the others are masked.
           - Apply the current hard-coded rules to compute next-frame state.
      4. After walking all frames, do a second pass to fill in lookahead-
         based labels using the per-track GT-id history.
    """
    run = ts.TrackSet(ubtrk2_path, decode_payloads=False, analysis_mode=True)
    gt = ts.TrackSet(gt_trackset_path)
    sequence_ctx = {
        "sequence_name": sequence_name,
        "run_trackset": run,
        "gt_trackset": gt,
    }

    tracks: Dict[int, TrackHist] = {}
    examples: List[Dict[str, Any]] = []
    # Cache: per-track GT-id at each frame. -1 = no GT alignment that frame.
    track_gt_history: Dict[int, Dict[int, int]] = defaultdict(dict)

    for frame_idx, frame in enumerate(run.frames):
        frame_time = float(frame.get("frame_time", 0.0))
        gt_curr = _gt_objects_at_time_class(sequence_ctx, frame_time, class_name)

        # --- 1) Build track_gt_history from this frame's run outputs.
        # The .ubtrk2 only carries TRACKED objects; UNCONFIRMED/LOST tracks
        # don't appear here. We use it to fix the GT-id label for TRACKED
        # tracks specifically. frame["objects"] is {track_id: {box,class,...}}.
        objects = frame.get("objects") or {}
        for tid, obj in objects.items():
            tid = int(tid)
            box = obj.get("box") if isinstance(obj, dict) else None
            if box is None:
                continue
            best_gt, best_iou = _best_iou_match(list(box), gt_curr)
            if best_gt is not None and best_iou >= seed_match_iou:
                track_gt_history[tid][frame_idx] = int(best_gt.track_id)
            else:
                track_gt_history[tid][frame_idx] = -1

        # --- 2) Decode this frame's pair-trace.
        debug = frame.get("debug") or {}
        trc = debug.get("match_cost_trace")
        if not isinstance(trc, dict):
            continue
        magic = int(trc.get("magic", 0)); version = int(trc.get("version", 0))
        if magic != PAIR_LOG_MAGIC or version != PAIR_LOG_VERSION:
            raise ValueError(f"{sequence_name}: trace magic/version mismatch")
        if int(trc.get("record_size", 0)) != record_size_bytes():
            raise ValueError(f"{sequence_name}: trace record_size mismatch")
        n_records = int(trc.get("n_records", 0))
        if n_records == 0:
            records = np.empty((0,), dtype=PAIR_LOG_DTYPE)
        else:
            records = decode_records(trc.get("data") or b"", n_records)

        # --- 3) Group records by track_id.
        per_track_recs: Dict[int, np.ndarray] = {}
        if records.size:
            tids = records["track_id"]
            for tid in np.unique(tids):
                per_track_recs[int(tid)] = records[tids == tid]

        # --- 4) Process every track that appears in this frame's trace.
        # (UNCONFIRMED/TRACKED/LOST: they all appear if they had any candidate
        # det this frame. A track absent from the trace this frame had no
        # candidate det; we still want to tick its state via timeout rules.)
        seen_this_frame: set = set()
        for tid, recs in per_track_recs.items():
            seen_this_frame.add(tid)

            # Was matched this frame? Pick the matched record (was_matched=1).
            matched_recs = recs[recs["was_matched"] == 1]
            matched = bool(len(matched_recs))
            match_rec = matched_recs[0] if matched else None

            if tid not in tracks:
                # First appearance: track was just created. State=UNCONFIRMED,
                # observations=1, num_missed=0. Prior_state for the example
                # is UNCONFIRMED (the state it was just created in).
                first_rec = recs[0]
                prior_state = STATE_UNCONFIRMED
                observations = 1
                num_missed = 0
                last_detect_time = frame_time
                prev_det_conf = float(first_rec["prev_det_conf"])
                last_box = (
                    float(first_rec["track_x0"]),
                    float(first_rec["track_y0"]),
                    float(first_rec["track_x1"]),
                    float(first_rec["track_y1"]),
                )
                e_track_cur = np.zeros(E_TRACK_DIM, dtype=np.float32)
                e_seen = False
            else:
                th = tracks[tid]
                prior_state = th.state
                # Tentative post-frame counters; finalised after rule replay.
                if matched:
                    observations = th.observations + 1
                    num_missed = 0
                    last_detect_time = frame_time
                else:
                    observations = th.observations
                    num_missed = th.num_missed + 1
                    last_detect_time = th.last_detect_time
                prev_det_conf = float(recs[0]["prev_det_conf"])
                last_box = (
                    float(recs[0]["track_x0"]),
                    float(recs[0]["track_y0"]),
                    float(recs[0]["track_x1"]),
                    float(recs[0]["track_y1"]),
                )
                e_track_cur = (th.e_track if th.e_track is not None
                               else np.zeros(E_TRACK_DIM, dtype=np.float32))
                e_seen = th.e_seen

            scene_density = int(recs[0]["scene_density"])
            time_since_det = float(frame_time - last_detect_time)
            det_conf = float(match_rec["det_conf"]) if matched else 0.0
            phase3_pair_score = float(match_rec["match_cost_score"]) if matched else 0.0

            # Spatial / detection-quality features. When matched, use the
            # detection's own box and pose; when unmatched, fall back to
            # the track box and a zero pose (no det this frame).
            if matched:
                bx0 = float(match_rec["det_x0"]); by0 = float(match_rec["det_y0"])
                bx1 = float(match_rec["det_x1"]); by1 = float(match_rec["det_y1"])
                pose_kp = int(match_rec["pose_kp_visible"])
            else:
                bx0 = float(recs[0]["track_x0"]); by0 = float(recs[0]["track_y0"])
                bx1 = float(recs[0]["track_x1"]); by1 = float(recs[0]["track_y1"])
                pose_kp = 0
            near_edge, det_w, det_h, log_aspect, log_pose_kp = _box_features(
                bx0, by0, bx1, by1, pose_kp,
            )

            # EMA-update e_track on matched obs using the trained Phase 3 f_obs.
            # Matches the offline frozen-accumulator update in train_phase3.py.
            # When no f_obs replay is configured, e_track stays zeros (v0).
            if matched and f_obs_replay is not None:
                obs_row = f_obs_replay.build_obs_row(match_rec)
                e_track_cur, e_seen = f_obs_replay.update(
                    e_track_cur, obs_row, e_seen,
                )

            # Determine which heads are valid for this prior_state.
            valid_promote = 1 if prior_state == STATE_UNCONFIRMED else 0
            valid_demote  = 1 if prior_state == STATE_TRACKED else 0
            valid_drop    = 1 if prior_state in (STATE_UNCONFIRMED, STATE_LOST) else 0

            examples.append(dict(
                sequence=sequence_name, track_id=int(tid),
                frame_idx=int(frame_idx), frame_time=float(frame_time),
                prior_state=int(prior_state),
                valid_promote=int(valid_promote),
                valid_demote=int(valid_demote),
                valid_drop=int(valid_drop),
                matched=int(matched),
                observations=int(observations),
                num_missed=int(num_missed),
                time_since_det=float(time_since_det),
                scene_density=int(scene_density),
                det_conf=float(det_conf),
                prev_det_conf=float(prev_det_conf),
                phase3_pair_score=float(phase3_pair_score),
                near_edge=near_edge, det_w=det_w, det_h=det_h,
                log_aspect=log_aspect, log_pose_kp=log_pose_kp,
                e_track=e_track_cur.copy(),
                # promote/demote/drop labels filled in pass 2
                promote_label=0, demote_label=0, drop_label=0,
                # GT alignment at this exact frame (filled in pass 2)
                gt_id_now=-1,
                weight=1.0,
            ))

            # --- 5) Apply C-style rules to compute next-frame state.
            #    UNCONFIRMED + matched + observations >= K_min  → TRACKED
            #    UNCONFIRMED + matched + observations <  K_min  → UNCONFIRMED
            #    UNCONFIRMED + not matched                       → REMOVED
            #    TRACKED + matched                               → TRACKED
            #    TRACKED + not matched + num_missed >= 2         → LOST
            #    TRACKED + not matched + num_missed <  2         → TRACKED
            #    LOST + matched                                  → TRACKED (recovery)
            #    LOST + not matched + (time>buffer or missed>=max) → REMOVED
            #    LOST + not matched + still in window            → LOST
            if prior_state == STATE_UNCONFIRMED:
                if matched:
                    next_state = (
                        STATE_TRACKED if observations >= k_min
                        else STATE_UNCONFIRMED
                    )
                else:
                    next_state = STATE_REMOVED
            elif prior_state == STATE_TRACKED:
                if matched:
                    next_state = STATE_TRACKED
                elif num_missed >= 2:
                    next_state = STATE_LOST
                else:
                    next_state = STATE_TRACKED
            elif prior_state == STATE_LOST:
                if matched:
                    next_state = STATE_TRACKED
                elif (time_since_det >= buffer_sec
                      or num_missed >= max_missed):
                    next_state = STATE_REMOVED
                else:
                    next_state = STATE_LOST
            else:
                next_state = STATE_REMOVED

            if next_state == STATE_REMOVED:
                if tid in tracks:
                    del tracks[tid]
            else:
                tracks[tid] = TrackHist(
                    track_id=tid,
                    state=int(next_state),
                    observations=int(observations),
                    num_missed=int(num_missed),
                    last_detect_time=float(last_detect_time),
                    last_box=last_box,
                    prev_det_conf=float(det_conf if matched else prev_det_conf),
                    gt_history={},
                    alive_frames=[],
                    match_frames=[],
                    e_track=e_track_cur,
                    e_seen=e_seen,
                    last_pose_kp=int(pose_kp if matched else (
                        tracks[tid].last_pose_kp if tid in tracks else 0)),
                )

        # --- 6) Tick state for tracks NOT seen in trace this frame
        # (no candidate dets at all). Treated as "not matched."
        absent_tids = [tid for tid in tracks.keys() if tid not in seen_this_frame]
        for tid in absent_tids:
            th = tracks[tid]
            prior_state = th.state
            observations = th.observations
            num_missed = th.num_missed + 1
            last_detect_time = th.last_detect_time
            time_since_det = float(frame_time - last_detect_time)

            valid_promote = 1 if prior_state == STATE_UNCONFIRMED else 0
            valid_demote  = 1 if prior_state == STATE_TRACKED else 0
            valid_drop    = 1 if prior_state in (STATE_UNCONFIRMED, STATE_LOST) else 0

            e_track_cur = (th.e_track if th.e_track is not None
                           else np.zeros(E_TRACK_DIM, dtype=np.float32))
            # Spatial features: use last known track box, last seen pose count.
            near_edge, det_w, det_h, log_aspect, log_pose_kp = _box_features(
                th.last_box[0], th.last_box[1], th.last_box[2], th.last_box[3],
                th.last_pose_kp,
            )

            examples.append(dict(
                sequence=sequence_name, track_id=int(tid),
                frame_idx=int(frame_idx), frame_time=float(frame_time),
                prior_state=int(prior_state),
                valid_promote=int(valid_promote),
                valid_demote=int(valid_demote),
                valid_drop=int(valid_drop),
                matched=0,
                observations=int(observations),
                num_missed=int(num_missed),
                time_since_det=float(time_since_det),
                scene_density=0,
                det_conf=0.0,
                prev_det_conf=float(th.prev_det_conf),
                phase3_pair_score=0.0,
                near_edge=near_edge, det_w=det_w, det_h=det_h,
                log_aspect=log_aspect, log_pose_kp=log_pose_kp,
                e_track=e_track_cur.copy(),
                promote_label=0, demote_label=0, drop_label=0,
                gt_id_now=-1,
            ))

            # Rule replay (matched=False).
            if prior_state == STATE_UNCONFIRMED:
                next_state = STATE_REMOVED
            elif prior_state == STATE_TRACKED:
                next_state = STATE_LOST if num_missed >= 2 else STATE_TRACKED
            elif prior_state == STATE_LOST:
                if (time_since_det >= buffer_sec
                    or num_missed >= max_missed):
                    next_state = STATE_REMOVED
                else:
                    next_state = STATE_LOST
            else:
                next_state = STATE_REMOVED

            if next_state == STATE_REMOVED:
                del tracks[tid]
            else:
                tracks[tid].state = int(next_state)
                tracks[tid].num_missed = int(num_missed)

    if not examples:
        return np.empty((0,), dtype=EXAMPLE_DTYPE)

    # --- second pass: lookahead-based labels --------------------------------
    # For each (track_id, frame_idx) pair in track_gt_history, the GT id at
    # that frame is known. We use this to label per-example.
    n_frames = len(run.frames)
    frame_times = np.asarray(
        [float(run.frames[i].get("frame_time", 0.0)) for i in range(n_frames)],
        dtype=np.float64,
    )

    # Per-track: sorted list of (frame_idx, gt_id) where gt_id != -1 (i.e.,
    # the run-track was aligned with some GT).
    track_aligned_frames: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for tid, hist in track_gt_history.items():
        for fi, gid in sorted(hist.items()):
            if gid != -1:
                track_aligned_frames[tid].append((int(fi), int(gid)))

    # Build per-(track, frame_idx) future-alignment indicator: does this track
    # have an aligned GT frame within H seconds of frame_idx?
    def future_alignment(tid: int, fi: int) -> Tuple[bool, Optional[int]]:
        """Returns (has_future_alignment, future_gt_id_or_None)."""
        aligned = track_aligned_frames.get(tid, [])
        if not aligned:
            return False, None
        t_now = frame_times[fi] if fi < n_frames else 0.0
        for fi2, gid2 in aligned:
            if fi2 <= fi:
                continue
            t2 = frame_times[fi2] if fi2 < n_frames else 0.0
            if t2 - t_now <= h_lookahead_sec:
                return True, gid2
            else:
                break  # aligned is sorted; no point looking further
        return False, None

    # Per-track current-or-recent GT id (any aligned frame so far).
    def historic_gt_id(tid: int, fi: int) -> Optional[int]:
        aligned = track_aligned_frames.get(tid, [])
        last = None
        for fi2, gid2 in aligned:
            if fi2 <= fi:
                last = gid2
            else:
                break
        return last

    out = np.zeros((len(examples),), dtype=EXAMPLE_DTYPE)
    for i, ex in enumerate(examples):
        tid = ex["track_id"]; fi = ex["frame_idx"]
        prior_state = ex["prior_state"]

        future_ok, future_gid = future_alignment(tid, fi)
        prev_gid = historic_gt_id(tid, fi)
        ex["gt_id_now"] = int(track_gt_history.get(tid, {}).get(fi, -1))

        promote_label = 0
        demote_label = 0
        drop_label = 0

        if prior_state == STATE_UNCONFIRMED:
            # Promote = "this is real, expose it." Real iff the run-track
            # accumulates ≥1 GT-aligned frame in the next H seconds (combined
            # with the current frame's matched/aligned status). Cleaner:
            # promote_label = 1 if there's a GT-alignment in [fi, fi+H].
            promote_label = int(
                future_ok or (
                    track_gt_history.get(tid, {}).get(fi, -1) != -1
                )
            )
            # Drop = phantom: no GT alignment now, none in lookahead, AND
            # no historical alignment.
            drop_label = int(
                not future_ok
                and track_gt_history.get(tid, {}).get(fi, -1) == -1
                and prev_gid is None
            )
        elif prior_state == STATE_TRACKED:
            # Demote = "this track is going to lose its GT id soon." We use:
            # the run-track's current GT alignment exists, but NOT in the
            # next H seconds. (i.e., the GT object is going away.) Or the
            # current frame already lost alignment.
            cur_gid_this_frame = track_gt_history.get(tid, {}).get(fi, -1)
            demote_label = int(
                (cur_gid_this_frame == -1)  # already lost it
                and not future_ok            # and not coming back
            )
        elif prior_state == STATE_LOST:
            # Drop = "GT object never reappears within H seconds."
            drop_label = int(not future_ok)

        for k, v in ex.items():
            out[i][k] = v
        out[i]["promote_label"] = promote_label
        out[i]["demote_label"] = demote_label
        out[i]["drop_label"] = drop_label

    return out


# --- label-driven corpus extraction ------------------------------------------
#
# Why this exists: the original `extract_sequence` replays state with a
# deterministic K_min / missed≥2 / time-buffer rule set. The C runtime
# (with the state head loaded) does not — the head decides at every step.
# Examples emitted by the deterministic replay therefore live on a strictly
# narrower (state, obs, missed) manifold than what the head sees at
# inference, making most of the head's runtime queries OOD.
#
# `extract_sequence_label_driven` replaces the deterministic replay with
# a label-driven oracle: the same lookahead-based labels that pass 2 already
# computes are used in pass 1 to decide transitions. Hard floors stay
# (corpus must not produce transitions the runtime can't legally take):
#   - TRACKED + missed≥2 → LOST regardless of label
#   - LOST + (time≥buffer or missed≥max) → REMOVED regardless of label
#
# Plus two corpus/runtime parity fixes:
#   - track_gt_history is also populated from pair-trace track boxes, not
#     just from emitted (TRACKED) objects, so UNCONFIRMED/LOST tracks get
#     real GT-id labels instead of "no GT" by default.
#   - Track creation can short-circuit to TRACKED if det_conf >
#     immediate_confirm_thr, mirroring `utrack.c` line 1707.
#
# e_track is forced to zero throughout, which matches the runtime fix
# (zero e_track at the call site to utrack_build_state_features) until
# Phase 2a re-enables it with a corresponding match-cost NN in training.


def extract_sequence_label_driven(
    sequence_name: str,
    ubtrk2_path: str,
    gt_trackset_path: str,
    *,
    k_min: int,
    buffer_sec: float,
    max_missed: int,
    h_lookahead_sec: float,
    immediate_confirm_thr: float = 0.93,
    seed_match_iou: float = 0.5,
    class_name: str = "person",
    delay_aug_offsets: Tuple[int, ...] = (),
    delay_aug_tau: float = 2.0,
    f_obs_replay: Optional["FObsReplay"] = None,
) -> np.ndarray:
    """Walk one sequence and emit label-driven examples.

    delay_aug_offsets: optional tuple of K values. For each promotable track,
    emit additional 'delayed promotion' trajectory copies where the track is
    held UNCONFIRMED for K extra frames before promoting. Each copy's
    examples carry weight = exp(-K/delay_aug_tau). Empty tuple = no
    augmentation (Phase 1b default; switch on after the bin-coverage check).
    """
    run = ts.TrackSet(ubtrk2_path, decode_payloads=False, analysis_mode=True)
    gt = ts.TrackSet(gt_trackset_path)
    sequence_ctx = {
        "sequence_name": sequence_name,
        "run_trackset": run,
        "gt_trackset": gt,
    }
    n_frames = len(run.frames)
    frame_times = np.asarray(
        [float(run.frames[i].get("frame_time", 0.0)) for i in range(n_frames)],
        dtype=np.float64,
    )

    # ------------------------------------------------------------------
    # Pass A: build per-(tid, frame_idx) GT alignment using BOTH emitted
    # objects and pair-trace track boxes. The latter is essential — without
    # it, UNCONFIRMED/LOST tracks (which never appear in `frame.objects`)
    # would have no GT alignment record and every label would default to
    # "phantom", which is exactly the bias we are trying to remove.
    # ------------------------------------------------------------------
    track_gt_history: Dict[int, Dict[int, int]] = defaultdict(dict)
    per_track_recs_per_frame: Dict[int, Dict[int, np.ndarray]] = {}
    for fi, frame in enumerate(run.frames):
        ft = float(frame_times[fi])
        gt_curr = _gt_objects_at_time_class(sequence_ctx, ft, class_name)

        # 1. From emitted objects (TRACKED tracks).
        objects = frame.get("objects") or {}
        for tid_raw, obj in objects.items():
            tid = int(tid_raw)
            if not isinstance(obj, dict):
                continue
            box = obj.get("box")
            if box is None:
                continue
            best_gt, best_iou = _best_iou_match(list(box), gt_curr)
            if best_gt is not None and best_iou >= seed_match_iou:
                track_gt_history[tid][fi] = int(best_gt.track_id)
            else:
                track_gt_history[tid].setdefault(fi, -1)

        # 2. From pair-trace track boxes (UNCONFIRMED/LOST tracks).
        debug = frame.get("debug") or {}
        trc = debug.get("match_cost_trace")
        if not isinstance(trc, dict):
            per_track_recs_per_frame[fi] = {}
            continue
        if int(trc.get("magic", 0)) != PAIR_LOG_MAGIC:
            per_track_recs_per_frame[fi] = {}
            continue
        if int(trc.get("record_size", 0)) != record_size_bytes():
            raise ValueError(f"{sequence_name}: pair-trace record_size mismatch")
        n_records = int(trc.get("n_records", 0))
        if n_records == 0:
            per_track_recs_per_frame[fi] = {}
            continue
        records = decode_records(trc.get("data") or b"", n_records)
        per_track: Dict[int, np.ndarray] = {}
        tids = records["track_id"]
        for tid_unique in np.unique(tids):
            per_track[int(tid_unique)] = records[tids == tid_unique]
        per_track_recs_per_frame[fi] = per_track

        for tid, recs in per_track.items():
            if fi in track_gt_history.get(tid, {}):
                # Already filled from emitted objects — emitted box is
                # what the runtime would actually emit, prefer it.
                continue
            box = [
                float(recs[0]["track_x0"]),
                float(recs[0]["track_y0"]),
                float(recs[0]["track_x1"]),
                float(recs[0]["track_y1"]),
            ]
            best_gt, best_iou = _best_iou_match(box, gt_curr)
            if best_gt is not None and best_iou >= seed_match_iou:
                track_gt_history[tid][fi] = int(best_gt.track_id)
            else:
                track_gt_history[tid][fi] = -1

    # ------------------------------------------------------------------
    # Pass B: build label lookups (future_alignment, historic_gt_id).
    # ------------------------------------------------------------------
    track_aligned_frames: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for tid, hist in track_gt_history.items():
        for fi, gid in sorted(hist.items()):
            if gid != -1:
                track_aligned_frames[tid].append((int(fi), int(gid)))

    def future_alignment(tid: int, fi: int) -> bool:
        aligned = track_aligned_frames.get(tid, [])
        if not aligned:
            return False
        t_now = frame_times[fi] if fi < n_frames else 0.0
        for fi2, _gid in aligned:
            if fi2 <= fi:
                continue
            t2 = frame_times[fi2] if fi2 < n_frames else 0.0
            if t2 - t_now <= h_lookahead_sec:
                return True
            else:
                break
        return False

    def historic_gt_id(tid: int, fi: int) -> Optional[int]:
        aligned = track_aligned_frames.get(tid, [])
        last = None
        for fi2, gid2 in aligned:
            if fi2 <= fi:
                last = gid2
            else:
                break
        return last

    def oracle_promote(tid: int, fi: int) -> bool:
        return (
            future_alignment(tid, fi)
            or track_gt_history.get(tid, {}).get(fi, -1) != -1
        )

    def oracle_drop_unconfirmed(tid: int, fi: int) -> bool:
        return (
            not future_alignment(tid, fi)
            and track_gt_history.get(tid, {}).get(fi, -1) == -1
            and historic_gt_id(tid, fi) is None
        )

    def oracle_demote(tid: int, fi: int) -> bool:
        cur = track_gt_history.get(tid, {}).get(fi, -1)
        return cur == -1 and not future_alignment(tid, fi)

    def oracle_drop_lost(tid: int, fi: int) -> bool:
        return not future_alignment(tid, fi)

    # ------------------------------------------------------------------
    # Pass C: walk frames, replay state with label-driven oracle, emit
    # examples. Does this once per `delay_offset` value:
    #   delay_offset=0  → natural trajectory (always emitted; weight 1.0)
    #   delay_offset=k>0 → 'forced hold' for k extra frames on UNCONFIRMED
    #                       even when oracle says promote (weight exp(-k/τ))
    # The k>0 copies generate obs≥2 UNCONFIRMED examples for tracks that
    # would otherwise immediately promote — without these, the head learns
    # 'obs≥2 UNCONFIRMED → don't promote' because the only obs≥2 UNCONFIRMED
    # examples in the natural trajectory are bad tracks.
    # ------------------------------------------------------------------
    examples: List[Dict[str, Any]] = []
    delay_iter = (0,) + tuple(delay_aug_offsets)

    for delay_offset in delay_iter:
        weight = 1.0 if delay_offset == 0 else float(np.exp(-delay_offset / max(1e-3, delay_aug_tau)))
        tracks: Dict[int, "_TH"] = {}
        # held_remaining[tid] = how many MORE frames to force UNCONFIRMED on a
        # promotable track. Set to delay_offset on each track's first
        # appearance; decrements each frame; once at 0 the natural oracle
        # transition runs.
        held_remaining: Dict[int, int] = {}
        # Per-track f_obs EMA accumulator (e_track[16]) — updated on every
        # matched obs when f_obs_replay is supplied. Zeros otherwise.
        e_track_state: Dict[int, Tuple[np.ndarray, bool]] = {}

        for fi in range(n_frames):
            frame = run.frames[fi]
            frame_time = float(frame_times[fi])
            per_track = per_track_recs_per_frame.get(fi) or {}

            seen_this_frame: set = set()
            for tid, recs in per_track.items():
                seen_this_frame.add(tid)
                matched_recs = recs[recs["was_matched"] == 1]
                matched = bool(len(matched_recs))
                match_rec = matched_recs[0] if matched else None
                first_rec = recs[0]

                if tid not in tracks:
                    # First appearance: track is brand new.
                    prior_state = STATE_UNCONFIRMED
                    observations = 1
                    num_missed = 0
                    last_detect_time = frame_time
                    prev_det_conf = float(first_rec["prev_det_conf"])
                    last_box = (
                        float(first_rec["track_x0"]),
                        float(first_rec["track_y0"]),
                        float(first_rec["track_x1"]),
                        float(first_rec["track_y1"]),
                    )
                    held_remaining[tid] = int(delay_offset)
                    e_track_state[tid] = (np.zeros(E_TRACK_DIM, dtype=np.float32), False)
                    # immediate-confirm short-circuit: matches utrack.c:1707.
                    # Only fires on the natural trajectory (delay_offset==0)
                    # and only when this frame's matched det conf is above
                    # the threshold. For delay augmentation we always force
                    # UNCONFIRMED on first appearance.
                    if (delay_offset == 0
                        and matched
                        and float(match_rec["det_conf"]) > immediate_confirm_thr):
                        prior_state = STATE_TRACKED
                else:
                    th = tracks[tid]
                    prior_state = th.state
                    if matched:
                        observations = th.observations + 1
                        num_missed = 0
                        last_detect_time = frame_time
                    else:
                        observations = th.observations
                        num_missed = th.num_missed + 1
                        last_detect_time = th.last_detect_time
                    prev_det_conf = float(first_rec["prev_det_conf"])
                    last_box = (
                        float(first_rec["track_x0"]),
                        float(first_rec["track_y0"]),
                        float(first_rec["track_x1"]),
                        float(first_rec["track_y1"]),
                    )

                scene_density = int(first_rec["scene_density"])
                time_since_det = float(frame_time - last_detect_time)
                det_conf = float(match_rec["det_conf"]) if matched else 0.0
                phase3_pair_score = float(match_rec["match_cost_score"]) if matched else 0.0

                if matched:
                    bx0, by0 = float(match_rec["det_x0"]), float(match_rec["det_y0"])
                    bx1, by1 = float(match_rec["det_x1"]), float(match_rec["det_y1"])
                    pose_kp = int(match_rec["pose_kp_visible"])
                else:
                    bx0, by0 = float(first_rec["track_x0"]), float(first_rec["track_y0"])
                    bx1, by1 = float(first_rec["track_x1"]), float(first_rec["track_y1"])
                    pose_kp = 0
                near_edge, det_w, det_h, log_aspect, log_pose_kp = _box_features(
                    bx0, by0, bx1, by1, pose_kp,
                )

                # Update per-track e_track EMA on matched obs (when f_obs
                # replay is configured). The state head will see the POST-
                # update value, matching what the C runtime would feed it.
                e_track_cur, e_seen = e_track_state.get(
                    tid, (np.zeros(E_TRACK_DIM, dtype=np.float32), False))
                if matched and f_obs_replay is not None and match_rec is not None:
                    obs_row = f_obs_replay.build_obs_row(match_rec)
                    e_track_cur, e_seen = f_obs_replay.update(
                        e_track_cur, obs_row, e_seen)
                e_track_state[tid] = (e_track_cur, e_seen)

                valid_promote = 1 if prior_state == STATE_UNCONFIRMED else 0
                valid_demote  = 1 if prior_state == STATE_TRACKED else 0
                valid_drop    = 1 if prior_state in (STATE_UNCONFIRMED, STATE_LOST) else 0

                examples.append(dict(
                    sequence=sequence_name, track_id=int(tid),
                    frame_idx=int(fi), frame_time=float(frame_time),
                    prior_state=int(prior_state),
                    valid_promote=int(valid_promote),
                    valid_demote=int(valid_demote),
                    valid_drop=int(valid_drop),
                    matched=int(matched),
                    observations=int(observations),
                    num_missed=int(num_missed),
                    time_since_det=float(time_since_det),
                    scene_density=int(scene_density),
                    det_conf=float(det_conf),
                    prev_det_conf=float(prev_det_conf),
                    phase3_pair_score=float(phase3_pair_score),
                    near_edge=near_edge, det_w=det_w, det_h=det_h,
                    log_aspect=log_aspect, log_pose_kp=log_pose_kp,
                    e_track=e_track_cur.copy(),
                    promote_label=0, demote_label=0, drop_label=0,
                    gt_id_now=-1,
                    weight=float(weight),
                ))

                # ----- label-driven transition -----
                if prior_state == STATE_UNCONFIRMED:
                    if matched:
                        if held_remaining.get(tid, 0) > 0:
                            # Forced hold for delay augmentation. The synthetic
                            # examples this generates are weight-discounted.
                            next_state = STATE_UNCONFIRMED
                            held_remaining[tid] -= 1
                        elif oracle_promote(tid, fi):
                            next_state = STATE_TRACKED
                        else:
                            next_state = STATE_UNCONFIRMED
                    else:
                        if oracle_drop_unconfirmed(tid, fi):
                            next_state = STATE_REMOVED
                        else:
                            next_state = STATE_UNCONFIRMED
                elif prior_state == STATE_TRACKED:
                    if matched:
                        next_state = STATE_TRACKED
                    elif num_missed >= 2:
                        next_state = STATE_LOST  # hard floor
                    elif oracle_demote(tid, fi):
                        next_state = STATE_LOST
                    else:
                        next_state = STATE_TRACKED
                elif prior_state == STATE_LOST:
                    if matched:
                        next_state = STATE_TRACKED
                    elif (time_since_det >= buffer_sec or num_missed >= max_missed):
                        next_state = STATE_REMOVED  # hard floor
                    elif oracle_drop_lost(tid, fi):
                        next_state = STATE_REMOVED
                    else:
                        next_state = STATE_LOST
                else:
                    next_state = STATE_REMOVED

                if next_state == STATE_REMOVED:
                    if tid in tracks:
                        del tracks[tid]
                    if tid in held_remaining:
                        del held_remaining[tid]
                    if tid in e_track_state:
                        del e_track_state[tid]
                else:
                    tracks[tid] = _TH(
                        state=int(next_state),
                        observations=int(observations),
                        num_missed=int(num_missed),
                        last_detect_time=float(last_detect_time),
                        last_box=last_box,
                        prev_det_conf=float(det_conf if matched else prev_det_conf),
                    )

            # Tracks alive but absent from this frame's trace — treat as
            # not matched and tick.
            absent = [tid for tid in list(tracks.keys()) if tid not in seen_this_frame]
            for tid in absent:
                th = tracks[tid]
                prior_state = th.state
                observations = th.observations
                num_missed = th.num_missed + 1
                last_detect_time = th.last_detect_time
                time_since_det = float(frame_time - last_detect_time)
                last_box = th.last_box
                bx0, by0, bx1, by1 = last_box
                near_edge, det_w, det_h, log_aspect, log_pose_kp = _box_features(
                    bx0, by0, bx1, by1, 0,
                )

                # Carry the track's e_track unchanged on absent frames (no
                # matched obs to update from). Matches what the C runtime
                # would feed the head — last EMA value.
                e_track_cur, _ = e_track_state.get(
                    tid, (np.zeros(E_TRACK_DIM, dtype=np.float32), False))

                valid_promote = 1 if prior_state == STATE_UNCONFIRMED else 0
                valid_demote  = 1 if prior_state == STATE_TRACKED else 0
                valid_drop    = 1 if prior_state in (STATE_UNCONFIRMED, STATE_LOST) else 0

                examples.append(dict(
                    sequence=sequence_name, track_id=int(tid),
                    frame_idx=int(fi), frame_time=float(frame_time),
                    prior_state=int(prior_state),
                    valid_promote=int(valid_promote),
                    valid_demote=int(valid_demote),
                    valid_drop=int(valid_drop),
                    matched=0,
                    observations=int(observations),
                    num_missed=int(num_missed),
                    time_since_det=float(time_since_det),
                    scene_density=0,
                    det_conf=0.0,
                    prev_det_conf=float(th.prev_det_conf),
                    phase3_pair_score=0.0,
                    near_edge=near_edge, det_w=det_w, det_h=det_h,
                    log_aspect=log_aspect, log_pose_kp=log_pose_kp,
                    e_track=e_track_cur.copy(),
                    promote_label=0, demote_label=0, drop_label=0,
                    gt_id_now=-1,
                    weight=float(weight),
                ))

                if prior_state == STATE_UNCONFIRMED:
                    if oracle_drop_unconfirmed(tid, fi):
                        next_state = STATE_REMOVED
                    else:
                        next_state = STATE_UNCONFIRMED
                elif prior_state == STATE_TRACKED:
                    if num_missed >= 2:
                        next_state = STATE_LOST
                    elif oracle_demote(tid, fi):
                        next_state = STATE_LOST
                    else:
                        next_state = STATE_TRACKED
                elif prior_state == STATE_LOST:
                    if (time_since_det >= buffer_sec or num_missed >= max_missed):
                        next_state = STATE_REMOVED
                    elif oracle_drop_lost(tid, fi):
                        next_state = STATE_REMOVED
                    else:
                        next_state = STATE_LOST
                else:
                    next_state = STATE_REMOVED

                if next_state == STATE_REMOVED:
                    del tracks[tid]
                    if tid in held_remaining:
                        del held_remaining[tid]
                    if tid in e_track_state:
                        del e_track_state[tid]
                else:
                    tracks[tid].state = int(next_state)
                    tracks[tid].num_missed = int(num_missed)

    if not examples:
        return np.empty((0,), dtype=EXAMPLE_DTYPE)

    # ------------------------------------------------------------------
    # Pass D: fill in label fields. Uses the same oracle calls so the
    # corpus is internally consistent (replay transitions and labels agree).
    # ------------------------------------------------------------------
    out = np.zeros((len(examples),), dtype=EXAMPLE_DTYPE)
    for i, ex in enumerate(examples):
        tid = int(ex["track_id"]); fi = int(ex["frame_idx"])
        prior_state = int(ex["prior_state"])

        ex["gt_id_now"] = int(track_gt_history.get(tid, {}).get(fi, -1))

        promote_label = 0
        demote_label = 0
        drop_label = 0

        if prior_state == STATE_UNCONFIRMED:
            promote_label = int(oracle_promote(tid, fi))
            drop_label = int(oracle_drop_unconfirmed(tid, fi))
        elif prior_state == STATE_TRACKED:
            demote_label = int(oracle_demote(tid, fi))
        elif prior_state == STATE_LOST:
            drop_label = int(oracle_drop_lost(tid, fi))

        for k, v in ex.items():
            out[i][k] = v
        out[i]["promote_label"] = promote_label
        out[i]["demote_label"] = demote_label
        out[i]["drop_label"] = drop_label

    return out


# Lightweight per-track replay state for the new extractor (the original
# TrackHist carried torch/EMA bookkeeping we don't need here).
@dataclass
class _TH:
    state: int
    observations: int
    num_missed: int
    last_detect_time: float
    last_box: Tuple[float, float, float, float]
    prev_det_conf: float


# --- driver -----------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair-log-dir", required=True,
                   help="phase1 output dir, e.g. runs/track_analysis/pair_log_v4_nopose")
    p.add_argument("--gt-config", required=True,
                   help="phase1 yaml (for dataset list + split tags)")
    p.add_argument("--out", default="bench/data/state_corpus",
                   help="output prefix (writes <prefix>_train.npz, _val.npz, _test.npz)")
    p.add_argument("--k-min", type=int, default=DEFAULT_K_MIN)
    p.add_argument("--buffer-sec", type=float, default=DEFAULT_TRACK_BUFFER_SEC)
    p.add_argument("--max-missed", type=int, default=DEFAULT_MAX_MISSED)
    p.add_argument("--h-lookahead-sec", type=float, default=3.0)
    p.add_argument("--phase3-model", default=None,
                   help="Optional Phase 3 .pt; when set, e_track[16] in each "
                        "example is filled by replaying the trained f_obs "
                        "subnet's EMA over matched obs. Without this flag, "
                        "e_track is zeros (the v0 fallback).")
    p.add_argument("--label-driven", action="store_true",
                   help="Use the label-driven oracle replay (Phase 1 fix). "
                        "Generates a corpus whose state distribution matches "
                        "what the runtime queries the head on, instead of "
                        "the deterministic K_min replay. Forces e_track to "
                        "zero (mirrors the runtime fix).")
    p.add_argument("--immediate-confirm-thr", type=float, default=0.93,
                   help="Mirror utrack.c param_immediate_confirm_thr. Only "
                        "used by --label-driven.")
    p.add_argument("--delay-aug", type=str, default="",
                   help="Comma-separated delay offsets for promotion-hold "
                        "augmentation (e.g. '1,2,3'). Empty = no aug.")
    p.add_argument("--delay-aug-tau", type=float, default=2.0,
                   help="Time-constant for the delay-aug weight decay "
                        "exp(-offset/tau).")
    args = p.parse_args()
    delay_aug_offsets: Tuple[int, ...] = tuple(
        int(x) for x in args.delay_aug.split(",") if x.strip()
    )

    f_obs_replay = None
    if args.phase3_model:
        f_obs_replay = FObsReplay(args.phase3_model)
        print(f"f_obs replay: {args.phase3_model}  α={f_obs_replay.alpha}  "
              f"e_dim={f_obs_replay.e_dim}", flush=True)

    cfg = yaml.safe_load(open(args.gt_config))
    by_split: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}

    ubtrk2_root = os.path.join(args.pair_log_dir, "ubtrk2")
    for seq_name, seq in cfg["dataset"].items():
        split = seq.get("split", "train")
        gt_path = seq["trackset"]
        ubtrk2_path = os.path.join(ubtrk2_root, f"{seq_name}.ubtrk2")
        if not os.path.exists(ubtrk2_path):
            print(f"[skip] {seq_name}: no ubtrk2 at {ubtrk2_path}")
            continue
        print(f"[{split:5s}] {seq_name}", flush=True)
        if args.label_driven:
            ex = extract_sequence_label_driven(
                seq_name, ubtrk2_path, gt_path,
                k_min=args.k_min, buffer_sec=args.buffer_sec,
                max_missed=args.max_missed,
                h_lookahead_sec=args.h_lookahead_sec,
                immediate_confirm_thr=args.immediate_confirm_thr,
                delay_aug_offsets=delay_aug_offsets,
                delay_aug_tau=args.delay_aug_tau,
                f_obs_replay=f_obs_replay,
            )
        else:
            ex = extract_sequence(
                seq_name, ubtrk2_path, gt_path,
                k_min=args.k_min, buffer_sec=args.buffer_sec,
                max_missed=args.max_missed,
                h_lookahead_sec=args.h_lookahead_sec,
                f_obs_replay=f_obs_replay,
            )
        n_pos = {
            "promote": int((ex["valid_promote"] & ex["promote_label"]).sum()),
            "demote":  int((ex["valid_demote"]  & ex["demote_label"]).sum()),
            "drop":    int((ex["valid_drop"]    & ex["drop_label"]).sum()),
        }
        n_total = {
            "promote": int(ex["valid_promote"].sum()),
            "demote":  int(ex["valid_demote"].sum()),
            "drop":    int(ex["valid_drop"].sum()),
        }
        print(f"        n={len(ex):6d}  promote={n_pos['promote']}/{n_total['promote']}  "
              f"demote={n_pos['demote']}/{n_total['demote']}  "
              f"drop={n_pos['drop']}/{n_total['drop']}")
        by_split[split].append(ex)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for split, chunks in by_split.items():
        if not chunks:
            print(f"[{split}] no data")
            continue
        arr = np.concatenate(chunks)
        out_path = f"{args.out}_{split}.npz"
        np.savez_compressed(out_path, records=arr)
        print(f"wrote {out_path}: {len(arr)} examples")


if __name__ == "__main__":
    main()
