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
  python -m ml.data_prep.build_state_corpus \\
      --pair-log-dir runs/track_analysis/pair_log_v4_nopose \\
      --gt-config ml/configs/pair_log_config_v3_p2off.yaml \\
      --out ml/data/state_corpus

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

import stuff

import src.trackset as ts
from src.pair_log import (
    best_iou_match as _best_iou_match,
    gt_objects_at_time_class as _gt_objects_at_time_class,
)
from src.pair_log_schema import (
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


# e_track dim. Match-cost NN's per-track accumulator size (kept in the
# corpus dtype for backward-compat with older .npz files; the 19-dim
# decoupled head doesn't read it). Must match utrack/nn.h NN_MAX_E_DIM.
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
    # 2026-05-07: track-history aggregate features. e_track encodes
    # match-cost residuals via a 16-dim learned compression; these are
    # explicit signals of accumulated track quality that the state-head
    # was previously seeing only through e_track and prior_state.
    #
    # The Bayesian-like log_sum_det_conf accumulator captures "has this
    # track been seen confidently many times". The match_score_*
    # aggregates capture how good the matches themselves were over the
    # track's history.
    ("ema_match_x_conf", "<f4"),  # EMA(match_cost_score * det_conf), α=0.3
    ("log_sum_det_conf", "<f4"),  # log1p(Σ det_conf over matched obs)
    ("min_match_score",  "<f4"),  # min match_cost_score over last 8 matches
    ("mean_match_score", "<f4"),  # mean match_cost_score over last 8 matches
    ("n_strong_matches", "<f4"),  # count where match_cost_score > 0.6 in last 8
    # 2026-05-09: track-age + state-age, in seconds (head sees log1p of these).
    # Lets the head distinguish 'rock-solid TRACKED that just lost a frame'
    # from 'barely-promoted track already faltering' — the signal the legacy
    # num_missed≥2 floor implicitly used.
    ("time_since_creation", "<f4"),
    ("time_in_state",       "<f4"),
    # 2026-05-11: face / subbox channel for the state head. The match head
    # already sees these but the state head was blind to them. Hypothesis:
    # consistent face detection on a track corroborates "real human" and
    # raises promote conviction; face failing mid-track signals occlusion
    # / possible FP. All three are 0 when no face was detected (subbox_conf
    # at 0 also flags absence).
    ("det_subbox_conf",     "<f4"),
    ("track_subbox_conf",   "<f4"),
    ("det_fiqa_score",      "<f4"),
    # 2026-05-11: scene-aggregate features for the V2 state head (in_dim=25).
    # Mirror the per-utrack EMAs that ubon_cstuff/src/track/utrack/utrack.c
    # maintains; populated by SceneStats.apply_to_examples() in a frame-major
    # post-pass. Trainer flag --with-scene now requires these fields (no more
    # silent constant fallback that produced the 0.29-fitness V2 catastrophe
    # diagnosed 2026-05-11 — see commit 877a2da).
    ("scene_promote_rate",            "<f4"),
    ("scene_mean_det_conf_TRACKED",   "<f4"),
    ("scene_mean_det_conf_unmatched", "<f4"),
    ("scene_track_density_smooth",    "<f4"),
    ("scene_mean_alive_track_age",    "<f4"),
    # Derived per-row: det_conf - scene_mean_det_conf_TRACKED (the
    # "how-confident is this det relative to the scene's TRACKED-track
    # average" signal). C runtime computes it on-the-fly at line 153 of
    # utrack_state.c; we precompute to keep the trainer column-major.
    ("det_conf_minus_scene_TP_avg",   "<f4"),
    # e_track[16] from Phase 3 f_obs replay — zeros if --phase3-model not given.
    # Kept in dtype for backward-compat (older consumers may still read it);
    # NOT included in the 23-dim runtime feature vector built by
    # train_state_head.py:build_input_matrix as of 2026-05-09. Inspection
    # showed all 16 cols sat at near-init magnitude across runs because
    # the match-cost NN failed to load during pair-log generation.
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


# --- scene-aggregate EMAs (Phase 29) -----------------------------------------
#
# Replays ubon_cstuff/src/track/utrack/utrack.c's `utrack_scene_stats_*`
# offline so the V2 (--with-scene) trainer sees the same input distribution
# the C runtime feeds the head at inference. The C source is authoritative
# (lines 605-707 of utrack.c); these constants must stay in lockstep.
#
# 2026-05-11: this class was deleted by commit e288eea (2026-05-10) during
# the "remove legacy bench scripts" cleanup, but the trainer's --with-scene
# flag wasn't disabled. Result: every V2 retrain since then trained on the
# trainer's constant fallback (SCENE_COL_DEFAULTS), yielding ~0.29 fitness
# below V1 on the same corpus. Restoring this fixes that catastrophe.

SCENE_STATS_EMA_ALPHA   = 0.05
SCENE_STATS_MIN_SAMPLES = 20
SCENE_PROMOTE_RATE_BOOTSTRAP     = 0.5
SCENE_MEAN_DET_TRK_BOOTSTRAP     = 0.7
SCENE_MEAN_DET_UNMTCHD_BOOTSTRAP = 0.3
SCENE_TRACK_DENSITY_BOOTSTRAP    = 5.0
SCENE_ALIVE_AGE_BOOTSTRAP        = 5.0


class SceneStats:
    """Per-sequence EMA replay matching utrack.c's scene-stats semantics.

    Five EMAs (alpha=0.05) over scene-level observations, each gated by a
    min-samples-20 bootstrap. `read()` returns the value the C runtime
    would expose to the state head at the same point.
    """

    __slots__ = (
        "promote_rate_ema", "promote_rate_updates",
        "mean_det_TRK_ema", "mean_det_TRK_updates",
        "mean_det_unmtchd_ema", "mean_det_unmtchd_updates",
        "track_density_ema", "track_density_updates",
        "alive_age_ema", "alive_age_updates",
    )

    def __init__(self) -> None:
        self.promote_rate_ema = SCENE_PROMOTE_RATE_BOOTSTRAP
        self.promote_rate_updates = 0
        self.mean_det_TRK_ema = SCENE_MEAN_DET_TRK_BOOTSTRAP
        self.mean_det_TRK_updates = 0
        self.mean_det_unmtchd_ema = SCENE_MEAN_DET_UNMTCHD_BOOTSTRAP
        self.mean_det_unmtchd_updates = 0
        self.track_density_ema = SCENE_TRACK_DENSITY_BOOTSTRAP
        self.track_density_updates = 0
        self.alive_age_ema = SCENE_ALIVE_AGE_BOOTSTRAP
        self.alive_age_updates = 0

    @staticmethod
    def _ema(prev: float, x: float) -> float:
        return (1.0 - SCENE_STATS_EMA_ALPHA) * prev + SCENE_STATS_EMA_ALPHA * x

    def update_promote_outcome(self, promoted: bool) -> None:
        """One UNCONFIRMED outcome (promote-to-TRACKED or drop)."""
        self.promote_rate_ema = self._ema(self.promote_rate_ema,
                                           1.0 if promoted else 0.0)
        self.promote_rate_updates += 1

    def update_unmatched_det(self, det_conf: float) -> None:
        """One detection that wasn't assigned to any existing track."""
        self.mean_det_unmtchd_ema = self._ema(self.mean_det_unmtchd_ema,
                                               float(det_conf))
        self.mean_det_unmtchd_updates += 1

    def update_per_frame(self, n_tracked: int, sum_age: float,
                          sum_det_conf_TRK: float,
                          n_det_conf_TRK: int) -> None:
        """One frame's end-of-frame walk over the TRACKED set. Mirrors
        utrack_scene_stats_update_per_frame: density is unconditional;
        alive-age fires only when n_tracked>0; mean-det-conf-TRACKED
        fires only when at least one TRACKED track also matched."""
        self.track_density_ema = self._ema(self.track_density_ema,
                                            float(n_tracked))
        self.track_density_updates += 1
        if n_tracked > 0:
            self.alive_age_ema = self._ema(self.alive_age_ema,
                                            sum_age / float(n_tracked))
            self.alive_age_updates += 1
        if n_det_conf_TRK > 0:
            self.mean_det_TRK_ema = self._ema(self.mean_det_TRK_ema,
                                               sum_det_conf_TRK
                                               / float(n_det_conf_TRK))
            self.mean_det_TRK_updates += 1

    def read(self) -> Tuple[float, float, float, float, float]:
        """Returns (promote_rate, mean_det_TRK, mean_det_unmtchd,
        track_density, alive_age) — bootstrap value substituted for any
        EMA with fewer than MIN_SAMPLES updates. Mirrors
        utrack_scene_stats_read."""
        return (
            self.promote_rate_ema
                if self.promote_rate_updates >= SCENE_STATS_MIN_SAMPLES
                else SCENE_PROMOTE_RATE_BOOTSTRAP,
            self.mean_det_TRK_ema
                if self.mean_det_TRK_updates >= SCENE_STATS_MIN_SAMPLES
                else SCENE_MEAN_DET_TRK_BOOTSTRAP,
            self.mean_det_unmtchd_ema
                if self.mean_det_unmtchd_updates >= SCENE_STATS_MIN_SAMPLES
                else SCENE_MEAN_DET_UNMTCHD_BOOTSTRAP,
            self.track_density_ema
                if self.track_density_updates >= SCENE_STATS_MIN_SAMPLES
                else SCENE_TRACK_DENSITY_BOOTSTRAP,
            self.alive_age_ema
                if self.alive_age_updates >= SCENE_STATS_MIN_SAMPLES
                else SCENE_ALIVE_AGE_BOOTSTRAP,
        )


def apply_scene_stats_to_examples(examples: np.ndarray) -> None:
    """Walk `examples` in (sequence, frame_time) order and fill the six
    scene-aggregate columns in-place. Mirrors the C runtime per-frame
    update timing:

      1. PRE-HEAD: for each detection that didn't assign to an existing
         track this frame, update mean_det_conf_unmatched. We
         approximate "unmatched det" as a row that is its track's
         first observation AND was matched this frame (i.e. the det that
         caused a new track to be created — which in the C runtime IS
         the unmatched-det that triggered the new-track loop).
      2. EMIT ROWS: read() the current EMAs into each example's scene
         cols. All rows at the same frame_time see the same value
         (ordering of the per-track head reads within a frame is below
         the discriminative noise of the head — see commit message).
      3. POST-HEAD per-row: if this row is an UNCONFIRMED outcome
         (prior_state==UNCONFIRMED and observations crossed K_min, OR
         this is the row at which the track was about to be dropped),
         feed update_promote_outcome(True/False).
      4. PER-FRAME: at end of frame, walk all examples for that frame
         to compute n_tracked, sum_age, sum_det_conf_TRK over rows whose
         prior_state==TRACKED (post-transition state isn't recorded
         per-row, so we use prior_state==TRACKED, which approximates
         "currently TRACKED at this frame" — close to C's `track_state`
         walk at end of utrack_run).
    """
    if len(examples) == 0:
        return
    seq = examples["sequence"].astype(str)
    fi  = examples["frame_idx"]
    ft  = examples["frame_time"].astype(np.float64)
    order = np.lexsort((fi, seq))

    i = 0
    n = len(order)
    while i < n:
        s_i = seq[order[i]]
        stats = SceneStats()
        # Find span of this sequence
        j = i
        while j < n and seq[order[j]] == s_i:
            j += 1
        seq_idx = order[i:j]
        # Group by frame_idx within sequence
        fis = fi[seq_idx]
        k = 0
        m = len(seq_idx)
        while k < m:
            f_k = fis[k]
            kk = k
            while kk < m and fis[kk] == f_k:
                kk += 1
            frame_idx = seq_idx[k:kk]
            rtp_time = float(ft[frame_idx[0]])

            # PHASE 1 — unmatched-det updates. Row is the det that
            # spawned a new track iff observations==1 and matched==1.
            for row_i in frame_idx:
                if (int(examples["observations"][row_i]) == 1
                        and int(examples["matched"][row_i]) == 1):
                    stats.update_unmatched_det(
                        float(examples["det_conf"][row_i]))

            # PHASE 2 — read current EMAs, write into rows' scene cols.
            pr, mt, mu, td, aa = stats.read()
            for row_i in frame_idx:
                det_c = float(examples["det_conf"][row_i])
                examples["scene_promote_rate"][row_i] = pr
                examples["scene_mean_det_conf_TRACKED"][row_i] = mt
                examples["scene_mean_det_conf_unmatched"][row_i] = mu
                examples["scene_track_density_smooth"][row_i] = td
                examples["scene_mean_alive_track_age"][row_i] = aa
                examples["det_conf_minus_scene_TP_avg"][row_i] = det_c - mt

            # PHASE 3 — promote-outcome updates. Approximate the C
            # runtime's "UNCONFIRMED → TRACKED" detection by the
            # promote_label on UNCONFIRMED rows. promote_label is 1
            # iff the future GT history validates that promote was
            # correct — but in the C runtime the EMA is fed by the
            # DECISION, not the truth. The label-driven extractor uses
            # the label AS the decision, so this is the right oracle
            # here. The deterministic extractor uses observations>=K_min
            # which is reflected by promote_label too (UNCONFIRMED rows
            # at the moment of crossing K_min). For UNCONFIRMED rows
            # that drop instead, the C feeds False — proxied by
            # drop_label==1 on the same prior_state==UNCONFIRMED row.
            for row_i in frame_idx:
                if int(examples["prior_state"][row_i]) == STATE_UNCONFIRMED:
                    if int(examples["valid_promote"][row_i]) == 1:
                        if int(examples["promote_label"][row_i]) == 1:
                            stats.update_promote_outcome(True)
                        elif int(examples["drop_label"][row_i]) == 1:
                            stats.update_promote_outcome(False)

            # PHASE 4 — per-frame walk over end-of-frame TRACKED set.
            # We approximate "currently TRACKED" with prior_state==TRACKED
            # (the state at start-of-frame — post-transition state isn't
            # stored). For ages, use time_since_creation if present;
            # else use a tdet-style proxy. For mean_det_conf_TRACKED,
            # only matched TRACKED rows contribute.
            n_tracked = 0
            sum_age = 0.0
            sum_det_TRK = 0.0
            n_det_TRK = 0
            for row_i in frame_idx:
                if int(examples["prior_state"][row_i]) == STATE_TRACKED:
                    n_tracked += 1
                    sum_age += float(examples["time_since_creation"][row_i])
                    if int(examples["matched"][row_i]) == 1:
                        sum_det_TRK += float(examples["det_conf"][row_i])
                        n_det_TRK += 1
            stats.update_per_frame(n_tracked, sum_age, sum_det_TRK, n_det_TRK)

            k = kk
        i = j


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
        magic = int(trc.get("magic", 0))
        if magic != PAIR_LOG_MAGIC:
            raise ValueError(f"{sequence_name}: trace magic mismatch")
        from src.pair_log_schema import PAIR_LOG_DTYPE_V2 as _V2_DT
        rs = int(trc.get("record_size", 0))
        if rs not in (record_size_bytes(), _V2_DT.itemsize):
            raise ValueError(
                f"{sequence_name}: trace record_size {rs} not in "
                f"{{{record_size_bytes()}, {_V2_DT.itemsize}}}")
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
            det_subbox_conf   = float(match_rec["det_subbox_conf"])   if matched else 0.0
            det_fiqa_score    = float(match_rec["det_fiqa_score"])    if matched else 0.0
            track_subbox_conf = float(recs[0]["track_subbox_conf"])

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
                det_subbox_conf=det_subbox_conf,
                track_subbox_conf=track_subbox_conf,
                det_fiqa_score=det_fiqa_score,
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
                det_subbox_conf=det_subbox_conf,
                track_subbox_conf=track_subbox_conf,
                det_fiqa_score=det_fiqa_score,
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
    fitness_fp_boost: float = 1.0,
) -> np.ndarray:
    """Walk one sequence and emit label-driven examples.

    delay_aug_offsets: optional tuple of K values. For each promotable track,
    emit additional 'delayed promotion' trajectory copies where the track is
    held UNCONFIRMED for K extra frames before promoting. Each copy's
    examples carry weight = exp(-K/delay_aug_tau). Empty tuple = no
    augmentation (Phase 1b default; switch on after the bin-coverage check).

    fitness_fp_boost: weight multiplier applied to all examples on tracks
    that never matched any GT (i.e., would count as `fp_tracks` in the
    user's fitness function `mota - 0.0005*fp_tracks - 0.002*fp_per_frame`).
    Default 1.0 = no change. Values >1 push the loss to focus on getting
    the FP-track decisions right (drop fast, don't promote), which matches
    what the search-time fitness penalises.
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
    # honest-fp-train-adopt-v4: per-(tid,fi) whether the track box
    # overlaps ANY GT at all (IoU>0), distinct from the >=seed_match_iou
    # alignment. A loose-near-real frame (0<IoU<0.5) is NOT a phantom
    # under the resolved honest ruler (counts only IoU==0 runs) — the
    # state-NN must NOT be taught to drop/demote it (= FN/MOTA loss for
    # zero honest-FP benefit: the v1 mechanism in state-NN form).
    track_overlaps_gt: Dict[int, Dict[int, bool]] = defaultdict(dict)
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
            track_overlaps_gt[tid].setdefault(
                fi, best_gt is not None and best_iou > 0.0)
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
        # decode_records accepts both v2 (152) and v3 (156) byte layouts.
        # Match either; raise on anything else.
        from src.pair_log_schema import PAIR_LOG_DTYPE_V2 as _V2_DT
        rs = int(trc.get("record_size", 0))
        if rs not in (record_size_bytes(), _V2_DT.itemsize):
            raise ValueError(
                f"{sequence_name}: pair-trace record_size {rs} not in "
                f"{{{record_size_bytes()}, {_V2_DT.itemsize}}}")
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
            track_overlaps_gt[tid].setdefault(
                fi, best_gt is not None and best_iou > 0.0)
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

    # honest-fp-train-adopt-v4: a frame where the track box overlaps a
    # real GT object at all (IoU>0) is NOT a phantom under the resolved
    # honest ruler (counts only IoU==0 runs). Do NOT hard-drop/demote it
    # — that is pure FN/MOTA loss for zero honest-FP benefit. Restrict
    # drop/demote pressure to genuinely far-from-all-GT (IoU==0) frames.
    def overlaps_gt(tid: int, fi: int) -> bool:
        return bool(track_overlaps_gt.get(tid, {}).get(fi, False))

    def oracle_drop_unconfirmed(tid: int, fi: int) -> bool:
        return (
            not future_alignment(tid, fi)
            and track_gt_history.get(tid, {}).get(fi, -1) == -1
            and historic_gt_id(tid, fi) is None
            and not overlaps_gt(tid, fi)        # v4: IoU==0 only
        )

    def oracle_demote(tid: int, fi: int) -> bool:
        cur = track_gt_history.get(tid, {}).get(fi, -1)
        return (cur == -1 and not future_alignment(tid, fi)
                and not overlaps_gt(tid, fi))   # v4: IoU==0 only

    def oracle_drop_lost(tid: int, fi: int) -> bool:
        return not future_alignment(tid, fi) and not overlaps_gt(tid, fi)

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
        # Per-track history accumulators for the v17.5 aggregate features.
        # See EXAMPLE_DTYPE: ema_match_x_conf etc.
        match_score_history: Dict[int, Dict] = {}

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
                    # 2026-05-09: track-age timestamps. Both equal frame_time
                    # at creation; state_entered_time will be re-set if the
                    # oracle promotes this frame.
                    creation_time = frame_time
                    state_entered_time = frame_time
                    # Pure-NN runtime no longer applies an immediate-confirm
                    # short-circuit at creation; the promote head decides at
                    # site B every time. Corpus now mirrors that — first
                    # appearance is always prior=UNCONFIRMED. (immediate_confirm_thr
                    # arg is retained for legacy-runtime corpus builds.)
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
                    # 2026-05-09: carry forward track-age timestamps. The
                    # head reads PRE-transition values (matching the
                    # runtime's pre-transition feature snapshot); transition-
                    # caused bumps to state_entered_time happen below in the
                    # state-update block.
                    creation_time = th.creation_time
                    state_entered_time = th.state_entered_time

                scene_density = int(first_rec["scene_density"])
                time_since_det = float(frame_time - last_detect_time)
                det_conf = float(match_rec["det_conf"]) if matched else 0.0
                phase3_pair_score = float(match_rec["match_cost_score"]) if matched else 0.0
                det_subbox_conf   = float(match_rec["det_subbox_conf"])   if matched else 0.0
                det_fiqa_score    = float(match_rec["det_fiqa_score"])    if matched else 0.0
                track_subbox_conf = float(first_rec["track_subbox_conf"])

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

                # Track-history aggregates: ema(match_score*det_conf),
                # log_sum(det_conf), min/mean/count over recent matches.
                # Stats are emitted PRE-update (state-head decision uses
                # accumulated history through previous frame, not this
                # one) — matches how a runtime would call the head.
                hist = match_score_history.get(tid, {
                    "ema_mxc": 0.0, "sum_det_conf": 0.0,
                    "recent_match_scores": [],   # rolling window of last 8
                })
                ema_mxc = hist["ema_mxc"]
                log_sum_dc = float(np.log1p(hist["sum_det_conf"]))
                rec_scores = hist["recent_match_scores"]
                if rec_scores:
                    min_match = min(rec_scores)
                    mean_match = sum(rec_scores) / len(rec_scores)
                    n_strong = sum(1 for s in rec_scores if s > 0.6)
                else:
                    min_match = 0.0; mean_match = 0.0; n_strong = 0
                # Update for next frame (NOT used in this row's emit).
                if matched and match_rec is not None:
                    s = float(match_rec["match_cost_score"])
                    c = float(match_rec["det_conf"])
                    hist["ema_mxc"] = 0.7 * hist["ema_mxc"] + 0.3 * (s * c)
                    hist["sum_det_conf"] += c
                    rec = list(rec_scores)
                    rec.append(s)
                    if len(rec) > 8: rec = rec[-8:]
                    hist["recent_match_scores"] = rec
                match_score_history[tid] = hist

                # valid_promote covers UNCONFIRMED examples (the head fires
                # on every UNCONFIRMED match in the runtime) AND LOST+matched
                # (recovery), since the new pure-NN runtime puts that
                # decision through the promote head too.
                valid_promote = 1 if (
                    prior_state == STATE_UNCONFIRMED
                    or (prior_state == STATE_LOST and matched)
                ) else 0
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
                    det_subbox_conf=det_subbox_conf,
                    track_subbox_conf=track_subbox_conf,
                    det_fiqa_score=det_fiqa_score,
                    ema_match_x_conf=float(ema_mxc),
                    log_sum_det_conf=float(log_sum_dc),
                    min_match_score=float(min_match),
                    mean_match_score=float(mean_match),
                    n_strong_matches=float(n_strong),
                    time_since_creation=float(frame_time - creation_time),
                    time_in_state=float(frame_time - state_entered_time),
                    e_track=e_track_cur.copy(),
                    promote_label=0, demote_label=0, drop_label=0,
                    gt_id_now=-1,
                    weight=float(weight),
                ))

                # ----- label-driven transition (mirrors the unified
                # state-pass C runtime semantics: 2 effective heads
                # (promote, demote); UNCONFIRMED has no exit-on-unmatched
                # other than state-age timeout; LOST timeout is
                # state-age based (matches that don't promote don't reset
                # the timer); drop_unconfirmed / drop_lost heads are not
                # consulted by the runtime so the oracles don't drive
                # transitions here either). -----
                state_age = float(frame_time - state_entered_time)
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
                        next_state = STATE_UNCONFIRMED
                    # Hard state-age timeout (matches utrack.c).
                    if (next_state == STATE_UNCONFIRMED
                        and state_age > buffer_sec):
                        next_state = STATE_REMOVED
                elif prior_state == STATE_TRACKED:
                    if matched:
                        next_state = STATE_TRACKED
                    elif oracle_demote(tid, fi):
                        next_state = STATE_LOST
                    else:
                        next_state = STATE_TRACKED
                elif prior_state == STATE_LOST:
                    if matched:
                        if oracle_promote(tid, fi):
                            next_state = STATE_TRACKED
                        else:
                            next_state = STATE_LOST
                    else:
                        next_state = STATE_LOST
                    # Hard state-age timeout (matches utrack.c).
                    if (next_state == STATE_LOST
                        and state_age > buffer_sec):
                        next_state = STATE_REMOVED
                else:
                    next_state = STATE_REMOVED

                # Track-age timestamp updates: state_entered_time bumps to
                # frame_time iff the state actually changed. creation_time
                # is preserved (unless the track is first appearance, in
                # which case it was set above).
                if next_state != prior_state:
                    new_state_entered_time = float(frame_time)
                else:
                    new_state_entered_time = float(state_entered_time)

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
                        creation_time=float(creation_time),
                        state_entered_time=new_state_entered_time,
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
                    det_subbox_conf=0.0,
                    track_subbox_conf=0.0,
                    det_fiqa_score=0.0,
                    time_since_creation=float(frame_time - th.creation_time),
                    time_in_state=float(frame_time - th.state_entered_time),
                    e_track=e_track_cur.copy(),
                    promote_label=0, demote_label=0, drop_label=0,
                    gt_id_now=-1,
                    weight=float(weight),
                ))

                # Pure-NN runtime transitions for absent-this-frame tracks
                # — same as the matched branch's policy: state-age timeout
                # is the only hard backstop; oracle_demote drives TRACKED.
                state_age = float(frame_time - th.state_entered_time)
                if prior_state == STATE_UNCONFIRMED:
                    next_state = (STATE_REMOVED if state_age > buffer_sec
                                  else STATE_UNCONFIRMED)
                elif prior_state == STATE_TRACKED:
                    if oracle_demote(tid, fi):
                        next_state = STATE_LOST
                    else:
                        next_state = STATE_TRACKED
                elif prior_state == STATE_LOST:
                    next_state = (STATE_REMOVED if state_age > buffer_sec
                                  else STATE_LOST)
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
                    if next_state != prior_state:
                        tracks[tid].state_entered_time = float(frame_time)

    if not examples:
        return np.empty((0,), dtype=EXAMPLE_DTYPE)

    # ------------------------------------------------------------------
    # Pass D: fill in label fields. Uses the same oracle calls so the
    # corpus is internally consistent (replay transitions and labels agree).
    # ------------------------------------------------------------------
    # Per-track FP flag: a track that never matched any GT in this sequence
    # is exactly what counts in `fp_tracks` in src/track_test.py
    # fitness_score (penalty 0.0005 each). Boost weight on those examples
    # so the loss tracks deployment fitness, not just per-head AUC.
    fp_track_ids = {tid for tid in track_gt_history.keys()
                    if not track_aligned_frames.get(tid)}
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
            # LOST+matched is a recovery candidate: the runtime calls the
            # promote head to decide whether to put it back into TRACKED.
            # Use the same lookahead oracle as UNCONFIRMED promotion.
            # Loss-masking via valid_promote=1 only when matched (set above)
            # makes this label only apply to the recovery regime.
            promote_label = int(oracle_promote(tid, fi))
            drop_label = int(oracle_drop_lost(tid, fi))

        if fitness_fp_boost != 1.0 and tid in fp_track_ids:
            ex["weight"] = float(ex["weight"]) * fitness_fp_boost

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
    # 2026-05-09: track-age and state-age timestamps — set at first
    # appearance, state_entered_time bumped on every state change. Mirror
    # the C runtime's utdet_t.creation_time / state_entered_time.
    creation_time: float = 0.0
    state_entered_time: float = 0.0


# --- driver -----------------------------------------------------------------

def _extract_seq_work_fn(work_item, mpwq_context, mpwq_progress_fn=None):
    """mp_workqueue worker: extract one sequence's state-corpus examples.

    CPU-bound (FObsReplay.f_obs is pure numpy). The phase3 replay weights
    are loaded once per worker process and cached in mpwq_context.
    """
    seq_name, seq, A = work_item
    delay_aug_offsets = A["delay_aug_offsets"]

    f_obs_replay = mpwq_context.get("process_setup_results")
    if f_obs_replay is None and A["phase3_model"]:
        f_obs_replay = FObsReplay(A["phase3_model"])
        mpwq_context["process_setup_results"] = f_obs_replay

    if A["label_driven"]:
        ex = extract_sequence_label_driven(
            seq_name, A["ubtrk2_path"], A["gt_path"],
            k_min=A["k_min"], buffer_sec=A["buffer_sec"],
            max_missed=A["max_missed"],
            h_lookahead_sec=A["h_lookahead_sec"],
            immediate_confirm_thr=A["immediate_confirm_thr"],
            delay_aug_offsets=delay_aug_offsets,
            delay_aug_tau=A["delay_aug_tau"],
            f_obs_replay=f_obs_replay,
            fitness_fp_boost=A["fitness_fp_boost"],
        )
    else:
        ex = extract_sequence(
            seq_name, A["ubtrk2_path"], A["gt_path"],
            k_min=A["k_min"], buffer_sec=A["buffer_sec"],
            max_missed=A["max_missed"],
            h_lookahead_sec=A["h_lookahead_sec"],
            f_obs_replay=f_obs_replay,
        )
    return {"seq_name": seq_name, "split": A["split"], "ex": ex}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair-log-dir", required=True,
                   help="phase1 output dir, e.g. runs/track_analysis/pair_log_v4_nopose")
    p.add_argument("--gt-config", required=True,
                   help="phase1 yaml (for dataset list + split tags)")
    p.add_argument("--out", default="ml/data/state_corpus",
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
    p.add_argument("--comment", required=True,
                   help="Free-form note recorded in every output .npz's "
                        "_meta. Required: every corpus must carry a human "
                        "description of what it is and why it was built. "
                        "The 2026-05-11 state_corpus_v18 reproducibility "
                        "incident was caused by v18 lacking this — no one "
                        "could reconstruct which pair-log dir it came from.")
    p.add_argument("--workers", default="auto",
                   help="parallel sequence workers (int or 'auto'). This "
                        "step is CPU-bound (numpy f_obs replay); 'auto' "
                        "resolves via stuff.resolve_num_workers. 0 = "
                        "synchronous single process.")
    p.add_argument("--fitness-fp-boost", type=float, default=1.0,
                   help="Weight multiplier for examples on tracks that "
                        "never match any GT (= fp_tracks in fitness "
                        "function). 1.0 = no boost (default). Values >1 "
                        "push the loss to weight FP-track-avoidance "
                        "harder, matching the search-time fitness "
                        "(0.0005 per FP track + 0.002 per FP per frame).")
    args = p.parse_args()
    delay_aug_offsets: Tuple[int, ...] = tuple(
        int(x) for x in args.delay_aug.split(",") if x.strip()
    )

    if args.phase3_model:
        # Validate up-front in the parent so a bad checkpoint fails fast
        # (workers re-load their own copy lazily).
        _probe = FObsReplay(args.phase3_model)
        print(f"f_obs replay: {args.phase3_model}  α={_probe.alpha}  "
              f"e_dim={_probe.e_dim}", flush=True)
        del _probe

    cfg = yaml.safe_load(open(args.gt_config))
    by_split: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}

    ubtrk2_root = os.path.join(args.pair_log_dir, "ubtrk2")
    common = {
        "k_min": args.k_min, "buffer_sec": args.buffer_sec,
        "max_missed": args.max_missed,
        "h_lookahead_sec": args.h_lookahead_sec,
        "immediate_confirm_thr": args.immediate_confirm_thr,
        "delay_aug_offsets": delay_aug_offsets,
        "delay_aug_tau": args.delay_aug_tau,
        "label_driven": args.label_driven,
        "phase3_model": args.phase3_model,
        "fitness_fp_boost": args.fitness_fp_boost,
    }
    work_items = []
    for seq_name, seq in cfg["dataset"].items():
        split = seq.get("split", "train")
        gt_path = seq["trackset"]
        ubtrk2_path = os.path.join(ubtrk2_root, f"{seq_name}.ubtrk2")
        if not os.path.exists(ubtrk2_path):
            print(f"[skip] {seq_name}: no ubtrk2 at {ubtrk2_path}")
            continue
        A = dict(common)
        A["split"] = split
        A["gt_path"] = gt_path
        A["ubtrk2_path"] = ubtrk2_path
        work_items.append((seq_name, seq, A))

    workers = stuff.resolve_num_workers(args.workers)
    print(f"extracting {len(work_items)} sequences with {workers} workers "
          f"(CPU-bound; no GPU)", flush=True)

    # CPU-bound numpy work — disable GPU sharding so workers aren't
    # needlessly pinned to one GPU each.
    results = stuff.mp_workqueue_run(
        work_items, _extract_seq_work_fn,
        num_workers=workers,
        desc="state-corpus extract",
        auto_gpu_shard=False,
    )

    # mp_workqueue returns results in completion order; restore the
    # gt-config sequence order so the concatenated corpus (and thus the
    # written npz row order) is deterministic regardless of worker count.
    seq_order = {item[0]: i for i, item in enumerate(work_items)}
    results.sort(key=lambda r: seq_order.get(r["seq_name"], 1 << 30))

    for r in results:
        ex = r["ex"]
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
        print(f"[{r['split']:5s}] {r['seq_name']}  n={len(ex):6d}  "
              f"promote={n_pos['promote']}/{n_total['promote']}  "
              f"demote={n_pos['demote']}/{n_total['demote']}  "
              f"drop={n_pos['drop']}/{n_total['drop']}")
        by_split[r["split"]].append(ex)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    from ml.util._artefact_meta import make_pt_meta, save_npz_with_meta
    for split, chunks in by_split.items():
        if not chunks:
            print(f"[{split}] no data")
            continue
        arr = np.concatenate(chunks)
        # Phase 29 scene-aggregate post-pass: replay the C runtime's
        # per-sequence EMAs (SceneStats) and fill the 6 scene_* columns
        # in-place. The trainer's --with-scene path requires these to
        # have been written by something other than the constant fallback
        # (see commit 877a2da for the silent-fallback bug + fail-loud
        # guard).
        apply_scene_stats_to_examples(arr)
        out_path = f"{args.out}_{split}.npz"
        meta = make_pt_meta(
            artefact_kind="state_corpus",
            args=args,
            hparams={
                "split": split,
                "n_examples": int(len(arr)),
                "n_sequences": int(len(np.unique(arr["sequence"]))) if len(arr) else 0,
                "extractor": "label_driven" if args.label_driven
                             else "deterministic_replay",
                "dtype_fields": list(arr.dtype.names),
                "k_min": args.k_min,
                "buffer_sec": args.buffer_sec,
                "max_missed": args.max_missed,
                "h_lookahead_sec": args.h_lookahead_sec,
                "delay_aug_offsets": list(delay_aug_offsets),
                "fitness_fp_boost": args.fitness_fp_boost,
                "phase3_model": args.phase3_model,
            },
            dataset_info={
                "pair_log_dir": args.pair_log_dir,
                "gt_config": args.gt_config,
            },
            comment=args.comment,
        )
        save_npz_with_meta(out_path, meta, records=arr)
        print(f"wrote {out_path}: {len(arr)} examples (with _meta)")


if __name__ == "__main__":
    main()
