"""Cost-rule arithmetic for the decoupled state head.

Single source of truth, mirrored byte-for-byte by the C runtime in
`utrack.c` (search for `delta_promote` / `delta_demote`). Exists as
its own module so the unit tests in `test_cost_rule.py` can lock in
both the formulas AND their calibration against the bench fitness
formula.

History of bugs this catches (2026-05-09):
1. `c_FP_track=0.10` used in shipped configs — 20× the actual bench
   fitness penalty of 0.005 per FP-track. Caused under-promotion on
   PP22 clips (5× MOTA gap on PP22_00206 alone).
2. `c_FP_frame=0.002` interpreted as fitness-loss-per-second-of-FP-life
   when it should be `fps · 0.002 / N_frames` ≈ 1e-5 (3+ orders of
   magnitude too high).
3. Unit confusion: μ_TP, μ_FP are SECONDS in the head's output.
   Coefficients must absorb the fps factor when converting from
   per-frame bench-fitness penalties.
"""
from __future__ import annotations
from dataclasses import dataclass


UNCONFIRMED, TRACKED, LOST = 0, 1, 2


@dataclass(frozen=True)
class CostCoefs:
    """Cost coefficients for the decoupled-head decision rule.

    All deltas are in BENCH-FITNESS UNITS (same as the eval formula
    `fitness = mota − 0.005·fp_tracks − 0.002·fp_per_frame`).

    - `c_MOTA`: fitness gain per second of TP-track-life that's matched.
      Maps to per-second MOTA gain = fps / N_GT.
    - `c_FP_track`: fitness loss for promoting an FP track.
      Maps directly to the bench formula's 0.005 coefficient.
    - `c_FP_frame`: fitness loss per second of FP-track-life kept alive.
      Maps to `fps · 0.002 / N_frames`.
    - `match_rate_TP`: empirical fraction of TP-track frames that match
      (≈ 0.95 — TP tracks aren't matched 100% of frames).
    """
    c_MOTA: float
    c_FP_track: float
    c_FP_frame: float
    match_rate_TP: float = 0.95


def calibrate_for_clip(fps: float, n_gt: float, n_frames: float,
                       match_rate_TP: float = 0.95) -> CostCoefs:
    """Derive cost coefficients for a specific clip's geometry.

    The bench fitness formula is
        fitness = MOTA − 0.005 · fp_tracks − 0.002 · fp_per_frame
    with `fp_per_frame = total_fps / n_frames` and
        MOTA = 1 − (FP + FN + IDS) / n_gt.

    Per-second fitness gain from a matched TP frame:
        d(fitness)/d(matched_TP_seconds) = fps · 1/n_gt

    Per-second fitness loss from a kept FP track:
        d(fitness)/d(FP_track_seconds) = fps · 0.002/n_frames
    """
    return CostCoefs(
        c_MOTA=fps / max(n_gt, 1.0),
        c_FP_track=0.005,
        c_FP_frame=fps * 0.002 / max(n_frames, 1.0),
        match_rate_TP=match_rate_TP,
    )


def delta_promote(p_TP: float, mu_TP: float, mu_FP: float,
                  c: CostCoefs) -> float:
    """Expected fitness change from FIRST-time promote (UNCONFIRMED→
    TRACKED). Pays the c_FP_track sunk cost — the bench fitness
    formula penalises every track that ever entered TRACKED."""
    p_FP = 1.0 - p_TP
    return (p_TP * c.c_MOTA * mu_TP * c.match_rate_TP
            - p_FP * (c.c_FP_track + c.c_FP_frame * mu_FP))


def delta_repromote(p_TP: float, mu_TP: float, mu_FP: float,
                    c: CostCoefs) -> float:
    """Expected fitness change from LOST→TRACKED. Does NOT pay
    c_FP_track again — the track already counts as 'promoted' in the
    bench fitness formula from its first TRACKED entry. Symmetric
    inverse of `delta_demote` (re-promoting reverses a demote)."""
    p_FP = 1.0 - p_TP
    return (p_TP * c.c_MOTA * mu_TP * c.match_rate_TP
            - p_FP * c.c_FP_frame * mu_FP)


def delta_demote(p_TP: float, mu_TP: float, mu_FP: float,
                 c: CostCoefs) -> float:
    """Expected fitness change from demoting (TRACKED→LOST). Mirror of
    `delta_repromote` — demoting now and re-promoting later should
    cancel exactly when the situation reverses."""
    p_FP = 1.0 - p_TP
    return (p_FP * c.c_FP_frame * mu_FP
            - p_TP * c.c_MOTA * mu_TP * c.match_rate_TP)


def cost_decision(p_TP: float, mu_TP: float, mu_FP: float,
                  current_state: int, matched: bool,
                  c: CostCoefs) -> int:
    """Pure expected-value state transition. Returns desired next
    state. Same formulation as the C runtime."""
    if current_state == UNCONFIRMED:
        if matched and delta_promote(p_TP, mu_TP, mu_FP, c) > 0:
            return TRACKED
        return UNCONFIRMED
    if current_state == TRACKED:
        if not matched and delta_demote(p_TP, mu_TP, mu_FP, c) > 0:
            return LOST
        return TRACKED
    if current_state == LOST:
        if matched and delta_repromote(p_TP, mu_TP, mu_FP, c) > 0:
            return TRACKED
        return LOST
    return current_state
