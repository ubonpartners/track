"""Unit tests for `bench/cost_rule.py`.

These tests are designed to catch the specific bugs that hit shipped
state heads in 2026-05-09:

1. `c_FP_track=0.10` mis-calibrated (was 20× actual fitness penalty)
2. `c_FP_frame=0.002` mis-calibrated (was 100-1000× actual penalty
   for typical clip lengths)
3. Demote/promote rule asymmetry around (state, matched) gating

Run with: `python -m bench.test_cost_rule`
"""
import math
import unittest

from bench.cost_rule import (
    CostCoefs, calibrate_for_clip,
    delta_promote, delta_repromote, delta_demote, cost_decision,
    UNCONFIRMED, TRACKED, LOST,
)


# Reference clip stats roughly matching PP22_00120
PP22_FPS, PP22_N_GT, PP22_N_FRAMES = 5.0, 4400.0, 700.0


class TestCalibration(unittest.TestCase):
    """Coefficients must follow the bench fitness formula
    `fitness = mota − 0.005·fp_tracks − 0.002·fp_per_frame`."""

    def test_c_FP_track_matches_bench_formula(self):
        """The bench penalises each fp_track by literal 0.005 — the
        cost rule must use the same number, NOT a hand-tuned 0.10."""
        c = calibrate_for_clip(PP22_FPS, PP22_N_GT, PP22_N_FRAMES)
        self.assertAlmostEqual(c.c_FP_track, 0.005)

    def test_c_MOTA_per_second_of_matched_TP(self):
        """Per matched TP-second fitness gain = fps / N_GT.
        For PP22-style (5 fps, 4400 GT): 5/4400 ≈ 0.001136."""
        c = calibrate_for_clip(PP22_FPS, PP22_N_GT, PP22_N_FRAMES)
        self.assertAlmostEqual(c.c_MOTA, 5.0 / 4400.0, places=6)
        # Sanity: must be on the order of 1e-4 to 1e-3, not 1e-2.
        self.assertLess(c.c_MOTA, 0.01)
        self.assertGreater(c.c_MOTA, 1e-4)

    def test_c_FP_frame_per_second_of_FP_life(self):
        """fp_per_frame in the bench formula is `total_fps/N_frames`.
        Per FP frame: 0.002/N_frames in fitness. Per FP SECOND
        (μ_FP units): fps · 0.002/N_frames. The bug: shipped configs
        used 0.002 directly, off by a factor of N_frames/fps ≈ 140
        for PP22 — three orders of magnitude."""
        c = calibrate_for_clip(PP22_FPS, PP22_N_GT, PP22_N_FRAMES)
        expected = 5.0 * 0.002 / 700.0  # ≈ 1.43e-5
        self.assertAlmostEqual(c.c_FP_frame, expected, places=8)
        # Must be far below the 0.002 that was used in production.
        self.assertLess(c.c_FP_frame, 1e-4)


class TestPromoteDelta(unittest.TestCase):
    """Δ_promote = p_TP · c_MOTA · μ_TP · match_rate
                 − p_FP · (c_FP_track + c_FP_frame · μ_FP)"""

    def setUp(self):
        self.c = calibrate_for_clip(PP22_FPS, PP22_N_GT, PP22_N_FRAMES)

    def test_high_confidence_TP_promotes(self):
        """p_TP=0.97 with reasonable μ_TP should yield Δ>0."""
        d = delta_promote(p_TP=0.97, mu_TP=2.0, mu_FP=2.5, c=self.c)
        self.assertGreater(d, 0,
            f"high-conf TP must promote, got Δ={d:.6f}")

    def test_low_confidence_FP_does_not_promote(self):
        """p_TP=0.10 must yield Δ<0 — strong FP signal."""
        d = delta_promote(p_TP=0.10, mu_TP=1.0, mu_FP=2.0, c=self.c)
        self.assertLess(d, 0,
            f"low-conf FP must NOT promote, got Δ={d:.6f}")

    def test_uncertainty_mid_threshold(self):
        """At p_TP=0.5 the rule should be near zero — it's the
        decision boundary between treating-as-TP and treating-as-FP.

        Exact crossover depends on μ_TP/μ_FP magnitudes, but the
        signed result must respond monotonically as p_TP rises."""
        d_low  = delta_promote(0.30, 1.5, 2.5, self.c)
        d_mid  = delta_promote(0.50, 1.5, 2.5, self.c)
        d_high = delta_promote(0.70, 1.5, 2.5, self.c)
        self.assertLess(d_low, d_mid)
        self.assertLess(d_mid, d_high)

    def test_old_buggy_coefs_would_reject_high_conf_TP(self):
        """Regression test: with the old shipped c_FP_track=0.10
        AND c_FP_frame=0.002 (both wrong), a track with p_TP=0.97
        was being rejected. Lock that in as a NEGATIVE control —
        future code that brings these magnitudes back will fail."""
        old = CostCoefs(c_MOTA=0.001, c_FP_track=0.10,
                        c_FP_frame=0.002)
        d = delta_promote(p_TP=0.97, mu_TP=2.4, mu_FP=2.5, c=old)
        # The bug was that this was negative when it should be positive.
        self.assertLess(d, 0,
            "old coefs SHOULD give Δ<0 for p_TP=0.97 — that's the bug "
            "we're documenting; if this fails the buggy regime moved")
        # Now show the fixed coefs flip the sign:
        fixed = CostCoefs(c_MOTA=0.001, c_FP_track=0.005,
                          c_FP_frame=1e-5)
        d_fixed = delta_promote(p_TP=0.97, mu_TP=2.4, mu_FP=2.5, c=fixed)
        self.assertGreater(d_fixed, 0,
            f"fixed coefs must promote a 0.97-conf track, got {d_fixed:.6f}")


class TestDemoteDelta(unittest.TestCase):
    """Δ_demote = p_FP · c_FP_frame · μ_FP
                − p_TP · c_MOTA · μ_TP · match_rate"""

    def setUp(self):
        self.c = calibrate_for_clip(PP22_FPS, PP22_N_GT, PP22_N_FRAMES)

    def test_strong_FP_with_long_remaining_demotes(self):
        d = delta_demote(p_TP=0.10, mu_TP=0.3, mu_FP=3.0, c=self.c)
        self.assertGreater(d, 0,
            f"clear FP with long remaining μ_FP should demote, Δ={d:.6f}")

    def test_strong_TP_does_not_demote(self):
        d = delta_demote(p_TP=0.97, mu_TP=3.0, mu_FP=2.0, c=self.c)
        self.assertLess(d, 0,
            f"clear TP must NOT demote, Δ={d:.6f}")

    def test_repromote_demote_are_exact_inverses(self):
        """Mathematical identity: Δ_demote = −Δ_repromote (no
        c_FP_track involved — that cost has already been paid when
        the track first promoted). Demoting now and re-promoting
        later should be a no-op in expected fitness terms."""
        for (p, mt, mf) in [(0.9, 2.0, 2.0), (0.3, 1.0, 3.0),
                            (0.55, 0.5, 1.5)]:
            dr = delta_repromote(p, mt, mf, self.c)
            dd = delta_demote(p, mt, mf, self.c)
            self.assertAlmostEqual(dr, -dd, places=12,
                msg=f"symmetry violated at p={p}, μ_TP={mt}, μ_FP={mf}")

    def test_promote_pays_c_FP_track_repromote_does_not(self):
        """First-time promotion must include the c_FP_track sunk
        cost (each new track adds one to fp_tracks). LOST→TRACKED
        re-promotion does NOT — the track is already counted."""
        for (p, mt, mf) in [(0.5, 1.0, 2.0), (0.7, 1.5, 1.5)]:
            dp = delta_promote(p, mt, mf, self.c)
            dr = delta_repromote(p, mt, mf, self.c)
            # The only difference must be a single -p_FP·c_FP_track term:
            expected_gap = -(1.0 - p) * self.c.c_FP_track
            self.assertAlmostEqual(dp - dr, expected_gap, places=12)

    def test_buggy_c_FP_frame_demotes_TP_on_one_miss(self):
        """The shipped 0.002 c_FP_frame caused TPs to demote on a
        single miss because the per-frame term was 100×+ inflated.
        Lock in that the bad coefficient produces the bad behaviour."""
        # Scenario: a TP track has had a miss, head's μ_TP collapsed
        # to 0.67 but μ_FP is still 3.23 (head says 'this might be
        # ending'). With shipped buggy c_FP_frame=0.002:
        bad = CostCoefs(c_MOTA=0.001, c_FP_track=0.005,
                        c_FP_frame=0.002)
        d_bad = delta_demote(p_TP=0.56, mu_TP=0.67, mu_FP=3.23, c=bad)
        self.assertGreater(d_bad, 0,
            "buggy 0.002 c_FP_frame SHOULD demote — that's the bug")
        # Calibrated c_FP_frame ≈ 1e-5 keeps the track alive longer,
        # giving the head time to recover.
        good = CostCoefs(c_MOTA=0.001, c_FP_track=0.005,
                         c_FP_frame=1.4e-5)
        d_good = delta_demote(p_TP=0.56, mu_TP=0.67, mu_FP=3.23, c=good)
        # Note: with the calibrated c_FP_frame, demote may still fire
        # because p_TP collapsed to 0.56. The fix to the demote
        # CADENCE has to come from the head's μ_TP being more robust
        # to single misses, not from cost-coefficient tuning. This
        # test documents the cost-rule arithmetic fix without claiming
        # it solves the head's brittleness.
        self.assertLess(abs(d_good), abs(d_bad),
            "calibrated c_FP_frame must reduce |Δ_demote| magnitude")


class TestStateTransitions(unittest.TestCase):
    """The state machine is gated by (current_state, matched), not
    just delta sign. Lock in those gates so a refactor doesn't drop
    them silently."""

    def setUp(self):
        self.c = calibrate_for_clip(PP22_FPS, PP22_N_GT, PP22_N_FRAMES)

    def test_unconfirmed_unmatched_stays_unconfirmed(self):
        """No promote allowed without a match — even if Δ_promote>0
        from prior history we don't transition without an observation."""
        s = cost_decision(p_TP=0.99, mu_TP=3, mu_FP=2,
                          current_state=UNCONFIRMED, matched=False, c=self.c)
        self.assertEqual(s, UNCONFIRMED)

    def test_unconfirmed_matched_high_conf_promotes(self):
        s = cost_decision(p_TP=0.99, mu_TP=3, mu_FP=2,
                          current_state=UNCONFIRMED, matched=True, c=self.c)
        self.assertEqual(s, TRACKED)

    def test_tracked_matched_stays_tracked(self):
        """Demote can ONLY fire on an unmatched frame."""
        s = cost_decision(p_TP=0.10, mu_TP=0.1, mu_FP=5.0,
                          current_state=TRACKED, matched=True, c=self.c)
        self.assertEqual(s, TRACKED)

    def test_tracked_unmatched_strong_FP_demotes(self):
        s = cost_decision(p_TP=0.05, mu_TP=0.1, mu_FP=5.0,
                          current_state=TRACKED, matched=False, c=self.c)
        self.assertEqual(s, LOST)

    def test_lost_unmatched_stays_lost(self):
        """LOST→TRACKED requires a match."""
        s = cost_decision(p_TP=0.99, mu_TP=3, mu_FP=2,
                          current_state=LOST, matched=False, c=self.c)
        self.assertEqual(s, LOST)

    def test_lost_matched_can_re_promote(self):
        s = cost_decision(p_TP=0.99, mu_TP=3, mu_FP=2,
                          current_state=LOST, matched=True, c=self.c)
        self.assertEqual(s, TRACKED)


if __name__ == "__main__":
    unittest.main(verbosity=2)
