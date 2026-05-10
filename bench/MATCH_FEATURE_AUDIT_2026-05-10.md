# 2026-05-10 — Match-cost NN feature audit

Permutation-importance audit of the deployed v10 match-cost head against
`bench/data/pairs_val.npz` (2.36M val pairs, 14% positives).

Tool: `bench/permute_match_features.py`. For each (view, column) it
shuffles that column across val rows and recomputes the combined score
(`pre_thr_score + λ·residual`) — twice, once at training λ (=1.0) and
once at deployment λ (=0.05 in `uc_v11.yaml`). Δ-AUC is the drop vs
the un-shuffled baseline. Higher Δ = feature is doing more work for
the model.

Baselines on `pairs_val.npz`:

|                                | AUC      | Δ vs pre_thr |
|--------------------------------|----------|--------------|
| pre_thr_score only             | 0.96504  | —            |
| residual alone                 | 0.96043  | -0.00461     |
| pre + λ_train(=1.00) · res     | 0.96674  | +0.00171     |
| pre + λ_deploy(=0.05) · res    | 0.97045  | +0.00541     |

The deployed λ=0.05 gives more headroom than training λ=1.0 because at
λ=1.0 the residual is heavy enough to *fight* the highly-correlated
pre_thr_score (the head was trained on this corpus which already had
v9's residual baked in). At λ=0.05 the residual is a gentle nudge on
top — that's the regime that matters for production fitness.

## Aggregated ranking (Σ over views, deployment-λ)

| Feature              | Views        | Σ Δ_deploy | Read                  |
|----------------------|--------------|-----------:|-----------------------|
| iou                  | pair         | +0.00229   | Box overlap dominates |
| of_score             | obs + pair   | +0.00087   | OF-warped motion fit  |
| log_num_missed       | obs          | +0.00045   | Track recency         |
| pre_thr_score        | pair         | +0.00044   | NN reads the prior    |
| log_observations     | obs          | +0.00037   | Track tenure          |
| h_ratio              | pair         | +0.00033   | Height consistency    |
| reid_cos_raw         | obs          | +0.00022   | Raw appearance        |
| a_ratio              | pair         | +0.00015   | Aspect consistency    |
| det_h                | det          | +0.00010   | Det size              |
| pass_0               | pair         | +0.00009   | Initial-pass flag     |
| pass_2               | pair         | +0.00008   | Low-pass flag         |
| sim_term             | obs + pair   | +0.00007   | Weighted appearance   |
| pose_kp_visible      | obs + det    | +0.00006   | Pose visibility       |
| reid_z_masked        | pair         | +0.00006   | Z-scored appearance   |
| prev_det_conf        | obs          | +0.00005   | Prior conf            |
| conf_delta           | pair         | +0.00004   | Δ conf                |
| det_aspect           | det          | +0.00004   | Det shape             |
| kf_score             | obs + pair   | +0.00004   | KF motion             |
| pass_1               | pair         | +0.00002   | High-pass flag        |
| track_subbox_conf    | obs + pair   | +0.00001   | Face conf, track      |
| track_speed          | obs          | +0.00001   | Track speed           |
| det_w                | det          | +0.00001   | Det width             |
| det_fiqa_score       | obs + pair   |  0.00000   | Face quality          |
| ocm_cos              | obs + pair   |  0.00000   | Motion dir cos        |
| kf_d2                | obs + pair   | -0.00000   | KF Mahalanobis²       |
| subbox_iou           | pair         | -0.00000   | Face-box overlap      |
| size_ratio           | pair         | -0.00001   | Area ratio (dup)      |
| det_conf             | obs + det    | -0.00001   | Det conf (3rd path)   |
| det_subbox_conf      | obs + pair   | -0.00001   | Face conf, det        |
| reid_z               | obs          | -0.00002   | Z-scored cos (obs)    |

## Reading

The top **7 features** together account for ~95% of the residual's
deployment Δ. The bottom **18 features** collectively contribute ≤0
(combined importance is within permutation noise of zero, with several
mildly negative).

Specifics worth calling out:

1. **`iou` is the single biggest contributor by 3×.** The model uses
   raw IoU as a direct anchor for the residual — even with
   `pre_thr_score` (which already contains a DIoU-based fusion) the
   head is asking for plain IoU separately. This is consistent: IoU
   has a sharper "definitely-the-same-track" signal than DIoU when
   boxes overlap, and the head leverages it.

2. **OCM (motion-direction cosine) is dead.** Both `ocm_cos` views show
   essentially zero importance. The C runtime only computes OCM when
   `pass>0 && observations≥3 && num_missed≤1`, so it's almost always
   masked to zero. Worth dropping from the model unless we change the
   masking rule.

3. **`reid_z` is anti-helpful in obs view (-0.00002).** The z-scored
   appearance feature has a `reid_stats_valid` mask, but the value is
   passed through unconditionally in obs (raw `reid_z`). When
   `reid_stats_valid=false` this is NaN→0; when it's true it's
   correlated with `sim_term`. The pair view's `reid_z_masked` (z·valid)
   is the cleaner version and is mildly helpful (+0.00006). Obs's
   `reid_z` should be dropped, or also masked.

4. **`det_conf` is redundant across three views**, and aggregated to
   ≈0. It's in obs (det_conf, prev_det_conf), det (det_conf), and pair
   (conf_delta). The head doesn't need all three slots.

5. **Face/subbox features (`det_subbox_conf`, `track_subbox_conf`,
   `det_fiqa_score`, `subbox_iou`) are essentially zero.** This is the
   v2 face schema — added during the v9 retrain. The pair-trace
   corpus has them but the model isn't extracting useful signal.
   Either the face-quality signal is too weak in the val mix, or it's
   correlated with features the head already uses (det_conf).

6. **`kf_d2` is dead in both views (-0.0 / -0.0).** Same story as OCM:
   it's a Mahalanobis distance only meaningful for tracks with ≥2
   observations and an initialised KF, otherwise zero. The head finds
   `kf_score` (the exp transform) marginally useful but not `kf_d2`
   raw. Likely the exp transform fully captures the signal.

7. **`size_ratio` is dead** (-0.00001 / +0.00056). The Δ_train shows
   the head learned a non-trivial use of it during training but at
   deployment-λ it doesn't matter. h_ratio + a_ratio already capture
   the dimension consistency.

8. **`track_speed` is essentially dead.** Same conditional-masking
   pattern as OCM — only filled when the motion-history conditions
   hold.

## Features in the corpus but NOT fed to the NN

The pair-trace record has additional fields the model never sees:

- **`time_since_det`** — seconds since track's last matched detection.
  This is the *time gap* feature StrongSORT / OC-SORT / DeepOCSORT all
  use directly. Currently unused by our NN.
- **`scene_density`** — number of dets in the current frame. Indicator
  of crowded-vs-sparse scene context. Unused.
- **`reid_stats_valid`** — currently used only to mask `reid_z_masked`.
  Could also be a flag input.
- **Absolute box position** (track and det x/y centres) — not fed.
  Could let the head learn edge-of-frame patterns.

## Proposed actions

- **A1**: drop the bottom-7 features and retrain — confirm AUC holds.
  Removable candidates (by aggregate Δ_deploy ≤ 0): `reid_z` (obs),
  `det_subbox_conf` (obs+pair), `det_fiqa_score` (obs+pair),
  `subbox_iou` (pair), `kf_d2` (obs+pair), `ocm_cos` (obs+pair),
  `det_conf` (det — keep in obs and pair).
- **A2**: add `time_since_det` (log1p-transformed) to obs+pair.
  Retrain. Measure AUC + fitness.
- **A3**: add `scene_density` (log1p-transformed) to pair. Retrain.
- **A4**: if A2/A3 don't improve, accept the lean A1 head as the new
  ship.

Order: A2 first (single highest-EV experiment), then A1, then A3.
