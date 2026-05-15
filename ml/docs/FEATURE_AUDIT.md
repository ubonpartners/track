# Match-cost NN feature audit

Two independent feature-importance analyses of the deployed match-cost
head: one **permutation-based** (2026-05-10 on v10) and one
**static weight-based** (2026-05-15 on v13). They agree on the
qualitative ranking. The static one is cheap (~5 s, no GPU); the
permutation one is closed-loop (~10 min on val corpus) and harder
to game.

This doc consolidates both, then summarises the four ablation
experiments (A1–A4 in the 2026-05-10 audit) and where each landed.

---

## v13 static analysis (2026-05-15)

L2/L1 norm of incoming first-layer weight columns for each input
feature. Higher norm = more influence on next layer's activations.
Coarse but useful prior before paying for permutation ablations.

### OBS tower (16 inputs, L2 norm of `f_obs.net.0.weight` cols)

| Rank | Feature              | L2     | L1      |
|------|----------------------|--------|---------|
| 1    | log_num_missed       | 4.256  | 17.798  |
| 2    | of_score             | 2.938  | 11.041  |
| 3    | prev_det_conf        | 2.620  | 10.974  |
| 4    | reid_cos_raw         | 1.969  | 6.706   |
| 5    | kf_score             | 1.718  | 6.623   |
| 6    | kf_d2                | 1.603  | 6.418   |
| 7    | det_conf             | 1.556  | 6.386   |
| 8    | log_observations     | 1.535  | 5.471   |
| 9    | sim_term             | 1.155  | 4.230   |
| 10   | reid_z               | 1.077  | 4.294   |
| 11   | pose_kp_visible      | 1.030  | 3.886   |
| 12   | track_subbox_conf    | 1.019  | 3.572   |
| 13   | **ocm_cos**          | 0.998  | 3.840   |
| 14   | **det_subbox_conf**  | 0.945  | 3.250   |
| 15   | **det_fiqa_score**   | 0.928  | 3.628   |
| 16   | **track_speed**      | 0.746  | 2.731   |

**Bottom 4 (drop candidates)**: `ocm_cos`, `det_subbox_conf`,
`det_fiqa_score`, `track_speed`.

### DET tower (5 inputs, L2 norm of `g_det.net.0.weight` cols)

| Rank | Feature              | L2     |
|------|----------------------|--------|
| 1    | det_aspect           | 2.062  |
| 2    | det_h                | 1.981  |
| 3    | det_w                | 1.530  |
| 4    | pose_kp_visible      | 1.485  |
| 5    | **det_conf**         | 1.163  |

**Bottom 1 (drop candidate)**: `det_conf`. Interesting — `det_conf`
is one of the most important signals on paper, but the model
effectively discards the det-tower copy because `pre_thr_score` in
PAIR already encodes the confidence-derived score.

### PAIR features (19 inputs, head's cols 32:51)

| Rank | Feature              | L2     |
|------|----------------------|--------|
| 1    | iou                  | 2.296  |
| 2    | of_score             | 2.117  |
| 3    | pre_thr_score        | 1.975  |
| 4    | size_ratio           | 1.734  |
| 5    | kf_d2                | 1.706  |
| 6    | h_ratio              | 1.323  |
| 7    | det_fiqa_score       | 1.229  |
| 8    | kf_score             | 1.079  |
| 9    | subbox_iou           | 0.961  |
| 10   | pass_0               | 0.903  |
| 11   | a_ratio              | 0.898  |
| 12   | sim_term             | 0.881  |
| 13   | pass_2               | 0.862  |
| 14   | reid_z_masked        | 0.831  |
| 15   | track_subbox_conf    | 0.819  |
| 16   | **ocm_cos**          | 0.815  |
| 17   | **det_subbox_conf**  | 0.625  |
| 18   | **conf_delta**       | 0.576  |
| 19   | **pass_1**           | 0.485  |

**Bottom 4 (drop candidates)**: `ocm_cos`, `det_subbox_conf`,
`conf_delta`, `pass_1`.

### Drop-list intersection (B2 ablation candidates)

Features that appear in the bottom quartile of TWO views are the
highest-confidence drop candidates:

1. **`ocm_cos`** (OBS + PAIR) — appears bottom in both. Likely safe
   to drop. The v10 permutation audit also flagged this as dead due
   to the conditional masking rule (only filled when
   `pass>0 && observations≥3 && num_missed≤1`).
2. **`det_subbox_conf`** (OBS + PAIR) — appears bottom in both.
3. **`det_fiqa_score`** (OBS) — bottom in one tower, but combined
   with the 2026-05-10 audit's permutation result this is a confident
   no-op.
4. **`pass_1`** (PAIR) — bottom in PAIR view. One-hot encoding of
   the high-thr pass; the model uses `pass_0` and `pass_2` but
   `pass_1` looks dead.

### Why ablation hasn't been run yet

The C runtime feature builder (`utrack_match.c` `nn_build_obs/det/pair`)
hardcodes the input vector layout. A model trained with
`--drop-features ocm_cos` produces a `.bin` expecting 15 obs inputs,
but the C runtime feeds it 16. Either the loader rejects on dim
mismatch or it silently feeds garbage into the missing column slot.

**Required C work** (~50 lines): when loading a `.bin`, read its
`obs_in/det_in/pair_in` dims from the bin header and zero the dropped
feature in the input vector builder. Then ablation becomes
`--drop-features X` + retrain + closed-loop eval.

Until that lands, ablation can only give **training-time AUC delta**
which doesn't predict deployment fitness (see
`feedback_offline_online_gap.md` and the v11_time A2 result below).

---

## v10 permutation importance (2026-05-10)

Tool: `ml.analysis.permute_match_features`. For each (view, column)
shuffles that column across val rows, recomputes the combined score
(`pre_thr_score + λ·residual`), reports Δ-AUC at both training-λ
(=1.0) and deployment-λ (=0.05).

Baselines on `pairs_val.npz` (2.36 M val pairs, 14% positive):

|                                | AUC      | Δ vs pre_thr |
|--------------------------------|----------|--------------|
| pre_thr_score only             | 0.96504  | —            |
| residual alone                 | 0.96043  | −0.00461     |
| pre + λ_train(=1.00) · res     | 0.96674  | +0.00171     |
| pre + λ_deploy(=0.05) · res    | 0.97045  | +0.00541     |

The deployed λ=0.05 gives more headroom than training λ=1.0 because at
λ=1.0 the residual is heavy enough to *fight* the highly-correlated
`pre_thr_score` (the head was trained on this corpus which already had
v9's residual baked in). At λ=0.05 the residual is a gentle nudge on
top — that's the regime that matters for production fitness.

### Top 10 features (Σ Δ_deploy across views)

| Feature              | Views        | Σ Δ_deploy | Read                  |
|----------------------|--------------|------------|-----------------------|
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

The **top 7 features account for ~95% of the residual's deployment Δ.**
The bottom 18 collectively contribute ≤ 0.

### Specific dead-or-near-dead features (v10 audit)

1. **`iou` is the single biggest contributor by 3×.** The model
   uses raw IoU as a direct anchor even with `pre_thr_score` (which
   already contains a DIoU-based fusion) — IoU has a sharper
   "definitely-the-same-track" signal than DIoU when boxes overlap.

2. **OCM (motion-direction cosine) is dead.** Both `ocm_cos` views
   show essentially zero importance. The C runtime only computes
   OCM when `pass>0 && observations≥3 && num_missed≤1`, so it's
   almost always masked to zero. Confirmed by v13 static analysis.

3. **`reid_z` is anti-helpful in obs view (−0.00002).** The z-scored
   appearance feature has a `reid_stats_valid` mask, but the value
   is passed through unconditionally in obs (raw `reid_z`). When
   `reid_stats_valid=false` this is NaN→0; when it's true it's
   correlated with `sim_term`. The pair view's `reid_z_masked`
   (z·valid) is the cleaner version and is mildly helpful (+0.00006).

4. **`det_conf` is redundant across three views**, and aggregated to
   ≈0. It's in obs (det_conf, prev_det_conf), det (det_conf), and
   pair (conf_delta). The head doesn't need all three.

5. **Face/subbox features (`det_subbox_conf`, `track_subbox_conf`,
   `det_fiqa_score`, `subbox_iou`) are essentially zero.** This is
   the v2 face schema. The pair-trace corpus has them but the model
   isn't extracting useful signal. Investigated in A5 below (the
   STATE head DOES use face features, but not the match NN at the
   v10 operating point).

6. **`kf_d2` is dead in both views.** Mahalanobis distance only
   meaningful for tracks with ≥ 2 observations + initialised KF.

7. **`size_ratio` is dead at deployment-λ.**

8. **`track_speed` is essentially dead.** Same conditional-masking
   pattern as OCM.

### Features in the pair-trace corpus but NOT fed to the NN

- **`time_since_det`** — seconds since track's last matched
  detection. The "time gap" feature StrongSORT / OC-SORT / DeepOCSORT
  all use directly. Tested in A2 below (doesn't ship).
- **`scene_density`** — number of dets in the current frame.
  Tested in A3 below (positive AUC, doesn't ship alone).
- **`reid_stats_valid`** — used only to mask `reid_z_masked`. Could
  be a flag input.
- **Absolute box position** (centres) — not fed.

---

## Ablation experiments (2026-05-10)

### A2 — add `time_since_det` (v11_time)

Trained `phase3_v11_time.pt` with `log1p(time_since_det)` in obs+pair
(v3 schema, obs_in=17 / pair_in=20). Permutation importance lands
`log_time_since_det` as the **#4 most important feature** (Σ Δ_deploy
+0.00259, ~25% of the residual signal).

Offline AUC was promising:

| Head            | λ_deploy | val AUC | Δ vs pre_thr |
|-----------------|----------|---------|--------------|
| pre_thr only    | —        | 0.96504 | —            |
| v10 (shipped)   | 0.05     | 0.97045 | +0.00541     |
| **v11_time**    | 0.20     | 0.97342 | +0.00839     |

But closed-loop fitness on `diverse-29` was the opposite:

| Recipe                | fitness | mota    | fp_tracks |
|-----------------------|---------|---------|-----------|
| v10 + λ=0.05 (shipped)| 0.6393  | 0.6559  | 24.0      |
| v11_time + λ=0.05     | 0.6369  | 0.6550  | 26.7      |
| v11_time + λ=0.10     | 0.6360  | 0.6539  | 26.0      |
| v11_time + λ=0.20     | 0.6337  | 0.6535  | 29.7      |

**v11 regresses fitness by 0.0015–0.0056** across the λ grid,
widening with λ. Reading: with the time-gap feature available, the
head re-attaches drifting tracks more confidently. That increases
TP-pair AUC offline but in deployment it manifests as lingering /
resurrected tracks scoring as FP — the case the fp_tracks coefficient
penalises.

**Textbook reproduction of the offline ≠ online gap** (user memory
`feedback_offline_online_gap.md`). pairs_val AUC is not a reliable
predictor of deployment fitness for the match-cost head.

**Decision**: v11 does NOT ship. v10 stays. The C runtime's v3 dim
acceptance was kept (additive, harmless) so a future fitness-aware
retrain can re-use the schema bump.

### A1 — leaner head with 8 dropped features (v12_lean)

Dropped: `reid_z` (obs), `det_subbox_conf`, `det_fiqa_score`,
`subbox_iou`, `ocm_cos`, `kf_d2`, `size_ratio`, `det_conf` (det copy
only — kept in obs and pair).

Result: 10/4/13-dim head vs v10's 16/5/19 — **~38% leaner**.

| Head        | dims      | λ=0.05 AUC | λ=0.10  | λ=0.20  |
|-------------|-----------|------------|---------|---------|
| v10 (ship)  | 16/5/19   | 0.97045    | 0.97150 | 0.97145 |
| v12_lean    | 10/4/13   | 0.97010    | 0.97097 | 0.97024 |

Lean head loses ~0.0004 AUC at every λ — within noise. The drop
happens because the head loses access to weak signals that were
additive across views (e.g. `det_conf` lives in three views — keeping
just two costs a thread).

**Decision**: don't ship v12_lean. No fitness motivation, only a
parameter-count argument, and a lean schema would need its own C dim
variant. Keep the `--drop-features` flag for future ablation work.

### A3 — add `scene_density` (v13_density)

Trained with `log1p(scene_density)` appended to pair vector (16/5/20).
+0.0001 to +0.0026 AUC, monotone in λ. Density slot is #5–#6 most
important feature.

**Decision**: don't ship v13_density alone. AUC gain too small for a
C-side schema bump on its own. Park unless combined with another
feature that makes the schema bump worth it.

### A4 — warped DIoU for subbox (v14_subdiou)

Triggered by the audit observation that `subbox_iou` is near-zero
importance because it's computed as `IoU(track.det.subbox, det.subbox)`
where `track.det.subbox` is the *last-observed* face box (no motion
warp). When the person moves more than a face-box width between
frames, IoU drops to 0 mechanically even on correct matches.

Implemented `compute_subbox_scores` in C runtime producing both
plain `subbox_iou` and `subbox_diou_warped` (DIoU after applying the
OF-predicted main-box translation to the track subbox). Pair-trace v3
schema (rec size 152→156). Yaml-gated chooser
`utrack.subdiou_warped: false|true`.

Standalone signal on the new corpus (subset where face detected):
- `subbox_iou` AUC 0.844
- `subbox_diou_warped` AUC 0.899 (~3× more non-zero rows)

But trained heads (`phase3_v10base_on_v14corpus.pt` plain vs
`phase3_v14_subdiou.pt` warped) **tie within noise**:

| Head                  | dims    | val AUC | fitness | fp_tracks |
|-----------------------|---------|---------|---------|-----------|
| v10 (shipped)         | 16/5/19 | 0.99219 | 0.6393  | 24.0      |
| v10base_on_v14corpus  | 16/5/19 | 0.99219 | 0.6350  | 29.7      |
| v14_subdiou (warped)  | 16/5/19 | 0.99197 | 0.6339  | 31.0      |

**The head doesn't use the warped subbox.** Within-corpus delta is
−0.0011 (within run noise σ ≈ 0.0008). Main-box IoU still dominates
(Σ Δ_deploy +0.01182 — 50× larger).

**Decision**: v14_subdiou does NOT ship. v10 stays default with
`subdiou_warped: false`. The C-runtime code is kept anyway (the fix
is correct; the bug was real) — a future architectural change might
make the head capable of using the subbox signal.

**The bigger story**: −0.0043 fitness drop from **corpus regen
alone** (v10base trained on regen corpus vs shipped v10). This is
the "corpus drift" problem that gates all retrains; see user memory
`feedback_track_corpus_drift.md`.

### A5 — face conf in state head (separate from match NN)

End-to-end plumbing of `det_subbox_conf` / `track_subbox_conf` /
`det_fiqa_score` into the state-head GRU input (V3 schema, 25 → 28).

**Within-corpus** comparison (face vs no-face, same corpus):
- v21 corpus: face = **+0.0178 fitness**
- v22 corpus: face = **+0.0722 fitness**

Face features add real, repeatable, fitness-positive signal in the
state head (unlike the match NN where they're dead). But both
retrains regressed vs shipped v20 by 0.04–0.17 fitness — same
corpus-drift symptom.

**Decision**: v22_face does NOT ship today. Keep v20 in production.
The C-runtime V3 path is kept (additive, in_dim ∈ {19, 25, 28}). A
future fitness-aware retrain that resolves the corpus drift can ship
v22_face by flipping the yaml `nn_state_path`.

---

## What still hasn't been tried

These are the ablation/feature experiments the v10 audit suggested
but no one has run yet, ordered by signal-to-effort ratio:

1. **C runtime feature-builder zero-injection (~50 lines)** —
   enables true closed-loop ablation. Without this, every "drop
   feature X and retrain" experiment is stuck reporting AUC delta,
   which doesn't predict fitness (A2's lesson).
2. **`reid_z` masking fix in obs view.** The pair view masks
   correctly; the obs view leaks NaN→0 unmasked.
3. **Architectural variant: down-weight main-box IoU during
   training** to force capacity allocation onto appearance + face
   features.
4. **Cross-attention path** where face features explicitly interact
   with the appearance channel before the head.
5. **Per-track learned face-quality gate** that decides per-pair
   whether to attend to the subbox features at all.
6. **Combine A2 + A3** (time_since_det AND scene_density) plus a
   fitness-aware loss that resists the v11 over-re-attachment failure
   mode — this is the angle most likely to make a feature-architecture
   move ship-worthy.
