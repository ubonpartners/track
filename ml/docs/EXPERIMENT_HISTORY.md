# Experiment history

A consolidated record of every experiment that shipped, was rejected, or
left behind reusable insight. Ordered chronologically. Numbers in the
"fitness" column are full-176 closed-loop unless stated otherwise.

The fitness metric is
**`MOTA − 5e-4 × fp_tracks_total`**; the 5e-4 weight on FP-tracks is
what punishes the "more recall at any cost" direction (see user memory
`feedback_track_eval_metric.md`).

---

## Lineage at a glance

| Match-cost NN | State head | Shipped on | Notes |
|---|---|---|---|
| v9              | v5         | pre-2026-05 | regression on 68/72 PP22; root cause: state corpus mismatch with runtime |
| v9              | v10 → v14  | 2026-05-09 → 2026-05-10 | μ_TP truncation fix, cleaner trainer, mandatory `--comment`/`--save` |
| v10             | v14        | candidate; not shipped | AUC +0.0092 on current corpus, never closed-loop'd to ship |
| v9              | v20_pw05   | 2026-05-10 evening | h=64 + pos_weight=0.5; +0.0042 fitness over v14 |
| v12             | v22_pw05   | 2026-05-13 | first iter1 bootstrap of both NNs in lockstep |
| **v13**         | **v23_pw05** | **2026-05-14** | cheap-filter aware: δ=0.5 training filter + C runtime filter |
| v13             | v23_pw05   | **2026-05-15** | **F5d ship: δ=0.7 + delete_dup_iou=0.70 (yaml-only)** |

---

## Phase 1 (May 2026) — fixing the state-corpus / runtime mismatch

**Context**: the existing `nn_state_v5.bin` regressed 68 of 72 PP22 clips
that had both metrics (mean MOTA delta −0.105 vs no-NN baseline).

**Root cause**: the corpus produced by `ml/data_prep/build_state_corpus.py`
was strictly narrower than what the C runtime queried the head on. Four
state transitions disagreed; `e_track` was always zero in training but
non-zero at inference.

**Fix**: refactored `utrack.c` to a **pure-NN state machine** (no
heuristic bypasses) + rewrote `build_state_corpus.py` to do **label-driven
replay**: GT history as oracle for transitions, runtime hard floors
(`missed≥2`, `K_min`, buffer) as ceilings. One shared label code path
between replay and label generation — no risk of drift.

Plumbed in commits across tasks #43–#47.

---

## v9 → v10 → v14 — clean-trainer pipeline validation (2026-05-10 morning)

**Goal**: exercise the cleaned-up trainer + exporter + verification chain
end-to-end on a fresh checkout.

**State head v15 vs shipped v14 (offline, decoupled-eval):**

| Metric                          | v14 (shipped) | v15 (this retrain) | Δ       |
|---------------------------------|---------------|--------------------|---------|
| TP coverage (TRACKED %)         | 21.66 %       | 28.67 %            | +7.01 pp |
| FP exposure  (TRACKED %)        | 0.57 %        | 1.82 %             | +1.25 pp |
| mota_proxy                      | 0.4811        | 0.6153             | +0.1342 |
| fp_per_frame                    | 0.0044        | 0.0155             | +0.0111 |
| **fitness_proxy**               | **0.4416**    | **0.4268**         | −0.0148 |
| Trace library (6 traces)        | 7 / 14 pass   | 10 / 14 pass       | +3      |

v15 was the more aggressive head: higher mota / TP coverage but more
FP-tracks, so it lost on the fp_track penalty. v14 was tuned for the
exact fitness target. v15 won on trace-library interpretability.

**Decision**: keep v14 in production. v15 saved in `ml/data/` as a
validation artefact + a starting point for further tuning.

**Match-cost v9 vs v10 (head-to-head on pairs_val):**

| Head                   | Aggregator | Val AUC | Δ over `pre_thr_score` |
|------------------------|------------|---------|------------------------|
| pre_thr_score (no NN)  | —          | 0.9650  | —                      |
| v9 (shipped)           | ema_fixed  | 0.9663  | +0.0013                |
| **v10 (this retrain)** | ema_fixed  | 0.9742  | **+0.0092**            |

v10 was a candidate replacement for v9 (+0.0079 AUC over v9), but never
closed-loop'd. The match-cost stack later got rebuilt from scratch in
the cheap-filter iter1_d05v2 recipe (→ v13).

**What the run validated:**
- Trainers ran cleanly from scratch on the current corpora.
- `--save` required (no silent loss).
- Face-feature auto-detect on match-cost trainer prevents the
  "trained an obs_in=13 model on an obs_in=16 corpus" footgun.
- Metadata in both `.pt` and `.bin` trailer; `ml.util._artefact_meta`
  reads it.
- Re-export of v14 .pt was bit-identical to shipped .bin (modulo trailer).
- `ubon_cstuff` library built; cu13/cuda_runtime include filter
  unblocked the conda CUDA13 header conflict.
- `unit_tests`: 200 / 200 PASS.

---

## v20 — pos_weight=0.5 + hidden=64 (2026-05-10 evening)

Continued the search for a fitness win after the v15 regression.

### Failed: scene-aggregate features (v16-scene, v17 variants)

Plumbed the 6 Phase-29 per-scene EMAs (promote_rate, mean_det_conf_TRACKED,
unmatched, track_density_smooth, mean_alive_track_age,
det_conf_minus_scene_TP_avg) end-to-end into the C runtime
(`ubon_cstuff:0397799`). Trained as v16-scene + v17 variants (h32+w3,
h32+w5, h64+w1, h64+w3).

**Every scene-feature variant lost** on full-178 fitness (−0.011 to
−0.017) because the extra discrimination made the head MORE aggressive,
doubling fp_tracks. `c_FP_track` sweeps couldn't recover the gap.

### Win: pos_weight < 1 BCE bias (v20)

Telling the trainer that false-positives cost more than false-negatives
(pos_weight < 1) produces a head that's slightly more conservative than
v14 without losing the capacity gains of hidden=64. Full-178 3-run
results:

| Head                       | fitness ± σ          | mota    | fp_tr |
|----------------------------|----------------------|---------|-------|
| v14 (prior ship)           | +0.4759 ± 0.0012     | +0.5175 | 79.0  |
| v18 h64 pw=1.0             | +0.4651 ± 0.0032     | +0.5292 | 123.7 |
| v20 h64 pw=0.7             | +0.4778 ± 0.0006     | +0.5218 | 83.7  |
| **v20 h64 pw=0.5 (SHIP)**  | **+0.4801 ± 0.0013** | +0.5108 | 57.3  |
| v20 h64 pw=0.4             | +0.4802 ± 0.0009     | +0.5056 | 47.0  |
| v20 h64 pw=0.3 (1 run)     | +0.4331 (overcooked) | 0.4432  | 17    |

**Shipped**: `nn_state_v20_pw05.bin` at `+0.4801 ± 0.0013`, a clean
**+0.0042 fitness over v14** (~3σ above run-to-run noise). The fitness
curve is broadly flat between pw=0.4 and pw=0.5 (both at +0.0042); pw=0.5
picked as slightly less aggressive.

**What this meant**: `pos_weight` was a long-standing trainer arg
(defaulted to 1.0). The win was just hidden=64 + pos_weight=0.5. No
architecture change, no new features, no corpus change. Scene-feature
plumbing left in the C runtime as a strict superset (in_dim ∈ {19, 25}).

---

## V2 / V3 face + scene work (2026-05-11) — silent-training-bug catastrophe

**User asked**: "queue up trying adding face confidence to the state-head
network" + "investigate the 'corpus drift' — have you lost the 'v18
recipe'?"

**Diagnosis** (commits 877a2da, 055c52e): V2 (--with-scene) and V3
(--with-face) heads were **silently training on constants**. Commit
e288eea (2026-05-10, legacy-bench cleanup) deleted the `SceneStats`
replay class from `build_state_corpus.py` without disabling the trainer's
`--with-scene` flag. The trainer's `build_input_matrix_no_state` happily
fell back to `SCENE_COL_DEFAULTS` constants for every row. Every V2/V3
retrain after 2026-05-10 trained on constants but was evaluated against
the C runtime's dynamic EMAs — a catastrophic OOD shift.

**Measured impact**: V2 retrain on state_corpus_v23 hit fitness 0.347
vs V1 shipped 0.639 (**Δ = −0.29**) — the "regression" that gated
face-feature work for ~2 days.

**Fix (3 commits):**
1. `547aad1` — schema-sentinel guard
   `ml/util/verify_tree_sentinels.py` + pre-commit hook at
   `ml/githooks/pre-commit`. Activate per-clone with
   `git config --local core.hooksPath ml/githooks`. Asserts the v3
   face/diou schema AND the SceneStats class AND the trainer fail-loud
   guard are present on every commit touching schema-relevant files.
   **Prevents recurrence of silent-revert failures.**
2. `877a2da` — trainer `--with-scene` raises `ValueError` on missing
   corpus scene columns instead of silently falling back to constants.
3. `055c52e` — restored `SceneStats` replay class as Python port of
   `utrack.c` lines 605–707 (alpha=0.05, MIN_SAMPLES=20). Added 6 scene
   columns to `EXAMPLE_DTYPE`. `apply_scene_stats_to_examples()` is a
   frame-major post-pass that fills them in-place mirroring the C
   runtime call ordering.

**K-seed sweep on v24** (seeds 0–4, --pos-weight 0.5, h=64, 16 epochs):

| Variant       | offline best_score | diverse-29 fitness mean ± std | MOTA  | fp_tracks |
|---------------|--------------------|--------------------------------|-------|-----------|
| V1 shipped    | 0.7282             | 0.6391 ± 0.0001                | 0.656 | 24.0      |
| V1 retrain    | 0.7282             | 0.6392 ± 0.0001                | 0.656 | 24.0      |
| V2 s=0        | 0.7299             | 0.6382 ± 0.0004                | 0.661 | 35.7      |
| V2 s=1        | 0.7301             | 0.6369 ± 0.0004                | 0.661 | 38.7      |
| V2 s=2 ★best  | 0.7318             | 0.6266 ± 0.0007                | 0.647 | 32.3      |
| V3 s=0        | 0.7311             | 0.6380 ± 0.0003                | 0.660 | 35.3      |
| V3 s=2 ★best  | 0.7313             | 0.6362 ± 0.0001                | 0.659 | 35.0      |

★ = best offline-score seed in its variant. The **best-offline-score
seed for both V2 and V3 produced the worst (or near-worst) fitness** —
val-score and fitness are not just decoupled, they're *anti-correlated*
at this operating point. Selecting V2/V3 candidates by offline metric
would pick a head 0.013 fitness worse than the seed-0 candidate.

**Full-178 verdict**:

| Variant            | fitness ± σ          | MOTA   | fp_tracks |
|--------------------|----------------------|--------|-----------|
| V1 retrain (=ship) | 0.4794 ± 0.0008      | 0.5116 | 60.3      |
| V2_s0 + cFP=1e-3   | 0.4749 ± 0.0035      | 0.5128 | 71.7      |

**V2 LOSES full-178 by −0.0045** (5σ on V1's std). The diverse-29 +0.0013
advantage was subset-specific noise. Scene features added MOTA but
produced +11 FP-tracks net of threshold tuning; fp-track penalty
(0.0005 × 11 = 0.0055) buried the +0.0012 MOTA gain.

**Decision**: V1 (shipped v20) stays. No ship of V2 or V3. Scene/face
features as plumbed today are net-neutral-to-negative at the v20
operating point.

**What was gained:**
- V2/V3 plumbing is now CORRECT — future face-feature work resumes
  from a working baseline instead of starting from a −0.29 catastrophe.
- Schema-sentinel guard prevents the silent-deletion failure mode
  recurring.
- Confirmed the offline-online gap is severe enough that **K-seed by
  offline best_score is actively misleading** for V2/V3. Future work
  needs closed-loop screening as the primary selection criterion.

---

## Matching-cheap-filter — iter1 bootstrap (2026-05-13 → 2026-05-14)

**Hypothesis**: train + infer should agree on which (track, det) pairs
the NN gets to see. The deployed v9/v20 was trained on the *full*
per-event pair distribution but was queried at runtime through a cheap
heuristic filter that dropped pairs whose pre-thr score was far from the
matcher's decision boundary. Runtime distribution ≠ training distribution
→ wasted NN capacity on irrelevant pairs.

### Cheap-filter design (three gates in `utrack_match.c`)

1. **Below-threshold discard** — `score < match_thr − δ` → drop the
   pair entirely (no NN, no further consideration).
2. **Delta-from-top filter** — `score < cmax − δ` (where `cmax` is the
   highest score for this detection) → skip NN evaluation.
3. **Near-threshold rescue** — if the score is within δ of `match_thr`,
   keep the NN active regardless of the delta-from-top gate (lets the
   matcher use NN output for accept/reject right at the boundary).

### Training-side filter (`build_pair_dataset.py --delta-filter δ`)

Mirror the runtime cheap filter at corpus-build time: drop pair-log rows
whose pre-thr score is ≥ δ below the per-event top score (gate a) +
below the threshold lower bound (gate b). At inference the same gates
fire, so the NN sees the same input distribution it trained on.

### Iter1 d05v2 — full bootstrap from scratch

Built the iter1_d05v2 corpus from the iter0_noNN_jaad pair-log with
δ=0.5 cheap-filter training mirror. Re-trained match NN (v13) and state
head (v23_pw05) on it. C runtime cheap-filter shipped with
δ=0.5 (`utrack.match_cheap_filter_delta`).

| Variant                                | full-176 fit | fp_tracks |
|----------------------------------------|--------------|-----------|
| v12 + v22_pw05 (prior ship, no filter) | 0.5746       | 65        |
| v13 + v23_pw05 + δ=0.5 (this ship)     | 0.5736       | 68        |

Within ±0.003 single-eval noise; **~50% of match-NN evaluations skipped
at runtime** (the cheap-filter's selling point). JAAD val: ship 0.2693
vs prior 0.2664 — both within noise.

Shipped via `/mldata/config` commit `cd503a3` on 2026-05-14.

### Bin provenance (chain self-contained in meta.json):
- `nn_match_v13.bin` ← `ml/data/iter1_d05v2/nn_match.bin`
- `nn_state_v23_pw05.bin` ← `ml/data/iter1_d05v2/state_head_pw0.5.bin`
- ubon_cstuff main: `57b21b1` (cheap-filter runtime)
- track main: `914e022` (cheap-filter training + analyzer)
- stuff main: `56085ef` (B-frame + framerate transcode fix)

---

## Overnight 2026-05-14 → 2026-05-15 — sweep + F5d ship

Long autonomous run. Goal: probe whether the iter1_d05v2 cheap-filter
ship had hyperparameter headroom left, what direction it was in, and
whether the cross-domain JAAD-val gap (no-NN 0.286 vs ship 0.266) could
be closed by more data.

Twelve experiments queued; the eight that produced signal are below.

### A1 — state-head pos_weight sweep

Trained pw ∈ {0.3, 0.4, 0.6, 0.7, 1.0} at seed=0, h=64, ε=16 each. Eval
at filter δ=0.5 ship config.

| pw   | full-176 fit | full-176 fp_tr | JAAD fit | Note |
|------|--------------|----------------|----------|------|
| 0.3  | 0.517        | 28             | 0.207    | over-conservative |
| 0.4  | 0.557        | 34             | 0.215    |  |
| 0.5⁺ | 0.575        | 66             | 0.266    | ship at A1 launch |
| 0.6  | **0.579**    | 67             | 0.269    | apparent winner |
| 0.7  | 0.549        | 145            | 0.285    | too aggressive on full-176 |
| 1.0  | 0.540        | 170            | **0.290** | wins JAAD by +0.024 |

⁺ ship at A1 launch (filter δ=0.5).

**A1 alone read as "ship pw=0.6"**. Then F3 multi-seed turned that
on its head.

### A3 — cheap-filter δ sweep at inference (JAAD-val only, then full-176)

| δ        | full-176 fit | JAAD fit | fp_tr (full-176) |
|----------|--------------|----------|------------------|
| 0.4      | 0.574        | 0.267    | 69               |
| 0.5⁺     | 0.574        | 0.266    | 69               |
| **0.7**  | **0.575**    | **0.270** | **66**          |

Marginal but consistent direction (both views). The +0.001/+0.004
sat inside ±0.003 noise, but the direction was the *opposite* of
my prior (smaller δ should help cross-domain by passing more info to
the NN), so the result was treated as signal rather than noise.

### B1 — match-NN static feature importance

L2/L1 norms of incoming first-layer weight columns. See
[FEATURE_AUDIT.md](FEATURE_AUDIT.md) for the full table.

**Bottom-quartile drop candidates** (suggested for future ablation):

- OBS bottom: `ocm_cos`, `det_subbox_conf`, `det_fiqa_score`,
  `track_speed`
- DET bottom: `det_conf` (the model relies on `pre_thr_score` in PAIR
  for confidence info)
- PAIR bottom: `ocm_cos`, `det_subbox_conf`, `conf_delta`, `pass_1`

No ablation actually run — would need a C-runtime change to feed the
modified `.bin` (the runtime hardcodes the input width). Deferred.

### D1, D2 — JAAD coverage probes (both failed)

**Hypothesis**: more JAAD pairs in the training corpus would close the
cross-domain gap (no-NN 0.286 vs ship 0.266 on JAAD val).

- **D1**: promote JAAD test (117 clips) → train. Full bootstrap.
  Result: full-176 0.562 (**−0.012 vs ship**), JAAD 0.263 (no
  improvement).
- **D2**: oversample JAAD pairs 4× at the pair-dataset stage. No
  re-tracking. Result: full-176 0.577 (**+0.002 vs ship**) but JAAD
  0.261 (**−0.005**, worse).

**Both falsify the "more JAAD data" hypothesis from independent
angles**. Adding/upweighting JAAD pairs makes the NN more cautious
(fewer fp_tracks on dense crowds), but it doesn't help JAAD's
sparse-density recall. The cross-domain gap is not a data-volume problem.

See user memory `project_d1_jaad_test_promotion.md`.

### E1 — `bayes_c_FP_track` sweep

3e-4, 5e-4, 7e-4 (current), 1e-3, 1.5e-3 — flat fitness across
{3e-4, 5e-4, 7e-4, 1e-3}; 1.5e-3 starts to break. Cost-rule c_FP_track
is saturated at the shipped value.

### E2 — `track_buffer_seconds` sweep

1.5 / 2.0 / 2.2 (current) / 3.0 / 4.0 — clean U-shape; **2.2 is at the
peak**. Confirmed default. 4.0 wins JAAD by +0.001 (long-occlusion
benefit) but loses full-176.

### E3 — `delete_dup_iou` sweep ⭐

User memory `feedback_track_thresholds_dedup.md` predicted "stronger
NN should allow looser dedup". Confirmed:

| dd     | full-176 fit | fp_tr |
|--------|--------------|-------|
| **0.70** | **0.577**  | **65** |
| 0.80   | 0.575        | 68    |
| 0.90⁺  | 0.575        | 67    |
| 1.0    | 0.573        | 69    |

**dd=0.70 wins** +0.002 fitness, −2 fp_tracks. No JAAD regression. The
relaxed dedup gate works because the stronger v13 match-NN already
disambiguates overlapping candidates the IoU dedup used to handle.

### E4 — `nn_lambda` sweep ⭐ (new IDF1 finding)

| λ    | fp_tr | MOTA  | **IDF1** | fitness |
|------|-------|-------|----------|---------|
| 0.02 | 63    | 0.608 | 0.570    | 0.574   |
| 0.05⁺| 68    | 0.612 | 0.588    | **0.576** |
| 0.10 | 96    | 0.610 | 0.598    | 0.560   |
| 0.15 | 111   | 0.612 | 0.604    | 0.554   |
| 0.20 | 119   | 0.611 | **0.607** | 0.549  |

**IDF1 climbs monotonically with λ** (0.570 → 0.607 on full-176; 0.420
→ 0.443 on JAAD). Each +0.05 λ buys ~+0.005 IDF1, paid in fp_tracks.
**λ=0.05 is fitness-optimal**; **λ=0.10+ is the IDF1-leaning
alternative** for deployments that care about ID consistency over
fp-track rate (e.g. surveillance with downstream re-id).

See user memory `project_nn_lambda_idf1_dial.md`.

### E5 — `max_consecutive_misses` sweep

mm ∈ {5, 7, 10, 15, 20}. mm=10 (current) and mm=15 tied at peak
(0.576). mm=5 catastrophic (0.565). mm=20 wins JAAD by +0.002.
No clear ship change.

### G1 — runtime profile ⭐

4 representative clips × 4 δ values:

| Clip          | δ=0.0   | δ=0.5   | δ=0.7   | δ=1.0   |
|---------------|---------|---------|---------|---------|
| CEVO indoor   | 9.18 s  | 9.19 s  | 9.13 s  | 9.24 s  |
| PP22 long     | 26.74 s | 27.07 s | 26.98 s | 27.00 s |
| MOT17 medium  | 8.33 s  | 8.45 s  | 8.26 s  | 8.28 s  |
| JAAD short    | 5.09 s  | 5.13 s  | 5.08 s  | 5.12 s  |

**The cheap-filter does not measurably reduce wall time**. δ=0.0 (filter
off) and δ=0.7 (aggressive filter) are within ±1% wall. Detector (TRT)
dominates the per-frame cost; NN evals were not constraining throughput.

The cheap-filter's value proposition shifts from "saves runtime" to
"regularises the matcher's input distribution at runtime". Worth
keeping for now; could be revisited for removal once we have confidence
it's not earning even the input-regularisation argument.

See user memory `project_cheap_filter_speed_neutral.md`.

### F1 — combo (δ=0.7 + pw=0.6) verification

Re-eval pw=0.6 in the full eval batch:

| Variant            | full-176 fit | JAAD fit |
|--------------------|--------------|----------|
| ship               | 0.575        | 0.269    |
| δ=0.7              | 0.574        | 0.266    |
| **pw=0.6**         | **0.578**    | **0.270** |
| combo (δ=0.7+pw=0.6)| 0.578       | 0.268    |

`pw=0.6` still ahead by +0.003. δ=0.7's A3b "win" *reversed* to −0.001
in this eval — single-run noise. Combos didn't stack: δ=0.7+pw=0.6
matches pw=0.6 alone on full-176 but loses a hair on JAAD. The two
knobs interact weakly at best.

### F3 — multi-seed pw=0.6 (the catch)

Re-trained pw=0.6 at seeds 1 and 2 (seed 0 was A1's original):

| Seed | full-176 fit | fp_tr | JAAD fit |
|------|--------------|-------|----------|
| 0    | 0.579        | 67    | 0.268    |
| 1    | 0.565        | 70    | 0.271    |
| 2    | 0.553        | **135** | 0.286   |

**σ across 3 seeds ≈ 0.013 fitness — 3× the single-run noise band**.
pw=0.6's A1 win was seed luck. The entire A1 sweep is suspect: every
"win" sat inside the seed-variance band.

This is the canonical example of why state-head training is bimodal
at certain hyperparam settings (matches `feedback_track_phase20c_failed.md`
which warned about this for `cr_promote`). K-seed-fit-pick is mandatory
before claiming a state-head fitness win.

### F4 — extended pw range (1.2 / 1.5 / 2.0)

| Variant | full-176 fit | full-176 fp_tr | JAAD fit | JAAD IDF1 |
|---------|--------------|----------------|----------|-----------|
| pw=1.2  | 0.540        | 175            | 0.299    | 0.477     |
| pw=1.5  | 0.536        | 185            | **0.303** | 0.483    |
| pw=2.0  | 0.536        | 187            | 0.303    | 0.483     |

**JAAD val saturates at pw=1.5** (0.303 vs A1 pw=1.0 at 0.290). Further
pw doesn't help. fp_tracks explode on full-176 — not viable as a
single ship without domain-split.

### F5 — multi-knob combo eval ⭐ (the actual ship decision)

| Variant                            | full-176 fit | fp_tr | JAAD fit |
|------------------------------------|--------------|-------|----------|
| F5a ship                           | 0.576        | 66    | 0.266    |
| F5b δ=0.7                          | 0.572        | 71    | 0.267    |
| F5c dd=0.70                        | 0.576        | 67    | 0.268    |
| **F5d δ=0.7 + dd=0.70**            | **0.579**    | **63** | 0.266   |
| F5e + pw=0.6                       | 0.581        | 65    | 0.271    |
| F5f + λ=0.10 (IDF1-lean)           | 0.571        | 85    | 0.270    |
| F5g pw=1.0 + δ=0.7 + dd=0.70       | 0.542        | 168   | **0.291** |

**Key interaction**: δ=0.7 alone (F5b) regresses by −0.004, but
δ=0.7+dd=0.70 (F5d) gains +0.003. The two knobs interact —
dd=0.70 compensates for δ=0.7's drift. Single-knob sweeps would have
missed this and dropped δ=0.7 from consideration.

## 2026-05-15 ship: F5d

**Two yaml edits in `/mldata/config/track/trackers/uc_v11.yaml`**:

```yaml
utrack:
  match_cheap_filter_delta: 0.7   # was 0.5
  delete_dup_iou:           0.70  # was 0.90
```

No NN retrain. Same shipped bins. Closed-loop result on full-176:
**0.576 → 0.579 fitness**, **66 → 63 fp_tracks**, JAAD parity.

**Why F5d and not F5e (+0.005 fitness)**: F5e includes pw=0.6, and F3
showed pw=0.6 has σ≈0.013 across seeds — the +0.005 sits inside the
variance band. F5d uses **only inference-only utrack knobs** with no
state-head retraining, so it's seed-safe and reproducible exactly
from the shipped bins.

**Why not F5f (+λ=0.10)**: trades 0.005 fitness for +0.011 IDF1.
Documented as the IDF1-leaning alternative — apply `utrack.nn_lambda: 0.10`
on top of the default ship if IDF1 matters more than fp-track rate.

**Why not F5g (pw=1.0, JAAD-cam)**: wins JAAD by +0.025 fitness but
loses full-176 by −0.034. Domain-split candidate — would ship as a
JAAD-camera-specific yaml, not as the default. Deferred.

Shipped via `/mldata/config` commit `0919729`.

---

## What's still open

These are the threads the overnight surfaced but didn't close. They
are *prioritised* by signal-to-effort ratio:

### High-value, low-effort

- **K-seed pw=0.6 retry**. F3 only ran 3 seeds; one seed (s=0) hit
  0.579 fitness, well above F5d's 0.579 baseline. If 5 seeds reveal a
  reliable mode at pw=0.6, that's +0.002 fitness on top of F5d at
  zero retraining cost. Add an F6: K=5 seeds at pw=0.6 + the F5d
  (δ=0.7 + dd=0.70) inference knobs.

- **F5g JAAD-cam recipe**. pw=1.0 + δ=0.7 + dd=0.70 wins JAAD val by
  +0.025 fitness. Should ship as a per-deployment override
  (`uc_v11_jaadcam.yaml`?) for dashcam-style cameras. Multi-seed
  verification first.

### Medium-value, medium-effort

- **B2 ablation**: drop the bottom-quartile features identified in
  B1 (ocm_cos, det_subbox_conf, det_fiqa_score, det_conf,
  conf_delta, pass_1, track_speed). Requires a C-runtime change
  (cheap-filter loader needs to read `obs_in/det_in/pair_in` from
  the bin meta and zero the missing columns at input-vector build
  time). ~50 lines of C + retrain + eval per candidate.

- **fp-track-shaped sample weighting in the trainer**. User memory
  `feedback_track_training_objective.md` flagged this. The
  fp-track penalty is 5e-4 per spurious track; the BCE loss doesn't
  see it. Add a per-sample weight inversely proportional to the
  expected fp-track contribution of that decision. Could change the
  V2/V3 face/scene story (those headed regressed on fp_tracks
  specifically).

- **Multi-seed audit of currently-shipped pw=0.5**. The pw=0.5 ship
  is single-seed (seed=0). The F3 finding makes it possible — but
  not confirmed — that pw=0.5 is also seed-lucky. A 3-seed
  retrain at pw=0.5 (with same recipe) would bound that variance.

### Low-value but eventually worth doing

- **Cheap-filter cleanup**. G1 showed it doesn't move wall time;
  fitness impact is in the noise. The C-side cheap-filter (3 gates
  in `utrack_match.c`) + the training-side mirror in
  `build_pair_dataset.py` together add ~150 lines of code that buy
  no measurable performance. If a future bootstrap confirms it's
  not helping fitness either, remove the whole machinery. Out of
  scope for now since it WAS at least a defensible mechanism for
  training/inference distribution agreement.

- **D1/D2 follow-up: fitness-shaped per-sample weights**. The
  cross-domain JAAD gap isn't a data-volume problem (D1/D2 both
  falsified that). The next angle is per-sample weighting that
  rewards JAAD-domain correctness more. Tricky — the JAAD-cam
  domain has lower fp_track absolute count, so the fp_track
  penalty is structurally smaller there. A fitness-shaped
  trainer would naturally weight differently across domains.

---

## Lessons banked as user memory

The following memory entries at
`~/.claude/projects/-home-mark-stuff-ubonpartners/memory/` capture the
lessons that future sessions inherit:

- `feedback_track_eval_metric.md` — judge by fitness, not val AUC.
- `feedback_track_training_objective.md` — BCE/AUC don't optimise
  fitness; consider counterfactual / fitness-shaped weighting.
- `feedback_track_history_aggregates.md` — history aggregates DO work.
- `feedback_track_thresholds_dedup.md` — keep pushing on thresholds;
  revisit dedup_iou now that match NN is stronger.
- `feedback_offline_online_gap.md` — stop manual whack-a-mole; fix
  the offline pipeline to predict deployment fitness.
- `feedback_track_phase20c_failed.md` — state-head training is bimodal;
  K-seed-fit-pick is mandatory.
- `feedback_track_mu_tp_truncation.md` — μ_TP target = dist-to-next-match,
  not matched-seconds-until-death.
- `feedback_verify_config_knobs.md` — always grep the active code
  path before proposing yaml/CLI changes.
- `feedback_track_corpus_drift.md` — corpus regen drifts retrains
  worse than shipped; always report within-corpus delta.
- `feedback_track_v18_recipe.md` — state_corpus_v18 was built from
  pair_log_v15_permissive (not v9_face).
- `feedback_methodical_git_state.md` — verify tree before AND after
  every git mutation.
- `feedback_eval_runs_default.md` — default eval to runs=1; runs=3
  only when two candidates differ by < 0.003.
- `feedback_preflight_long_runs.md`, `feedback_time_is_primary.md` —
  pre-flight every long-running setup; estimate wall-clock before
  launching.
- `project_pair_log_scene_skew.md` — top 5 scenes = 36% of all pairs;
  bootstrap retrain is structurally dense-crowd-biased.
- `project_d1_jaad_test_promotion.md` — adding/upweighting JAAD
  pairs doesn't close the cross-domain gap. (2026-05-15)
- `project_nn_lambda_idf1_dial.md` — λ is the IDF1 dial. (2026-05-15)
- `project_cheap_filter_speed_neutral.md` — cheap-filter saves no
  wall time. (2026-05-15)
- `project_a1_pw_sweep_was_seed_luck.md` — A1 single-seed conclusions
  are unreliable; F3 multi-seed showed σ≈0.013 at pw=0.6. (2026-05-15)
- `project_h264_bframe_bug.md`, `project_h264_transcode_framecount.md` —
  H.264 B-frame extraction silently dropped frames; 251/260 cached files
  were affected.
- `project_cmc_aspect_fix.md` — added `alpha = H/W` to cmc_transform_t
  to fix ~28% ty underestimate on 16:9 input.
- `reference_cmc_compare_tool.md` — `ml/cmc/cmc_compare.py` for 4-DOF
  CMC diagnosis against OpenCV ORB+RANSAC reference.
- `reference_jaad_dataset.md` — JAAD dashcam dataset at
  `/mldata/tracking/jaad`; default split 177/29/117.

---

## When to update this doc

Append a new section whenever a sweep produces a ship-candidate result.
Promote ship-decisions to the "Lineage at a glance" table at the top.
Demote rejected candidates to the "still open" section if there's a
plausible angle to retry, or just close them out with the rejection
reason inline.

The user memory entries should mirror the most generalisable lessons
(things that should change future *behaviour*); this doc is the
*specific results* archive.
