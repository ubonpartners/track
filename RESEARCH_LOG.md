# RESEARCH_LOG.md — Idea Bank + Experiment Log

Living companion to [`RESEARCH.md`](RESEARCH.md). **This file changes every
iteration.** RESEARCH.md is the fixed methodology; this file is the state:
decision thresholds, the ranked idea bank, and the append-only experiment
log. Nothing here is edited after an experiment's `decision.md` is written —
the log is append-only; only the bank section is curated.

---

## Decision thresholds (front-matter)

Authoritative copy of the §5 defaults. A change here is itself a logged
event in the Experiment Log (with reason), never a silent edit.

```yaml
sigma:            0.003   # single-eval fitness noise; replace with bootstrap-measured value
win_sigmas:       2.0     # win bar = +win_sigmas * sigma  (=> +0.006)
delta_min:        0.001   # loss ceiling
epsilon_family:   0.010   # max tolerated per-family fitness regression
mota_guard:      -0.004
idf1_guard:      -0.005
fp_tracks_guard: +3       # absolute integer
K_seed:           3       # seeds for any training step (escalate to 5 if straddling)
K_loss_stop:      8       # consecutive losses => halt + ask user
ev_floor:         0.0005  # fitness per cheap-unit; below this, no open idea => consider stop
cost_units:       {cheap: 1, medium: 8, heavy: 30}
baseline_ref:     null    # set at bootstrap: {fitness, mota, idf1, fp_tracks, provenance}
```

> Bootstrap has not yet run. `sigma` is the prior 0.003 until measured;
> `baseline_ref` is null until the bootstrap eval is recorded.

---

## Idea Bank

Ranked working set. Schema = RESEARCH.md §6. `EV = prior_p_win *
max(0, prior_effect.mean) / cost_units[cost_class]`. Re-rank after every
experiment; update `correlated_with` posteriors per §7.2; `killed` ideas
stay listed (with the kill citation) so a mechanism can't be silently
resurrected.

Priors are calibrated, not vibes: `uc_v11.yaml` is auto-tuned, so an
isolated single-knob nudge near the optimum has **low** `prior_p_win` and
small `prior_effect`; ideas backed by a *documented train/infer mismatch* or
a *structural* mechanism get more prior mass with wider `sd`.

```yaml
- slug: cheap-filter-delta-realign
  category: inference
  cost_class: cheap
  touches: [uc_v11.yaml: utrack.match_cheap_filter_delta]
  mechanism: >
    The cheap-filter delta gates which candidate pairs the match-NN scores
    at all. Ship is 0.7 (F5d). The shipped match-NN (nn_match_v13) was
    trained from a pair-log built with --delta-filter 0.5. At inference
    delta=0.7 feeds the NN a wider pair distribution than it was trained
    on — PIPELINE.md common-pitfall #1 (train/infer distribution mismatch).
    Sweeping delta toward 0.5 should realign inference with training; the
    *correct* long-term fix (if direction confirms) is match-nn-retrain at
    the shipped delta, not living with the mismatch.
  prior_p_win: 0.45
  prior_effect: {mean: 0.004, sd: 0.007}
  prediction: >
    Sweep {0.5,0.6,0.7(control)}. Expect monotone-ish; best at 0.5 or 0.6
    with Δfit in [-0.002,+0.012]; MOT/PP22 move most, JAAD ~flat; fp_tracks
    down 2-6 at tighter gate. If 0.7 is already best, the mismatch story is
    falsified and match-nn-retrain-at-ship-delta is demoted.
  correlated_with: [match-nn-retrain-at-ship-delta, dedup-iou-sweep]
  status: open

- slug: new-track-thr-sweep
  category: inference
  cost_class: cheap
  touches: [uc_v11.yaml: utrack.new_track_thr]
  mechanism: >
    new_track_thr (ship 0.77) is the confidence gate to spawn a new track.
    It is the most direct fp_tracks lever: raising it suppresses spurious
    new tracks (fitness has a -0.0005*fp_tracks term) but risks late
    initialisation -> misses (mota down). Auto-tuning optimised the full
    vector; a 1-D re-probe around the current point tests whether the
    fitness FP term wants it tighter than the tuner left it.
  prior_p_win: 0.30
  prior_effect: {mean: 0.002, sd: 0.005}
  prediction: >
    Sweep {0.74,0.77(control),0.80,0.83}. fp_tracks strictly decreasing in
    the threshold; mota turning over somewhere; fitness a shallow hump.
    Win only if a point clears +2σ with mota_guard intact.
  correlated_with: [track-initial-thr-sweep, dedup-iou-sweep]
  status: open

- slug: dedup-iou-sweep
  category: inference
  cost_class: cheap
  touches: [uc_v11.yaml: utrack.delete_dup_iou]
  mechanism: >
    delete_dup_iou (ship 0.70, lowered from 0.9 in F5d) removes
    near-duplicate tracks above this IoU. F5d lowering it was part of a
    confirmed ship gain, which suggests the fitness surface is still
    sensitive here. Probe both sides of 0.70 to check F5d didn't stop short
    of / overshoot the optimum.
  prior_p_win: 0.35
  prior_effect: {mean: 0.003, sd: 0.006}
  prediction: >
    Sweep {0.55,0.65,0.70(control),0.80}. Lower => fewer duplicate tracks
    => fp_tracks down but risk of merging distinct identities (idf1 down,
    switches up). Win needs idf1_guard intact.
  correlated_with: [cheap-filter-delta-realign, new-track-thr-sweep]
  status: open

- slug: nn-lambda-sweep
  category: inference
  cost_class: cheap
  touches: [uc_v11.yaml: utrack.nn_lambda]
  mechanism: >
    nn_lambda (ship 0.05) blends the learned match-NN cost with the
    geometric/heuristic cost. It is the single knob that says "how much do
    we trust the NN vs geometry". If the NN generalised well, more weight
    helps; if it overfits the corpus, more weight hurts JAAD (cross-domain).
    A sweep is a cheap, high-information probe of NN trust.
  prior_p_win: 0.30
  prior_effect: {mean: 0.002, sd: 0.007}
  prediction: >
    Sweep {0.0,0.05(control),0.10,0.20}. 0.0 ~= no-NN (sanity anchor).
    Expect a hump; JAAD vs full176 may peak at different lambda (overfit
    signature). A JAAD/full176 split in the optimum is itself the finding.
  correlated_with: [match-nn-retrain-at-ship-delta]
  status: open

- slug: track-buffer-seconds-sweep
  category: inference
  cost_class: cheap
  touches: [uc_v11.yaml: utrack.track_buffer_seconds]
  mechanism: >
    track_buffer_seconds (ship 2.2) is how long a lost track is kept alive
    for re-association before deletion. Longer buffer recovers occlusions
    (fewer switches/misses, idf1 up) but keeps stale tracks around longer
    (fp_tracks / fp_per_frame up). Directly trades the two halves of the
    fitness formula; worth a deliberate sweep rather than trusting the
    tuner's single point.
  prior_p_win: 0.30
  prior_effect: {mean: 0.002, sd: 0.005}
  prediction: >
    Sweep {1.5,2.2(control),3.0,4.0}. idf1 increasing then flat; fp_tracks
    increasing throughout; fitness a hump. Family split likely (JAAD
    dashcam has fast egomotion -> shorter buffer better there).
  correlated_with: [new-track-thr-sweep]
  status: open

- slug: state-head-poswfit-kseed
  category: state_nn
  cost_class: medium
  touches: [run_pipeline.sh: --pw ; train_state_head_decoupled]
  mechanism: >
    pos_weight for the state head is shipped at 0.5. The "pw=0.6 wins"
    result (A1) was FALSIFIED by F3 multi-seed (seed σ≈0.013) — see
    feedback_track_phase20c_failed.md / project_a1_pw_sweep_was_seed_luck.
    This idea is NOT "try 0.6 again"; it is: re-decide pw by a *proper*
    K-seed-median sweep so the choice is evidence-based rather than
    seed-luck-based. Seed-sensitive => K=3 medians, never single seed.
  prior_p_win: 0.25
  prior_effect: {mean: 0.002, sd: 0.010}
  prediction: >
    Sweep pw {0.4,0.5(control),0.6,0.8}, K=3 seeds each, compare medians
    within the same corpus build. Expect the per-seed spread (~0.013) to
    swamp most pw differences -> likely inconclusive, which would itself
    confirm pw is not a real lever and free the bank of pw ideas.
  correlated_with: [state-corpus-fp-boost]
  status: open

- slug: state-corpus-fp-boost
  category: state_nn
  cost_class: medium
  touches: [run_pipeline.sh: --fp-boost ; build_state_corpus.fitness_fp_boost]
  mechanism: >
    --fitness-fp-boost (default 1.0) up-weights training examples on tracks
    that never match any GT (the fp_tracks term in fitness). Raising it
    makes the state head's loss surface match the eval's fitness weighting
    more closely (FP-track avoidance). Mechanistically aligned with the
    headline metric; medium cost, seed-sensitive (training step).
  prior_p_win: 0.35
  prior_effect: {mean: 0.004, sd: 0.010}
  prediction: >
    Sweep boost {1.0(control),1.5,2.0}, K=3 seeds. fp_tracks should fall
    monotonically with boost; mota may erode (over-conservative promotion).
    Win = a boost where fp_tracks drop buys Δfit≥+2σ with mota_guard intact.
  correlated_with: [state-head-poswfit-kseed]
  status: open

- slug: match-nn-retrain-at-ship-delta
  category: match_nn
  cost_class: heavy
  touches: [pipeline: build_pair_dataset --delta-filter ; bootstrap iter]
  mechanism: >
    The principled fix for the train/infer δ mismatch (see
    cheap-filter-delta-realign). Rebuild the pair dataset with
    --delta-filter set to the *shipped* inference delta and retrain the
    match-NN, so train and inference see the same pair distribution
    (PIPELINE.md pitfall #1, stated remedy). Heavy + seed-sensitive +
    corpus-drift exposed => strict within-corpus, K-seed.
  prior_p_win: 0.40
  prior_effect: {mean: 0.006, sd: 0.012}
  prediction: >
    Retrain at δ matched to whatever cheap-filter-delta-realign identifies
    as best inference δ. Expect Δfit≥+2σ vs the same-corpus control NN if
    the mismatch story (from the cheap probe) held. If cheap-filter-delta-
    realign falsified the mismatch, this idea is auto-demoted to killed.
  correlated_with: [cheap-filter-delta-realign, nn-lambda-sweep]
  status: open
  blocked_by: cheap-filter-delta-realign   # run the cheap probe first; it gates this

- slug: dagger-iter4
  category: corpus
  cost_class: heavy
  touches: [bootstrap_recipe.sh: a 4th DAgger pass]
  mechanism: >
    bootstrap_recipe.sh runs 3 DAgger iterations and explicitly flags iter3
    as an open question ("does another pass help or are we at the head's
    representational ceiling?"). A 4th pass tests for residual DAgger gain.
    Heavy; corpus-drift exposed => must compare iter4 vs iter3 built from
    the same iter3-NN pair-log, within-corpus, K-seed.
  prior_p_win: 0.20
  prior_effect: {mean: 0.003, sd: 0.012}
  prediction: >
    iter3->iter4 within-corpus Δfit. If |Δfit| < 2σ we have evidence the
    head is at its representational ceiling and DAgger depth is exhausted —
    a valuable negative result that kills further-iteration ideas.
  correlated_with: [match-nn-retrain-at-ship-delta]
  status: open
```

### Current ranking (recompute each iteration)

| Rank | slug | EV | cost | P(win) | prior Δfit | note |
|---|---|---|---|---|---|---|
| 1 | cheap-filter-delta-realign | 0.0020 | cheap | 0.45 | +0.004 | high info; gates the heavy NN retrain |
| 2 | dedup-iou-sweep | 0.0011 | cheap | 0.35 | +0.003 | F5d showed surface still live here |
| 3 | new-track-thr-sweep | 0.0006 | cheap | 0.30 | +0.002 | direct fp_tracks lever |
| 4 | nn-lambda-sweep | 0.0006 | cheap | 0.30 | +0.002 | probes NN trust / overfit |
| 5 | track-buffer-seconds-sweep | 0.0006 | cheap | 0.30 | +0.002 | idf1↔fp_tracks trade |
| 6 | state-corpus-fp-boost | 0.00018 | medium | 0.35 | +0.004 | metric-aligned; seed-sensitive |
| 7 | match-nn-retrain-at-ship-delta | 0.00008 | heavy | 0.40 | +0.006 | blocked by #1 |
| 8 | state-head-poswfit-kseed | 0.00006 | medium | 0.25 | +0.002 | likely confirms "not a lever" |
| 9 | dagger-iter4 | 0.00002 | heavy | 0.20 | +0.003 | ceiling probe |

First pick = **cheap-filter-delta-realign**: top EV, and win-or-lose it
sharply updates `match-nn-retrain-at-ship-delta` and `nn-lambda-sweep`
(highest information value). It is also the gate on the most expensive idea
in the bank, so resolving it cheaply first is correct sequencing.

---

## Experiment Log (append-only)

One entry per experiment, newest at the bottom. Never edited after written.
Each entry must contain the explicit Bayesian update
(`μ0,τ0 → Δfit,σ_obs → μ1,τ1 → P(win)`), the per-family / fp_tracks check,
the decision, and the bank curation that followed.

Entry schema (copy for each new experiment):

```
### <YYYYMMDD>-<slug>   [win|loss|inconclusive|errored]
- selected: EV=<v> (runner-up <slug> EV=<v>); reason=<why this pick>
- prior: p_win=<>, effect~N(μ0=<>, τ0=<>)
- change: <one line + path to change.patch>
- eval: <path to results-<ts>.json>; corpus=eval_ship_baseline (frozen)
- evidence:
    candidate fitness=<>  control(same-batch) fitness=<>  Δfit=<>
    mota Δ=<>  idf1 Δ=<>  fp_tracks Δ=<>
    families: full176 Δ=<>  jaad_val Δ=<>
    (seed-sensitive? per-seed: [..]; median used)
- update: σ_obs=<>; μ1=<> τ1=<>; posterior P(win)=<>
- prediction check: <held / partially / FALSIFIED — what it implies>
- decision: <win|loss|inconclusive|errored> + (confirmed? how)
- baseline: <unchanged | promoted: new control provenance=<path>>
- bank curation: <correlated updates w/ rationale; ideas added/demoted/killed>
- artifacts: RESEARCH_OUT/<YYYYMMDD>-<slug>/
```

> Worked example of the format (NOT a real result — illustrative only,
> `status: example`, never counted in the progress curve):
>
> ```
> ### 20260515-cheap-filter-delta-realign   [example]
> - selected: EV=0.0020 (runner-up dedup-iou-sweep EV=0.0011);
>   reason=top EV + gates the heavy match-NN retrain
> - prior: p_win=0.45, effect~N(μ0=+0.004, τ0=0.007)
> - change: uc_v11.yaml utrack.match_cheap_filter_delta swept {0.5,0.6,0.7};
>   one eval, 4 tests (3 candidates + 0.7 control). change.patch attached
> - eval: RESEARCH_OUT/20260515-.../eval/results-<ts>.json (frozen corpus)
> - evidence: [fill from the JSON sidecar's tests.*.overall + .groups]
> - update: σ_obs=0.003; [Normal update]; posterior P(win)=...
> - prediction check: [held / FALSIFIED -> demote match-nn-retrain-at-ship-delta]
> - decision: [win => confirm re-run before promotion | loss | inconclusive]
> - baseline: [unchanged | promoted + re-pinned]
> - bank curation: [update correlated nn-lambda / match-nn-retrain priors]
> - artifacts: RESEARCH_OUT/20260515-cheap-filter-delta-realign/
> ```

--- (no real experiments yet — bootstrap §11 then first pick above) ---

## Progress curve

Chain of **confirmed within-batch Δfit** through promotions (immune to
corpus drift — see RESEARCH.md §4.4). Empty until the first confirmed win.

```
baseline (bootstrap) ──> [no confirmed wins yet]
```
