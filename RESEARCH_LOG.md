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
sigma:             0.003  # single-eval fitness noise; replace w/ bootstrap-measured
sigma_idf1:        0.003   # single-eval IDF1 noise; replace w/ bootstrap-measured
win_sigmas:        2.0     # win bar = +win_sigmas * sigma  (=> +0.006)
delta_min:         0.001   # loss ceiling
epsilon_family:    0.010   # max tolerated per-family fitness regression
mota_guard:       -0.004
idf1_guard:       -0.005
fp_tracks_guard:  +3       # absolute integer
speed_improve_min: 0.05    # fractional speedup bar (pinned conditions) for a speed win
K_seed:            3       # seeds for any training step (escalate to 5 if straddling)
K_loss_stop:       8       # consecutive losses => halt + ask user
ev_floor:          0.0005  # fitness per cheap-unit; below => consider stop
cost_units:        {cheap: 1, medium: 8, heavy: 30}
objective_order:   [fitness, idf1, speed, simplicity]  # lexicographic; fitness never bought
artifact_root:     RESEARCH_OUT          # one folder per experiment under here; zero scatter
baseline_ref:      null   # set at bootstrap: {fitness, mota, idf1, fp_tracks, speed, provenance}
promotion_gate:    [candidate_confirmed, full_pipeline_clean, overall_better_metrics]
honest_ruler:                            # FROZEN exp#6 20260515-honest-fp-frame-metric
  metric:          fp_frames_honest      # src.track_test._honest_fp_frames_core
  theta:           0.5                    # box-diagonals; joint-OK band θ∈[0.3,0.6]
  nm_policy:       track                  # never-matched mirrors gamed (=1 each)
  status:          frozen                 # live==offline gate PASS (smoke 3/3, 0 mismatch)
  fitness_uses_it: false                  # side-channel until clean full-pipeline re-pin (§3/§4.5)
  change_is:       logged_experiment_event # editing θ/nm re-opens the ruler — never silent
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
  primary_axis: idf1
  mechanism: >
    nn_lambda (ship 0.05) blends the learned match-NN cost with the
    geometric/heuristic cost. Documented finding
    `project_nn_lambda_idf1_dial.md` (2026-05-15): λ is effectively the
    IDF1 dial — it trades identity consistency against fitness rather than
    moving fitness much. So this is targeted at the **IDF1 axis** (§2.1):
    find a λ that lifts IDF1 at statistically-similar fitness.
  prior_p_win: 0.35
  prior_effect: {mean: 0.001, sd: 0.005}   # fitness ~flat by the documented finding
  prediction: >
    Sweep {0.0,0.05(control),0.10,0.20} in ONE eval. 0.0 ~= no-NN anchor.
    Per the IDF1-dial finding: fitness ~flat (|Δfit|<2σ), IDF1 monotone-ish
    in λ. Win if some λ gives Δidf1 ≥ +2σ_idf1 at similar fitness with
    guards intact. JAAD vs full176 optimum split is an overfit signature.
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

- slug: dagger-multiiter-regression
  category: corpus
  cost_class: medium       # diagnostic: reuse the already-produced iter1/2/3 artefacts
  source: user
  primary_axis: fitness    # understanding -> recover the lost ~0.02 fitness
  touches: [analysis only: bootstrap iter1/2/3 pair-logs + corpora + evals]
  mechanism: >
    "dagger-iter4" (go deeper) is KILLED — the opposite is true. Direct
    evidence (2026-05-15 optimized run, frozen corpus, within-batch):
    full-config fitness iter1 0.5610 -> iter2 0.5395 (-0.0215, >>2sigma),
    MOTA 0.6148 -> 0.5654; earlier runs showed the same direction. DAgger
    iters 2/3 *regress hard and repeatably*. Hypothesised mechanism: the
    pair-log relabel under the iter-1 NN collapses/【biases】 the candidate
    distribution (the NN only surfaces pairs it already likes -> the next
    corpus is self-confirming, not corrective; classic DAgger-without-
    expert-correction failure). This idea is to *diagnose* it, not to run
    more iters. Default is now ONE iteration (bootstrap_recipe.sh
    STOP_AFTER=1); multi-iter only for this investigation.
  prior_p_win: 0.60        # P(produces an actionable root-cause finding)
  prior_effect: {mean: 0.000, sd: 0.004}   # diagnostic; payoff is via downstream fixes
  prediction: >
    Compare iter1 vs iter2 pair-logs from the same run: expect iter2 to
    show (a) lower pair diversity / entropy, (b) collapsed score
    distribution toward the iter1-NN's preferred region, (c) fewer
    hard-negative pairs. If confirmed, the fix is corpus-side (keep iter0
    non-NN candidate generation; never relabel candidacy with the learned
    NN — only relabel *targets*), feeding a concrete corpus idea. If the
    diversity story is falsified, escalate to a class-balance / label-noise
    hypothesis. Either way it produces a child idea, not a dead end.
  correlated_with: [scene-density-pair-rebalance, match-nn-retrain-at-ship-delta]
  status: open

# ---- user-seeded ideas (2026-05-15) ---------------------------------------

- slug: honest-fp-track-metric
  category: measurement        # SPECIAL: audits/repairs the headline ruler itself
  cost_class: cheap            # DIAGNOSIS is cheap (post-process existing ubtrk2+JSON);
                               # the metric *fix* is medium + campaign-resetting
  source: user
  primary_axis: measurement    # protects every other axis; not a tracker win itself
  touches: [analysis: track_test fp_tracks accounting vs GT spans ; ubtrk2 runs]
  mechanism: >
    fitness penalises *unique* FP tracks (-0.0005*fp_tracks). A unique FP
    is a whole output track that never matches any GT. The optimizer can
    cut that count WITHOUT removing false positives — by ID-merging an
    (in reality unrelated) FP segment onto a true GT track, or stitching
    two FP tracks into one. The FP frames still exist on screen; they just
    stop being counted as a *separate* track. That is Goodhart gaming, not
    real improvement. SMOKING GUN in our own data: 2026-05-15 optimized
    bootstrap iter2 drove fp_tracks 103 -> 48 while MOTA collapsed
    0.6148 -> 0.5654 — the head learned the exploit, not the task.
    Some gamed merges are detectable post-hoc:
      (a) lead-in/lag-out: an output track scored as a GT identity but
          whose lifetime starts noticeably BEFORE that GT first appears
          (or ends well AFTER it leaves) — the extra span is a hidden FP
          absorbed by the ID match.
      (b) spatial excursion: a "matched" track whose box is far from ANY
          GT for a large fraction of its frames (covers background then
          briefly touches a GT).
      (c) teleport/merge: one output track whose trajectory has implausible
          jumps between two disjoint FP regions (two FPs stitched to read
          as <=1 unique FP), or MOT merge/fragment events linking an
          unmatched segment to a matched one.
    An "honest" FP accounting charges FP by track-segment / FP-frames that
    survive these checks, not only by whole-never-matched tracks.
  prior_p_win: 0.70            # P(diagnosis shows material, detectable gaming)
  prior_effect: {mean: 0.000, sd: 0.000}   # not a tracker win; recalibrates the ruler
  prediction: >
    DECISIVE CHEAP TEST (zero new compute — use existing iter1/2/3 eval
    JSON): if fp_tracks is being gamed, the unique-FP count falls but the
    actual FP *volume* does NOT — i.e. fp_tracks drops sharply while
    `num_false_positives` / `fp_per_frame` stay ~flat (the FP frames are
    just absorbed into fewer tracks). Real FP reduction drops BOTH
    together. So compare iter1->iter2 (fp_tracks 103->48, -53%): if
    num_false_positives / fp_per_frame fell by ≪53% (≈flat) that is the
    smoking gun; if it fell ~proportionally, the FP reduction is real and
    the hypothesis is FALSIFIED (fitness trustworthy — re-validates prior
    fp_tracks wins). The heuristics (a)/(b)/(c) are the follow-up
    attribution once the volume-vs-count decoupling is established.
  correlated_with: [dagger-multiiter-regression, adaptive-nn-prior, state-corpus-fp-boost]
  status: confirmed   # 20260515: gaming material (decoupling 0.33); see Experiment Log
  handling_note: >
    Special scope. RESEARCH.md §2/§3 freeze the fitness formula and eval
    for a campaign — so the DIAGNOSIS (read-only audit of how gameable the
    current metric is) is in-scope and top-priority *because* it tests the
    validity of the ruler every other idea is judged by. But a confirmed
    defect does NOT silently change the metric mid-campaign: per RESEARCH.md
    §3 it triggers a campaign reset — freeze the honest metric, re-pin the
    baseline by re-measuring under it, and re-judge open results. Until
    then it cannot "promote"; its win condition is producing the audit
    verdict, not moving fitness. This is the highest-leverage idea in the
    bank: if the headline is gameable, prior fp_tracks-driven wins
    (incl. F5d ship, DAgger conservatism) may be partly illusory.
  evidence_2026_05_15: >
    CONFIRMED by experiment 20260515-honest-fp-track-metric (read-only,
    iter1->iter2 of the optimized bootstrap, frozen corpus): unique
    fp_tracks -53.4% (103->48) but FP *volume* num_FP only -17.9%
    (98497->80909); decoupling ratio |ΔFPvol|/|Δfp_tracks| = 0.33;
    misses +19.1% (recall collapsed), MOTA 0.6148->0.5654. The gameable
    terms handed +0.0279 fitness credit that masked 56% of the -0.0494
    MOTA collapse. Gaming is material and the optimizer already found it.

- slug: honest-fp-track-metric-definition
  category: measurement        # constructive successor to the (confirmed) diagnosis
  cost_class: heavy            # metric def + TRAINING alignment + revalidation + re-pin
  source: user
  primary_axis: measurement
  touches: [track_test: per-frame GT<->hyp match stream / fp_tracks accounting ;
            build_state_corpus + fitness-shaped loss (training must follow the
            new definition) ; campaign baseline re-pin]
  depends_on: honest-fp-track-metric   # diagnosis CONFIRMED; this builds the fix
  mechanism: >
    Replace the gameable "1 unique FP iff the track NEVER matches any GT
    over its whole life" with a SEGMENT-based count. Decompose each output
    track into maximal contiguous matched / unmatched segments vs GT
    (the per-frame TP/FP/IDSW stream motmetrics already produces), then
    count a unique FP for each *material* unmatched segment:
      - lead-in : unmatched run BEFORE the track's first GT match,
                  length >= L_lead  -> +1 FP
      - lag-out : unmatched run AFTER the track's last GT match,
                  length >= L_lag   -> +1 FP
      - bridge  : unmatched run BETWEEN two matched sections that is
                  either longer than a legitimate occlusion gap
                  (> G_max) OR spatially excursive (box far from the
                  matched GT's extrapolated location, IoU/centroid gate)
                  -> +1 FP   (this is the "stitch two unrelated things"
                  case the optimizer exploited)
      - a fully-unmatched track stays exactly 1 FP (unchanged).
    Must degrade to ~the current metric in the clean / no-gaming limit
    (a well-behaved tracker scores ~same fp_tracks under both).
  training_cascade: >
    Per user: "we might have to adjust the training etc to account for
    this." The honest definition is NOT just an eval swap — every
    training signal that references fp_tracks must be re-derived under
    it or training optimises the OLD gameable target while we evaluate
    on the honest one (a train/infer objective mismatch, the documented
    hazard class). Concretely: build_state_corpus `fitness_fp_boost`
    example weighting, the fitness-shaped-trainer-loss term, and the
    DAgger relabel all currently reward whole-track-never-matched; they
    must instead reward removing the lead-in/lag-out/bridge FP *frames*.
    This couples honest-fp-track-metric-definition with
    state-corpus-fp-boost and fitness-shaped-trainer-loss — they should
    be designed together, not serially.
  prior_p_win: 0.45            # P(a validated, robust, not-trivially-regameable def exists)
  prior_effect: {mean: 0.000, sd: 0.000}   # measurement; payoff is a trustworthy ruler
  prediction: >
    Validation (NOT just "define and ship" — user flagged it needs
    validation):
      1. Re-scores the gaming contrast honestly: under the new def the
         iter1->iter2 unique-FP delta TRACKS the FP-volume delta
         (decoupling ratio -> ~1, from the gamed 0.33). If it still
         shows the iter2 collapse, the def FAILED to capture the exploit.
      2. Threshold robustness: sweep L_lead/L_lag/G_max + the spatial
         gate; the verdict (gamed vs clean) must be stable across a
         sensible range, not knife-edge.
      3. No false alarms: a clean reference config's honest fp_tracks
         ~= its old fp_tracks (legit short occlusion bridges NOT charged).
      4. New exploit surface reviewed: trimming lead-in/lag-out is now
         the optimizer's cheapest move — that is the DESIRED behaviour
         (it removes real FP frames), confirm no perverse residual.
    FALSIFIED if no threshold set satisfies (1)&(3) simultaneously
    (the segment criterion can't separate gaming from legit occlusion)
    -> fall back to an FP-frame / FP-track-seconds metric instead.
  handling_note: >
    Campaign-resetting + training-coupled. On a validated def: freeze it
    as the new honest ruler, re-derive the training targets under it,
    re-pin the baseline by re-measuring, and re-judge open fp_tracks
    results (RESEARCH.md §3/§4.5). Sequenced AFTER the (confirmed)
    diagnosis; co-designed with the training-objective ideas. It does
    not promote a tracker — it produces the ruler the rest of the
    campaign is then run against.
  correlated_with: [honest-fp-track-metric, state-corpus-fp-boost,
                    fitness-shaped-trainer-loss, dagger-multiiter-regression]
  status: validating   # 20260515: criterion-1 (decoupling) PASSED+reproduced
                        # (gamed 0.33 -> honest 0.88); crit-2/3 + spatial
                        # gate + training cascade outstanding. See Exp Log.

# ---- children spawned by honest-fp-track-metric-definition (crit-2/3) ------

- slug: honest-fp-cleanconfig-falsealarm
  category: measurement
  cost_class: cheap            # one 2-test eval, frozen corpus, no retrain
  source: agent
  primary_axis: measurement
  depends_on: honest-fp-track-metric-definition
  touches: [analysis: honest vs gamed fp_tracks on the CURRENT ship uc_v11]
  mechanism: >
    Criterion 3: the honest metric must NOT over-charge a well-behaved
    tracker (routine short post-track tails / legitimate occlusion
    bridges are not real FP). Run gamed vs honest on the current ship
    config (uc_v11, v13+v23_pw05) on the frozen corpus. If honest
    fp_tracks >> gamed for a clean tracker, the thresholds are too
    aggressive and the metric would punish good behaviour — it must be
    re-tuned (feeds honest-fp-threshold-sweep) before it can be frozen.
  prior_p_win: 0.55            # P(a sane threshold band gives honest≈gamed clean)
  prior_effect: {mean: 0.000, sd: 0.000}
  prediction: >
    Expect honest/gamed ratio on the clean ship to be MUCH closer to 1
    than the iter2-gamed case; if it is still huge (e.g. >5x) the v0
    thresholds (5/5/10 eval-frames) are too tight — quantify the gap and
    hand the target band to the threshold sweep. Falsified-as-blocking
    if no threshold band can make clean≈honest while still catching the
    iter2 exploit (then the temporal-only def is insufficient — escalate
    to the spatial gate / FP-frame metric).
  correlated_with: [honest-fp-threshold-sweep, honest-fp-spatial-gate]
  status: confirmed   # 20260515 crit-3 FAIL@v0: clean ship honest/gamed=31x; see Exp Log

- slug: honest-fp-threshold-sweep
  category: measurement
  cost_class: cheap            # re-score existing eval JSON at varied thresholds
  source: agent
  primary_axis: measurement
  depends_on: honest-fp-track-metric-definition
  touches: [_honest_fp_tracks l_lead/l_lag/g_max/theta ; re-score, no re-eval]
  decisive: true        # 20260515 exp#4: THE crux. crit-3 (clean≈gamed) and
                        # crit-1 (decoupling≥~0.8) are in TENSION along θ —
                        # θ=2.0 gave 0.8x clean but decoupling 0.40; temporal-
                        # only gave 0.88 decoupling but 31x clean. Joint
                        # search over (θ,l_lead,l_lag,g_max) for BOTH.
  mechanism: >
    Criterion 2 + the exp#4 tension: find (θ,l_lead,l_lag,g_max) giving
    clean-ship honest/gamed ∈ ~[0.7,2] AND iter1→iter2 honest decoupling
    ≥ ~0.8 *simultaneously*. Cheap if the per-frame event+centroid data
    is cached so thresholds re-score without re-eval. If no joint band
    exists the count formulation can't separate gaming from legit
    occlusion -> pivot to honest-fp-frame-metric.
  prior_p_win: 0.35    # exp#4 lowered: genuine chance no joint band exists
  prior_effect: {mean: 0.000, sd: 0.000}
  prediction: >
    Either a contiguous (θ,thresholds) band satisfies BOTH crit-3 and
    crit-1 (report it + operating point) OR none does (FALSIFIES the
    count formulation -> honest-fp-frame-metric).
  correlated_with: [honest-fp-cleanconfig-falsealarm, honest-fp-spatial-gate,
                    honest-fp-frame-metric]
  status: open   # DECISIVE next pick

- slug: honest-fp-spatial-gate
  category: measurement
  cost_class: medium           # needs box geometry vs matched-GT extrapolation
  source: agent
  primary_axis: measurement
  depends_on: honest-fp-track-metric-definition
  touches: [_honest_fp_tracks: add spatial-excursion criterion to bridge rule]
  mechanism: >
    Documented follow-up to the temporal v0: a bridge is most clearly a
    stitched-unrelated-FP when the box during the unmatched run is far
    from the matched GT's extrapolated location (IoU/centroid gate),
    not merely long. Adding the spatial gate should let the temporal
    thresholds relax (fewer false alarms on legit long occlusions that
    stay spatially consistent) while still catching teleport/merge.
  prior_p_win: 0.45
  prior_effect: {mean: 0.000, sd: 0.000}
  prediction: >
    With the spatial gate, the clean≈honest band widens (crit-3 easier)
    while iter2 decoupling stays ~0.88. Falsified if the gate doesn't
    separate legit-long-occlusion from teleport (then fall back to an
    FP-frame / FP-track-seconds metric).
  correlated_with: [honest-fp-cleanconfig-falsealarm, honest-fp-threshold-sweep]
  status: confirmed   # 20260515 exp#4: crit-3 FIXED (31x->0.8x) but crit-1
                      # REGRESSED (decoupling 0.88->0.40 @θ2.0). θ tension.
                      # Prediction only HALF held. See Experiment Log.

- slug: honest-fp-frame-metric
  category: measurement
  cost_class: medium
  source: agent
  primary_axis: measurement
  depends_on: honest-fp-threshold-sweep   # contingency if no count θ-band exists
  mechanism: >
    exp#4 showed segment-COUNT honest-FP has crit-3 vs crit-1 in tension
    along θ. Contingency: charge FP by *surviving spatially-excursive FP
    frames* (FP-track-seconds), not per-segment units. Merging an FP into
    a matched track keeps the FP frames so they still count (hard to
    game); a short benign coast contributes few frames (graceful on clean
    trackers). Closest to the user's original "count FP for detections
    not matching a GT before/after a matching section".
  prior_p_win: 0.45
  prior_effect: {mean: 0.000, sd: 0.000}
  prediction: >
    Clean ship ≈ small multiple of gamed FP-frames (sane) AND iter1→iter2
    honest-frame Δ tracks FP-volume Δ (decoupling →~1) — satisfies BOTH
    where the count formulation could not. Falsified if it also decouples
    or over-charges (then need GT-trajectory-aware matching).
  correlated_with: [honest-fp-threshold-sweep, honest-fp-track-metric-definition]
  status: confirmed   # 20260515 exp#6: FRAME formulation VALIDATED — joint
                      # gate PASSED, nm='track' θ∈[0.3,0.6], predicted
                      # crossover present, P 0.55→~0.90. Candidate frozen
                      # ruler θ=0.5/nm=track. See Experiment Log. Spawns
                      # child honest-fp-frame-metric-wire-live (GATED next).

- slug: honest-fp-frame-metric-wire-live
  category: measurement
  cost_class: medium           # 1 smoke eval + 1 clean full-pipeline cycle
  source: agent
  primary_axis: measurement
  depends_on: honest-fp-frame-metric
  mechanism: >
    exp#6 validated the FRAME honest-FP ruler offline (θ=0.5,
    nm_policy='track'). Make it real: wire `_honest_fp_frames_core` into
    live compute_metrics as side-channel `fp_frames_honest`(+breakdown);
    re-run offline==live correctness gate on a smoke set (guaranteed by
    construction but verify §8.1); FREEZE θ/nm/code-hash in front-matter;
    one clean full-pipeline cycle reporting the frozen ruler beside
    fitness (re-pin baseline); then adjust training to optimise the
    honest ruler instead of the gamed fp_tracks term.
  prior_p_win: 0.85           # mostly engineering; risk = a live≠offline
                              # surprise or pipeline integration breakage
  prior_effect: {mean: 0.000, sd: 0.000}
  prediction: >
    Live fp_frames_honest == offline _honest_fp_frames_core on every
    smoke clip (0 mismatch). Clean full cycle completes; frozen ruler
    reproduces exp#6 clean_frac≈0.06–0.16 on the fresh ship eval.
    Falsified if live≠offline, or the fresh clean ratio departs the
    exp#6 band (→ corpus-sensitivity; re-open formulation).
  correlated_with: [honest-fp-frame-metric, honest-fp-threshold-sweep]
  status: open

- slug: poseflow-box-warp
  category: inference          # ubon_cstuff motion-warp change; no NN retrain
  cost_class: medium           # C rebuild + 1 eval; NO training loop added
  source: user
  primary_axis: fitness
  touches: [ubon_cstuff: optical-flow box warp (motiontrack/box_prediction)]
  mechanism: >
    The OF box-warp currently displaces a box by the mean motion of 5
    fixed grid points inside it. Fixed grid points sit on background /
    occluder pixels in crowds, biasing the warp. Pose keypoints track the
    actual articulated person; the upper body (head/shoulders/torso) is
    far less often occluded than legs in dense scenes. Use a top-weighted
    average of available pose keypoints for the warp when pose exists;
    fall back to the existing 5-point scheme only when pose is absent.
    `pose_kp_visible` is already a tracked feature, so the signal exists.
  prior_p_win: 0.35
  prior_effect: {mean: 0.004, sd: 0.008}
  prediction: >
    Gains concentrate in crowded/occlusion families (MOT20, dense PP22):
    fewer ID switches, idf1 up, fp_tracks ~flat-to-down; ~flat where pose
    is sparse (some JAAD). Risk: pose jitter on low-res far targets adds
    warp noise -> guard mota/idf1. Falsified if no crowded-family lift.
  correlated_with: [adaptive-nn-prior, ocm-why-no-gain]
  status: open

- slug: adaptive-nn-prior
  category: inference          # runtime prior schedule; NO extra training loop
  cost_class: medium
  source: user
  primary_axis: fitness
  touches: [ubon_cstuff: nn_state/utrack prior injection ; maybe uc_v11 knob]
  mechanism: >
    The NN scheme was designed to be "learning over time" by updating an
    injected prior belief; in practice a *fixed* prior is injected, so the
    Bayesian-adaptivity the design intended is unused. Make the prior
    adapt at *inference* (no retrain, no DAgger loop — hard constraint):
    (a) strengthen a track's prior with clean accumulated history
    (age/observation-count), (b) set the *starting* prior from scene
    density (dense scene => higher FP-track base rate => more conservative
    start). Scene density is already computed (SceneStats / scene_*).
    Initial track generation stays non-NN (constraint honoured).
  prior_p_win: 0.40
  prior_effect: {mean: 0.005, sd: 0.010}
  prediction: >
    fp_tracks down in dense families (MOT20/PP22) without hurting sparse;
    fitness up via the -0.0005*fp_tracks term. Risk: over-conservative
    start suppresses slow-to-confirm real tracks -> mota/idf1 guard.
    Falsified if the density-conditioned start shows no fp_tracks/density
    interaction.
  correlated_with: [state-corpus-fp-boost, dagger-multiiter-regression, poseflow-box-warp]
  status: open

- slug: literature-feature-scan
  category: match_nn
  cost_class: cheap            # web research + writing child bank entries; no eval
  source: user
  primary_axis: fitness        # generator: spawns concrete feature ideas
  touches: [RESEARCH_LOG bank (generates children) ]
  mechanism: >
    The match-NN input set (16 OBS + 5 DET + 19 PAIR) is fixed. The
    tracking-by-detection literature has association cues that may be
    absent or weak here: camera-motion-compensated IoU / GMC (BoT-SORT),
    observation-centric momentum done right (OC-SORT), low-score-detection
    recovery (ByteTrack), appearance-gallery / long-term ReID memory,
    velocity-direction consistency, size/scale priors. A structured
    literature pass enumerates candidates, each with a mechanism + citation,
    and emits them as child bank entries (not a single blob).
  prior_p_win: 0.60            # P(yields >=1 child with prior_p_win >= 0.3)
  prior_effect: {mean: 0.004, sd: 0.010}   # option value of the best child
  prediction: >
    Produces >=3 child ideas, each with mechanism + citation + a
    feature-importance plan (zero-out screen first, §ablation). The scan
    "wins" if >=1 child clears prior_p_win 0.3 after mechanism review.
  correlated_with: [ocm-why-no-gain, feature-ablation-prune]
  status: open

- slug: ocm-why-no-gain
  category: match_nn
  cost_class: cheap            # uses existing pair-log + ml.analysis tooling
  source: user
  primary_axis: simplicity     # likely outcome: justified removal of ocm_cos
  touches: [analysis: ocm_cos in pair-log ; ml.analysis.permute_match_features]
  mechanism: >
    OC-SORT's observation-centric momentum is a strong cue elsewhere, yet
    `ocm_cos` ranks 13/16 in the v13 OBS feature audit (FEATURE_AUDIT.md)
    and is a documented drop-candidate. Hypotheses: (a) `ocm_cos` is
    degenerate in our pipeline (mostly zero/constant — detector cadence or
    a compute bug), or (b) it is collinear with kf_d2 / track_speed /
    kf_score so the NN extracts no marginal value. Diagnose with: value
    distribution + NaN/zero rate in the pair-log; permutation importance;
    pairwise collinearity vs the motion features; compare our formula to
    the OC-SORT reference.
  prior_p_win: 0.65            # P(produces a clear degenerate-or-redundant verdict)
  prior_effect: {mean: 0.001, sd: 0.004}
  prediction: >
    Outcome A (degenerate): fixing the computation gives a real fitness
    lever -> spawn a fix idea. Outcome B (collinear/low-value): confirms
    redundancy -> feeds feature-ablation-prune as a simplicity removal.
    Falsified-as-uninformative only if ocm_cos is mid-importance AND
    independent, which the audit already makes unlikely.
  correlated_with: [feature-ablation-prune, literature-feature-scan]
  status: open

- slug: feature-ablation-prune
  category: match_nn
  cost_class: medium           # zero-out screen is cheap; retrain-without confirm = medium
  source: user
  primary_axis: simplicity
  touches: [train_phase3 / build_pair_dataset feature set ; eval]
  mechanism: >
    Wide NN input (16+5+19) has documented low-value dims. FEATURE_AUDIT
    bottom-4 OBS = {ocm_cos, det_subbox_conf, det_fiqa_score, track_speed};
    DET-tower det_conf is redundant with PAIR pre_thr_score. Pruning
    fitness-neutral features = a simplicity-axis win (fewer params, smaller
    feature pipeline, less overfit surface, marginally faster) per §2.1.
    Method: cheap screen first — zero the feature in BOTH train inputs and
    eval and check fitness stays within 2sigma; only then pay for a
    retrain-without-the-column to confirm and to realise the param saving.
  prior_p_win: 0.55            # P(>=1 feature prunes fitness-neutral)
  prior_effect: {mean: 0.000, sd: 0.004}   # neutral by design; payoff is simplicity
  prediction: >
    Zero-out screen on the audit bottom-4 + det_conf: expect >=2 to be
    fitness-neutral (|Δfit|<2σ, guards intact). Those become a simplicity
    win after the confirming retrain (record params↓, LOC↓). A feature
    whose zeroing *improves* fitness was actively harmful -> fitness win.
    Falsified if every audited-low feature is actually load-bearing.
  correlated_with: [ocm-why-no-gain, literature-feature-scan, remove-cheap-filter-machinery]
  status: open

# ---- agent-proposed, grounded in documented history (2026-05-15) ----------

- slug: fitness-shaped-trainer-loss
  category: state_nn           # also applies to match_nn; start with state head
  cost_class: heavy            # retrain + K-seed + corpus-drift exposed
  source: agent
  primary_axis: fitness
  touches: [train_state_head_decoupled / train_phase3 loss ; build_*_corpus weights]
  mechanism: >
    Documented as THE next angle in EXPERIMENT_HISTORY ("D1/D2 follow-up:
    fitness-shaped per-sample weights") and feedback memories
    (`feedback_track_training_objective`: BCE/AUC don't optimise fitness;
    `feedback_track_eval_metric`: judge by fitness not val-AUC). The heads
    train on BCE; the eval is fitness (mota - .0005*fp_tracks -
    .002*fp_per_frame). Per-sample weights that mirror the fitness
    cost (counterfactual: how much does this example's error move
    fp_tracks/mota) align the surrogate with the true objective, and the
    cross-domain JAAD gap is structurally a fitness-weighting artefact
    (lower fp_track base rate there), not a data-volume one (D1/D2
    falsified volume).
  prior_p_win: 0.35
  prior_effect: {mean: 0.006, sd: 0.014}
  prediction: >
    K=3-seed medians, within-corpus vs same-corpus BCE control. Expect
    fp_tracks down and fitness up, with the JAAD/full176 gap narrowing
    (the discriminating prediction — a pure data effect would not narrow
    it). Falsified if fitness-shaping moves val-AUC but not fitness, or
    widens the domain gap.
  correlated_with: [state-corpus-fp-boost, dagger-multiiter-regression]
  status: open

- slug: scene-density-pair-rebalance
  category: corpus
  cost_class: heavy
  source: agent
  primary_axis: fitness
  touches: [build_pair_dataset: per-scene pair sampling/weight ; NO extra loop]
  mechanism: >
    `project_pair_log_scene_skew`: the top-5 scenes are 36% of all pairs,
    so the bootstrap corpus is structurally dense-crowd-biased — the head
    over-fits crowded static-cam scenes and under-serves sparse / dashcam.
    Cap or inverse-frequency-weight per-scene pair contribution at
    corpus-build time (single pass, NO added training loop, initial
    candidate generation stays non-NN — honours the same constraint as
    adaptive-nn-prior). NOTE: naive JAAD up-weight was already FALSIFIED
    (`project_d1_jaad_test_promotion`); this is scene-frequency
    rebalancing of the *whole* corpus, a different mechanism, and must
    cite that it is not the killed D1 idea.
  prior_p_win: 0.30
  prior_effect: {mean: 0.004, sd: 0.012}
  prediction: >
    Within-corpus, K-seed. Expect modest full176 change but a narrowed
    JAAD/full176 gap if the skew story holds. Falsified (and demoted
    toward the D1 kill) if rebalancing behaves like the failed JAAD
    up-weight (no gap change).
  correlated_with: [fitness-shaped-trainer-loss, dagger-multiiter-regression]
  status: open

- slug: remove-cheap-filter-machinery
  category: inference
  cost_class: medium
  source: agent
  primary_axis: simplicity
  touches: [ubon_cstuff utrack_match.c + build_pair_dataset.py cheap-filter (~150 LOC)]
  mechanism: >
    EXPERIMENT_HISTORY records the cheap-filter machinery (~150 LOC across
    the C matcher and the training-side mirror) "buys no measurable
    performance" and `project_cheap_filter_speed_neutral`: it saves no
    wall time either. If a fitness-neutral check confirms, deleting the
    whole mechanism is a pure simplicity win (§2.1 axis 4): less code, one
    fewer train/infer-coupled knob (removes the δ-mismatch class of bugs
    entirely). Gated behind cheap-filter-delta-realign: if δ turns out to
    matter, the machinery can't be removed, so resolve that cheap probe
    first.
  prior_p_win: 0.50           # P(simplicity win: fitness-neutral removal)
  prior_effect: {mean: 0.000, sd: 0.004}
  prediction: >
    With cheap-filter disabled/removed: |Δfit|<2σ, guards intact, ~0 speed
    change (consistent with the speed-neutral finding) -> simplicity win,
    record LOC↓ ≈150 and one knob removed. Falsified if removal regresses
    fitness beyond 2σ (then δ *did* matter — promote cheap-filter-delta
    work instead).
  correlated_with: [cheap-filter-delta-realign, feature-ablation-prune]
  status: open
  blocked_by: cheap-filter-delta-realign

# ---- killed (kept so a falsified mechanism cannot be silently revived) -----

- slug: jaad-pair-upweight
  category: corpus
  source: agent
  status: killed
  killed_by: project_d1_jaad_test_promotion.md (2026-05-15)
  mechanism: >
    Adding / up-weighting JAAD pairs to close the cross-domain gap.
    FALSIFIED: D1/D2 showed the gap is not a data-volume problem. Listed
    here per RESEARCH.md §7.2 so it cannot be resurrected under a new slug;
    the legitimate successors are fitness-shaped-trainer-loss and
    scene-density-pair-rebalance (different mechanisms, must cite this kill).
```

### Current ranking (recompute each iteration)

Non-fitness-primary ideas (`idf1`/`speed`/`simplicity`/diagnostic) have
`prior_effect.mean≈0` by design, so the fitness-only EV in §7.1 understates
them; they are ranked by **information / option value** per the §7.1
tie-breaker and flagged `(info)`.

| Rank | slug | EV | cost | P(win) | axis | note |
|---|---|---|---|---|---|---|
| 1 | honest-fp-track-metric | (info,max) | cheap | 0.70 | measurement | audits the ruler every fitness idea uses; iter2 is a smoking gun |
| 2 | cheap-filter-delta-realign | 0.0020 | cheap | 0.45 | fitness | gates remove-cheap-filter + match-nn-retrain |
| 3 | literature-feature-scan | 0.0024(info) | cheap | 0.60 | gen | cheap generator; high option value |
| 3 | ocm-why-no-gain | 0.0013(info) | cheap | 0.65 | simplicity | audit already flags ocm_cos low; near-certain verdict |
| 4 | dedup-iou-sweep | 0.0011 | cheap | 0.35 | fitness | F5d showed surface still live |
| 5 | feature-ablation-prune | (info) | medium | 0.55 | simplicity | bottom-4 named in FEATURE_AUDIT |
| 6 | new-track-thr-sweep | 0.0006 | cheap | 0.30 | fitness | direct fp_tracks lever |
| 7 | nn-lambda-sweep | (info) | cheap | 0.35 | idf1 | documented IDF1 dial |
| 8 | track-buffer-seconds-sweep | 0.0006 | cheap | 0.30 | fitness | idf1↔fp_tracks trade |
| 9 | dagger-multiiter-regression | (info) | medium | 0.60 | fitness | diagnose the −0.02 iter2 regression |
| 10 | adaptive-nn-prior | 0.00025 | medium | 0.40 | fitness | design intent unused; honours no-loop rule |
| 11 | poseflow-box-warp | 0.00018 | medium | 0.35 | fitness | crowded-family motion fix |
| 12 | state-corpus-fp-boost | 0.00018 | medium | 0.35 | fitness | metric-aligned; seed-sensitive |
| 13 | remove-cheap-filter-machinery | (info) | medium | 0.50 | simplicity | ~150 LOC; blocked by #1 |
| 14 | fitness-shaped-trainer-loss | 0.00009 | heavy | 0.35 | fitness | documented "next angle" |
| 15 | match-nn-retrain-at-ship-delta | 0.00008 | heavy | 0.40 | fitness | blocked by #1 |
| 16 | scene-density-pair-rebalance | 0.00005 | heavy | 0.30 | fitness | structural skew; not the killed D1 |
| 17 | state-head-poswfit-kseed | 0.00006 | medium | 0.25 | fitness | likely confirms "not a lever" |
| — | jaad-pair-upweight | — | — | — | killed | falsified by project_d1 |

First pick = **honest-fp-track-metric**. It is cheap (read-only audit of
existing ubtrk2 + GT) and has the highest possible information value: it
tests whether the headline `fitness` — the ruler *every* other idea is
judged by — is being gamed via FP-track merges. If it is, prior
fp_tracks-driven "wins" (F5d ship, DAgger conservatism) are partly
illusory and the campaign re-pins under an honest metric (RESEARCH.md §3);
if not, fitness is re-validated and every fitness idea proceeds with
confidence. You fix the ruler before measuring more with it. Then
**cheap-filter-delta-realign** (gates two heavier ideas), and the cheap
generators/diagnostics (literature-feature-scan, ocm-why-no-gain) which
are ~zero compute and curate the rest of the bank.

### Curation note — 2026-05-15 (DAgger multi-iter regression)

Direct evidence from the 2026-05-15 optimized bootstrap (frozen corpus,
within-batch), `full` config: fitness **iter1 0.5610 → iter2 0.5395**
(−0.0215 ≫ 2σ), MOTA 0.6148 → 0.5654, fp_tracks 103 → 48 (the head got
far more conservative — buying fp_tracks by collapsing recall). Same
direction in earlier runs. Consequences applied:

- `dagger-iter4` ("go deeper") **killed** — the data falsifies the
  premise; replaced by the diagnostic `dagger-multiiter-regression`.
- `bootstrap_recipe.sh` default changed **STOP_AFTER 3 → 1**; multi-iter
  is now investigation-only.
- `nn-lambda-sweep` re-tagged to the **IDF1 axis** per
  `project_nn_lambda_idf1_dial.md` (λ moves IDF1, not fitness).

---

## Experiment Log (append-only)

One entry per experiment, newest at the bottom. Never edited after written.
Each entry must contain the explicit Bayesian update
(`μ0,τ0 → Δfit,σ_obs → μ1,τ1 → P(win)`), the per-family / fp_tracks check,
the decision, and the bank curation that followed.

Entry schema (copy for each new experiment):

```
### <YYYYMMDD>-<slug>   [win(fitness|idf1|speed|simplicity)|loss|inconclusive|errored]
- selected: EV=<v> (runner-up <slug> EV=<v>); reason=<why this pick>
- primary_axis: <fitness|idf1|speed|simplicity>
- prior: p_win=<>, effect~N(μ0=<>, τ0=<>)
- change: <one line + path to change.patch>
- preflight (§8.1): active-path re-checked=<y/n>; redundant steps cut=<what skipped & why>;
    wall-clock est=<>; launch liveness-confirmed=<y/n @ ~t>; smoke=<n/a | result>
- eval: <path to results-<ts>.json>; corpus=eval_ship_baseline (frozen)
- evidence (within-batch vs same-batch control):
    fitness Δ=<>   mota Δ=<>   idf1 Δ=<>   fp_tracks Δ=<>
    families: full176 Δ=<>  jaad_val Δ=<>
    speed: <elapsed_s cand/control, pinned GPU/workers>  (if speed axis)
    simplicity: <LOC↓ / params↓ / knobs removed / module deleted>  (if simplicity axis)
    (seed-sensitive? per-seed: [..]; median used)
- update: σ_obs=<>; μ1=<> τ1=<>; posterior P(win)=<>
- prediction check: <held / partially / FALSIFIED — what it implies>
- decision: <outcome> + (candidate-confirmed? how)
- promotion gate: confirmed=<y/n>  pipeline_clean=<y/n, log path>  overall_better=<y/n>
- containment: artifacts only under RESEARCH_OUT/<slug>/ ? <verified y/n>
- baseline: <unchanged | promoted: commit=<sha> invocation=<cmd> control provenance=<path>>
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

### 20260515-honest-fp-track-metric   [win(measurement)]
- selected: rank #1 (info,max); runner-up cheap-filter-delta-realign EV=0.0020;
  reason=cheap read-only audit of the ruler every fitness idea is judged by
- primary_axis: measurement
- prior: p_win=0.70 (P material+detectable gaming), effect~N(μ0=0, τ0=0)
- change: none (read-only); tooling=RESEARCH_OUT/20260515-honest-fp-track-metric/audit.py
- preflight (§8.1): active-path re-checked=y (iter1/2/3 `full` JSON carry
  num_FP/misses/switches/frames); redundant steps cut=NO eval/pipeline/retrain
  (pure arithmetic on existing artifacts); wall-clock est=seconds;
  launch liveness=immediate; smoke=n/a (read-only)
- eval: existing iter1/2/3 results-*.json (optimized bootstrap, frozen corpus)
- evidence (iter1→iter2, the −53% fp_tracks drop):
    fp_tracks −53.4% (103→48) but num_FP only −17.9% (98497→80909);
    fp_per_frame −17.9%; misses +19.1%; switches −29.2%;
    MOTA 0.6148→0.5654; fitness 0.5610→0.5395.
    decoupling |ΔFPvol|/|Δfp_tracks| = 0.33; gameable terms gave +0.0279
    fitness credit masking 56% of the −0.0494 MOTA collapse.
- update: decisive pre-registered test; H2(trustworthy)=FALSIFIED,
  H1(gaming)=supported by 3 independent signals; posterior P(metric
  materially gameable) ≈ 0.97
- prediction check: HELD decisively (predicted ≪53% FPvol drop ⇒ smoking gun)
- decision: win(measurement) — verdict delivered; NOT promotable (handling_note)
- promotion gate: n/a (measurement; does not promote a tracker)
- containment: artifacts only under RESEARCH_OUT/<slug>/ — verified y
- baseline: unchanged; triggers campaign-fix track (RESEARCH.md §3/§4.5):
  prior fp_tracks-dominated wins (F5d ship, DAgger) now SUSPECT pending re-audit
- bank curation: honest-fp-track-metric → confirmed (evidence block added);
  honest-fp-track-metric-definition promoted to next pick; state-corpus-fp-boost
  / fitness-shaped-trainer-loss / dagger-multiiter-regression flagged
  training-coupled to the new ruler
- artifacts: RESEARCH_OUT/20260515-honest-fp-track-metric/

### 20260515-honest-fp-track-metric-definition   [win(measurement, partial)]
- selected: promoted by exp#1 (ruler confirmed gamed); campaign-fix rank #1
- primary_axis: measurement (produces ruler; no tracker promotion)
- prior: p_win=0.45, effect~N(0,0)
- change: track_test.py _honest_fp_tracks() segment count, side-channel
  (fitness untouched, §3); change.patch = git show of the two commits
- preflight (§8.1): smoke one clip (gamed0→honest1, invariant ok); dict-agg
  bug fixed pre-launch; 1st launch errored (missing logs/ redirect) caught
  by liveness check, re-run; one 2-test sharded eval, frozen corpus, ~1m38s
- eval: RESEARCH_OUT/20260515-honest-fp-track-metric-definition/eval/results-20260515-0930.json
- evidence (full176, iter1→iter2; reproduced cross-run):
    gamed fp_tracks 102→47 (−53.9%); FP-volume 98485→80835 (−17.9%);
    honest fp_tracks 2703→2151 (−20.4%); MOTA 0.6147→0.5660.
    decoupling |ΔFPvol|/|Δuniq|: gamed 0.33 → HONEST 0.88 (jaad 0.66→0.88).
    breakdown: hidden FP is mostly lag-out+bridge inside matched tracks.
- update: prediction held; gamed 0.33 reproduced independently across
  exp#1 and exp#2 ⇒ P(exploit real & seg-def captures it) ≈ 0.95
- prediction check: HELD (honest→0.88, gamed stays 0.33)
- decision: win(measurement, partial) — criterion 1 PASSED+reproduced;
  criteria 2 (threshold robustness) & 3 (no false-alarm on clean tracker)
  + spatial gate + training cascade OUTSTANDING. Does NOT freeze the ruler.
- promotion gate: n/a (measurement)
- containment: all under RESEARCH_OUT/<slug>/ — verified y; track_test
  side-channel committed (frozen fitness untouched)
- baseline: unchanged; campaign stays in §3/§4.5 metric-fix track — no
  fitness-measured tracker promotion until honest ruler frozen
- bank curation: idea → status validating; spawned children
  honest-fp-threshold-sweep (crit-2), honest-fp-cleanconfig-falsealarm
  (crit-3), honest-fp-spatial-gate; state-corpus-fp-boost /
  fitness-shaped-trainer-loss / dagger-multiiter-regression remain
  training-coupled to the eventual frozen honest ruler
- artifacts: RESEARCH_OUT/20260515-honest-fp-track-metric-definition/

### 20260515-honest-fp-cleanconfig-falsealarm   [win(measurement)]
- selected: next gating validation after exp#2 (crit-3 false-alarm)
- primary_axis: measurement
- prior: p_win=0.55 (P a sane band gives clean honest≈gamed), effect~N(0,0)
- change: none (read-only eval of deployed ship uc_v11); 1-test frozen eval
- preflight (§8.1): containment tree incl logs/ created BEFORE launch
  (prior lesson applied); sharded 1-test, frozen corpus, ~1m34s;
  liveness-confirmed; metric already crit-1 validated → no new smoke
- eval: RESEARCH_OUT/20260515-honest-fp-cleanconfig-falsealarm/eval/results-20260515-0937.json
- evidence (CLEAN ship uc_v11=v13+v23_pw05, frozen):
    full176 gamed fp_tracks 84 vs honest 2608 = 31.0x (lagout 1098 +
    bridge 1364 dominate; fully=53≈gamed); jaad 2 vs 7 = 3.5x.
- update: false-alarm worry CONFIRMED not falsified; P(temporal-only v0
  usable)≈0.05; P(spatial gate required)≈0.85
- prediction check: predicted "≫1x ⇒ thresholds too aggressive" — HELD,
  severely (31x). Key inference: gaming and legit occlusion share the
  SAME temporal signature; only SPATIAL info separates them ⇒ spatial
  gate is critical-path, temporal-only band likely doesn't exist.
- decision: win(measurement) — decisive verdict prevents freezing a
  broken ruler. Ruler NOT frozen; campaign stays §3/§4.5.
- baseline: unchanged; no fitness-measured tracker promotion
- bank curation: honest-fp-cleanconfig-falsealarm → confirmed;
  honest-fp-spatial-gate prior↑ + re-tagged critical-path (next pick);
  honest-fp-threshold-sweep prior↓ + sequenced AFTER the spatial gate
  (run jointly); honest-fp-track-metric-definition stays validating
  (crit-1 pass, crit-3 fail@v0)
- artifacts: RESEARCH_OUT/20260515-honest-fp-cleanconfig-falsealarm/

### 20260515-honest-fp-spatial-gate   [win(measurement)]
- selected: critical-path pick after crit-3 (temporal-only false-alarmed 31x)
- primary_axis: measurement
- prior: p_win=0.70 (spatial gate makes def freezable), effect~N(0,0)
- change: _honest_fp_tracks spatial-excursion gate + per-frame hyp
  centroid capture in compute_metrics (HId==track_id verified);
  side-channel only, frozen fitness untouched
- preflight (§8.1): logs/ pre-created; smoke 1 clip (v0 honest1 -> spatial0,
  legit bridge forgiven, invariant ok); 2 sharded evals reuse prior
  cfg/yaml, no retrain, ~1m32s+1m39s; liveness-confirmed
- eval: RESEARCH_OUT/20260515-honest-fp-spatial-gate/eval/{ship,iter1v2}/results-*.json
- evidence (full176, θ=2.0):
    crit-3 clean ship honest/gamed 31.0x -> 0.8x (gamed83 honest68;
      lag/bridge false alarms 1098/1364 -> 11/2) — PASSED
    crit-1 iter1->iter2 honest -44.3% vs FP-vol -17.9% -> decoupling
      0.88(v0) -> 0.40 — REGRESSED (gate too forgiving @θ=2.0)
- update: gamed 0.33 reproduced a 4th time (exploit P≈0.99). spatial-gate
  v1@θ2.0 freezable P≈0.05; some θ-band satisfies both P≈0.35;
  need FP-frame metric P≈0.45
- prediction check: HALF held (clean fixed; decoupling did NOT stay)
- decision: win(measurement) — pinpoints a real θ tension (crit-3 vs
  crit-1); prevents freezing a ruler that re-admits the exploit
- baseline: unchanged; ruler NOT frozen; campaign stays §3/§4.5
- nuance: honest<gamed possible (SWITCH treated as matched-assoc, not
  pure FP) — defensible, breaks the naive invariant, must be documented
- bank curation: honest-fp-spatial-gate → confirmed(caveat);
  honest-fp-threshold-sweep → DECISIVE next, prior↓0.35, joint objective;
  added contingency honest-fp-frame-metric; honest-fp-track-metric-
  definition stays validating
- artifacts: RESEARCH_OUT/20260515-honest-fp-spatial-gate/

### 20260515-honest-fp-threshold-sweep   [win(measurement) — FALSIFICATION]
- selected: DECISIVE crux after exp#4 (crit-3 vs crit-1 θ tension)
- primary_axis: measurement
- prior: p_win=0.35 (a joint band exists), effect~N(0,0)
- change: refactor honest-FP -> pure _honest_fp_core (live==offline);
  env-gated atomic per-clip dump (events+centroids) in compute_metrics;
  offline sweep.py. Frozen fitness untouched.
- preflight (§8.1): correctness gate offline==live 615 clips 0 mismatch;
  dump completeness 205/410 asserted; caught+fixed silent mp dump loss;
  extended θ grid to include θ→0 after noticing truncation of the
  temporal-only regime — completed the frontier before concluding
- eval: RESEARCH_OUT/20260515-honest-fp-threshold-sweep/eval_{ship,iter}/results-*.json (frozen corpus); sweep 225 cells
- evidence (full176): as θ rises clean honest/gamed 31.8x->1.7x while
  crit-1 decoupling 0.90->0.57 — opposite directions, NO crossover.
  Over 225 cells max decoupling while clean-OK = 0.65 (<0.8 needed).
  NO joint band. θ→0 reproduces exp#2 (0.88≈0.90) AND exp#3 (31x≈31.8x).
- update: pre-registered falsification condition MET across the full θ
  axis; P(count formulation freezable) 0.35 -> ~0.02. Exploit 0.33
  signature now seen in 5 independent measurements.
- prediction check: HELD — no band exists (the falsification branch)
- decision: win(measurement) — decisive complete negative; rules out the
  entire segment-COUNT family; prevents endless threshold-fiddling
- baseline: unchanged; ruler still unfrozen; campaign stays §3/§4.5
- bank curation: honest-fp-threshold-sweep -> confirmed(count FALSIFIED);
  honest-fp-track-metric-definition + -spatial-gate -> superseded
  (count line closed; their cache/excursion machinery REUSED);
  honest-fp-frame-metric -> DECISIVE next, prior 0.45->0.60
- artifacts: RESEARCH_OUT/20260515-honest-fp-threshold-sweep/

### 20260515-honest-fp-frame-metric   [win(measurement) — VALIDATION]
- selected: DECISIVE next after exp#5 falsified the segment-COUNT family
- primary_axis: measurement
- prior: p_win=0.55 (a joint band exists for the FRAME formulation),
  effect~N(0,0)
- change: add pure `_honest_fp_frames_core` to track_test.py (side-
  channel; fitness UNTOUCHED) — count surviving FP *frames*, per-frame
  spatial-gated, in lead/lag/bridge of matched tracks; NO length
  threshold (exp#5's L-vs-θ coupling removed). nm_policy='track'
  mirrors gamed (never-matched not the gaming vector). Offline sweep
  imports the live core (no reimplementation).
- preflight (§8.1): chain-integrity gate PASS — |dFPvol|/|dGamed|=0.331
  reproduces the exp#3-5 exploit signature exactly → inputs are the
  exp#5-validated chain (offline==live, 615 clips 0 mismatch there);
  fixed a misleading hard-coded annotation before trusting the log
- eval: reuse exp#5 cached dumps+JSONs via symlink (no scatter);
  deterministic offline recompute, 18-cell sweep + breakdown probe
- evidence (full176, nm='track'): joint-OK at θ∈[0.3,0.6] —
  clean_frac 0.163/0.061 (≤0.50), crit-1 0.992/0.864 (∈[0.8,1.25]).
  The pre-registered clean-vs-decoupling **crossover EXISTS**: clean
  falls monotone 0.95→0.16→0.06→0.02 while crit-1 stays ≈1 then
  collapses at θ≥1.0 (vs exp#5's monotone-opposite no-crossover).
  Non-degenerate: nm = 52/15750 (0.3%); honest reports the REAL
  dFPvol=−0.179 not the gamed dGamed=−0.539 (smoking-gun property).
  nm='frames' has no band (never-matched dominates) — confirms the
  principled nm='track' choice.
- update: pre-registered joint gate MET with margin and the
  specifically-predicted crossover; P(FRAME formulation freezable)
  0.55 → ~0.90. First honest-FP formulation to PASS.
- prediction check: HELD — removing the length threshold broke the
  exp#5 coupling exactly as hypothesised; a θ window with clean low
  AND crit-1≈1 exists
- decision: win(measurement) — promote FRAME formulation to **candidate
  frozen honest ruler** (θ=0.5, nm_policy='track'). fitness still
  untouched; ruler not yet frozen (needs the downstream gated steps).
- baseline: unchanged; campaign stays §3/§4.5 until ruler frozen
- bank curation: honest-fp-frame-metric → confirmed(FRAME VALIDATED,
  candidate ruler θ=0.5/nm=track); honest-fp-threshold-sweep family
  closed; spawn child honest-fp-frame-metric-wire-live (next).
- artifacts: RESEARCH_OUT/20260515-honest-fp-frame-metric/

### 20260515-honest-fp-frame-metric-wire-live   [win(measurement) — GATED]
- selected: pre-registered child of exp#6 (make the validated ruler real)
- primary_axis: measurement
- prior: p_win=0.85 (mostly engineering; risk = live≠offline surprise)
- change: live compute_metrics now emits side-channel `fp_frames_honest`
  (+fp_fhonest_nm/leadin/lagout/bridge) via the SAME pure
  `_honest_fp_frames_core` on the in-memory ev/cd inputs; frozen-ruler
  constants HONEST_FP_FRAME_THETA=0.5 / _NM='track' with a "changing
  this re-opens the ruler" banner; keys added to _summary_metric_keys;
  except-branch zero-fills. fitness_score UNTOUCHED (grep-verified).
- preflight (§8.1): re-read the live honest block, spliced into the
  existing try (pure reuse, no new compute path); smoke = 3 datasets,
  num_workers 3, ~7s
- eval: RESEARCH_OUT/20260515-honest-fp-frame-metric/eval_smoke +
  dump_smoke; correctness gate verify_live.py
- evidence: live `fp_frames_honest` == offline `_honest_fp_frames_core`
  @ frozen θ=0.5/nm='track', 3/3 clips, 0 mismatch (FP clip 5==5, clean
  clips 0==0). Live wiring faithful; ruler correctly frozen.
- update: prediction (live==offline) HELD; P 0.85→~0.98 (residual =
  the fresh-corpus clean-ratio check, deferred to the re-pin cycle)
- prediction check: HELD — 0 mismatch as constructed
- decision: ruler FROZEN in front-matter (honest_ruler: status frozen,
  fitness_uses_it false). win(measurement).
- baseline: unchanged; campaign stays §3/§4.5 (fitness still untouched)
- bank curation: honest-fp-frame-metric-wire-live → confirmed; spawn
  child honest-fp-frame-metric-repin (clean full-pipeline re-pin, next)
- artifacts: RESEARCH_OUT/20260515-honest-fp-frame-metric/ (smoke.yaml,
  verify_live.py, eval_smoke/, dump_smoke/, logs/)

--- next: honest-fp-frame-metric-repin (GATED) — one clean full-pipeline
    cycle (retrain → build → eval, STOP_AFTER=1) with the now-frozen
    side-channel ruler reported beside fitness, to (a) re-pin baseline_ref
    {fitness,mota,idf1,fp_tracks,fp_frames_honest,speed,provenance} and
    (b) confirm the fresh-corpus clean fp_frames_honest reproduces the
    exp#6 band (clean_frac ~0.06–0.16) — falsified ⇒ corpus-sensitive,
    re-open formulation. Pre-flight §8.1 on the pipeline entrypoint
    (fail-fast liveness). THEN honest-fp-train-adopt: switch the training
    objective from gamed fp_tracks to the frozen ruler (campaign reset —
    its own pre-registered experiment, K-seed). fitness stays untouched
    until the re-pin cycle promotes the ruler (§3/§4.5). ---

## Progress curve

Chain of **confirmed within-batch Δfit** through promotions (immune to
corpus drift — see RESEARCH.md §4.4). Empty until the first confirmed win.

```
baseline (bootstrap) ──> [no confirmed tracker wins yet]

measurement track:
  20260515 honest-fp-track-metric  [win(measurement)]
    => fitness `fp_tracks` term materially gameable (decoupling 0.33,
       P≈0.97). Campaign pivots to honest-metric-definition before
       further fitness-measured tracker work is trusted. Prior
       fp_tracks-dominated wins (F5d ship, DAgger) flagged SUSPECT.
  20260515 honest-fp-threshold-sweep  [win(measurement) — FALSIFICATION]
    => segment-COUNT honest-FP family ruled out (no joint band over
       225 cells; clean vs decoupling monotone-opposite, no crossover).
  20260515 honest-fp-frame-metric  [win(measurement) — VALIDATION]
    => FRAME formulation PASSES the joint gate (first to do so):
       nm='track' θ∈[0.3,0.6] clean_frac 0.16/0.06, crit-1 0.99/0.86,
       predicted crossover present, P 0.55→~0.90. Candidate frozen
       honest ruler θ=0.5/nm='track'. Next: wire-live + freeze +
       clean re-pin + training adoption (fitness untouched until then).
```
