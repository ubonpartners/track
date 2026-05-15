# RESEARCH.md — Tracker Improvement Research Loop

A self-contained, **Bayesian, evidence-first** loop for proposing, running,
judging, and either promoting or discarding **tracker improvement ideas**
(inference config, match-NN, state-head, corpus). The goal is *monotonic,
reproducible* improvement of a single headline score on a **frozen** corpus,
with every attempt — win or loss — left behind as a written artifact.

This document is the **methodology**. The living idea bank and the
experiment log are in a separate file: [`RESEARCH_LOG.md`](RESEARCH_LOG.md).
This file changes rarely; the log file changes every iteration.

The non-negotiable principles:

- **No guessing.** Every experiment starts from a written hypothesis with a
  stated *mechanism* and a *falsifiable numeric prediction*. An idea with no
  mechanism does not enter the bank.
- **Be a good Bayesian.** Every idea carries an explicit prior (probability
  it is a real win + expected effect size with uncertainty). Evidence
  updates the posterior of that idea *and* of correlated ideas. Selection is
  by expected value, not by hunch.
- **Evidence beats intuition, and noise is the enemy.** This domain has
  three quantified evidence hazards (eval noise, seed luck, corpus drift —
  §4). The decision rule is built around them. A number that does not clear
  the noise model is not evidence.
- **Reproducibility is a precondition, not a nicety.** Every experiment
  pins config diff, git SHA, engine/bin versions, seed(s), and keeps the
  machine-readable eval JSON sidecar.
- **One experiment, one folder. No artifact scatter.** Everything an
  experiment generates — diffs, eval JSON/TXT, pipeline outputs, retrained
  bins/checkpoints, logs, temp files — lives under that experiment's single
  directory and nowhere else (§9). The repo and the rest of the filesystem
  stay clean. An experiment that wrote outside its folder is treated as
  `errored` and re-run.
- **Improvement is multi-objective, fitness-first.** A win is *any* of:
  higher fitness; or — at statistically *similar* fitness — higher IDF1,
  or faster execution, or simpler code/architecture (§2.1). Fitness can
  never be *bought* with the others: a fitness regression beyond noise is
  a loss regardless of IDF1/speed/simplicity gains.
- **Steady forward progress.** The baseline only ever moves up (on the
  active objective), and only on a *confirmed* win. The baseline is a
  re-measured number on the frozen corpus, never a remembered one.

---

## 1. Mission

Improve the tracker's headline **fitness** on the frozen full-176 + JAAD-val
corpus, at fixed evaluation settings, by a methodical search over a ranked
idea bank. Keep what is *proven* to help, discard what is not, and leave a
paper trail that lets any future run reconstruct exactly what was tried, why,
and what the evidence said.

The agent is autonomous between user checkpoints. Each iteration it:

1. Picks the highest expected-value idea from `RESEARCH_LOG.md`.
2. Writes the hypothesis (prior, mechanism, prediction, cost).
3. Executes the change (config edit, or a retrain via the pipeline).
4. Evaluates on the frozen corpus with the single unified evaluator.
5. Decides win / loss / inconclusive against the noise model.
6. Updates the posterior for this idea and correlated ideas.
7. Promotes the baseline only on a *confirmed* win.
8. Re-ranks the bank; adds, demotes, or kills ideas from the new evidence.

The user reviews progress through `RESEARCH_LOG.md` and the per-experiment
artifact directories.

---

## 2. Headline metric and guards

### Headline: fitness

The single optimized scalar is the tracker `fitness` already defined in
`src/track_test.py` (`fitness_score`):

```
fitness = mota
          - 0.0005 * fp_tracks       # false-positive tracks (integer count)
          - 0.002  * fp_per_frame    # false-positive detections per frame
```

This is computed by the unified evaluator over the corpus and reported as
the `__ovr<group>` rollup. Fitness is the headline because it already encodes
the product trade-off (identity quality vs. false-positive tracks vs.
per-frame FP rate) the tracker is tuned for.

### Guard metrics (regression gates, not optimized directly)

| Guard | Why | Default gate |
|---|---|---|
| `mota` | Don't buy fitness by trading away raw accuracy | no worse than −0.004 |
| `idf1` | Identity consistency; fitness is FP-weighted and can hide ID churn | no worse than −0.005 |
| `fp_tracks` | Integer, deterministic — the *sharpest* signal we have | no worse than +3 absolute |
| per-family fitness | A change that helps the mean but tanks low-light/JAAD is a loss | no family worse than −ε (§5) |

`fp_tracks` is integer and (modulo the FP-thread ordering noted in §4)
deterministic, so it is the highest-information single number in the report.
When fitness is marginal, look at `fp_tracks` first.

### 2.1 Objectives (strict priority order)

Improvement is multi-objective but **lexicographic, fitness-first**. An
experiment is a win if it improves the highest-priority objective it can
reach *without regressing any higher one beyond noise*:

1. **Fitness.** `Δfit ≥ +2σ` with all guards intact → win (primary axis).
2. **IDF1 at similar fitness.** If fitness is *statistically unchanged*
   (`|Δfit| < 2σ`, i.e. neither a fitness win nor a fitness loss beyond
   noise) and no guard regressed, then `Δidf1 ≥ +2σ_idf1` → win
   (secondary axis: better identity consistency for free).
3. **Execution speed at similar fitness.** Similar fitness (as above),
   IDF1 not regressed beyond noise, and a **meaningful, pinned-condition**
   speedup (`Δspeed ≥ speed_improve_min`, default +5%, measured on the
   frozen corpus at fixed hardware + worker count) → win.
4. **Simpler code/architecture at similar fitness.** Similar fitness, IDF1
   and speed not regressed beyond noise, and a *recorded, quantified*
   simplification → win. "Simpler" is not a vibe: it must cite at least one
   concrete reduction — net-negative code diff (LOC), fewer model
   parameters, removed tunable knobs, or a deleted code path/module — and
   the number goes in `decision.md`.

Hard rule: **fitness is never bought.** `Δfit ≤ −2σ` (or any guard past its
gate, or any family past `ε`) is a **loss** no matter how much IDF1, speed,
or simplicity improved. The objective hierarchy only lets a *fitness-neutral*
change win on a lower axis; it never trades fitness away.

Each experiment declares, up front in its hypothesis, which axis it targets
(`primary_axis: fitness|idf1|speed|simplicity`); the decision records which
axis it actually won on. The progress curve (§7.3) is per-axis.

### What is *out of scope* for this loop

These are real levers but are owned by other tracks and would break the
"frozen corpus / frozen evaluator" invariant:

- The detector TRT engine (built by `quant/` + `make_data/`).
- Detector/CLIP/face model weights or quantization.
- The eval corpus contents, the eval framerate/IoU settings, the fitness
  formula itself.

If a hypothesis can only be tested by changing one of these, it does **not**
enter this loop — it goes to the user as a separate track. The line is:
*if it changes a field the tracker reads from `uc_v11.yaml` at runtime, or
a NN bin the tracker loads, it is in scope; otherwise it is not.*

---

## 3. The frozen corpus and the evaluator

### Corpus (frozen)

`/mldata/config/track/eval/eval_ship_baseline.yaml` — full-176 (cevo / MOT /
PP22, `group: full176`) plus JAAD-val (`group: jaad_val`). This file is
**frozen for the duration of a research campaign**. Changing it invalidates
every prior comparison; if it must change, that starts a new campaign with a
re-measured baseline and a note in `RESEARCH_LOG.md`.

### The single evaluator

```
python track.py --eval /mldata/config/track/eval/eval_ship_baseline.yaml
```

There is exactly **one** evaluator (`track_search.eval_track` →
`track_test.track_test`). It:

- shards across all visible GPUs via `stuff.mp_workqueue_run`
  (`num_workers: auto`), so a full pass is ~1.5–2.5 min on the 8-GPU box;
- aborts loudly if any NN bin silently fails to load (the C-runtime
  `failed to load` / `in_dim mismatch` guard) — a "no head" bench is never
  silently scored;
- writes both a human `results-<ts>.txt` and a machine
  `results-<ts>.json` sidecar. The JSON is the experiment artifact:
  `tests.<key>.overall` carries `fitness`, `mota`, `idf1`, `fp_tracks`,
  `fp_per_frame`; `tests.<key>.groups` carries the per-family rollups;
  `tests.<key>.clips` carries per-clip rows for regression triage.

Multiple variants (e.g. candidate vs. control) go in **one** eval call as
multiple `tests:` entries so they share the dataset load and run against the
identical corpus snapshot in the same batch. This is mandatory for any
comparison (§4, within-corpus rule).

---

## 4. The evidence model (read this twice)

This domain has three quantified ways to fool yourself. The decision rule
(§5) exists to defeat exactly these. They are documented from prior
incidents in `ml/docs/PIPELINE.md` and the `feedback_track_*` memories.

### 4.1 Single-eval noise

The C tracker's FP-thread ordering and float accumulation give **≈ ±0.003
fitness** noise on a single eval pass. `mota` and `fp_per_frame` fluctuate;
`fp_tracks` is integer and deterministic. Therefore:

- A fitness delta below **2σ ≈ +0.006** is *not* a win on a single run.
- `fp_tracks` moving is believable at smaller magnitude than fitness moving.

If you have measured this machine's actual σ (see Bootstrapping §11), use
the measured value; otherwise use σ = 0.003.

### 4.2 Seed luck (NN training only)

State-head / match-NN training has **σ ≈ 0.013 across seeds** (seeds 0/1/2,
per the F3 multi-seed finding that *falsified* the single-seed "pw=0.6 wins"
result — `feedback_track_phase20c_failed.md`,
`project_a1_pw_sweep_was_seed_luck.md`). Therefore:

- Any idea whose execution includes a NN training step is **seed-sensitive**.
- A seed-sensitive change is judged on the **median of K = 3 seeds**
  (escalate to K = 5 if the K = 3 spread straddles the decision threshold),
  never a single seed. Report the per-seed values and the spread.
- A single-seed "win" on a seed-sensitive change is recorded as
  *inconclusive — needs K-seed*, never as a win.

### 4.3 Corpus drift (pipeline regens)

Re-running the corpus build drifts the retrained head **0.04–0.17 fitness**
vs. the shipped baseline *even with identical args*
(`feedback_track_corpus_drift.md`). Therefore the cardinal rule:

> **Within-corpus comparison only.** Never compare a candidate's absolute
> fitness to a number remembered from a previous campaign or a doc. A
> candidate is judged only against a **control evaluated in the same eval
> batch** (or, for retrains, built from the **same pair-log / corpus** as the
> candidate). Always report `Δ = candidate − control_same_batch`, not
> absolute fitness.

The unified evaluator makes this cheap: put the control config and the
candidate config as two `tests:` in one eval yaml.

### 4.4 Consequence for "baseline"

"Baseline" is therefore **not a remembered number**. It is the control
config, re-measured *in the same batch* every time a candidate is judged.
The headline progress curve in `RESEARCH_LOG.md` is built from
within-batch deltas chained through confirmed promotions, not from raw
absolute numbers across campaigns.

### 4.5 Metric integrity (the Goodhart hazard)

The deepest way to fool yourself is for the **headline metric itself** to
be gameable — then a real, confirmed, reproducible win can still be a
*proxy* win, and the whole loop optimises an exploit. fitness's
`-0.0005·fp_tracks` term counts *unique never-matched output tracks*; the
optimizer can cut that count by ID-merging an unrelated FP segment onto a
true track (or stitching two FPs into one) rather than removing the false
positives — the FP pixels stay on screen, they just stop counting. Direct
evidence this is real here: the 2026-05-15 DAgger iter2 drove fp_tracks
103→48 while MOTA *collapsed* 0.615→0.565 — classic Goodhart.

Rules:

- A **metric-integrity audit is always in scope and high priority**, even
  though §2/§3 freeze the metric for a campaign. Auditing the ruler is not
  changing it; it is read-only diagnosis (see `honest-fp-track-metric` in
  `RESEARCH_LOG.md`).
- A confirmed metric defect does **not** trigger a silent mid-campaign
  metric change. It triggers a **campaign reset** (§3): freeze the
  honest metric, re-pin the baseline by re-measuring under it, and
  re-judge open results. Prior wins driven by the defective term are
  re-examined, not assumed valid.
- When a win is *dominated by the gameable term* (e.g. almost all of
  `Δfit` comes from `fp_tracks` while MOTA/IDF1 sag within guard), treat
  it as **suspect** and require the integrity audit on that result before
  promotion, regardless of the §5 arithmetic.

---

## 5. Decision rule

Every experiment produces exactly one of: **win / loss / inconclusive /
errored**. Thresholds live at the top of `RESEARCH_LOG.md` so they can be
retuned without editing this doc; defaults below.

Let `Δfit = fitness(candidate) − fitness(control_same_batch)`, both from the
same eval batch (or, for seed-sensitive changes, `Δfit` of the per-seed
medians).

Let "**similar fitness**" mean `|Δfit| < 2σ` *and* no guard past its gate
*and* no family past `ε` (a genuinely fitness-neutral change, not a hidden
regression). Secondary axes are only consulted under similar fitness.

| Outcome | Condition |
|---|---|
| **win (fitness)** | `Δfit ≥ +2σ` (default +0.006) **and** no guard regresses past its gate (§2) **and** no per-family fitness regresses by more than `ε` (default 0.010) **and**, if seed-sensitive, this is the K-seed median. |
| **win (idf1)** | similar fitness **and** `Δidf1 ≥ +2σ_idf1` (default +0.006). |
| **win (speed)** | similar fitness **and** `Δidf1 ≥ −2σ_idf1` **and** pinned-condition speedup `≥ speed_improve_min` (default +5%). |
| **win (simplicity)** | similar fitness **and** idf1 & speed not regressed beyond their noise **and** a quantified simplification recorded in `decision.md` (LOC↓ / params↓ / knobs removed / module deleted). |
| **loss** | `Δfit ≤ −2σ` **or** any guard regresses past its gate **or** any family regresses by more than `ε` **or** (for a declared secondary-axis experiment) the targeted axis did not clear its bar and no higher axis did either. |
| **inconclusive** | fitness in `(−2σ, +2σ)` but the experiment's declared axis is unresolved at the measured precision (e.g. `+Δmin < Δfit < +2σ` for a fitness experiment, or a seed-sensitive change measured at a single seed). Logged; escalated to confirmation / K-seed if it is the best lead. |
| **errored** | Eval crashed, NN bin failed to load (evaluator self-aborts), a pipeline step failed, the config was rejected, **or the experiment wrote artifacts outside its own folder** (§9). Logs preserved; the loop continues. |

**Confirmation before promotion.** A win is *candidate-confirmed* only after:
- a re-run in a fresh eval batch reproduces `Δfit ≥ +2σ` (config-only
  changes), **or**
- the K-seed median reproduces `Δfit ≥ +2σ` (seed-sensitive changes).

Only a candidate-confirmed win promotes the baseline (§7).

Defaults (mirror these into `RESEARCH_LOG.md` front-matter):
`sigma = 0.003`, `sigma_idf1 = 0.003` (measure at bootstrap, floor 0.003),
`win_sigmas = 2.0`, `delta_min = 0.001`, `epsilon_family = 0.010`,
`mota_guard = -0.004`, `idf1_guard = -0.005`, `fp_tracks_guard = +3`,
`speed_improve_min = 0.05` (fractional, pinned conditions), `K_seed = 3`.

**Speed measurement** must be apples-to-apples: same machine, same visible
GPU count, same `num_workers`, frozen corpus, eval cache cleared. Use the
evaluator's own wall time (`elapsed_seconds` in the JSON sidecar) and/or
mean `tracked_fps`; take the median of 3 timed runs since wall time is
noisier than fitness. A speed claim without pinned conditions recorded in
`provenance.json` is rejected.

---

## 6. Hypotheses as Bayesian objects

Every idea in the bank is a structured object, not a sentence. Required
fields (schema enforced in `RESEARCH_LOG.md`):

```yaml
- slug: cheap-filter-delta-0p6
  category: inference            # inference | match_nn | state_nn | corpus
  cost_class: cheap              # cheap (~2 min, 1 eval) | medium (~15 min, 1 pipeline iter) | heavy (~1 h, 3-iter bootstrap)
  touches: [uc_v11.yaml: utrack.match_cheap_filter_delta]
  mechanism: >
    The cheap-filter delta gates which candidate pairs the match-NN even
    scores. Ship is 0.7; the v13 NN was trained with delta-filter 0.5, so
    at inference 0.7 feeds the NN a wider pair distribution than it saw in
    training (train/infer mismatch, PIPELINE.md common-pitfall #1).
    Lowering toward 0.5 should realign infer with train.
  prior_p_win: 0.45              # P(this is a real win) — calibrated, not vibes
  prior_effect: {mean: +0.004, sd: 0.006}   # prior over Δfit (fitness units)
  prediction: >
    Δfit between -0.002 and +0.012; MOT/PP22 families move most; JAAD ~flat;
    fp_tracks drops 2-6 (tighter gate => fewer spurious confirmations).
  correlated_with: [cheap-filter-delta-0p5, match-nn-retrain-delta0p7]
  status: open                   # open | running | win | loss | inconclusive | killed
```

Rules:

- **`mechanism` is mandatory and must reference how the tracker/NN actually
  works** (a knob's role, a train/infer mismatch, a documented pitfall).
  "Might help" is not a mechanism → the idea is rejected from the bank.
- **`prediction` must be falsifiable**: a numeric `Δfit` interval *and* which
  families/`fp_tracks` should move and in which direction. If the result
  contradicts the prediction, that is itself evidence (the mechanism is
  wrong) and must be written up — a "right answer for the wrong reason" is
  flagged, not silently kept.
- **`prior_p_win` and `prior_effect` are calibrated estimates.** Anchor them:
  inference-knob nudges near a tuned optimum have low `prior_p_win` and small
  `prior_effect` (the config is auto-tuned, so most single-knob moves are
  near-neutral); structural changes with a strong mechanism get higher mass
  but wider `sd`. Record the anchoring reasoning in the experiment write-up.
- **`correlated_with`** lists ideas whose posterior should move when this one
  resolves (§7.2).

---

## 7. Selection, update, and promotion

### 7.1 Selection — expected value

Each iteration, rank `open` ideas by **expected fitness gain per unit cost**,
then pick the max:

```
EV = prior_p_win * max(0, prior_effect.mean) / cost_units
cost_units: cheap = 1, medium = 8, heavy = 30   (≈ wall-time ratios)
```

Tie-breakers, in order: (1) higher information value — an idea that, win or
lose, sharply updates many `correlated_with` ideas beats an isolated one;
(2) lower `cost_class`; (3) older idea (avoid starvation). The chosen idea's
EV, the runner-up, and the reason for the pick are written into the
experiment entry — selection itself is evidence-logged.

### 7.2 Posterior update — explicit

After the eval, update beliefs. Use a plain Normal conjugate update on the
effect; this is deliberately simple and written down, not implicit.

For the tested idea, treat the measured `Δfit` as an observation of the true
effect with observation noise `σ_obs` (= σ for a single config eval; = the
K-seed standard error for seed-sensitive changes):

```
prior:      effect ~ Normal(μ0, τ0²)        # from prior_effect
likelihood: Δfit   ~ Normal(effect, σ_obs²)
posterior:  μ1 = (μ0/τ0² + Δfit/σ_obs²) / (1/τ0² + 1/σ_obs²)
            τ1² = 1 / (1/τ0² + 1/σ_obs²)
posterior P(win) = P(effect ≥ +2σ under Normal(μ1, τ1²))
```

Write `μ0,τ0 → Δfit,σ_obs → μ1,τ1 → P(win)` explicitly in the experiment
entry. Then:

- **Tested idea**: set `status` and replace `prior_effect`/`prior_p_win`
  with the posterior (so a re-test starts from the updated belief).
- **Correlated ideas**: shift their `prior_effect.mean` toward the observed
  direction by a stated fraction (default 0.3 for same-mechanism siblings,
  0.1 for loosely related), and *widen or narrow* `sd` per the result's
  consistency with their mechanism. Every correlated update is logged with
  its rationale — no silent re-weighting.
- **Mechanism falsified** (result contradicts the prediction's *sign* or the
  family pattern): mark the idea `killed`, and demote every idea that shared
  the falsified mechanism. A killed mechanism cannot be silently resurrected
  under a new slug — the log must cite the kill.

### 7.3 Baseline promotion

Promotion has a **hard three-part gate**. All three must pass, in order;
failing any one means *no promotion* (the change stays a logged result, not
the baseline):

1. **Candidate-confirmed win** on the declared axis (§5) — the re-run /
   K-seed reproduction of the within-batch delta.
2. **Clean full pipeline cycle.** The change is run through the *entire
   reproducible pipeline* end-to-end — `ml/orchestration/bootstrap_recipe.sh`
   (or `run_pipeline.sh` for the relevant scope) — with **zero `errored`
   steps**. A config-only change still goes through this: the pipeline is
   re-run with the new config so the baseline is always a fully-reproduced
   artifact, never a hand-edited config that never survived the real cycle.
   This also re-exposes corpus drift (§4.3) — the win must survive a fresh
   build, not just a fresh eval.
3. **Overall better metrics, not just the headline.** On the frozen corpus
   *after* the clean pipeline run: the targeted axis clears its bar **and**
   no guard regresses past its gate **and** no family past `ε` **and** the
   metric set is net-not-worse (fitness, IDF1, fp_tracks, per-family all
   either improved or within noise). "Won on one number, quietly worse on
   three" is **not** a promotion.

Only when all three pass:

- **Config win** — apply to `uc_v11.yaml` (or flip its pointer).
- **NN win** — copy the pipeline-produced bin to
  `/mldata/config/track/trackers/` under a bumped versioned name, flip the
  `uc_v11.yaml` pointer; keep the source `.pt` + its `_artefact_meta`.
- **Reproducibly commit.** One commit on `research/<date>-<slug>`
  containing: the config/code diff, `provenance.json` (three repo SHAs,
  engine/bin sha256s, seed(s), the *exact pipeline invocation* that
  produced the promoted artifact), and the post-pipeline eval JSON. The
  commit message states the slug, the win axis, and the within-batch
  delta. A cold checkout of that commit + the recorded invocation must
  regenerate the baseline. Shipping stays human-gated — promotion sets the
  research baseline, it does not auto-ship.
- **Re-pin the baseline.** The post-pipeline eval becomes the new control:
  store its `results-<ts>.json` as `baseline/current.json` + its
  `provenance.json`; record the new absolute numbers in `RESEARCH_LOG.md`
  *for reference only* — future comparisons stay within-batch against this
  new control.
- **Update the per-axis progress curve** (chain of confirmed within-batch
  deltas through promotions).

A loss, inconclusive, an unconfirmed win, a pipeline run with any `errored`
step, or a "headline-only" win that fails the overall-better-metrics check
all → **no promotion**: the change is reverted (config restored / pipeline
bins pruned per §9), and the artifact directory + log entry are kept.

---

## 8. The loop

```
while not should_stop():                       # §10 stop conditions
    idea  = pick_max_EV(open_ideas)            # §7.1
    slug  = f"{date}-{idea.slug}"
    mkdir RESEARCH_OUT/<slug>
    write_hypothesis(slug, idea)               # prior, mechanism, prediction, EV, runner-up

    branch research/<slug>
    apply_change(idea)                         # edit uc_v11.yaml, or run the pipeline
    save_diff(slug)

    preflight(slug, idea)                      # §8.1 — MANDATORY gate before any long run
    metrics = run_eval(slug)                   # subagent §12; one eval call,
    #   candidate + control as tests, frozen corpus; returns results-<ts>.json
    #   seed-sensitive idea => K-seed medians (§4.2)
    #   launch is liveness-checked, not fire-and-forget (§8.1)

    decision = decide(slug, metrics)           # §5
    posterior = bayes_update(idea, metrics)    # §7.2, written explicitly

    if decision.startswith("win"):
        confirmed = confirm(slug)              # re-run / K-seed (§5)
        clean = confirmed and run_full_pipeline_clean(slug)   # §7.3 gate 2
        overall_ok = clean and overall_better_metrics(slug)   # §7.3 gate 3
        if confirmed and clean and overall_ok:
            promote_baseline(slug, metrics)    # §7.3 + reproducible commit
        else:
            decision = "inconclusive"          # win not promotable (yet)
    if not decision.startswith("win") or decision == "inconclusive":
        revert_change(slug)                    # bins pruned per §9

    append_log(slug, idea, prior, metrics, posterior, decision)  # RESEARCH_LOG.md
    rerank_and_curate_bank(metrics)            # update correlated, add/kill ideas
```

One experiment in flight at a time (the eval already saturates all GPUs;
parallel experiments would contend and break the within-batch invariant).

### 8.1 Pre-flight (mandatory before every experiment)

A long run that fails in the first 10 seconds but is only discovered an
hour later (timeout) is the single most expensive mistake in this loop.
Before committing to *any* eval or pipeline run, do a final review — this
gate is not optional, and it is itself logged in the experiment entry.

1. **Re-check the code & setup against the active path.** Grep the
   *actually-executed* code/config path for the knob or function the
   change touches and confirm it is live and spelled right — yaml keys
   silently ignored by dead code paths have burned us before
   (`feedback_verify_config_knobs.md`). Confirm every input exists and
   resolves: frozen eval yaml parses; corpus paths, NN bins, TRT engines
   present; control config is the *current* baseline; artifact dir created
   with `results_location` / `TMPDIR` / logs redirected into it (§9);
   enough free disk + GPUs idle.

2. **Reduce execution time — omit redundant steps.** Estimate wall-clock
   before launching (`feedback_time_is_primary.md`,
   `feedback_preflight_long_runs.md`) and ask what can be skipped without
   changing the result:
   - config-only change ⇒ **no retrain, no C rebuild** — just the eval;
   - only the metric/analysis changed, not the tracker ⇒ reuse cached
     ubtrk2 / pair-log (`--no-regen`); don't regenerate the corpus;
   - candidate **and** control in **one** eval call (shared dataset load),
     never sequential passes;
   - `num_workers: auto` (all GPUs); `runs=1` unless two candidates differ
     by < σ (`feedback_eval_runs_default.md`);
   - rebuild `ubon_cstuff` only if a C source actually changed; reuse the
     existing engine/bins otherwise;
   - DAgger: **1 iteration by default** (iters 2/3 regress — §4 evidence);
     multi-iter only for the explicit investigation idea.
   If a cheaper equivalent measurement exists, take it and record why.

3. **Fail fast, don't fire-and-forget.** After launch, verify within
   ~1–2 min that it is *actually progressing* — worker processes spawned,
   GPUs busy, the progress bar advancing, no immediate traceback / NN-load
   abort / config reject — before walking away. Never sleep on a long
   timeout hoping it worked.

4. **Smoke-test when risk is non-trivial.** If the change touches a new
   or rarely-exercised code path, a schema, the first use of a knob, new
   C code, or anything where a silent-wrong or late failure is plausible:
   run a **tiny** smoke first (1–2 clips, or one short pipeline step) and
   confirm it produces a sane, well-formed result *before* committing to
   the full frozen-corpus run or the full pipeline. The minutes spent
   here are cheap against an hour-long dead run.

Only when 1–4 pass does the experiment proceed to the full run. The
preflight outcome (what was checked, what was skipped to save time, smoke
result if any) goes in `decision.md`.

---

## 9. Reproducibility & artifact containment

**Containment is a hard invariant.** One experiment ⇒ one directory:
`RESEARCH_OUT/<YYYYMMDD>-<slug>/`. *Every* byte the experiment generates
lands under it and nowhere else. The repo working tree and the rest of the
filesystem stay clean. Concretely, before any work the experiment sets:

- the eval yaml's `results_location` → `<slug>/eval/`
- any pipeline `--out` / data dir (`DDIR`, `EVAL_OUTDIR`, pair-log
  `output_root`, state-corpus prefix, retrained `.pt`/`.bin`) →
  `<slug>/pipeline/`
- `TMPDIR` → `<slug>/tmp/` (so library temp files don't litter `/tmp`)
- all stdout/stderr logs → `<slug>/logs/`

Then it asserts containment: after the run, *nothing* outside `<slug>/`
changed except the intended git-tracked config/code diff (which is also
saved as `<slug>/change.patch`). A stray write outside the folder ⇒ the
experiment is `errored` and re-run (§5). Generated heavyweight artifacts
(retrained bins, corpora) on a **loss** are deleted after the decision is
written; `decision.md`, `provenance.json`, the eval JSON, and the patch
are kept forever.

Per-experiment layout, `RESEARCH_OUT/<YYYYMMDD>-<slug>/`:

```
hypothesis.md      # prior, mechanism, falsifiable prediction, primary_axis,
                   #   EV vs runner-up
change.patch       # exact uc_v11.yaml / code diff (git diff)
provenance.json    # git SHA (track + stuff + ubon_cstuff), engine + bin
                   #   versions/sha256, seed(s), eval yaml sha256, hostname,
                   #   UTC, and the pinned speed conditions (GPU count,
                   #   num_workers) if speed is the axis
eval/              # results-<ts>.json + .txt sidecars (kept)
pipeline/          # retrain intermediates (.pt/.bin/corpora) — pruned on loss
logs/              # all captured stdout/stderr
tmp/               # TMPDIR for this experiment — wiped after decision
decision.md        # win-axis + the explicit Bayesian update:
                   #   μ0,τ0 → Δfit,σ_obs → μ1,τ1 → P(win); per-family /
                   #   fp_tracks check; idf1/speed/simplicity deltas if a
                   #   secondary axis was targeted; and what it implies
```

`RESEARCH_OUT/` itself lives outside the git tree (or is git-ignored) so
experiment bulk never pollutes commits; only confirmed-win config/code
changes are committed, on their `research/<slug>` branch.

Reproducibility preconditions, asserted at bootstrap and never assumed:

- The eval is deterministic up to the documented ±σ. Verified by running
  the control twice at bootstrap and recording the observed σ.
- NN bins carry their `_artefact_meta` `META` trailer; the pipeline already
  refuses silent-discard. The eval refuses silent NN-load failure.
- `provenance.json` must let a cold reader rebuild the exact tracker:
  config diff + the three repo SHAs + the engine/bin sha256s + seeds.

Never edit a `<slug>/` directory after `decision.md` is written. Never
rewrite git history. Never silently retune the thresholds — a threshold
change is itself a logged event in `RESEARCH_LOG.md`.

---

## 10. Stop conditions

The loop halts cleanly at the end of the current experiment when any of:

- `RESEARCH_OUT/STOP` exists.
- `K_loss` consecutive losses (default 8) — the search is in a dead region;
  pause and request user direction rather than thrash.
- The bank has no `open` idea with `EV > ev_floor` (default 0.0005 fitness
  per cheap-unit) **and** the agent, after a web-research + mechanism pass,
  declines to add a credible new one.
- Two consecutive `errored` from the same suspected root cause.
- A promotion would require touching an out-of-scope lever (§2) — surface
  to the user instead.

On halt, write `RESEARCH_OUT/summary.md`: confirmed wins, the current
baseline (absolute, for reference) and its `provenance.json`, the chained
within-batch fitness gain, and the live state of the bank.

---

## 11. Bootstrapping (once, before the loop)

1. Confirm the unified evaluator runs and emits the JSON sidecar on this
   machine (`python track.py --eval <frozen yaml>`); record wall time.
2. **Measure σ.** Run the *current shipped config* through the corpus twice
   in one batch (two identical `tests:`); record per-metric spread and the
   observed fitness σ in `RESEARCH_OUT/baseline/noise.json`. Use
   `max(measured σ, 0.003)` as σ in §5.
3. Record the **baseline** (current `uc_v11.yaml`) eval as
   `RESEARCH_OUT/baseline/current.json` + a `provenance.json`. This is the
   first control.
4. Confirm a seed-sensitive retrain reproduces within `σ_seed` by training
   the state head at two seeds (sanity that K-seed is actually needed and
   the harness is deterministic otherwise). Record in `baseline/`.
5. Seed `RESEARCH_LOG.md`: thresholds front-matter (defaults from §5), the
   idea bank (each entry with all §6 fields), an empty experiment log with
   the schema header.
6. Begin the loop. No approval pause.

---

## 12. Agent architecture

**Main agent** owns the reasoning and the history: picking the max-EV idea,
writing the hypothesis, editing config / launching the pipeline, reading the
subagent's compact JSON, doing the explicit Bayesian update, deciding,
git/promotion, and curating `RESEARCH_LOG.md`. Its context stays full of
hypotheses and posteriors, not eval logs.

**Subagent** (one per experiment, fresh, `Agent` tool, `general-purpose`)
owns the run: launch `track.py --eval` on the frozen yaml (or a pipeline
iter for retrains), wait, and return a single compact JSON blob — the path
to `results-<ts>.json` plus the headline `overall` numbers for candidate and
control, per-family deltas, `fp_tracks`, and any error/abort detail. The
main agent never tails eval/pipeline output; that is the subagent's job.
This split keeps the main agent's context on the science.

Web research is encouraged *between* experiments to source mechanisms for
new ideas (tracking-by-detection literature, association/NN-cost papers,
Kalman/ReID tuning). A new idea must still carry a mechanism and a
falsifiable prediction, with the reference cited in its bank entry.

---

## 13. Locked-in settings

| Setting | Value |
|---|---|
| Objectives (lexicographic) | fitness ≫ IDF1@≈fit ≫ speed@≈fit ≫ simplicity@≈fit; fitness never bought |
| Headline metric | `fitness` (`track_test.fitness_score`), `__ovr` rollup |
| Guards | `mota`, `idf1`, `fp_tracks`, per-family fitness |
| Artifact rule | one folder per experiment (`RESEARCH_OUT/<slug>/`); zero scatter; outside-write ⇒ errored |
| Pre-flight | mandatory §8.1 gate: re-check active path/setup, cut redundant steps, liveness-check the launch, smoke-test if risky |
| Evaluator | single: `python track.py --eval <frozen yaml>` (GPU-sharded) |
| Corpus | frozen `eval_ship_baseline.yaml` (full176 + jaad_val) |
| Comparison | within-batch only (candidate vs control in one eval) |
| Eval noise σ | measured at bootstrap, floor 0.003 fitness |
| Seed policy | K=3 median (escalate 5) for any training step |
| Win bar | Δfit ≥ +2σ, guards hold, confirmed before promotion |
| Promotion | baseline re-pinned by re-measurement; research branch; ship human-gated |
| Parallelism | one experiment in flight (eval saturates all GPUs) |
| Compute budget | none (run to a §10 stop condition) |
| Out of scope | detector engine/weights, corpus, fitness formula, eval settings |

---

## 14. Why this is "a good Bayesian", concretely

- Beliefs are **explicit and numeric** (`prior_p_win`, `prior_effect`), not
  adjectives.
- Evidence updates are **mechanical and written** (the §7.2 Normal update,
  logged each time) — not "this feels promising now".
- The decision rule is **calibrated to the measured noise**, and the three
  domain hazards (eval noise, seed luck, corpus drift) each have a specific
  defense, so a number is only treated as evidence once it clears them.
- **Falsification is first-class**: a contradicted prediction kills a
  mechanism and demotes its siblings; you cannot quietly keep a win that
  happened for a reason you predicted wrong.
- Progress is **a chain of confirmed within-batch deltas**, immune to corpus
  drift, so "steady forward progress" is a measurable invariant, not a hope.
