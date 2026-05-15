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
- **Steady forward progress.** The baseline only ever moves up, and only
  on a *confirmed* win. The baseline is a re-measured number on the frozen
  corpus, never a remembered one.

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

---

## 5. Decision rule

Every experiment produces exactly one of: **win / loss / inconclusive /
errored**. Thresholds live at the top of `RESEARCH_LOG.md` so they can be
retuned without editing this doc; defaults below.

Let `Δfit = fitness(candidate) − fitness(control_same_batch)`, both from the
same eval batch (or, for seed-sensitive changes, `Δfit` of the per-seed
medians).

| Outcome | Condition |
|---|---|
| **win** | `Δfit ≥ +2σ` (default +0.006) **and** no guard regresses past its gate (§2) **and** no per-family fitness regresses by more than `ε` (default 0.010) **and**, if seed-sensitive, this is the K-seed median, not a single seed. |
| **loss** | `Δfit ≤ +Δmin` (default +0.001) **or** any guard regresses past its gate **or** any family regresses by more than `ε`. |
| **inconclusive** | `+Δmin < Δfit < +2σ` (clears the floor but not the noise gate) **or** a seed-sensitive change measured at a single seed. The loop logs it and moves on; if it is the best available lead it is escalated to a confirmation run / K-seed before any promotion. |
| **errored** | Eval crashed, NN bin failed to load (the evaluator aborts itself), pipeline step failed, or the config was rejected. Logs preserved; the loop continues. |

**Confirmation before promotion.** A win is *candidate-confirmed* only after:
- a re-run in a fresh eval batch reproduces `Δfit ≥ +2σ` (config-only
  changes), **or**
- the K-seed median reproduces `Δfit ≥ +2σ` (seed-sensitive changes).

Only a candidate-confirmed win promotes the baseline (§7).

Defaults (mirror these into `RESEARCH_LOG.md` front-matter):
`sigma = 0.003`, `win_sigmas = 2.0`, `delta_min = 0.001`,
`epsilon_family = 0.010`, `mota_guard = -0.004`, `idf1_guard = -0.005`,
`fp_tracks_guard = +3`, `K_seed = 3`.

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

On a **candidate-confirmed win** (§5):

1. **Config-only win** — apply the change to `uc_v11.yaml` (or flip its
   pointer), commit on a branch `research/<date>-<slug>` with the slug and
   the within-batch `Δfit`, merge to the research branch. The shipping path
   stays human-gated (a confirmed win is a *promotion to baseline*, not an
   automatic ship).
2. **NN win** — copy the new bin to `/mldata/config/track/trackers/` under a
   bumped versioned name, flip the `uc_v11.yaml` pointer, commit as above.
   Keep the source `.pt` + its `_artefact_meta` trailer.
3. **Re-pin the baseline.** Run one more eval where the *new* config is the
   control, store its `results-<ts>.json` as `baseline/current.json`, and
   record the new absolute number in `RESEARCH_LOG.md` *for reference only* —
   future comparisons are still within-batch against this new control.
4. Update the progress curve (chain of within-batch deltas through
   promotions).

A loss or inconclusive reverts the change (config restored / bin discarded);
the artifact directory and the log entry are kept.

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

    # one eval call, candidate + control as two tests, frozen corpus
    metrics = run_eval(slug)                   # subagent §12; returns results-<ts>.json
    #   seed-sensitive idea => K-seed medians (§4.2)

    decision = decide(slug, metrics)           # §5
    posterior = bayes_update(idea, metrics)    # §7.2, written explicitly

    if decision == "win":
        confirmed = confirm(slug)              # re-run / K-seed (§5)
        if confirmed:
            promote_baseline(slug, metrics)    # §7.3
        else:
            decision = "inconclusive"
    if decision != "win":
        revert_change(slug)

    append_log(slug, idea, prior, metrics, posterior, decision)  # RESEARCH_LOG.md
    rerank_and_curate_bank(metrics)            # update correlated, add/kill ideas
```

One experiment in flight at a time (the eval already saturates all GPUs;
parallel experiments would contend and break the within-batch invariant).

---

## 9. Reproducibility & artifacts

Per experiment, under `RESEARCH_OUT/<YYYYMMDD>-<slug>/`:

```
hypothesis.md      # prior, mechanism, falsifiable prediction, EV vs runner-up
change.patch       # exact uc_v11.yaml / code diff (git diff)
provenance.json    # git SHA (track + stuff + ubon_cstuff), engine + bin
                   #   versions/sha256, seed(s), eval yaml sha256, hostname, UTC
eval/results-<ts>.json   # the machine sidecar from track.py --eval (kept)
eval/results-<ts>.txt    # human report (kept)
decision.md        # win/loss/inconclusive + the explicit Bayesian update:
                   #   μ0,τ0 → Δfit,σ_obs → μ1,τ1 → P(win), and the
                   #   per-family / fp_tracks check, and what it implies
```

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
| Headline metric | `fitness` (`track_test.fitness_score`), `__ovr` rollup |
| Guards | `mota`, `idf1`, `fp_tracks`, per-family fitness |
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
