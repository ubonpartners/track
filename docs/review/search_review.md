# Search code + algorithm review

Written 2026-07-23, against `src/track_search.py` @ f529a69 (post
multi-class/§15 landing) and the v11 / v11_mc search yamls. Every number
below is measured, not estimated (GT boxes sampled at the ~10 fps eval
rate — the actual mota denominator).

## Verdict in one paragraph

The coordinate-descent search is sound and appropriately simple for a
deterministic, expensive objective — but it is optimizing a **badly
weighted objective**: `_overall` is box-count-weighted, so a handful of
crowd clips own it (MOT20-05 is **75.6% of v11's validation signal** by
itself; MEVA is **62–70% of v11_mc's**, while bwc — the stated tuning
target — is **2%**). Fixing the objective weighting is worth more than
any algorithmic improvement. Second-order wins: parallelising the
up/down probes, per-param convergence, a val-gated best checkpoint, and
a crash-resume journal. The text log is good; a self-contained HTML
report from the existing JSON sidecar is cheap and worth it.

## 1. Are the metrics favouring MOT20-05 too much? YES — and it's worse than that

`_overall` mota/idf1 are recomputed from **summed counts**, so every
clip's weight is its GT box count at eval instants. Measured shares of
the mota denominator:

**v11** (include_families active):

| split | dominant clips | share |
|---|---|---|
| train | MOT20-03 26.7%, MOT20-02 15.2% (mot20 total ≈ 42%) | 533k boxes total |
| val | **MOT20-05 alone 75.6%** | 397k boxes total |

Every validate line in every v11 search log has effectively been "how
did MOT20-05 feel about it". A parameter change that wins +2 MOTA on
crowds and loses everywhere else validates as a win.

**v11_mc** makes this worse, not better — the new sets shift the mass to
MEVA schoolyards:

| split | family shares of mota denominator |
|---|---|
| train | meva **61.6%**, mot20 13.9%, pp22 9.6%, … **bwc 2.0%**, bdd 0.7% |
| val | meva **70.2%**, mot20 (=MOT20-05) 18.8%, … **bwc 2.1%**, bdd 0.6% |

Three single MEVA clips (G330/G299 pairs) are >30% of each split. A
search on v11_mc as-is would tune the tracker for distant schoolyard
crowds and be statistically indifferent to bodycam and dashcam — the
exact opposite of the goal.

**Component incoherence, same root cause.** fitness =
`mota − 0.35·(Σhonest/Σduration) − 0.002·(Σfp/Σframes)` mixes three
DIFFERENT implicit weightings on the aggregate row: mota is box-weighted
(meva 62%), the honest-FP rate is duration-weighted (meva 30%, pp22 24%,
otw 15%), fp_per_frame is frame-weighted. The same behavioural change
moves each component through a different dataset lens.

**Recommendation (STATUS 2026-07-23: items 1 and 3 IMPLEMENTED — the
`_groupmean` row + `group_weights` and `clip_weight_cap_pctl` are live
and wired into track_search_v11_mc.yaml; item 2's baseline run is
deferred by choice — protect stays commented until someone spends the
eval time):**

1. Add a **balanced objective row**: `_groupmean` = unweighted (or
   `group_weights:`-weighted) mean of the per-group `__ovr<group>`
   fitness_multi values, and point `result_dataset_opt_key` at it. The
   `__ovr` rows already exist; this is ~15 lines in `display_results`.
   Micro-average within a group (keeps clip volume meaningful inside a
   domain), macro-average across groups (no domain buys weight with
   density). With v11_mc's groups that gives cctv/pp22/dashcam/bwc/
   meva/otw/movie roughly equal say, adjustable by `group_weights`.
2. Keep `protect:` as the hard floor regardless — a balanced mean can
   still trade cctv down.
3. Optional refinement: cap any single CLIP's share within its group
   (e.g. weight = min(boxes, P95)) so MOT20-05 doesn't own the cctv
   group either. Cheaper alternative: leave MOT20-05 out of val and
   spread mot20 across splits — but the cap fixes the class of problem.

## 2. Can the search algorithm be improved?

What it is today: cyclic coordinate descent; probe ±(mult·step) on one
param per iteration; 3 successive improvements → move to next param; a
full quiet cycle halves the multiplier; stop below `final_mult`.
Deterministic objective, exact-vector memoisation, train/val split with
periodic validate logging.

That shape is defensible: ~20 dims, expensive evals (minutes each),
deterministic — gradient-free coordinate methods are a reasonable
choice, and its trajectory is human-readable (a real virtue given how
much these logs get eyeballed). Improvements in value order
(STATUS 2026-07-23: 1, 4, 5 IMPLEMENTED; 2 retracted; 3 downgraded —
the caching insight applies here too: a retry probe at an unchanged
vec_best is a cache hit, so the annealing tail only pays after real
movement or a multiplier change; 6 half-covered by 1; 7 unchanged):

1. **Parallelise the probes.** `score_up` and `score_down` are two
   sequential full evals, each of which parallelises over clips but
   leaves the pool idle between candidates. Evaluate both candidates in
   ONE `track_test` call (tests = {up, down} against the same dataset
   set — the cartesian machinery already exists). Straight ~2×
   wall-clock. Batching the next K params' probes into one call goes
   further (up to ~2K× when clips ≫ workers is false, i.e. small
   dataset runs), at the cost of stale-best probing — standard
   block-coordinate descent, still convergent for this use.
2. ~~Direction memory~~ — RETRACTED (2026-07-23, MB): on a monotone
   stretch the backward probe is exactly the PREVIOUS vec_best, which is
   already in `all_results` — the cache returns it for free. Direction
   memory would only save the down-probe immediately after a multiplier
   change or a param switch, which is noise. No action.
3. **Per-param convergence.** One global multiplier means a converged
   param keeps costing 2 evals per cycle until EVERY param is quiet:
   with 20+ params the annealing tail is the most expensive phase.
   Track per-param multipliers; retire params whose step has annealed
   below their yaml `step`; finish when all are retired.
4. **Val-gated checkpoint.** Validation is currently logged but never
   used. Keep `best_by_val` alongside `best_by_train` and REPORT both
   at the end — with MOT20-05 fixed (§1) val becomes meaningful, and
   train/val divergence is the overfitting alarm for free.
5. **Resume journal.** `all_results` lives in memory; a crash 300 evals
   in loses everything. Append (vec, split, score) JSONL per eval;
   preload on start. Trivial and transformative for multi-day runs.
6. **Engine reload amortisation.** Every candidate eval spins a fresh
   mp workqueue → every worker reloads TRT engines per candidate.
   Persistent workers across `search_test` calls (pool owned by
   search_track, work items = (candidate, clip)) removes a fixed
   ~seconds×workers tax per iteration. Meaningful once probes are
   batched (item 1 subsumes half of it).
7. **Algorithm swap? Not yet.** CMA-ES / TPE (Optuna) handle correlated
   params better than coordinate descent (our thresholds ARE
   correlated: track_* vs match_* vs kf_*), typically 2–5× fewer evals
   to comparable optima in this dim range. But they cost trajectory
   readability and reproducibility-by-eye. Verdict: do items 1–5 first;
   if searches still take days, run ONE Optuna A/B on a frozen dataset
   set before committing. The plumbing (`search_test` as a pure
   f(vec) → score) is already the right interface for it.

## 3. Code review (src/track_search.py)

STATUS 2026-07-23: everything below is FIXED except the SearchParam
dataclass refactor (parallel lists kept — the loop now has behavioural
tests, which was the risk the dataclass was meant to reduce).

- **Dead duplicate imports** at file bottom (`import copy / stuff /
  track_test / datetime` after `search_track`) — delete.
- **In-place base-config mutation**: `search_test` loads
  `c["config"]` once then mutates it for every candidate. Works because
  every search param is re-set every eval, but any key OUTSIDE
  `search_params` that a candidate ever touches (protect rejects leave
  variant blocks behind, §15 create-on-write) persists into subsequent
  candidates. Deep-copy the loaded base per eval — the cost is nil next
  to an eval.
- **Validation evals are never memoised** (`is_train` gates the cache):
  the same `vec_best` re-runs the full val set on every validate tick
  even when unchanged since the last one. Key the cache by
  (split, vec).
- **Out-of-range rejects aren't cached** (early return) — harmless
  (no eval behind them) but inconsistent with the protect-reject path,
  which now caches. Unify.
- `c = config["tests"][result_test_opt_key]` re-fetched inside the
  param loop — cosmetic.
- **`search_track` is one 190-line function** — state (names/initial/
  step/min/max/is_int as parallel lists) begs for a small SearchParam
  dataclass; would have prevented the index-juggling class of bug.
- Rounding: `_normalise_param_value` rounds floats to 3 dp — any future
  param with step < 0.001 silently collapses. Assert step ≥ 0.001 at
  load, or derive rounding from the step.
- Tests: §15 machinery is covered (test_search_params.py); the LOOP
  (annealing, direction, termination) has none. A fake-objective test
  (quadratic bowl over 3 params, monkeypatched track_test) would pin
  the algorithm's contract cheaply.

## 4. Logs and reporting (the HTML question)

The text search log is genuinely good (validate blocks, cumulative
per-param improvement attribution). Gaps: nothing machine-readable per
iteration, no per-group visibility during a search (you can't see "bwc
got better, cctv flat" without rerunning), no visual trajectory.

Recommended, in cost order (STATUS 2026-07-23: ALL IMPLEMENTED —
search_journal_*.jsonl doubles as resume_from state; validate blocks log
per-group levels + deltas; search_report_*.html regenerates each
validate; eval runs emit sortable results-*.html; colorize now shows
negatives in red instead of blanking them):

1. **JSONL search journal** (doubles as the resume journal, §2.5): one
   line per eval — {iter, param, direction, vec, split, score, and the
   per-group fitness values from the rollups}. The text log stays; this
   is for machines.
2. **Per-group deltas in the validate block**: one extra line per group
   (`bwc +0.012, cctv −0.001, …` vs previous validate) — the single
   highest-value legibility improvement, ~10 lines, no new deps.
3. **Self-contained HTML report** `search-<ts>.html` generated at end
   (and rewritten at each validate, so it's live during a run): vanilla
   JS + inline data, no server —
   - score-vs-iteration chart (train + val), per-group score traces;
   - per-param panel: value trajectory + cumulative improvement share
     (the data the text log already attributes);
   - final results table = the existing JSON sidecar, sortable,
     with per-clip drill-down and the worst-regressing clips ranked.
   The eval side should emit the same table for `--eval` runs
   (`results-<ts>.html` beside the .txt/.json).
4. **Fix the invisible negatives** while at it: `stuff.datatable.
   colorize` blanks every cell < 0.01 — zeros AND honest negative
   MOTAs render as empty cells (a bwc clip at mota −0.109 shows a
   blank, which reads as "no data" rather than "bad"). Show negatives
   in red instead of hiding them; keep blank-for-zero if you like it.

## 5. Other observations

- **Record the protect baseline before the first mc search** (one
  eval_track of production config over v11_mc train+val); until then
  `protect` is commented and cctv is only defended by the balanced
  objective.
- **match_iou is global** (0.45). Vehicles are large and rigid; 0.45 is
  loose for them and standard practice is 0.5–0.7 — when vehicle
  numbers start mattering, make match_iou per-class (plumbing exists:
  the per-class compute_metrics call).
- ~~`max_age_days` / `regen_tests` / `regen_datasets`~~ — confirmed
  unconsumed, REMOVED from both yamls (2026-07-23). Per-dataset
  `regenerate:` remains and works. autolabel's `eval/score_vehicles.py`
  deleted (plan P2 item 8 — `mota_vehicle` replaces it).
- The mc yaml carries `include_families` listing every family — it is
  now a no-op allow-everything list; keep it only if the subset
  workflow (comment a family out) is actually used.
- `num_workers: auto` resolves from GPU count; with batched candidates
  (§2.1) the right worker count becomes candidates×clips-aware —
  revisit then.


## 6. Post-review rationalisation (2026-07-23, MB direction)

The family/group/stream_hint triple had grown seven groups and an
implicit hint mapping — too much taxonomy. Rationalised in
track_search_v11_mc.yaml (full explanation in its header):

- **group is now the ONE behavioural axis that matters**: `static`
  (fixed mounts), `moving` (ego-motion: bodycam + dashcam + handheld
  pp22), `movie` (cut scenes, visibility only). _groupmean therefore
  votes static/moving/movie equally — the old 7-group mean gave
  static-style content 4 of 7 votes.
- **one hint value**: every `moving` clip carries `stream_hint:
  bodycam`, read as "the moving-camera profile" (the vocabulary
  operators already have). Static = unhinted default. A movie profile
  only appears if cut handling ever needs its own parameters.
- family stays as pure provenance/reporting.
