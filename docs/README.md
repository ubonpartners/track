# Docs index

Start with the user guides. Specs are the contracts other repos rely on.
Plans record design decisions and their execution. Research is dated
investigation logs, kept for the numbers; read the status line at the
top of each before trusting a figure.

## user_guides/ — how to do things

- `import_and_annotation.md` — the three data tiers, what derive does, the annotation json format, importing a new dataset step by step, autolabel for unlabelled footage.
- `optimization_flow.md` — tuning tracker parameters with `--search`: the objective, the algorithm, running and watching a search, reading the result, applying it via `--eval` and `eval_compare`.
- `capacity_curve.md` — remaking the quality-vs-concurrent-streams chart: the x86 quality grid, the Jetson performance-mode sweep, the join and plot.

## specs/ — contracts

- `data_tiers_and_corpus_registry.md` — tier 0/1/2 layout, MANIFEST.json and the capability registry shared with the autolabel repo, tier-2 derivation rules.
- `TRACKER_DEBUG.md` — the UBTRK2 tracker-result and debug format shared by track, ubon_cstuff and stuff.

## plans/ — designs and their execution records

- `repo_cleanup.md` — staged restructuring of this repo with per-stage adversarial review (2026-09-06).
- `multi_class_and_hints.md` — multi-class scoring, clip-type hints, the v11_mc search set and path-addressed search parameters (implemented 2026-07-23).
- `cadence_test.md` — the cadence test set for measuring tracking quality against analytics cadence (2026-07-28).
- `autolabel.md` — design notes for auto-labelling arbitrary mp4s into eval GT.

## review/

- `search_review.md` — review of the search code and objective weighting (2026-07-23) and the rationalisation that followed.

## research/ — dated investigation logs

- `eval-revalidation-2026-08-02.md` — every optical-flow and CMC result re-measured on the real objective; the earlier numbers were on the wrong config.
- `sw-optical-flow.md` — a software dense optical flow as a stand-in for NVOF. Scores in it were measured on the wrong config; see the revalidation.
- `detection-free-frames.md` — the skip chain end to end: decoder cadence, PM controller, capacity tooling. Long.
- `cadence_test_results.md` — phase 1 results of the cadence test set.
