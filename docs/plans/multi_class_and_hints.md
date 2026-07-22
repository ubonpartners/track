# Multi-class tracking + clip-type hints: what to change in this repo

Written 2026-07-22. Context: utrack now emits **vehicle and animal**
tracks and supports **per-"clip type" parameter optimization**. Goal:
run `track.py --search` against the new datasets (bwc-videotext, BDD
vehicles, reduced MEVA/OTW) and optimize BWC/vehicle performance
**without regressing CCTV**. This is the audit of what stands in the way
and the recommended changes, in priority order.

## Where the repo is today (audit results)

| concern | current state | where |
|---|---|---|
| tracker output classes | `target_classes=["person","face"]` **hardcoded** — utrack's new vehicle/animal tracks are dropped at import | `src/trackset.py:528` (`import_create`) |
| metric classes | `compute_metrics(classes_to_test=["person"])` — single class per call; vehicle GT never scored | `src/track_test.py:243` |
| det-AP classes | `classes_for_det_map=["person","face"]` | `src/track_test.py:89` |
| fitness | single scalar: `mota − K·honest_fp_rate − 0.002·fp_per_frame`, person-only | `src/track_test.py:49` |
| search objective | one cell: `result_dataset_opt_key: _overall`, `result_dataset_opt_param: fitness` | `track_search_v11.yaml:14-15` |
| per-scene aggregation | datasets already take `group:`; report gets `__ovr<group>` rows | `src/track_test.py:842-843,585` |
| dataset→tracker params | only `path/split/group` consumed; other dataset keys do NOT reach the tracker config | `src/track_test.py:~800` |
| config overrides | `main_config_override` deep-merge exists (2026-07-22) at test level | `src/trackset.py` (import_create) |
| GT class scheme | `["person","vehicle","other"]`; vehicle GT exists in bdd100k_mot (dense), bwc-videotext + augmented MEVA/OTW (autolabelled); **no animal GT anywhere** | importers |

## P0 — enable multi-class end to end

### 1. Un-hardcode `target_classes` in `import_create`
`target_classes = param_dict.get("target_classes", ["person","face"])`
so a test entry (or `main_config_override`) can request
`["person","face","vehicle","animal"]`. Keep the default as-is for
back-compat. Map tracker class names → GT scheme at import:
person→person, vehicle→vehicle, animal→(see §4), face stays a sub-object
as today.

### 2. Per-class metrics in one result row
`compute_metrics` already accepts `classes_to_test`. Change
`track_test_work_fn` to loop `["person","vehicle"]` (skip a class when
the GT trackset has zero boxes of it) and emit **namespaced keys**:
`mota_vehicle`, `idf1_vehicle`, `fp_per_frame_vehicle`,
`num_misses_vehicle`, … while keeping today's un-suffixed person keys
untouched (nothing downstream breaks; person remains the default
reading). The `_overall`/`__ovr<group>` aggregator sums count keys
generically already — extend the derived-metric recompute block
(`idf1/mota/motp/...`) to also recompute each `_vehicle` suffix from its
own counts. Columns are already config-driven (`columns:` strings), so
reports pick up `mota_vehicle,vMOTA,{:6.3f}` with zero code.

Notes:
- the `c_mota` C metrics path is person-shaped; run vehicle scoring on
  the python path until/unless it matters for speed (per-clip vehicle GT
  volumes are modest except BDD).
- `classes_for_det_map` gets `vehicle` (and later `animal`) so
  `det_ap_vehicle` appears alongside `det_ap_person`.

### 3. Multi-class fitness (the search objective)
Add a config-driven combined fitness; keep `fitness` (person) unchanged
and add:

```yaml
fitness_weights:            # in the search/eval yaml, not code
  person: 1.0
  vehicle: 0.3              # start low; vehicle GT quality is mixed
```
`fitness_multi = Σ_c w_c · fitness_c` where `fitness_c` reuses the same
de-gamed form per class (honest-FP penalty computed per class — the
honest FP ruler in `track_test.py` needs the per-class hyp/gt geometry
split, which the class loop in §2 provides naturally). Search then sets
`result_dataset_opt_param: fitness_multi`.

### 4. Animal: output yes, scoring no (for now)
No dataset has animal GT, so animal tracks must not touch any metric
(they'd be free — or worse, matched against person GT by a bug). Extend
the GT class scheme to `["person","vehicle","other","animal"]` in
imported *outputs* only; scoring loops only over classes present in GT.
Flag: the cheapest path to animal GT later is the autolabel pipeline
gaining an animal channel and the augment bridge — out of scope here.

## P1 — clip-type hints

### 5. Dataset-level hint + params flow
One-line schema addition per dataset entry:

```yaml
datasets:
  bwc_video116: {path: ..., group: bwc, clip_type: bwc}
  bdd_00c12bd0: {path: ..., group: dashcam, clip_type: dashcam}
  MOT20-01:     {path: ..., group: cctv,   clip_type: cctv}
```
Change `track_test` to merge all *extra* dataset keys (anything not
path/split/group/regenerate) into `params`, so `clip_type` reaches
`import_create` → `param_dict` → tracker config. **Confirm the exact
key upyc/utrack expects for the hint** (this repo side is naming-agnostic;
whatever the C side reads, pass it through verbatim). Convention:
`group == clip_type` by default so the existing `__ovr<group>` rows give
per-type report lines for free.

### 6. Hint-scoped search parameters
`_set_nested_param` already addresses dotted paths into the tracker
config, so if utrack's per-hint parameters live under nested sections
(e.g. `hints.bwc.<param>`), today's `search_params` can already search
them — nothing to build, just name them in the yaml:

```yaml
search_params:
  hints.bwc.track_threshold: {initial: ..., step: ..., min: ..., max: ...}
```
This is the mechanism for "optimize BWC without touching CCTV": search
ONLY `hints.bwc.*` / `hints.dashcam.*` keys; base/cctv parameters stay
frozen at production values.

### 7. Regression guard for protected groups
Even with frozen base params, shared-code effects can leak. Add an
optional guard evaluated inside `search_test`:

```yaml
protect:
  - {group: cctv, param: fitness, floor: <recorded baseline − 0.005>}
```
If a candidate's `__ovrcctv` fitness drops below the floor, return the
`-10000` reject (same path as out-of-range params). Cheap, explicit, and
it turns "don't break CCTV too much" into a hard constraint instead of a
weight-tuning hope.

## P2 — cleanups and de-duplication

8. **`eval/score_vehicles.py` (autolabel repo)** becomes redundant once
   §2 lands — delete it and read `mota_vehicle` from the normal report.
9. **`summary_string`/`display_results`**: add the per-class fields
   only when present; keep person-first ordering (avoid 30-column soup —
   suggest one `columns:` preset per use case in the yaml).
10. **Search cache keys**: `all_results` caches by `param_vec` only —
    fine, but note the results-cache pickle (`results_cache_file`) rows
    predate multi-class keys; bump/segregate the cache file name when §2
    lands so stale single-class rows can't merge into new runs
    (`results_cache_file: ..._mc.pkl`).
11. **Autolabelled-GT hygiene**: bwc-videotext GT boxes carry autolabel
    confidences (<1.0) and per-track review flags in metadata. For search
    use: (a) score against them as-is to start; (b) if noise becomes the
    limiter, drop GT boxes with conf < 0.5 at load (one line in
    `objects_at_time` callers) or exclude review-flagged spans — the
    flags are already in `metadata.autolabel.review`. Don't build this
    until a search actually shows GT-noise sensitivity.
12. **`import_bdd_mot_sequence` fps note**: BDD GT is 5 fps keyframes on
    30 fps video — `eval_min_framerate: 9.9` in the current search yaml
    would evaluate at ~10 fps against interpolated GT; that's fine, but
    document that BDD rows are interpolation-scored (same as pp22).
13. **meva/otw eval enablement**: after the augment pass completes, the
    `diagnostic_only` exclusions for MEVA/OTW (autolabel repo
    `eval/sequences.py`) can be revisited — the augmented sets are
    "fully annotated" for person+vehicle. Keep them out of *frozen* eval
    sets; they're optimization fodder (that's their purpose here).

## Suggested first search (once P0+P1 land)

```yaml
tests:
  search_config:
    config: /mldata/config/track/trackers/uc_v11.yaml
    main_config_override: {faces: {embeddings_enabled: false, jpegs_enabled: false},
                           clip: {frame_embeddings_enabled: false,
                                  object_embeddings_enabled: false, jpegs_enabled: false},
                           audio: {events: {enabled: false}},
                           inference_config: {face: {enabled: false}, clip: {enabled: false},
                                              audio_event: {enabled: false}}}
datasets:   # ~25 clips per type keeps one search iteration < a few min
  # 15-20 bwc-videotext (group/clip_type bwc)
  # 15-20 bdd100k_mot   (group/clip_type dashcam)  <- vehicle-rich
  # existing cctv staples (group/clip_type cctv)   <- protected
fitness_weights: {person: 1.0, vehicle: 0.3}
result_dataset_opt_param: fitness_multi
protect: [{group: cctv, param: fitness, floor: TBD_baseline_minus_0.005}]
search_params:   # hint-scoped ONLY
  hints.bwc....
  hints.dashcam....
```

Baseline first: one `eval_track` run of the production config over that
dataset mix to record per-group fitness (fills in the `protect` floor and
gives the before/after table).

## Effort estimate
P0 ≈ a day (metrics loop + aggregation suffixes + fitness_multi + tests);
P1 ≈ half a day (params flow + guard; hint key name pending utrack-side
confirmation); P2 items are individually < 1 hour. Nothing here touches
the tracker itself — it's all metrics/plumbing in `track_test.py`,
`track_search.py`, `trackset.py:import_create`, and yaml.
