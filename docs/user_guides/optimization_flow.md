# Optimising tracker parameters with `--search`

How a tracker config gets tuned: what `track.py --search` does, how to
run it, how to read what it produces, and how to get the result back
into the production config without fooling yourself. The design
history is in `docs/review/search_review.md` and
`docs/plans/multi_class_and_hints.md`; this page is the operating
manual.

## The pieces

| thing | where | role |
|---|---|---|
| tracker config | `/mldata/config/track/trackers/uc_v11.yaml` | the production config being tuned. Shared by every deployed box. Never edit it without explicit approval. |
| search yaml | `/mldata/config/track/search/track_search_v11_mc.yaml` | THE objective: which clips, how they are weighted, which parameters move and by how much. There is exactly one. |
| results dir | `/mldata/results/track_v11_mc/` (`result_log_file_path` in the yaml) | per-run text log, jsonl journal and live HTML report |
| eval | `python -m src.cli eval` | runs the same yaml once, no search, for before/after comparisons |
| comparator | `python -m src.eval_compare` | prints two eval runs side by side |

Search and eval read the same yaml, so they cannot describe different
datasets. Do not copy the yaml to change a field. Every knob an eval
run needs is a command-line override.

## What the search optimises

Each candidate config is run over every clip in the train split. The
score is built up in layers:

1. **Per clip, per class**: `fitness = MOTA - 0.35 * honest_fp_tracks_per_second - 0.002 * fp_per_frame`.
   `fitness_multi` is the weighted sum over classes present in the
   clip, using `fitness_weights` (person 1.0, vehicle 0.3 today). A
   class with no GT in a clip contributes nothing rather than zero.
2. **Per group**: clips are tagged `group: static | moving | movie`
   (plus the cadence groups). Counts are summed within a group, with
   `clip_weight_cap_pctl` scaling down any clip whose GT volume is
   above the 90th percentile of its peers, so one crowd clip cannot
   own a group.
3. **Across groups**: `_groupmean` is the weighted mean of the group
   rollups using `group_weights`. Groups not listed weigh 1.0; movie
   and the cadence groups are down-weighted to 0.05 and 0.1.

The yaml names the objective explicitly:

```yaml
result_test_opt_key: search_config      # which test block
result_dataset_opt_key: _groupmean      # which rollup row
result_dataset_opt_param: fitness_multi # which column
```

`_groupmean` is the objective. `_overall` is box-count weighted and is
not. Quoting the wrong row has invalidated results three times, so
read these three lines before quoting any number.

The `tests.search_config` block carries the eval conditions: the
tracker config path, the scoring frame rate floor, the match IoU, the
minimum person height, and a `main_config_override` that switches off
faces, CLIP and audio and forces exact-shape detector batching. Exact
batching removes most of the run-to-run noise but is a different
operating point from production, so search and eval scores are only
comparable with each other, never with production numbers.

## Which parameters move

`search_params` lists each tunable with `min`, `max` and `step`:

```yaml
search_params:
  track_low_thr:
    min: 0.02
    max: 1.0
    step: 0.01
  kf_weight:
    min: 0.02
    max: 20
    step: 0.02
    split_hints: [bodycam]       # also tune kf_weight(hint:bodycam)
    split_classes: [vehicle]     # also tune kf_weight(class:vehicle)
```

- A bare name is found anywhere in the tracker config tree. A dotted
  path (`motiontrack.mad_delta`) is explicit.
- `split_hints` adds an independent `(hint:bodycam)` variant, so
  moving-camera clips can settle on a different value from static
  ones. `split_classes` adds a `(class:vehicle)` variant. A variant
  that does not yet exist in the config starts at the base value, so
  iteration zero behaves exactly like the unsplit config.
- The starting value comes from the tracker config. Add `initial:` to
  override it.
- Float values are rounded to three decimals, so `step` must be at
  least 0.001.
- `protect: [{group, param, floor}]` at the top level rejects any
  candidate whose group rollup drops below a floor. Use it to say
  "do not break static CCTV" as a hard constraint.

## The algorithm

Coordinate descent, one parameter at a time:

1. Score the starting vector on train and on val.
2. For the current parameter, try `value + mult*step` and
   `value - mult*step` in one eval pass. `mult` starts at
   `initial_mult` (4).
3. If either probe beats the best train score, keep it. After three
   improvements in a row on the same parameter, move on. Otherwise
   move to the next parameter.
4. After a full lap with no improvement, halve `mult`. Stop when it
   drops below `final_mult` (0.5).
5. Once at least four iterations have passed since the last validate
   and the best vector has changed, re-score it on the val split and
   log per-group deltas.

Everything scored is memoised by (split, vector), so bouncing off a
range limit or re-visiting a value costs nothing.

Cost: one iteration is two candidates over the train split, about
three minutes on this box with the current 500-clip set. A full run
over the current 30-odd parameters is roughly 500 iterations, or a
day. Plan for that. Only one heavy job at a time on this machine.

## Running it

```
python -m src.cli search /mldata/config/track/search/track_search_v11_mc.yaml
```

Before launching:

- Check nothing else heavy is running on the GPU.
- If you changed the yaml or a clip, run a one-clip smoke test first
  so a bad path or a broken annotation fails in a minute, not at
  iteration 40.
- Decide the split. `do_train_split: true` optimises on train and
  validates on val. Setting it false uses every clip for both, which
  gives no overfitting signal at all.

Optional overrides: `--pm N` (before the verb) sets the detector performance tier for
every eval stream (0 is full resolution). `--tracker-config <path>`
on `eval` substitutes a tracker config without editing the yaml.

To resume a crashed run, add `resume_from: <path to search_journal_*.jsonl>`
to the yaml. Every candidate already scored is loaded into the cache
and the search continues from the same starting vector.

## Watching it

Three files appear in the results dir, all stamped with the start
time:

- `search_log_<stamp>.txt`. One line per iteration: which parameter,
  which direction won, the new score. Validate blocks show the val
  score, per-group levels and deltas since the last validate, and the
  cumulative improvement attributed to each parameter. The log opens
  with registry warnings for corpora not approved for tuning, which
  is informational today.
- `search_report_<stamp>.html`. Self-contained, regenerated at every
  validate: train trace, val markers, per-group traces, best vector.
  Open it in a browser during the run.
- `search_journal_<stamp>.jsonl`. Every evaluation as one json line.
  This is what `resume_from` reads.

Progress is measured in iterations done and elapsed time, not
estimates.

## Reading the result

The log ends with two vectors: best by train and best by val. They
differ. Prefer the val vector unless a parameter moved in a way that
makes no physical sense, and look at the per-group levels before
trusting a headline gain. A gain on `_groupmean` that comes entirely
from movie clips while moving cameras got worse is not a gain.

Run-to-run spread on the objective is real, about 0.005 on the
groupmean row with adaptive batching and about 0.0003 with the exact
batching the yaml now forces. Any improvement inside that band is
noise. Do not "fix" the spread. Replicate instead.

## Getting the result into production

1. Do not edit `uc_v11.yaml` from a search log by reflex. Show the
   proposed values and get approval per change.
2. Establish the baseline first:

   ```
   python -m src.cli eval --split val --results-location /mldata/results/eval/before
   ```

3. Apply the change to the tracker config, then:

   ```
   python -m src.cli eval --split val --results-location /mldata/results/eval/after
   python -m src.eval_compare /mldata/results/eval/before /mldata/results/eval/after
   ```

   The comparator prints both objective rows, the group breakdown and
   the biggest per-clip movers. The first directory is the baseline
   for deltas.

4. For a tracker A/B that should not touch the production file, pass
   `--tracker-config <candidate.yaml>` to `eval` instead.

`eval` with no yaml path runs the objective config. Passing a path
is allowed for one-off probes and prints a loud warning that the
numbers are not comparable with search scores.

## Things that go wrong

- **Wrong row.** `_overall` is not the objective. Read
  `result_dataset_opt_key` in the yaml.
- **Family not included.** A dataset entry whose `family` is missing
  from `include_families` is skipped silently.
- **Knob not live.** A parameter name that matches nothing in the
  config raises at startup. A parameter that matches but is ignored by
  the tracker build does not. Prove a new knob changes behaviour with
  a log line or counter before spending a day searching it.
- **Regulated quantities.** Real-time capacity and GPU seconds per
  frame are pinned by the controller and are not A/B material. The
  capacity curve tooling in `docs/user_guides/capacity_curve.md` is the path for
  those.
- **Comparing with production numbers.** Search and eval use exact
  detector batching and disable faces, CLIP and audio. Production
  does not. Compare eval runs with eval runs only.
- **Two configs called canonical.** If you find yourself wanting a
  second search yaml with different weights, stop. That is how the
  three invalidated results happened.
