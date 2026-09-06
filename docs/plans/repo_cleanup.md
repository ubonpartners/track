# Repo cleanup plan

Status: PROPOSED 2026-09-06. A review of the track repo's structure with
a target layout and a staged order of operations. Nothing here changes
behaviour. Every stage ends with the unit tests, a fixed three-clip
smoke eval, and an adversarial review by someone other than the author
(section 3).

## 1. What the survey found

Numbers from the tree on 2026-09-06 (10.8k lines of Python).

**Five kinds of code share one flat `src/` directory.**

| kind | files | lines |
|---|---|---|
| core data model (trackset, track_util, trackers, upyc_tracker) | 4 | 1.5k |
| eval and search engine (track_test, track_search, eval_compare) | 3 | 2.7k |
| data pipeline (trackset_import, dataset_lite, corpus_manifest, import_antare, autolabel_bridge) | 5 | 3.9k |
| research and one-off tooling (cadence_*, capacity_*, quality_*, gpu_attrib, make_rt_configs, camera_motion, jetson_pm_sweep.sh) | 10 | 1.5k |
| tests (test_*.py) | 8 | 1.2k |

Nothing in the directory name or the file name says which kind a file
is. `track_test.py` is the eval engine, not a test, and sits beside
eight `test_*.py` files.

**Three god modules carry most of the churn.** In the last 60 days
`trackset_import.py` was touched by 26 commits, `track_test.py` by 22.

- `trackset_import.py` (1862 lines): nine per-format parsers as methods
  of a mixin, thirteen `convert_*` drivers, four ffmpeg helpers,
  dataset reduction, and three one-off migration fixes (`dofix`,
  `fix_cevo25_vfr_times`, `estimate_bdd_time_offsets`).
- `track_test.py` (1712 lines): box matching and convention-permissive
  IoU, MOT metrics and the honest-FP ruler, fitness, the multiprocess
  work queue, the shared-stream tracker runner, result tables, and the
  json and html writers.
- `trackset.py` (992 lines): storage format encode and decode, time
  interpolation, importer dispatch, and an OpenCV viewer.

**The class hierarchy is a namespace trick.** `TrackSet` inherits
`TrackSetImportersMixin` from `trackset_import`, which in turn imports
`trackset`. The mixin exists so that parsers can call `self.add_frame`.
Plain functions taking a `TrackSet` would do the same without the
cycle.

**Paths and environments are baked in.** Seventy-odd literal `/mldata`
paths across fifteen files, and `sys.path.insert` of absolute home
directories in `cadence_diag.py`, `cadence_test.py` and
`trackset_import.py`. The package is not installable, so every module
is run as `python -m src.x` and every import is prefixed `src.`.

**The CLI is two things.** `track.py` mixes dataset conversion flags
(`--mot`, `--jaad`, ...) with eval, search, view and track. Only six of
the thirteen converters are reachable from it; the rest are called from
`python -c`.

**Helpers are duplicated.** Four ffprobe wrappers (`dataset_lite.probe`,
`import_antare.probe`, `_native_fps` and `_video_codec` in
`trackset_import`, another in `cadence_test`), `scale_dims` twice, and
four ffmpeg transcode recipes with slightly different flags.

**Dead or orphaned code.** Unreferenced top-level functions:
`search_test`, `result_string`, `object_class_remap`, `dofix`,
`estimate_bdd_time_offsets`, `densify_sparse_gt` (its only producer was
retired 2026-09-06), and `allows` / `set_file_source` in the registry
(spec'd API, no caller yet). `convert_chirla`, `convert_roundabouthd`
and `convert_bwc_videotext` are legitimate one-shot importers with no
CLI route.

**Documentation lives in three places.** `docs/` is well organised
(specs, plans, research, review, user_guides) apart from
`capacity_curve.md` at its root. The README's layout table is current
except one dead link. The third place is inside the code: long
narrative comments recording dates, incidents and who asked for what.
`dataset_lite.py` is 20 percent comment lines and `track_test.py` 12
percent. The invariants in them are valuable. The history around them
belongs in a ledger, not next to the code that has to be read to be
changed.

**Tests are in good shape and in the wrong place.** 51 tests run in 3
seconds with no GPU or data dependency except one that skips without
the autolabel checkout. They live in `src/`, there is no `conftest.py`
or `pytest.ini`, no markers for GPU or data tests, and no CI.

## 2. Target layout

```
track/
  pyproject.toml              installable package `track`; console script `track`
  README.md                   short: what it is, install, the five commands, links into docs/
  src/                        (rename to track/ is a later decision, see rules below)
    paths.py                  the /mldata roots and tracker-config path, env-overridable
    core/
      trackset.py             TrackSet: storage format, frames, interpolation
      objects.py              Object, interpolate (today track_util)
      display.py              the viewer (today trackset.display_trackset)
    formats/                  one pure parser per source format, returning a TrackSet
      mot.py personpath22.py meva.py otw.py chirla.py roundabouthd.py
      uvg_vcm.py jaad.py bdd100k.py antare.py
    corpus/
      media.py                ffprobe/ffmpeg helpers: probe, probe_audio, scale_dims, transcode, remux
      manifest.py             tier-1 registry (today corpus_manifest minus derive)
      derive.py               tier-2 derivation and check (today dataset_lite + derive_tracking)
      importers.py            the convert_* drivers: tier 0 -> tier 1
      autolabel_bridge.py
    eval/
      matching.py             IoU, convention-permissive matching, ignore regions
      metrics.py              MOT metrics, honest FP, fitness, fitness_multi
      runner.py               work queue, shared-stream runner, packed results
      report.py               tables, rollups, _groupmean, json + html writers
      compare.py              (today eval_compare)
    search/
      params.py               path-addressed keys, variants, split sugar
      loop.py                 coordinate descent, journal, resume
      report.py               the live html
    tracker/
      upyc.py                 (today upyc_tracker) plus the factory in trackers.py
    cli.py                    subcommands: view, track, eval, search, compare,
                              import <corpus>, corpus build|derive|check
  tools/                      research tooling, not packaged, may import track
    cadence_test.py cadence_diag.py capacity_curve.py capacity_plot.py
    quality_grid.py quality_table.py gpu_attrib.py make_rt_configs.py
    camera_motion.py jetson_pm_sweep.sh
  tests/
    conftest.py               fixtures: tiny synthetic TrackSets, tmp corpora
    unit/                     the current 51, split to mirror the package
    data/                     tests that need /mldata (marked `data`)
    gpu/                      tests that need the tracker (marked `gpu`)
  docs/
    README.md                 index of what is where
    user_guides/ specs/ plans/ research/ review/
    ledger.md                 dated decisions and incidents, moved out of code comments
```

Rules that keep it this way:

- `core` imports nothing else in the package. `formats` imports `core`.
  `corpus` imports `core`, `formats`. `eval` imports `core`, `tracker`.
  `search` imports `eval` and `corpus` (the registry consultation in the
  tiers spec). `cli` imports everything. `tools` is outside the package.
  `tests/test_import_graph.py` asserts this graph.
- Every `/mldata` path goes through `paths.py`, in `tools/` and `tests/`
  too. `tests/test_no_literal_paths.py` scans code (not docstrings or
  comments) for `/mldata`, `/home/`, `~/` and `expanduser(`.
- A code comment states an invariant and links to the ledger entry for
  its history. Dates and names go in the ledger.
- The package stays `src` for now: a package called `track` cannot
  coexist with `track.py` at the repo root, and the rename is a
  separate decision once stage 6 has made `track.py` a shim.

## 3. Verification and review protocol

Every stage below ends the same way. Nothing moves to the next stage
until all four are done.

1. **Tests green.** `python -m pytest tests -q -m "not gpu and not data"`.
2. **Smoke eval.** Three clips through the objective config (one static,
   one moving, one whose source is 4K; all evaluated on their 1280x720
   tier-2 copies), through the same shared-stream runner the objective
   uses, about 15 seconds: `python tests/smoke_eval.py --out <dir>`.
   Then `python tests/smoke_eval.py --compare <previous stage dir> <dir>`
   must report every clip and rollup cell identical. Repeat runs of the
   same code were shown identical at stage 0 (exact-shape batching), so
   any change is a defect, not noise. The script builds its yaml from
   the objective at run time and records the sha256 of the objective
   and tracker yamls plus the git revision, so a config edit under the
   repo's feet shows up in the comparison rather than being blamed on
   the stage.
3. **Adversarial review.** A second reader (a fresh agent session or a
   colleague, never the author of the stage) is given the diff and the
   stage's checklist below, and told to find what is wrong or missing
   rather than to approve. The reviewer reports findings; the author
   fixes or rebuts each in writing; the stage is not closed while any
   finding is open. The reviewer works from the questions in section 4.
4. **Commit.** One commit per stage, message naming the stage and
   listing the rebutted findings so the reasoning is not lost.

The stage checklists are the minimum. The reviewer is expected to go
beyond them.

## 4. Adversarial review questions

The reviewer applies every question to every stage. They are phrased
as attacks.

Completeness

- Which callers of a moved or renamed symbol were missed? Prove it with
  `grep -rn` across `track.py`, `src/`, `tools/`, `tests/`, `docs/`, the
  yaml configs under `/mldata/config/track`, and shell scripts.
  `python -c` invocations recorded in docs and ledgers count as callers.
- Which module-level side effects moved with the code, or failed to?
  Look for import-time registration, `sys.path` edits, logging config,
  multiprocessing fork-before-import ordering (there is one in
  `track_test.py`, marked `noqa`).
- Which docs, README tables, docstrings and comments still name the old
  path or the old command?
- Which files were left behind empty, or as re-export shims that were
  supposed to be removed in this stage?

Correctness

- Does any moved function now resolve a different default argument,
  global, or module constant than before? Check every default that
  points at a path, a config, or another module's constant.
- Do relative imports inside moved code still resolve to the same
  object? Circular import broken in one direction can silently reorder
  initialisation.
- Did a data path change under a running process's feet: any file the
  eval scheduler caches by path (`.meta.json` sidecars), any absolute
  path in a results json, any `derive_recipe.json`?
- Does the smoke eval prove what it claims? Read the rows, not the
  summary line. Confirm the three clips actually exercise the changed
  code path (a change to the MOT parser is not tested by three antare
  clips).
- For anything deleted: name the last commit that used it and show
  that no yaml, script, doc or ledger entry still refers to it.

Reviewability

- Does the diff mix a move with an edit? Any stage that says "move"
  must produce a diff where `git diff -M` shows renames with near-zero
  content change. Edits belong in their own commit.
- Would a reader six months from now know why the shim, the forwarder,
  or the compatibility import exists and when it can go? If not, the
  comment is missing.

## 5. Stages

### Stage 0. Scaffolding

Goal: the tools the later stages rely on exist before anything moves.
`pyproject.toml` is a pytest configuration carrier at this stage; it
does not declare a build system or make the package installable.

Steps

1. Add `pyproject.toml` declaring package `track` (currently `src`),
   with `[tool.pytest.ini_options]` setting `testpaths = ["tests"]` and
   markers `gpu` and `data`.
2. Add `tests/conftest.py` with fixtures for a tiny synthetic
   `TrackSet` (two frames, two objects, one vehicle) and a temporary
   corpus tree (tier 1 and tier 2 with one clip).
3. Add `tests/smoke_eval.py`: builds the smoke yaml from the objective
   at run time (three antare clips, the objective's test block and
   `single_shared_streams`), writes provenance, and has an exact
   `--compare` mode.
4. Add `tests/test_import_graph.py`: parses every module's imports and
   asserts the allowed edges from section 2. It fails today; mark it
   `xfail` with a note until stage 5 closes.
5. Add `tests/test_no_literal_paths.py`: greps the package for
   `/mldata` and `/home/` literals outside `paths.py`. Also `xfail`
   until stage 2.

Exit criteria: both structure tests report `xfail` for the right
reasons (layer violations, literal paths) and their live-guards pass;
two consecutive smoke runs compare identical. (The existing 51 tests
still run from `src/` until stage 1 moves them.)

Review checklist

- Do the two xfail tests actually run the check and fail for the right
  reason, or do they pass vacuously because they scan the wrong tree?
- Does the smoke yaml pull its test block from the objective yaml at
  run time, or is it a frozen copy that will drift?

### Stage 1. Mechanical moves

Goal: files land in their final directories with no content change.

Steps

1. `git mv src/test_*.py tests/unit/`. Fix their `import src.x` lines
   only if the move breaks discovery; otherwise leave imports alone.
   No `__init__.py` under `tests/` (another repo on PYTHONPATH owns a
   top-level `tests` package) and unique test basenames across
   subdirectories. Narrow `testpaths` to `tests` only.
2. `git mv` the ten research tools to `tools/`:
   `cadence_test.py cadence_diag.py capacity_curve.py capacity_plot.py
   quality_grid.py quality_table.py gpu_attrib.py make_rt_configs.py
   camera_motion.py jetson_pm_sweep.sh`. Update the one intra-tool
   import (`capacity_plot` imports `capacity_curve`) and the commands
   quoted in `docs/capacity_curve.md` and `docs/research/*.md`.
3. `git mv docs/capacity_curve.md docs/user_guides/capacity_curve.md`.
4. Fix the README dead link (`stuff/stuff/ubtrk2.py`) and its layout
   table.
5. Add `docs/README.md`: one line per document, grouped by folder.

Exit criteria: `git diff -M --stat` shows only renames plus the README
and doc edits. Tests green. Smoke eval unchanged.

Review checklist

- Run every command quoted in the moved docs. Do they still work from
  the repo root?
- Did any tool rely on being importable as `src.x` from another tool or
  from a saved yaml? (`quality_table` is imported by nothing but check
  the results tooling under `/mldata/tracking/results/qtab`.)
- Was anything moved that is not research tooling? `eval_compare.py`
  is not; it stays.

### Stage 2. Paths

Goal: one module owns every filesystem root.

Steps

1. Create `src/paths.py` exposing `MLDATA`, `DOWNLOADS`, `TIER1`,
   `TIER2`, `RESULTS`, `CONFIG_DIR`, `TRACKER_CONFIG`,
   `SEARCH_YAML`, each read from an environment variable with the
   current literal as the default. Add a `describe()` that prints the
   resolved values for `track.py --paths`.
2. Replace the literals file by file, starting with the smallest
   (`eval_compare`, `track_search`, `make_rt_configs`) and ending with
   `trackset_import` (31) and `corpus_manifest` (15). Default arguments
   that are paths become `None` resolved inside the function, so the
   environment is read at call time rather than import time.
3. Delete the three `sys.path.insert` lines in `cadence_diag`,
   `cadence_test` and `trackset_import`. The autolabel checkout
   location becomes `paths.AUTOLABEL`, read from `AUTOLABEL_PATH` as
   `autolabel_bridge` already does.
4. Flip `tests/test_no_literal_paths.py` from `xfail` to live.

Exit criteria: the literal-path test passes. Tests green. Smoke eval
unchanged. `track.py --paths` prints the same roots the code used
before.

Review checklist

- Diff every replaced literal against its default. A single changed
  character in a path is a silent redirect to an empty directory that
  fails only on the next import run.
- Which defaults were evaluated at import time and are now evaluated
  at call time? Any that were captured into a closure or a
  multiprocessing worker at fork time need a second look.
- Do string-built paths (`f"{root}/annotation"`) all go through
  `os.path.join` on the new roots, or did one keep a trailing slash
  assumption?

### Stage 3. Break the mixin cycle

Goal: `TrackSet` no longer inherits its parsers.

Steps

1. Create `src/formats/` with one module per format:
   `mot.py personpath22.py meva.py otw.py chirla.py roundabouthd.py
   uvg_vcm.py jaad.py bdd100k.py antare.py`. Each exposes
   `read(...) -> TrackSet` built by moving the body of the matching
   `import_*` mixin method and replacing `self` with a fresh
   `TrackSet()`.
2. Replace each mixin method with a two-line forwarder that calls the
   new function and copies its frames and metadata into `self`, marked
   `# compatibility: remove in stage 7`.
3. Remove `TrackSetImportersMixin` from `TrackSet`'s bases once every
   forwarder is on `TrackSet` directly. Delete the class.
4. `trackset.py` no longer imports `trackset_import`. Check that
   `trackset_import` importing `trackset` is now the only edge.
5. Move `src/import_antare.py`'s parser into `formats/antare.py`; its
   discovery and copy logic goes to `corpus/importers.py` in stage 4.
   (`formats/bdd100k.py` is deferred: the BDD parse is inline in
   `convert_bdd100k_kaggle` and comes out in stage 4b.)

Exit criteria: no import cycle (`python -X importtime` or the graph
test's cycle check). Tests green, including a new unit test per format
that parses a five-line fixture. Smoke eval unchanged.

Review checklist

- Every `self.` inside a moved parser body: does it still refer to the
  new `TrackSet`, or did one refer to a mixin attribute that no longer
  exists (`self.metadata` set before `add_frame`, `self._decode_...`)?
- Which parsers mutate metadata after adding frames? Order matters for
  `frame_rate`-dependent time computation.
- Are the forwarders byte-for-byte equivalent for callers that passed
  keyword arguments?
- Did any test only pass because the mixin's method resolution order
  hid a name clash between two `import_*` helpers with the same nested
  function name (`out_class`, `mapped_track_id`)?

### Stage 4. Split the god modules

Goal: `track_test.py`, `trackset_import.py` and `trackset.py` become
packages of single-purpose modules. Done one module at a time, each its
own sub-stage with the full protocol.

4a. `track_test.py` to `eval/`

1. `eval/matching.py`: `mot_obj`, `permissive_iou_matrix`,
   `permissive_iou`, `_box_in_ignore`.
2. `eval/metrics.py`: `fitness_score`, `fitness_multi_score`,
   `gt_class_box_counts`, `annotation_floors`,
   `compute_detection_metrics`, `_events_by_hid_from_df`,
   `_honest_fp_runs_core`, `compute_metrics`, `score_tracksets`.
3. `eval/runner.py`: `track_test_work_fn`, `_clip_meta`,
   `_parse_packed_results`, the two metrics workers, `_resolve_pm`,
   `run_single_shared`, `on_result_callback`, `track_test`. The
   fork-before-import line moves with `track_test`.
4. `eval/report.py`: `summary_string`, `result_string`,
   `get_avg_scores`, `display_results`, the rollup code including
   `_groupmean`, `_summary_metric_keys`, `_result_subset`, the json and
   html writers.
5. `track_test.py` becomes `from .eval.matching import *` style
   re-exports, marked for removal in stage 7. `track_search`,
   `cadence_diag` and the tests switch to the new modules in the same
   sub-stage.

4b. `trackset_import.py` to `corpus/` and `formats/`

1. `corpus/media.py`: `_frame_pts_monotonic`, `_remux_to_mp4`,
   `_video_codec`, `_native_fps`, `_transcode_h264`, and the
   roundabouthd `_transcode` closure (in practice the closure stayed
   inline in `convert_roundabouthd`; stage 5 lifts it with the rest).
2. `corpus/importers.py`: every `convert_*`, `convert_autolabel_folder`,
   `reduce_dataset`, `apply_reduction`.
3. `corpus/migrations.py`: `estimate_bdd_time_offsets`,
   `fix_cevo25_vfr_times`, `dofix`, each with a docstring stating the
   date it ran and the corpus it touched. Candidates for deletion in
   stage 7 once ledgered.
4. `trackset_import.py` becomes re-exports.

4c. `trackset.py` to `core/`

1. `core/objects.py`: `Object`, `object_interpolate`,
   `object_class_remap` (today `track_util`).
2. `core/display.py`: `display_trackset`, `onoff`.
3. `core/trackset.py`: everything else.

4d. `dataset_lite.py` plus `derive_tracking` and `check_tracking` to
`corpus/derive.py`; `corpus_manifest.py` keeps the registry, `build`,
`verify`, `load_capabilities`, `set_audit`, `set_file_source`, `allows`
and becomes `corpus/manifest.py`.

Exit criteria per sub-stage: for a 1-to-N split `git diff -M` cannot
show renames, so the reviewer reconstructs the old file from the new
modules (concatenate the top-level function/assignment bodies in the
original order) and diffs it against `git show <commit>~1:<old file>`;
every non-import, non-docstring difference must be explained. Tests green. Smoke eval unchanged. For 4a also a full val eval
compared with any saved run of the same config using `eval_compare`:
every per-clip row identical.

Review checklist

- For each moved function, which module-level names did it read
  (`_FP_TRACK_COEF`, `PM_OVERRIDE`, `_FORCE_DICT`)?
  Each must be reachable from its new home by the same name, and
  anything set from outside (`track_test.PM_OVERRIDE = opt.pm` in
  `track.py`) must still land where the reader looks.
- Which functions are called by name inside a multiprocessing worker
  and must be picklable at their new import path? Old pickles of
  `search_journal_*.jsonl` do not carry functions, but check the
  work-queue payloads.
- Is the `import src.trackset as _  # noqa` fork-ordering line still
  executed before the pool is created?
- After 4a, does `search_test_batch` still find `_groupmean` and
  `__ovr<group>` rows under the same keys?
- Did the re-export module keep `__all__` or does `from x import *`
  now drop the underscore-prefixed helpers that tests import directly?

### Stage 5. One media module

Goal: one ffprobe wrapper, one transcode function, one remux function.

Steps

1. In `corpus/media.py`, define `probe(video) -> VideoInfo` (width,
   height, fps, r_fps, n_frames, duration, audio codec, has_b_frames)
   and make the four existing wrappers thin views over it.
2. Define `transcode(src, dst, *, dims, divisor, out_fps, max_seconds,
   audio, gop, encoder)` with the nvenc-then-x264 fallback, and express
   `dataset_lite.transcode`, `_transcode_h264`, the roundabouthd
   recipe and `cadence_test`'s recipe as calls with different
   arguments. Record the argument set each caller used before the
   change in a table in the module docstring.
3. Delete the four originals.
4. Flip the import-graph test to live if 4 is complete.

Exit criteria: for each of the four callers, transcode one reference
clip with the old code (from the previous commit) and the new, and
compare with `ffprobe -show_streams` plus a frame-count and
first-and-last-pts check. Audio present in the output whenever present
in the input.

Review checklist

- Which flags did each original recipe pass that the unified call does
  not? `-pix_fmt yuv420p`, `-movflags +faststart`, `-g 30` versus
  `2*out_fps`, `-cq 22` versus `23`, `-preset p4` versus `p5`. Every
  difference must be either reproduced through an argument or ledgered
  as an intentional change with the affected corpora named.
- Does the unified probe still refuse VFR sources where the antare
  importer did, and still accept them where `dataset_lite` did?
- Does the fallback from nvenc to x264 still trigger on exit code 187
  and not on a genuine input error?

### Stage 6. CLI subcommands

Goal: `track <verb>` replaces the flag soup.

Steps

1. `src/cli.py` with argparse subparsers: `view`, `track`, `eval`,
   `search`, `compare`, `import <corpus>`, `corpus build|verify|derive|check`,
   `paths`. Each subcommand's function is the existing entry point;
   no logic moves.
2. `track.py` becomes a shim that maps old flags to subcommands and
   prints a one-line deprecation pointing at the new form.
3. Console script `track` in `pyproject.toml`.
4. Update both user guides and the README to the new commands, keeping
   the old forms in a short "if you have an old note" table.

Exit criteria: every command in the user guides runs as written. The
old `track.py --eval` still works through the shim.

Review checklist

- Does every old flag combination map? `--eval` with no path, `--eval
  <path>`, `--eval-split`, `--eval-permissive`, `--pm`,
  `--tracker-config`, `--results-location`, `--search`, `--test`,
  `--track` with `--display` `--output` `--save-trackset` `--proxy`,
  `--view` with `--trackset`, `--compare`, and the six conversion
  flags.
- Does `--pm` still set `PM_OVERRIDE` before the runner imports it?
- Is `tmp/log` still created before logging is configured?

### Stage 7. Delete

Goal: remove what nothing uses.

Steps

1. Delete the re-export shims from stages 3 and 4 and the mixin
   forwarders. Fix every caller the deletion breaks.
2. Delete `search_test`, `result_string`, `object_class_remap`,
   `densify_sparse_gt` and its tests, `dofix`,
   `estimate_bdd_time_offsets`, `fix_cevo25_vfr_times`, after writing a
   ledger entry for each migration stating when it ran and on what.
3. Decide `allows` and `set_file_source`: wire the registry check in
   `track_search._registry_check` to `allows`, or delete both and
   amend the tiers spec.
4. `_FORCE_DICT` in `eval/runner.py` is a debug hook read through
   `globals()` that nothing sets anywhere: delete it or make it a
   parameter.
5. Before deleting the shims: the autolabel repo imports
   `src.track_test` (compute_metrics, summary_string) and
   `src.corpus_manifest` (set_audit); switch those four files first.

Exit criteria: `vulture`-style pass (or the grep from section 1)
reports no unreferenced top-level function. Tests green. Smoke eval
unchanged.

Review checklist

- For every deletion, the reviewer independently repeats the reference
  search including yaml configs, shell history notes in docs, and the
  autolabel repo, which imports this repo's registry loader.
- Was a function deleted that is only reached by name through a yaml
  key or a string (`getattr`, `globals()`)? `track_test.py` uses
  `globals().get("_FORCE_DICT")`.

### Stage 8. Comments to ledger

Goal: code comments state invariants; history lives in one dated file.

Steps

1. Create `docs/ledger.md` with one entry per dated decision, newest
   first: date, one-line title, the reasoning, the files it touched.
2. Walk `eval/`, `corpus/derive.py`, `corpus/manifest.py`,
   `tracker/upyc.py` and `search/`. For each comment block that
   contains a date, a person, or an incident, move the narrative to the
   ledger and leave one sentence stating the rule plus `see ledger
   YYYY-MM-DD <title>`.
3. Do the same for the header comments in the search yaml and the
   tracker config only if the config owners agree; otherwise leave
   them.

Exit criteria: no comment in the package contains a date or a name
except as a ledger reference. The ledger has an entry for every removed
block.

Review checklist

- Read each shortened comment in isolation. Does the rule still make
  sense without the story? If a reader would ask "why", the sentence
  is too short.
- Did any comment block contain a constraint that is really an
  assertion or a test? Those become code, not ledger prose.

### Stage 9. CI

Steps

1. A workflow running `pytest -m "not gpu and not data"` and the
   import-graph and literal-path tests on every push.
2. A documented manual command for the `gpu` and `data` suites on this
   box, in `docs/user_guides/`.

Review checklist

- Does the CI environment have `stuff` and `ubon_pycstuff` importable?
  If not, which modules must the unit tests avoid importing, and does
  a test currently pull them in at module level?

## 6. Effort

| stage | size | notes |
|---|---|---|
| 0, 1, 2 | one afternoon together | mechanical |
| 3 | half a day | one parser at a time |
| 4a | a day | the eval engine; full val eval at the end |
| 4b, 4c, 4d | a day together | |
| 5 | half a day | the transcode comparison table is most of it |
| 6 | half a day | |
| 7, 8 | a day together | the ledger writing dominates |
| 9 | an hour | |

## 7. What not to do

- Do not rewrite the metrics or the search loop while moving them.
  Moving and changing in the same commit is how a smoke-eval difference
  becomes impossible to attribute.
- Do not "tidy" the tracker config or the search yaml as part of this.
  Both are production artefacts with their own approval rules.
- Do not delete the narrative comments. Move them. Several encode
  decisions that were expensive to learn.
