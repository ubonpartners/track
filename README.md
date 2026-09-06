# Track

`track` is the evaluation, tuning and dataset harness for the Ubon
tracking pipeline (`ubon_cstuff` / `ubon_pycstuff`). It answers three
questions:

- **How good is this tracker config?** Run it over a fixed set of
  labelled clips and score it with MOT metrics against one agreed
  objective (`eval`).
- **Which parameter values make it better?** Coordinate-descent search
  over the tracker config, scored on the same objective (`search`).
- **Where do the labelled clips come from?** Importers that turn raw
  dataset drops into a canonical corpus, and a derivation step that
  produces the copies the tracker actually evaluates (`import`, `corpus`).

It also replays ground truth and tracked runs with overlays (`view`) and
reads and writes the UBTRK2 tracker-run format shared with the other
repos.

The tracker itself lives in `ubon_cstuff`; this repo only drives it.
Older trackers (`utrack`, `bytetrack`, `botsort`) are no longer supported.

## Quick start

```bash
git clone https://github.com/ubonpartners/track.git
cd track
conda env create -f environment.yml      # or: pip install -r requirements.txt
conda activate track
python -m src.cli paths                  # where the code will look for data and configs
python -m pytest -q                      # unit suite, a few seconds, no GPU or data
python -m src.cli --help
```

You need, beyond the Python environment:

- `ubon_pycstuff` importable (the compiled tracker runtime) for anything
  that runs the tracker: `track`, `compare`, `eval`, `search`, `test`.
- `ffmpeg` and `ffprobe` on `PATH` for imports and corpus derivation.
- The data tree under `/mldata` (or your own layout, see
  [Filesystem roots](#filesystem-roots)).
- The `autolabel` repo checked out beside this one only for the
  importers that auto-label unlabelled footage.

The five commands you will use most:

```bash
python -m src.cli eval --split val --results-location /mldata/results/eval/<name>   # score the current tracker config
python -m src.eval_compare /mldata/results/eval/<before> /mldata/results/eval/<after>  # compare two eval runs
python -m src.cli search /mldata/config/track/search/track_search_v11_mc.yaml       # tune parameters (about a day)
python -m src.cli view /mldata/tracking/antare_bwc/annotation/pub-garden.json       # look at a clip's GT
python tests/smoke_eval.py --out /mldata/results/cleanup/<name>                     # 15-second end-to-end check
```

## Concepts

### Data tiers

```
/mldata/downloaded_datasets/        tier 0   raw drops, never modified
/mldata/tracking_original/<corpus>/ tier 1   canonical import: video + GT + MANIFEST.json
/mldata/tracking/<corpus>/          tier 2   derived eval copies: what eval and search read
```

Tier 1 is the source of truth and the only input the autolabel repo
reads. Tier 2 is disposable: every clip is the tier-1 clip capped at
1280 on the long side, decimated to the tracker's analytics frame grid,
re-encoded I+P with audio preserved, with the annotation subset to the
retained frames. `python -m src.cli corpus derive <corpus>` regenerates
it; `corpus check` verifies it is on spec. Full contract:
`docs/specs/data_tiers_and_corpus_registry.md`.

### Annotations

One json per clip, the same shape in every tier and for tracked output:

```jsonc
{"metadata": {"frame_rate": 10.0, "width": 1280, "height": 960,
              "classes": ["person", "vehicle", "other"],
              "original_video": "/mldata/tracking/antare_bwc/video/pub-garden.mp4",
              "box_convention": "visible", "hint": "bodycam", "lite": {...}},
 "frames": [{"frame_id": 1, "frame_time": 0.0,
             "objects": {"0": {"box": [x1, y1, x2, y2], "class": 0, "conf": 1.0}}}]}
```

Boxes are normalised to the frame, so a resolution change never touches
the GT. `frame_time` is seconds on the timeline the tracker sees; eval
interpolates GT between frames, so sparse keyframe GT is fine. Class
`other` is an ignore region: neither a false negative when missed nor a
false positive when matched.

### Corpora and the registry

Every tier-1 corpus carries a `MANIFEST.json` with file hashes and a
capability block: box convention, completeness, and what the corpus may
be used for. The registry is shared with the autolabel repo.

| corpus | camera | box convention | notes |
|---|---|---|---|
| mot (MOT17, MOT20) | static and moving | fullbody | crowd scenes, CC BY-NC-SA |
| personpath22 | handheld | visible | keyframe GT |
| jaad | dashcam | fullbody | selected subjects only |
| cevo, cevo_april25 | static indoor | visible | internal captures |
| chirla | static indoor | visible | multi-camera re-id |
| uvg_vcm | static 4K | visible | professional labels |
| antare_bwc | body-worn and fixed | visible | 25 staged-incident clips, dense GT, kept at native 10 fps |
| bdd100k_mot | dashcam | fullbody | detection gating only |
| roundabouthd | static 4K | visible | vehicles only |
| meva, otw | static | fullbody / visible | actor GT augmented by autolabel; training only |
| bwc-videotext, raw_movies | body-worn / movies | fullbody | autolabel output, no human GT |

### The objective

There is exactly one objective config:
`/mldata/config/track/search/track_search_v11_mc.yaml`. `eval` with no
path and `search` both read it, so they cannot describe different
datasets. It lists 565 clips (339 train, 226 val), tags each with a
`group` (`static`, `moving`, `movie`, plus cadence-test groups) and a
`stream_hint`, names the 20 parameters the search may move, and names
the score:

- per clip and class, `fitness = MOTA - 0.35 * honest_fp_tracks_per_second - 0.002 * fp_per_frame`,
  combined over classes as `fitness_multi` (person weight 1.0, vehicle 0.3);
- summed within each group with a cap on any one clip's weight;
- averaged across groups with `group_weights` into the `_groupmean` row.

`_groupmean` is the number to quote. `_overall` is box-count weighted
and is not the objective. Do not copy the yaml to change a field; every
knob an eval needs is a command-line option. Details and history:
`docs/user_guides/optimization_flow.md`.

Eval output columns, in order: frames, unique objects, mostly / partially
tracked and mostly lost fractions, missed fraction, false-positive
tracks (raw and honest), FP per frame, FN per object, switches and
fragmentations per object, IDF1, MOTA, MOTP, fitness, the vehicle
variants, and `fitness_multi`.

## Command line

`python -m src.cli <verb> ...`. `--logging level[:console|file]` and
`--pm N` (detector performance tier for eval/search/test streams, 0 is
full resolution) go before the verb.

| verb | what it does | main options |
|---|---|---|
| `eval [yaml]` | the measurement path; no yaml means the objective config | `--split train\|val\|both`, `--results-location DIR`, `--tracker-config YAML` (A/B a tracker without editing the objective), `--permissive auto\|on\|off` |
| `search YAML` | coordinate-descent parameter search | see the optimisation guide |
| `test YAML` | benchmark several tracker configs over datasets from a test yaml | |
| `track GT.json` | track one clip with one config and print metrics | `--config`, `--display`, `--output mp4`, `--save-trackset run.ubtrk2`, `--proxy addr:port` |
| `compare YAML` | several configs over one clip with per-frame MOTA | `--no-display` |
| `view PATH` | replay a GT json or a `.ubtrk2` run with boxes | |
| `import CORPUS` | tier 0 to tier 1 for a known dataset (`mot`, `jaad`, `meva`, `antare`, ...) | `--amodal` (personpath22) |
| `corpus build\|verify\|derive\|check CORPUS...` | manifest, hash check, tier-2 derivation, conformance check | `--hint static\|bodycam`, `--divisor N`, `--max-seconds S`, `--purge-legacy` |
| `paths` | print every filesystem root in effect | |

`python track.py --flag` forms from before the CLI rewrite still work
and print the equivalent verb.

`python -m src.eval_compare DIR [DIR ...]` prints two or more eval runs
side by side: both objective rows, the group breakdown and the biggest
per-clip movers. The first directory is the baseline.

## Workflows

**Evaluate a tracker change.** Run the objective on val before and
after, compare, and quote the group-mean row. Eval runs use exact-shape
detector batching and disable faces, CLIP and audio, so their scores
are comparable with each other but not with production numbers. Repeat
runs of the same code agree to about 0.0003.

```bash
python -m src.cli eval --split val --results-location /mldata/results/eval/before
python -m src.cli eval --split val --results-location /mldata/results/eval/after --tracker-config /path/candidate.yaml
python -m src.eval_compare /mldata/results/eval/before /mldata/results/eval/after
```

**Tune parameters.** One iteration probes one parameter up and down
over the train split, about three minutes; a full search over the
current parameter set is roughly a day. The results directory gets a
text log, a live HTML report and a journal that `resume_from` can
continue. Apply the winning values to the production config only with
explicit approval. Guide: `docs/user_guides/optimization_flow.md`.

**Import a dataset.** Write a parser in `src/formats/<name>.py`
(`read(...) -> TrackSet`, unit-tested on a five-line fixture) and a
`convert_<name>` driver in `src/corpus/importers.py`, declare the corpus
in `src/corpus/manifest.py`, then build, derive, check and register the
clips in the objective yaml. Check the frame-index convention of every
new source by drawing its boxes on extracted frames; it has been wrong
before. Guide: `docs/user_guides/import_and_annotation.md`.

**Debug one clip.** `view` for the GT, `track --display --save-trackset`
to watch and keep a run, `view run.ubtrk2` to replay it later with the
debug overlays (detections, ROI, box prediction, motion field, cost
map).

## Repository layout

| path | contents |
|---|---|
| `src/cli.py` | the command line; `track.py` at the root is the compatibility shim |
| `src/paths.py` | every filesystem root, read from `TRACK_*` variables |
| `src/core/` | `trackset` (TrackSet: json/yaml and UBTRK2 storage, time interpolation), `objects` (Object, drawing), `display` (viewer) |
| `src/formats/` | one parser per dataset format; `formats.load(path)` dispatches by extension |
| `src/corpus/` | `importers` (tier 0 to 1), `manifest` (registry, build, verify), `derive` (tier 2 and check), `media` (every ffprobe/ffmpeg call) |
| `src/tracker/` | `upyc` (wrapper over `ubon_pycstuff`), `factory`, `run` (`import_create(ts, ...)` drives a tracker into a TrackSet) |
| `src/eval/` | `matching` (IoU, box conventions), `metrics` (MOT metrics, honest FP, fitness), `runner` (work queue and shared-stream runner), `report` (rollups, json and html) |
| `src/track_search.py`, `src/eval_compare.py` | the search loop and the run comparator |
| `src/autolabel_bridge.py`, `src/import_antare.py` | the autolabel bridge and the antare importer |
| `src/track_test.py`, `src/trackset.py`, `src/corpus_manifest.py` | re-export shims kept for the autolabel repo; do not import them |
| `tools/` | research tooling outside the package: cadence tests, capacity curve, quality grid, GPU attribution |
| `tests/` | pytest suite; `tests/smoke_eval.py` is the end-to-end smoke |
| `docs/` | guides, specs, plans, research logs and the decision ledger; start at `docs/README.md` |

Layering is enforced by `tests/test_import_graph.py`: `core` imports
only `paths`; `formats` imports `core`; `corpus` imports `core` and
`formats`; `eval` imports `core` and `tracker`; `search` imports `eval`
and `corpus`; `cli` imports everything. `tests/test_no_literal_paths.py`
keeps every `/mldata` literal inside `paths.py`, and
`tests/test_comment_hygiene.py` keeps dated stories out of code comments
and in `docs/ledger.md`.

## Configuration

### Filesystem roots

`src/paths.py` resolves every root at call time from the environment,
defaulting to the dev-box layout:

| variable | default |
|---|---|
| `TRACK_MLDATA` | `/mldata` |
| `TRACK_DOWNLOADS` | `$TRACK_MLDATA/downloaded_datasets` |
| `TRACK_TIER1` | `$TRACK_MLDATA/tracking_original` |
| `TRACK_TIER2` | `$TRACK_MLDATA/tracking` |
| `TRACK_RESULTS` | `$TRACK_MLDATA/results` |
| `TRACK_CONFIG_DIR` | `$TRACK_MLDATA/config/track` |
| `TRACK_TRACKER_CONFIG` | `$TRACK_CONFIG_DIR/trackers/uc_v11.yaml` |
| `TRACK_SEARCH_YAML` | `$TRACK_CONFIG_DIR/search/track_search_v11_mc.yaml` |
| `AUTOLABEL_PATH` | the `autolabel` directory beside this repo |

`python -m src.cli paths` prints what is in effect.

### The tracker config

`/mldata/config/track/trackers/uc_v11.yaml` is the production tracker
config, shared by every deployed box. Nothing in this repo edits it.
Eval-time changes go into the objective yaml's `main_config_override`
block or into `--tracker-config`; production changes need explicit
per-change approval.

## Stored run format (UBTRK2)

Tracked runs are stored in the UBTRK2 container implemented in the
`stuff` repo (`stuff.ubtrk2`): a `ubtf` header box, a `meta` box with a
UTF-8 YAML payload (`schema_version`, `kind`, `source_video`,
`frame_rate`, `width`, `height`, `classes`, `payload_encoding`), then one
`fram` box per processed frame holding `fhdr` (frame time, result type,
motion score, motion and inference ROIs), `trks` (objects by track id
with detector-aligned fields: class, confidence, box, subbox, face and
pose points, attributes, re-id, face and CLIP embeddings, JPEG crops),
optional `dets`, `dbug` and `imgp`, and an `xtra` extension map. Large
arrays and blobs are stored inline as typed payloads. The container is
documented and tested in `stuff`; how this repo uses it for replay,
metrics and analysis is `docs/specs/TRACKER_DEBUG.md`.

## Testing

```bash
python -m pytest -q                                   # unit suite, seconds, no GPU or data
python tests/smoke_eval.py --out /mldata/results/cleanup/<name>          # 3 clips through the objective, ~15 s
python tests/smoke_eval.py --compare <previous dir> /mldata/results/cleanup/<name>
python -m src.cli corpus verify antare_bwc && python -m src.cli corpus check antare_bwc
```

The smoke compare must report every cell identical; differences under
1e-9 relative are summation noise and reported as such. CI
(`.github/workflows/tests.yml`) runs the structure tests on a hosted
runner and the full unit suite on a self-hosted one, since the package
needs the private `stuff` and `ubon_pycstuff`. The full ladder is in
`docs/user_guides/testing.md`.

## Documentation

- `docs/README.md` — index of everything below.
- `docs/user_guides/import_and_annotation.md` — tiers, derive, the annotation format, importing a dataset.
- `docs/user_guides/optimization_flow.md` — the objective, the search algorithm, running and reading a search, applying the result.
- `docs/user_guides/testing.md` — the test ladder and CI.
- `docs/user_guides/capacity_curve.md` — quality versus concurrent streams.
- `docs/specs/data_tiers_and_corpus_registry.md` — the tier contract and the registry shared with autolabel.
- `docs/specs/TRACKER_DEBUG.md` — the UBTRK2 result and debug format.
- `docs/ledger.md` — dated decisions and incidents; code comments point here.
- `docs/plans/repo_cleanup.md` — the 2026-09 restructuring and its reviews.

## License

Dual license:

- **AGPL** (non-commercial)
- **Ubon Cooperative License** (https://github.com/ubonpartners/license/blob/main/LICENSE)

Questions: bernandocribbenza@gmail.com (subject `yolo-dpa question`).
