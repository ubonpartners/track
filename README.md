# Track

`track` is now a thin Python tool around the `ubon_cstuff` / `ubon_pycstuff`
tracking pipeline. It is responsible for:

- loading input sequence metadata files such as MOT-style JSON/INI annotations
- running the `upyc` tracker on those inputs
- evaluating results with MOT metrics
- replaying tracked runs with debug overlays
- comparing configs and searching parameters
- reading and writing the canonical UBTRK2 tracker-run format

The repo no longer supports `utrack`, `bytetrack`, or `botsort`.

## Getting Started

```bash
git clone https://github.com/ubonpartners/track.git
cd track
conda env create -f environment.yml
conda activate track
python track.py --help
```

Runtime requirements:

- `ubon_pycstuff` / `ubon_cstuff` for tracking and optional C metrics
- `ffmpeg` on `PATH` for MP4 → H.264 conversion in the `upyc` path
- datasets reachable under `/mldata/...` or via equivalent local layout

## Repository Layout

| Path | Purpose |
|------|---------|
| `track.py` | CLI entry point for view/track/test/compare/search and dataset conversion helpers |
| `src/trackset.py` | `TrackSet` input loader, UBTRK2 run import/export, and replay viewer |
| `src/track_util.py` | Detection/track object helpers, interpolation, drawing |
| `src/trackers.py` | `upyc`-only tracker factory |
| `src/upyc_tracker/` | Python wrapper over `ubon_pycstuff` tracking results |
| `src/track_test.py` | MOT metrics, detection metrics, benchmark/test orchestration (the eval engine, not a test) |
| `src/track_search.py` | Parameter search over `ubon_cstuff` config values |
| `src/eval_compare.py` | Canonical comparator for two eval runs |
| `src/trackset_import.py`, `src/import_antare.py` | Dataset importers: tier 0 -> tier 1 |
| `src/corpus_manifest.py`, `src/dataset_lite.py` | Tier-1 manifest/registry and tier-2 eval-spec derivation |
| `src/autolabel_bridge.py` | Optional bridge to the autolabel repo (auto-labelling, GT augmentation) |
| `tools/` | Research and one-off tooling (cadence tests, capacity curve, quality grid, GPU attribution); not part of the package |
| `tests/` | pytest suite (`python -m pytest`); `tests/smoke_eval.py` is the three-clip eval smoke |
| `docs/` | Guides, specs, plans, research logs; start at `docs/README.md` |

## Supported Formats

### Input sequence metadata

These remain supported as tracking inputs:

- MOT `seqinfo.ini` + `gt/gt.txt`
- JSON/YAML annotation files used as input sequence metadata, for example `/mldata/tracking/mot/annotation/MOT17-05.json`
- Caltech `.vbb`

These are input formats, not the canonical stored tracker-result format.

### Stored tracker-result format

Tracked runs are stored in the canonical UBTRK2 container implemented in
`stuff.ubtrk2`.

Container layout:

1. top-level `ubtf` box (file header/version)
2. top-level `meta` box (UTF-8 YAML metadata payload)
3. repeated top-level `fram` boxes (one frame per processed frame)

The same frame-record schema is intended for:

- disk persistence
- network transport
- offline tracker-component analysis

### Metadata (`meta` box)

The metadata payload is UTF-8 YAML with stable string keys. Current keys written by `track` include:

- `schema_version`
- `kind`
- `source_video`
- `frame_rate`
- `width`
- `height`
- `classes`
- `payload_encoding`

### Frame record

Each `fram` box contains typed child boxes. Current frame payload includes:

- `fhdr`: frame header (`frame_time`, `result_type`, `motion_score`, `motion_roi`, `inference_roi`)
- `trks`: tracked objects map
- `dets`: optional detector output list
- `dbug`: debug entry map
- `imgp`: source image path (optional)
- `xtra`: extension map for forward-compatible fields

### Objects

`objects` is a map keyed by `track_id`.

Current stored object fields use detector-aligned names:

- `class`
- `confidence`
- `box`
- `subbox`
- `subbox_conf`
- `face_points`
- `pose_points`
- `attrs`
- `reid_vector`
- `face_embedding`
- `clip_embedding`
- `face_jpeg`
- `clip_jpeg`
- `fiqa_score`

### Debug

`debug` is a map of named debug entries. Viewer-supported typed entries currently include:

- `detections`
- `roi`
- `box_prediction`
- `motion_field`
- `cost_map`

Large arrays, vectors, embeddings, and JPEG blobs are stored inline as typed
payload wrappers rather than in sidecar files.

For low-level container and payload wrapper documentation, see the
`stuff` repo README and its `ubtrk2` module.

## CLI Workflows

### View a sequence or tracked run

```bash
python track.py --view --trackset /path/to/input.json
python track.py --view --trackset /path/to/run.ubtrk2
```

The viewer accepts:

- input sequence metadata files
- UBTRK2 tracked-run files

### Run tracking, evaluate, and save a tracked run

```bash
python track.py \
  --track \
  --trackset /mldata/tracking/mot/annotation/MOT17-05.json \
  --config /mldata/config/track/trackers/uc_v10.yaml \
  --save-trackset /tmp/MOT17-05.ubtrk2 \
  --display
```

This will:

- load the input sequence metadata
- run the `upyc` tracker
- compute metrics
- optionally display the run
- optionally save the full tracked run in UBTRK2 format

### Benchmark suites

```bash
python track.py --test configs/tests.yaml
```

### Compare configs

```bash
python track.py --compare compare.yaml --trackset /mldata/.../annotation.json
```

### Parameter search

```bash
python track.py --search configs/search.yaml
```

Search now assumes `upyc`-style configs only. Parameters are found by key name
within the loaded config tree, and ambiguous repeated keys are rejected.

## Testing And Validation

- `python -m pytest` runs the unit suite (`tests/`; no GPU or data needed)
- `python tests/smoke_eval.py --out DIR` runs a three-clip eval through the objective config; `--compare A B` diffs two runs exactly
- `--test` benchmarks tracker configs over datasets with caching and summary tables
- the replay viewer can inspect saved UBTRK2 runs directly
- the UBTRK2 run format is tested in the shared `stuff` repo

## Format Documentation

Format documentation is intentionally split by ownership:

- `stuff`: authoritative low-level UBTRK2 container and payload encoding docs
- `track`: how tracker runs use that format for replay, metrics, and analysis
- `ubon_cstuff`: runtime result/debug schema emitted before serialization

See also:

- `docs/specs/TRACKER_DEBUG.md`
- `docs/user_guides/import_and_annotation.md` — how video and GT are imported (tiers, derive, annotation format)
- `docs/user_guides/optimization_flow.md` — tuning tracker parameters with `--search`, reading the result, applying it via `--eval` + eval_compare

## License

Dual license:
- **AGPL** (non-commercial)
- **Ubon Cooperative License** ([https://github.com/ubonpartners/license/blob/main/LICENSE](https://github.com/ubonpartners/license/blob/main/LICENSE))

Questions? contact bernandocribbenza@gmail.com (subject `yolo-dpa question`).
