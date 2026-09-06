# Importing video and annotations

A short guide to how a labelled video gets from a download to something
`track.py` can evaluate or search on. The formal spec is
`docs/specs/data_tiers_and_corpus_registry.md`; this page is the
working summary.

## The three places data lives

```
/mldata/downloaded_datasets/        tier 0   raw downloads, never modified
/mldata/tracking_original/<corpus>/ tier 1   canonical import: video + GT + manifest
/mldata/tracking/<corpus>/          tier 2   eval copy: what track.py actually reads
```

These are the defaults. The code resolves every root through
`src/paths.py`, which reads `TRACK_MLDATA`, `TRACK_TIER1`, `TRACK_TIER2`
and friends from the environment; `python track.py --paths` shows what
is in effect.

**Tier 0** is whatever you fetched: zips, AVIs, PNG sequences, MOT text
files. Leave it exactly as it arrived.

**Tier 1** (`tracking_original`) is the source of truth. Each corpus
holds `video/<clip>.mp4`, `annotation/<clip>.json` and a
`MANIFEST.json`. Video here is the mezzanine: a lossless remux where
the source codec allows, otherwise a pinned transcode recorded in the
manifest. GT here is at the native frame grid. This tier is the only
input the autolabel repo reads, and it is append-only: a re-import
bumps the per-file version in the manifest rather than silently
replacing bytes.

**Tier 2** (`tracking`) is a derived, disposable copy laid out the same
way (`video/`, `annotation/`, plus `derive_recipe.json`). It is what
the search and eval yamls point at. Anything here must be regenerable
from tier 1 plus the recipe. If losing it would hurt, it belongs in
tier 1.

## What the tier-2 derive does

`python -m src.corpus_manifest derive <corpus> --hint=static|bodycam`
walks every tier-1 clip and produces the eval-spec version:

- longest side capped at 1280, never upscaled;
- I+P only h264 (no B-frames); audio kept when the source has it
  (AAC copied, anything else re-encoded to AAC);
- frame rate decimated by an integer divisor to the tracker's analytics
  grid: the divisor is the smallest N with N/fps at or above
  `min_time_delta_process` in the tracker config for that camera hint.
  Retained frames are restamped to an exact N/fps grid;
- the annotation subset to the retained frames, with `lite` provenance,
  the `hint`, the corpus `box_convention`, and a `source_video` pointer
  back to tier 1.

Pass `--divisor=N` to force the divisor instead (for example
`--divisor=1` keeps the native frame timing untouched; antare_bwc is
derived this way). The choice is recorded in `derive_recipe.json`, so a
bare `derive <corpus>` later reuses it and `check` judges against it.

Why tier 2 exists at all: decoding 4K B-framed video at 30 fps for every
search iteration is wasted work when the tracker processes one frame in
N anyway, and the ingest path can feed an I+P mp4 straight to the
tracker with container timestamps driving frame timing. B-framed
sources fall back to a slower elementary-stream path.

Because boxes in the annotation are normalised to [0,1], a resolution
change never touches the GT. Only the `width`/`height` metadata moves.

## The annotation format

One json per clip, the same shape in both tiers:

```jsonc
{
  "metadata": {
    "frame_rate": 10.0,
    "width": 1280, "height": 960,          // normalisation basis for boxes
    "classes": ["person", "vehicle", "other"],
    "original_video": "/mldata/tracking/antare_bwc/video/pub-garden.mp4",
    "source_video":   "/mldata/tracking_original/antare_bwc/video/pub-garden.mp4",
    "box_convention": "visible",           // or "fullbody"; stamped from the manifest
    "hint": "bodycam",
    "lite": {"source_fps": 10.0, "divisor": 1, "max_seconds": null, ...}
  },
  "frames": [
    {"frame_id": 1, "frame_time": 0.0,
     "objects": {"0": {"box": [x1, y1, x2, y2], "class": 0, "conf": 1.0},
                 "1": {...}}},
    ...
  ]
}
```

Points that matter when writing an importer:

- `frame_time` is seconds on the timeline the tracker will see. For a
  constant-rate source that is `(frame_index)/fps`. Get this wrong and
  every metric is wrong while nothing crashes.
- `frame_rate` must be the true rate of `original_video`. Eval divides
  it down to pick its scoring cadence and the ingest uses it to stamp
  frames.
- Object keys are track ids as strings. `class` indexes `classes`.
- Class `other` is an ignore region: a miss is not a false negative and
  a match is not a false positive. Use it for labelled things the
  tracker is not expected to report.
- Dense GT should emit every frame, including frames with no objects.
  An empty frame is labelled absence. Sparse GT (keyframes only) is
  fine too, as eval interpolates GT between frames, but say so in the
  manifest capabilities.

## Importing a new dataset, step by step

1. **Download into tier 0** under `/mldata/downloaded_datasets/`. Check
   the three acquisition gates first: complete GT (every person in
   frame is labelled, not only scripted actors), unblurred faces, and a
   licence that permits our use.

2. **Write the importer.** Two parts: a parser in `src/formats/<name>.py`
   exposing `read(...) -> TrackSet` (one module per source format, unit
   tested on a tiny fixture in `tests/unit/test_formats.py`), and a
   `convert_<name>` driver in `src/corpus/importers.py` (or a standalone
   file such as `src/import_antare.py`) that walks the drop. The driver
   reads tier 0, copies or remuxes video into
   `/mldata/tracking_original/<corpus>/video/` and writes annotation
   json beside it. Keep ffmpeg out of it unless the codec forces a
   transcode; the derive step does the eval-spec encoding.

3. **Declare the corpus** in `src/corpus_manifest.py`: a
   `CAPABILITIES_SEED` block (box convention, completeness, density,
   what the corpus may be used for) and a `CORPUS_INFO` block
   (licence, source root, import recipe). `build` refuses an
   undeclared corpus.

4. **Build the manifest**:
   `python -m src.corpus_manifest build <corpus>`
   hashes every file and stamps the capabilities.

5. **Derive tier 2**:
   `python -m src.corpus_manifest derive <corpus> --hint=bodycam`
   (or `static`).

6. **Check it**:
   `python -m src.corpus_manifest check <corpus>`
   verifies resolution, B-frames, frame grid, and that every annotation
   carries the provenance fields. Run `verify <corpus>` for a full
   tier-1 hash check.

7. **Register the clips** in the search yaml, for example
   `/mldata/config/track/search/track_search_v11_mc.yaml`:

   ```yaml
   datasets:
     antare_pub_garden:
       split: val               # train | val
       family: antare_bwc       # must also appear in include_families
       group: moving            # static | moving | a cadence group
       stream_hint: bodycam     # binds the (hint:bodycam) config variants
       path: /mldata/tracking/antare_bwc/annotation/pub-garden.json
   ```

   Entries whose family is not in the yaml's `include_families` line
   are silently skipped.

8. **Smoke test one clip** before any batch run.

## Videos with no labels: autolabel

For footage with no GT at all, `convert_autolabel_folder` in
`src/corpus/importers.py` runs the autolabel pipeline over every mp4 in
a folder and writes the result straight into tier 1 in the annotation
format above. It needs the autolabel checkout next to this repo and a
GPU environment. Autolabel always runs at the native frame rate; the
frame-rate reduction happens afterwards in derive, never on the
autolabel input. Corpora produced this way are registered as derived
GT and must not be used to gate autolabel itself.

The same bridge (`src/autolabel_bridge.py`) can augment a partially
labelled corpus with high-confidence autolabel tracks for unlabelled
bystanders (MEVA and OTW). Those are canonical GT passes: they mutate
tier 1, get an entry in the manifest's `gt_passes`, and require a
re-build.

## Worked example: the antare clips

Tier 0 is two drops under `/mldata/downloaded_datasets/antare/`: a flat
one (`<name>.mp4` beside `<name>/gt/gt.txt` in MOT format) and a nested
one where each staged incident folder holds several cameras, body-worn
`antare-bwc-NN` and fixed `nc/sm/wh-cam-NN`. Every camera is imported
as its own clip named `<incident>-<bwc|fixed>-<NN>`, for example
`knife-drawn-fixed-06`, with the incident slug taken from a table in
the importer and NN the source camera number. All clips are 10 fps
constant-rate with a label on every frame.

```
python -m src.import_antare                                   # copies mp4, writes json
python -m src.corpus_manifest build antare_bwc
python -m src.corpus_manifest derive antare_bwc --hint=bodycam --divisor=1
python -m src.corpus_manifest check antare_bwc
```

The corpus mixes moving and fixed cameras, so the importer writes the
camera class into each tier-1 json as `metadata.hint` and derive reads
it per clip. The `--hint` on the command line is only the default for
clips that do not declare one. Register the yaml entries to match:
`group: moving` plus `stream_hint: bodycam` for body-worn clips,
`group: static` with no hint for fixed ones. Cameras of the same incident
were kept in the same train or val split so an incident cannot leak
across the split.

The importer's only real work was establishing that MOT frame k is
video frame k-1, which was checked by drawing the boxes on extracted
frames from both drops. Do that check for every new source. The timing
convention is never documented and has been wrong before.

## Looking at what you imported

- `python track.py --view --trackset <annotation.json>` plays the
  video with the GT boxes drawn.
- `python track.py --eval` runs the one objective config the search
  optimises (`--eval-split val` for search-comparable scores; compare
  runs with `python -m src.eval_compare`). `--search <yaml>` runs the
  parameter search over the registered datasets.
