# 🛰️ Track

Track is a comprehensive toolkit for working with annotated tracking datasets. It provides:

- **Trackset management**: import/export MOT, Ubon, or Caltech sequences, trim intervals, and replay frames with annotation overlays.
- **Tracker execution**: drive Ultralytics/ByteTrack, the motion-aware Python `utrack`, or the optimized C `upyc` tracker through a consistent YAML config interface.
- **Evaluation & comparison**: compute MOT metrics (MOTA, IDF1, switches, FP/FN, detection AP) in Python or C, run benchmark suites, and display per-frame debugging events.
- **Optimization**: automatically jitters tracker/motion parameters using coordinate descent until a chosen metric (fitness, MOTA, etc.) is maximized.

This README covers:

1. Getting started (env + dependencies)
2. Repository layout
3. Supported formats (MOT & Ubon explained)
4. CLI workflows (view, track, test, compare, search)
5. Parameter optimization details
6. Ubon format deep dive
7. Testing & licenses

---

## 1. Getting Started

1. **Clone the repo**
   ```bash
   git clone https://github.com/ubonpartners/track.git
   cd track
   ```

2. **Create the Conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate track
   ```
   - `environment.yml` pins Python ≥3.10, PyTorch ≥2.4, CUDA 12.4, numpy 1.26.4, pandas, matplotlib, plus pip extras (OpenCV, scipy, motmetrics, lap/`lapx`, tabulate, albumentations, etc.).
   - The pip extras are also listed in `requirements.txt` for pip-only installations; duplicate entries (e.g., `cython_bbox`, `numpy=1.26.4`) have been deduplicated.

3. **Optional dependencies**
   - Install `ubon_pycstuff`, `ubon_cproxy`, and `ubon_cstuff` to run the `utrack` and `upyc` pipelines.
   - Ensure `ffmpeg` is accessible on your `PATH`; `utrack` converts MP4 → H.264 for the decoder.
   - Point datasets at `/mldata/...` (internal default) or override via config/symlink.

4. **Verify the CLI**
   ```bash
   python track.py --help
   ```

---

## 2. Repository Layout

| Path | Purpose |
|------|---------|
| `track.py` | CLI entry point that dispatches to visualizing, tracking, testing, comparing, searching, and dataset conversions (`--caltech`, `--mot`, `--cevo`). |
| `environment.yml` & `requirements.txt` | Conda and pip dependency manifests (pip file mirrors the pip block). |
| `src/trackset.py` | `TrackSet`: import/export MOT/MOT→JSON/Ubon, trim time ranges, handle frames/objects, and visualize overlays (`display_trackset`). |
| `src/track_util.py` | Detection `Object`, interpolation helpers, class remapping, and utilities for pose/face/subbox data. |
| `src/trackers.py` | Tracker factory (`create_tracker`) that initializes `ultralytics_tracker`, `utrack`, or `upyc_tracker` from YAML configs. |
| `src/track_test.py` | MOT metric computation (Python `motmetrics` accumulator / optional C via `ubon_pycstuff`), test runner, caching, and result display. |
| `src/track_search.py` | Parameter search logic that tweaks tracker/motion settings and optimizes a chosen dataset metric. |
| `src/utrack/` | Motion tracker, Kalman filter, and matching logic for the custom Python tracker (`utracker`, `motion_track`, `kalman`). |
| `src/upyc_tracker/` | Wrapper for the optimized C-based Ubon tracker (`upyc_tracker`). |
| `REVIEW.md` | Living review covering system understanding, discovered bugs, refactors, and fixes. |

---

## 3. Supported Trackset Formats

### 3.1 MOT format

- Standard MOT layout: `seqinfo.ini`, `gt/gt.txt`, `img1/`.
- `TrackSet.import_mot` normalizes boxes, sets metadata (`classes` include `person`, `vehicle`, `other`), and stores `frame_times` + per-frame `objects`.
- Good for benchmarking against MOT Challenge data; you can also convert MOT sequences into the Ubon format via `TrackSet.export_yaml`.

### 3.2 Ubon format (native)

The “Ubon” format pairs a video (MP4 or similar) with a JSON/YAML annotation file produced by `TrackSet.export_yaml`.

#### Annotation structure

- `metadata`: `frame_rate`, `width`, `height`, `classes`, optional `original_video`.
- `frames`: each entry contains `frame_id`, `frame_time`, `objects`, `image_path`, `tracker_debug`.
- `objects`: map from `track_id` to detection dicts with `box`, `class`, `conf`, optional pose/face, subboxes, embeddings, attributes, and JPEG crops.
- `tracker_debug`: stores ROIs, detection arrays, motion fields, and predictions for `display_trackset`.

Use this format to persist full tracker outputs, replay them later, or share annotated videos without rerunning detection.

---

## 4. CLI Workflows

### 🔍 View annotated sequences

```bash
python track.py --view --trackset /path/to/annotation.json
```

- Opens the GUI with GT/detection overlays, MOT event stats, and debug toggles (`g`, `d`, `h`, numbers to toggle overlays).
- Works with MOT `.ini` (imports before rendering) or Ubon `.json`.

### 🚀 Run tracking + evaluation

```bash
python track.py --track --trackset /mldata/.../annotation.json --config configs/trackers/uc_v8.yaml
```

- Instantiates the YAML-defined tracker, runs it over the trackset, computes metrics, and optionally displays the result (`--display`).
- Supported tracker types: `bytetrack`/`botsort` (Ultralytics) and the custom `utrack` and `upyc` pipelines.
- Metrics include MOTA, IDF1, switches/fragmentations, `fp_per_frame`, detection AP, ROI area, etc.

### 📊 Benchmark suites

```bash
python track.py --test configs/tests.yaml
```

- Executes dataset/test pairs defined in YAML via a multiprocessing workqueue.
- Caches results in pickle files, aggregates across datasets/groups (`overall`, `__mean`, `_arithmean`), and writes tables via `display_results`.
- Columns can include fitness, mismatches, `fp_tracks`, detection AP, and more.

### 📈 Compare tracker outputs

```bash
python track.py --compare compare.yaml --trackset /mldata/.../annotation.json
```

- Runs all configs from `configs_to_compare` on the same GT and prints per-frame MOTA deltas, summary tables, and optionally visualizes them sequentially.

### 🔧 Parameter search

```bash
python track.py --search configs/search.yaml
```

- Executes `search_track`, which tweaks parameters from a YAML-defined search space and optimizes a selected metric.
- Logs every iteration, validates on a split (`train`/`val`), and caches evaluated vectors.
- Uses helper functions (`_update_initial_parameters`, `_round_numeric_metrics`) to keep initial values consistent and avoid rounding non-numeric fields.

---

## 5. Parameter Optimization Flow

1. **Define the search**
   - `search_params`: name + `min`/`max`/`step`/`initial`.
   - `tests.search_config.config`: base tracker config. Parameters can appear at the top level or inside `utrack`/`motiontrack`.
   - `result_test_opt_key`, `result_dataset_opt_key`, `result_dataset_opt_param`: specify the metric to optimize.

2. **Evaluate candidate vectors**
   - `search_test` updates the config with the candidate vector, runs `track_test`, and inspects the selected metric.
   - Metrics are rounded only for numeric values via `_round_numeric_metrics`.
   - Results map tuples → scores in `all_results` to dedupe repeats.

3. **Coordinate descent**
   - Loop over parameters, attempting `+step` and `-step` (scaled by a dynamic multiplier starting at 4 and halving when no improvements occur).
   - Accept moves that improve the metric. After a prescribed number of improvements, move to the next parameter.
   - Optionally validate the best vector on a split (`val`) every few iterations.

4. **Logging**
   - `search_log_<timestamp>.txt` records the vector, score, validation summary, and parameter progression.
   - `track_test.summary_string` highlights MOTA, fitness, FP/FN counts, etc., for more context.

---

## 6. Ubon Format Deep Dive

The Ubon format is the repository’s native serialized annotation style.

- Track IDs persist across frames for consistent referencing.
- Each object can carry pose keypoints, face points, embeddings, JPEG crops, attributes, and subboxes.
- Per-frame `tracker_debug` includes detection arrays, ROI/motion fields, Kalman predictions, and additional overlays used by `display_trackset`.
- Timestamps (`frame_time`) align with the original video so you can replay recorded outputs or compute per-frame metrics without rerunning the tracker.

Use `TrackSet.export_yaml` to regenerate this format from any tracker run (including `utrack`/`upyc`), optionally writing a rendered video via OpenCV.

---

## 7. Testing & Validation

- Use `--test` to benchmark trackers across datasets with caching and summarized tables.
- The `--metrics c` option leverages `ubon_pycstuff.c_mota_metrics` for faster metric computation (requires the Ubon C stack).
- `--compare` prints per-frame stats and invoked a visualization workflow for side-by-side diagnostics.
- `--motiontracker-test` exercises the `MotionTracker` to verify ROI predictions and optical flow behavior.

---

## 8. License

Dual license:
- **AGPL** (non-commercial)
- **Ubon Cooperative License** ([https://github.com/ubonpartners/license/blob/main/LICENSE](https://github.com/ubonpartners/license/blob/main/LICENSE))

Questions? contact bernandocribbenza@gmail.com (subject `yolo-dpa question`).
