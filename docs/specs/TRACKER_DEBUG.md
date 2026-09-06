# Tracker Result + Debug Format (UBTRK2)

This document defines the tracker run format shared by:

- `track` (Python orchestration, replay, metrics)
- `ubon_cstuff` (runtime tracking + debug emit)
- `stuff` (generic UBTRK2 container utilities)

The design goal is one extensible, efficient, append-friendly format for:

- persisted tracker runs
- network transport
- offline tracker analysis (Kalman behavior, matching behavior, similarity vectors)

---

## Hard Cut-Over Rules

- Canonical stored run format is **UBTRK2** only.
- Old framed MessagePack format is removed.
- Input sequence metadata files remain supported (`.ini`, `.json/.yaml`, `.vbb`).
- No sidecar debug files: frame debug is stored inline in each frame box.

---

## Runtime Schema (`ubon_cstuff` -> Python)

`c_track_stream.get_results(wait, include_full_debug=False)` returns a list of per-frame dicts:

- `result_type`
- `time`
- `motion_score`
- `motion_roi`
- `inference_roi`
- `track_dets`
- `inference_dets`
- `debug`

Detection dicts can include:

- `class`, `confidence`, `track_id`, `box`
- `subbox`, `subbox_conf`
- `face_points`, `pose_points`, `attrs`
- `reid_vector`, `face_embedding`, `clip_embedding`
- `face_jpeg`, `clip_jpeg`, `fiqa_score`

`include_full_debug=True` currently adds extra debug overlays including:

- `tracking_output` (`detections`)
- `motion_field` (dense flow payload)

---

## UBTRK2 Container

UBTRK2 is a BMFF-like hierarchical TLV container:

- big-endian length prefix (`uint32`) including header
- 4CC box type (`char[4]`)
- payload bytes

Top-level box order:

1. `ubtf` : format/version header
2. `meta` : UTF-8 YAML metadata
3. repeated `fram` boxes (one per processed frame)

### Frame (`fram`) Child Boxes

- `fhdr`: frame header map
  - `frame_time`, `result_type`, `motion_score`, `motion_roi`, `inference_roi`
- `trks`: tracked objects map (`track_id` key -> object record)
- `dets`: optional detector output list
- `dbug`: debug map (`name -> {type, data}`)
- `imgp`: optional source image path
- `xtra`: optional forward-compatible extra fields map

### Payload Encoding

`stuff.ubtrk2` uses a compact built-in typed value codec (no third-party serializer):

- null / bool / int64 / float64
- UTF-8 string
- raw bytes
- list
- dict (string keys)

Large tensors/blobs use inline payload wrappers:

- ndarray payload:
  - `{"__payload_kind__":"ndarray","dtype":"...","shape":[...],"codec":"raw","data":<bytes>}`
- bytes payload:
  - `{"__payload_kind__":"bytes","mime":"...","codec":"raw","data":<bytes>}`

Wrappers are decoded by `stuff.decode_payload` / `stuff.decode_nested_payloads`.

---

## TrackSet Mapping (`track/src/core/trackset.py`)

`TrackSet.export_track_file(path)` writes:

- file metadata (`schema_version`, dimensions, class list, source video)
- one UBTRK2 frame per processed frame

`TrackSet.import_track_file(path)` reads UBTRK2 and rebuilds:

- `metadata`
- `frame_times`
- `frames` (`objects`, debug maps, ROIs, frame status)

Track object IDs are serialized as string keys in `trks` and restored to `int` on read.

---

## Debug Overlay Types Used By Viewer

The viewer currently renders these debug types when present:

- `detections`
- `roi`
- `box_prediction`
- `motion_field`
- `cost_map`

`motion_roi` and `inference_roi` are auto-exposed as `roi` overlays during replay.

---

## Ownership

- `stuff`: UBTRK2 container and payload codec implementation
- `ubon_cstuff`: runtime result/debug emission
- `track`: TrackSet schema mapping, replay behavior, metrics workflows

