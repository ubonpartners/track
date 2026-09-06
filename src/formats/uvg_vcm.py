"""UVG-VCM annotation JSON -> TrackSet.

Moved verbatim from TrackSetImportersMixin.import_uvg_vcm (repo_cleanup.md
stage 3): `read_into(ts, ...)` is the original body with `self` renamed
`ts`; `read(...)` builds a fresh TrackSet and fills it.
"""
import json

from src.trackset import TrackSet


def read(annotation_json_path, video_path, width, height, fps):
    ts = TrackSet()
    read_into(ts, annotation_json_path, video_path, width, height, fps)
    return ts


def read_into(ts, annotation_json_path, video_path,
                   width, height, fps):
    """Import one UVG-VCM sequence (annotation JSON + mp4).

    UVG-VCM (Tampere UVG, CC BY 4.0): 4K@60fps codec-benchmark
    sequences with professional tracking annotations. JSON is
    {"version": "1.0", "1": [obj, ...], ...} keyed by 1-based frame
    number; objects carry COCO class_id, persistent track_id and
    normalized x_min/y_min/x_max/y_max corners. COCO 1 -> person,
    {2,3,4,6,8} (bicycle/car/motorcycle/bus/truck) -> vehicle,
    everything else dropped. video_path is the pre-transcoded mp4
    (the dataset ships raw YUV; see convert_uvg_vcm).
    """
    with open(annotation_json_path) as fh:
        anno = json.load(fh)
    frame_keys = sorted((int(k) for k in anno if k.isdigit()))
    n = frame_keys[-1] if frame_keys else 0
    class_map = {1: 0, 2: 1, 3: 1, 4: 1, 6: 1, 8: 1}

    ts.metadata = {
        "frame_rate": float(fps),
        "width": int(width),
        "height": int(height),
        "classes": ["person", "vehicle", "other"],
        "original_video": video_path,
        "box_convention": "visible",
    }
    ts.frames = []
    for i in range(n):
        objects = {}
        for o in anno.get(str(i + 1), []):
            cl = class_map.get(int(o["class_id"]))
            if cl is None:
                continue
            x1, y1 = float(o["x_min"]), float(o["y_min"])
            x2, y2 = float(o["x_max"]), float(o["y_max"])
            if x2 <= x1 or y2 <= y1:
                continue
            objects[str(o["track_id"])] = {
                "box": [round(x1, 4), round(y1, 4),
                        round(x2, 4), round(y2, 4)],
                "class": cl,
                "conf": 1.0,
            }
        ts.frames.append({"frame_id": i,
                            "frame_time": i / float(fps),
                            "objects": objects})
