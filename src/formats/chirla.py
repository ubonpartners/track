"""CHIRLA per-frame JSON -> TrackSet.

Moved verbatim from TrackSetImportersMixin.import_chirla (repo_cleanup.md
stage 3): `read_into(ts, ...)` is the original body with `self` renamed
`ts`; `read(...)` builds a fresh TrackSet and fills it.
"""
import json
import cv2

from src.core.trackset import TrackSet


def read(annotation_json_path, video_path):
    ts = TrackSet()
    read_into(ts, annotation_json_path, video_path)
    return ts


def read_into(ts, annotation_json_path, video_path):
    """Import one CHIRLA camera clip (per-frame JSON + avi).

    CHIRLA (CC BY 4.0, bdager/CHIRLA on HF): indoor multi-camera
    re-id/tracking; annotation is {frame_number(1-based, str):
    [{"id": person_id, "BboxP": [x1,y1,x2,y2] pixels}, ...]} with
    one key per video frame (empty list = nobody visible). Person
    ids are globally consistent across cameras/sequences and serve
    as track ids within a clip. Persons only.
    """
    with open(annotation_json_path) as fh:
        anno = json.load(fh)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Could not read frame size of {video_path}")

    n = max(frame_count, max((int(k) for k in anno), default=0))
    if frame_count and abs(n - frame_count) > 2:
        raise ValueError(
            f"annotation/video frame count mismatch: {n} keys vs "
            f"{frame_count} frames in {video_path}")

    ts.metadata = {
        "frame_rate": fps,
        "width": width,
        "height": height,
        "classes": ["person", "vehicle", "other"],
        "original_video": video_path,
        "box_convention": "visible",
    }
    ts.frames = []
    for i in range(n):
        objects = {}
        for o in anno.get(str(i + 1), []):
            x1, y1, x2, y2 = o["BboxP"]
            if x2 <= x1 or y2 <= y1:
                continue
            objects[str(o["id"])] = {
                "box": [round(x1 / width, 4), round(y1 / height, 4),
                        round(x2 / width, 4), round(y2 / height, 4)],
                "class": 0,
                "conf": 1.0,
            }
        ts.frames.append({"frame_id": i,
                            "frame_time": i / fps,
                            "objects": objects})
