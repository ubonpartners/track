"""RoundaboutHD SCT_GT.txt -> TrackSet.

Moved verbatim from TrackSetImportersMixin.import_roundabouthd (repo_cleanup.md
stage 3): `read_into(ts, ...)` is the original body with `self` renamed
`ts`; `read(...)` builds a fresh TrackSet and fills it.
"""
import cv2

from src.trackset import TrackSet


def read(sct_path, video_path):
    ts = TrackSet()
    read_into(ts, sct_path, video_path)
    return ts


def read_into(ts, sct_path, video_path):
    """Import one RoundaboutHD camera (SCT_GT.txt + 4K mp4).

    RoundaboutHD (Bath research data 1574, MIT): 4 non-overlapping
    4K@15fps cameras over a roundabout, 10 min each, human-inspected
    vehicle tracking GT. SCT_GT rows are space-separated
    `frame_id track_id xmin ymin xmax ymax cls` in pixels, frames
    1-based (SCT frame 1 == video frame 0). Vehicles only.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 15.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Could not read frame size of {video_path}")

    by_frame = {}
    max_frame = 0
    with open(sct_path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 7:
                continue
            fi = int(parts[0]) - 1
            x1, y1, x2, y2 = (float(v) for v in parts[2:6])
            if x2 <= x1 or y2 <= y1:
                continue
            by_frame.setdefault(fi, {})[parts[1]] = {
                "box": [round(x1 / width, 4), round(y1 / height, 4),
                        round(x2 / width, 4), round(y2 / height, 4)],
                "class": 1,
                "conf": 1.0,
            }
            max_frame = max(max_frame, fi)

    n = frame_count if frame_count > 0 else max_frame + 1
    ts.metadata = {
        "frame_rate": fps,
        "width": width,
        "height": height,
        "classes": ["person", "vehicle", "other"],
        "original_video": video_path,
        "box_convention": "visible",
    }
    ts.frames = [{"frame_id": i,
                    "frame_time": i / fps,
                    "objects": by_frame.get(i, {})}
                   for i in range(n)]
