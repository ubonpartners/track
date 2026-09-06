"""Out the Window (OTW) annotations.csv rows -> TrackSet.

Moved verbatim from TrackSetImportersMixin.import_otw (repo_cleanup.md
stage 3): `read_into(ts, ...)` is the original body with `self` renamed
`ts`; `read(...)` builds a fresh TrackSet and fills it.
"""
import cv2

from src.core.trackset import TrackSet


def read(video_id, rows, video_path):
    ts = TrackSet()
    read_into(ts, video_id, rows, video_path)
    return ts


def read_into(ts, video_id, rows, video_path):
    """Import a single Out the Window (OTW) clip into TrackSet format.

    OTW (https://stresearch.github.io/otw/) is crowd-sourced surveillance
    footage shot from building windows. Each collection (homes/, lots/)
    has one annotations.csv covering all its videos, with rows:
      video_id, activity_id, actor_id, type, frame, xmin, ymin, xmax,
      ymax, labeled
    Boxes are pixel xyxy at native video resolution. Object boxes are
    keyframed at activity begin/end and machine-interpolated between
    (labeled=False rows), so they arrive densely per-frame. A row whose
    type is an activity name (e.g. "Opening Door") is the enclosing
    activity region, not an object, and is skipped — as are prop objects
    (cell phone, bags, carts, doors), which overlap the people holding
    them and would poison person eval if kept as ignore regions.

    Class scheme matches the MOT/PersonPath/MEVA importers
    (["person", "vehicle", "other"]).

    Track ids: OTW actor ids cover every object of the actor's activity —
    actor 00039 has both a "person" and a "bicycle" box — and the same
    actor re-appears in overlapping activities (person carrying + talking
    on phone gives two person rows for one frame). Tracks are therefore
    keyed by (actor id, object type) and renumbered sequentially, which
    merges duplicate rows and keeps person/vehicle tracks separate. Where
    a human-labeled and an interpolated row collide on the same frame,
    the human-labeled box wins.

    Note the lots/ collection annotates only vehicle activity regions
    (actor id None, no object boxes) — importing it yields an empty
    trackset; only homes/ carries object tracks.

    video_id: the CSV video id (used only for error messages)
    rows: this video's CSV rows, each a list of the 10 fields as strings
    video_path: filesystem path to the source mp4
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open OTW video at {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0:
        fps = 30.0
    if width <= 0 or height <= 0:
        raise ValueError(f"Could not read frame size of {video_path}")

    object_classes = {
        "person": 0,
        "car": 1,
        "truck": 1,
        "vehicle": 1,
        "motorcycle": 1,
        "scooter": 1,
        "motorcycle/scooter": 1,
        "bicycle": 1,
    }

    track_id_map = {}
    labeled_at = set()
    by_frame = {}
    for row in rows:
        if len(row) < 9 or str(row[0]).startswith("#"):
            continue
        obj_type = str(row[3]).strip().lower()
        out_cl = object_classes.get(obj_type)
        if out_cl is None:
            continue  # activity region, prop object, or unknown label
        actor = str(row[2]).strip()
        if actor == "" or actor.lower() == "none":
            continue
        try:
            frame_zero = int(float(row[4]))
            x1p, y1p, x2p, y2p = (float(v) for v in row[5:9])
        except (TypeError, ValueError):
            continue
        frame_id = frame_zero + 1  # 1-based, matching the other importers
        if frame_id < 1:
            continue
        key = (actor, obj_type)
        if key not in track_id_map:
            track_id_map[key] = len(track_id_map) + 1
        track_id = track_id_map[key]
        labeled = str(row[9]).strip().lower() == "true" if len(row) > 9 else False
        if (frame_id, track_id) in labeled_at and not labeled:
            continue  # keep the human-labeled box over an interpolated one
        if labeled:
            labeled_at.add((frame_id, track_id))
        x1 = round(max(0.0, min(1.0, x1p / width)), 4)
        y1 = round(max(0.0, min(1.0, y1p / height)), 4)
        x2 = round(max(0.0, min(1.0, x2p / width)), 4)
        y2 = round(max(0.0, min(1.0, y2p / height)), 4)
        if x2 <= x1 or y2 <= y1:
            continue
        by_frame.setdefault(frame_id, {})[track_id] = {
            "box": [x1, y1, x2, y2],
            "class": out_cl,
            "conf": 1.0,
        }

    ts.metadata = {
        "frame_rate": fps,
        "width": width,
        "height": height,
        "classes": ["person", "vehicle", "other"],
        # OTW keyframed boxes cover the visible extent only
        "box_convention": "visible",
        "original_video": video_path,
    }
    ts.frames = []
    ts.frame_times = []
    for frame_id in sorted(by_frame.keys()):
        frame_time = (frame_id - 1) / fps
        ts.frames.append({
            "frame_id": frame_id,
            "frame_time": frame_time,
            "objects": by_frame[frame_id],
        })
        ts.frame_times.append(frame_time)
