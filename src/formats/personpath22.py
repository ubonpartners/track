"""PersonPath22 (gluoncv-motion JSON) -> TrackSet.

Moved verbatim from TrackSetImportersMixin.import_personpath22 (repo_cleanup.md
stage 3): `read_into(ts, ...)` is the original body with `self` renamed
`ts`; `read(...)` builds a fresh TrackSet and fills it.
"""
from src.core.trackset import TrackSet


def read(sample_uid, sample_dict, video_path):
    ts = TrackSet()
    read_into(ts, sample_uid, sample_dict, video_path)
    return ts


def read_into(ts, sample_uid, sample_dict, video_path):
    """Import a single PersonPath22 sample (gluoncv-motion JSON format).

    sample_uid: the key from the top-level samples dict (e.g. "uid_vid_00001.mp4")
    sample_dict: the corresponding value with "metadata" and "entities"
    video_path: filesystem path to the source mp4 (will be copied by export_yaml)
    """
    meta = sample_dict["metadata"]
    fps = float(meta["fps"])
    resolution = meta.get("resolution", {})
    width = int(resolution.get("width", meta.get("width")))
    height = int(resolution.get("height", meta.get("height")))
    n_frames = int(meta.get("number_of_frames", meta.get("num_frames", 0)))
    entities = sample_dict.get("entities", []) or []

    # PersonPath22 annotates only sparse keyframes (~every 5th frame).
    # We emit just those keyframes — TrackSet.objects_at_time already
    # interpolates between bracketing frames, so dense filling here is
    # redundant and would also leak interpolated boxes into the gt.
    #
    # Class scheme matches MOT (["person","vehicle","other"]) so the same
    # downstream eval/training code works:
    #   - person / sitting_person / standing_person / etc. → class 0
    #   - crowd boxes (region covering an unannotated group) → class 2,
    #     intended to be used as ignore regions during evaluation
    #   - reflection entities are dropped entirely
    by_frame = {}
    for ent in entities:
        bb = ent.get("bb")
        if bb is None or len(bb) != 4:
            continue
        labels = ent.get("labels") or {}
        if labels.get("person"):
            out_cl = 0
        elif labels.get("crowd"):
            out_cl = 2
        else:
            # reflection or other non-person/non-crowd — drop
            continue
        blob = ent.get("blob") or {}
        if "frame_idx" in blob:
            frame_id = int(blob["frame_idx"]) + 1
        else:
            time_ms = ent.get("time")
            if time_ms is None:
                continue
            frame_id = int(round(float(time_ms) * fps / 1000.0)) + 1
        if frame_id < 1:
            frame_id = 1
        if n_frames and frame_id > n_frames:
            continue
        try:
            track_id = int(ent["id"])
        except (KeyError, TypeError, ValueError):
            continue
        x, y, w, h = (float(v) for v in bb)
        x1 = round(x / width, 4)
        y1 = round(y / height, 4)
        x2 = round((x + w) / width, 4)
        y2 = round((y + h) / height, 4)
        confidence = ent.get("confidence", 1.0)
        try:
            confidence = round(float(confidence), 4)
        except (TypeError, ValueError):
            confidence = 1.0
        by_frame.setdefault(frame_id, {})[track_id] = {
            "box": [x1, y1, x2, y2],
            "class": out_cl,
            "conf": confidence,
        }

    ts.metadata = {
        "frame_rate": fps,
        "width": width,
        "height": height,
        "classes": ["person", "vehicle", "other"],
        "original_video": video_path,
        "box_convention": "visible",
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
