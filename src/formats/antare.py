"""Antare labelled clips: MOT16-style gt.txt + labels.txt with dense
per-frame human GT -> track annotation dict (repo_cleanup.md stage 3;
moved verbatim from src/import_antare.py). The importer that walks the
drops, copies video and writes files stays in src/import_antare.py.

gt.txt frame k is 1-based and IS the video frame ordinal (frame k ==
decoded frame k-1); coordinates are source pixels.
"""
import os

# class-name -> our GT scheme. Anything unlisted -> "other" (ignore-region
# semantics: not an FN when missed, matches not charged as FP).
CLASS_MAP = {
    "person": "person",
    "bicycle": "vehicle",     # REVISABLE: rideables as vehicles
    "car": "vehicle",
    "motorbike": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "other": "other",
}
GT_CLASSES = ["person", "vehicle", "other"]


def load_gt(gt_dir):
    """gt.txt + labels.txt -> [(frame_1based, track_id, x, y, w, h, cls_name)]."""
    labels = [l.strip() for l in open(os.path.join(gt_dir, "labels.txt"))
              if l.strip()]
    rows = []
    for line in open(os.path.join(gt_dir, "gt.txt")):
        parts = line.strip().split(",")
        if len(parts) < 8:
            continue
        fr, tid = int(parts[0]), int(parts[1])
        x, y, w, h = (float(v) for v in parts[2:6])
        cls_i = int(parts[7])
        name = labels[cls_i - 1] if 1 <= cls_i <= len(labels) else "other"
        rows.append((fr, tid, x, y, w, h, name))
    return rows


def _clip01(v):
    return min(1.0, max(0.0, v))


def build_annotation(rows, n_frames, fps, src_w, src_h, video_path,
                     hint="bodycam", scene=None):
    """Dense GT rows -> trackset json dict with one record per video frame
    (frame_id k, frame_time (k-1)/fps), boxes normalised+clipped."""
    by_frame = {k: {} for k in range(1, n_frames + 1)}
    dropped = 0
    for (fr, tid, x, y, w, h, name) in rows:
        if fr < 1 or fr > n_frames:
            dropped += 1
            continue
        cls = CLASS_MAP.get(name, "other")
        box = [_clip01(x / src_w), _clip01(y / src_h),
               _clip01((x + w) / src_w), _clip01((y + h) / src_h)]
        by_frame[fr][str(tid)] = {"box": [round(v, 5) for v in box],
                                  "class": GT_CLASSES.index(cls), "conf": 1.0}
    frames = [{"frame_id": k, "frame_time": round((k - 1) / fps, 6),
               "objects": by_frame[k]} for k in range(1, n_frames + 1)]
    doc = {
        "metadata": {
            "frame_rate": fps,
            "width": src_w, "height": src_h,
            "classes": GT_CLASSES,
            "original_video": video_path,
            "hint": hint,                    # camera class; derive reads it per clip
            "gt_source": {"kind": "human_dense_mot",
                          "source": "antare labelled clips",
                          "scene": scene,
                          "frame_mapping": "gt frame k == video frame k-1"},
        },
        "frames": frames,
    }
    return doc, dropped
