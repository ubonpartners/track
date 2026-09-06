"""JAAD CVAT XML -> TrackSet.

Moved verbatim from TrackSetImportersMixin.import_jaad (repo_cleanup.md
stage 3): `read_into(ts, ...)` is the original body with `self` renamed
`ts`; `read(...)` builds a fresh TrackSet and fills it.
"""
import cv2
import xml.etree.ElementTree as ET

from src.trackset import TrackSet


def read(annotation_xml_path, video_path):
    ts = TrackSet()
    read_into(ts, annotation_xml_path, video_path)
    return ts


def read_into(ts, annotation_xml_path, video_path):
    """Import a single JAAD clip (XML + mp4) into TrackSet format.

    JAAD has three pedestrian-like labels:
      - pedestrian / ped: individual people
      - people: grouped crowd region
    We map grouped crowd regions to class "other" so they can be consumed
    as MOT-style ignore regions during metric evaluation.
    """
    tree = ET.parse(annotation_xml_path)
    root = tree.getroot()

    task = root.find("meta/task")
    xml_seq_length = 0
    xml_width = 0
    xml_height = 0
    if task is not None:
        try:
            xml_seq_length = int(task.findtext("size") or 0)
        except (TypeError, ValueError):
            xml_seq_length = 0
        original_size = task.find("original_size")
        if original_size is not None:
            try:
                xml_width = int(original_size.findtext("width") or 0)
            except (TypeError, ValueError):
                xml_width = 0
            try:
                xml_height = int(original_size.findtext("height") or 0)
            except (TypeError, ValueError):
                xml_height = 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open JAAD video at {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0:
        fps = 30.0
    if frame_width <= 0:
        frame_width = xml_width
    if frame_height <= 0:
        frame_height = xml_height
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(f"Could not infer frame size for {annotation_xml_path}")

    seq_length = xml_seq_length if xml_seq_length > 0 else frame_count
    if seq_length <= 0:
        seq_length = 0

    label_to_class = {
        "pedestrian": 0,
        "ped": 0,
        "people": 2,  # grouped pedestrians -> ignore region ("other")
    }
    track_id_map = {}
    next_track_id = 1
    by_frame = {}
    max_frame_seen = 0

    def mapped_track_id(raw_track_id):
        nonlocal next_track_id
        if raw_track_id not in track_id_map:
            track_id_map[raw_track_id] = next_track_id
            next_track_id += 1
        return track_id_map[raw_track_id]

    for track_idx, track in enumerate(root.findall("track")):
        label = (track.attrib.get("label") or "").strip().lower()
        out_cl = label_to_class.get(label)
        if out_cl is None:
            continue

        boxes = track.findall("box")
        if len(boxes) == 0:
            continue

        raw_track_key = None
        for box in boxes:
            for attr in box.findall("attribute"):
                if attr.attrib.get("name") == "id":
                    text = (attr.text or "").strip()
                    if text:
                        raw_track_key = text
                        break
            if raw_track_key is not None:
                break
        if raw_track_key is None:
            raw_track_key = f"{label}_track_{track_idx}"
        track_id = mapped_track_id(raw_track_key)

        for box in boxes:
            try:
                if int(box.attrib.get("outside", "0")) != 0:
                    continue
                frame_zero = int(box.attrib["frame"])
                xtl = float(box.attrib["xtl"])
                ytl = float(box.attrib["ytl"])
                xbr = float(box.attrib["xbr"])
                ybr = float(box.attrib["ybr"])
            except (KeyError, TypeError, ValueError):
                continue

            frame_id = frame_zero + 1
            if frame_id < 1:
                continue
            if seq_length > 0 and frame_id > seq_length:
                continue

            max_frame_seen = max(max_frame_seen, frame_id)

            x1 = round(max(0.0, min(1.0, xtl / frame_width)), 4)
            y1 = round(max(0.0, min(1.0, ytl / frame_height)), 4)
            x2 = round(max(0.0, min(1.0, xbr / frame_width)), 4)
            y2 = round(max(0.0, min(1.0, ybr / frame_height)), 4)
            if x2 <= x1 or y2 <= y1:
                continue

            by_frame.setdefault(frame_id, {})[track_id] = {
                "box": [x1, y1, x2, y2],
                "class": out_cl,
                "conf": 1.0,
            }

    if seq_length <= 0:
        seq_length = max_frame_seen

    ts.metadata = {
        "frame_rate": fps,
        "width": frame_width,
        "height": frame_height,
        "classes": ["person", "vehicle", "other"],
        "original_video": video_path,
        "box_convention": "fullbody",
    }
    ts.frames = []
    ts.frame_times = []
    for frame_id in range(1, seq_length + 1):
        frame_time = (frame_id - 1) / fps
        ts.frames.append({
            "frame_id": frame_id,
            "frame_time": frame_time,
            "objects": by_frame.get(frame_id, {}),
        })
        ts.frame_times.append(frame_time)
