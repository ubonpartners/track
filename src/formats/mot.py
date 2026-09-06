"""MOTChallenge seqinfo.ini + gt/gt.txt -> TrackSet.

Moved verbatim from TrackSetImportersMixin.import_mot (repo_cleanup.md
stage 3): `read_into(ts, ...)` is the original body with `self` renamed
`ts`; `read(...)` builds a fresh TrackSet and fills it.
"""
import configparser
import os
import numpy as np

from src.trackset import TrackSet


def read(seqinfo_path):
    ts = TrackSet()
    read_into(ts, seqinfo_path)
    return ts


def read_into(ts, seqinfo_path):
    config = configparser.ConfigParser()
    config.read(seqinfo_path)

    seq_dir = os.path.dirname(seqinfo_path)
    frame_rate = int(config['Sequence']['frameRate'])
    seq_length = int(config['Sequence']['seqLength'])
    frame_height = int(config['Sequence']['imHeight'])
    frame_width = int(config['Sequence']['imWidth'])
    image_dir = os.path.join(seq_dir, config['Sequence']['imDir'])
    image_ext = config['Sequence']['imExt']

    # Load the ground truth annotations (gt.txt)
    gt_path = os.path.join(seq_dir, "gt/gt.txt")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found at {gt_path}")

    # Parse ground truth annotations
    data = np.loadtxt(gt_path, delimiter=',')
    # Columns: frame_id, track_id, bb_left, bb_top, bb_width, bb_height, confidence, class, visibility

    ts.frames = []
    ts.metadata={
            "frame_rate": frame_rate,
            "width": frame_width,
            "height": frame_height,
            "classes": ["person", "vehicle", "other"],
            "box_convention": "fullbody",
        }

    for frame_id in range(1, seq_length):
        frame_time = (frame_id-1) / frame_rate
        objects = {}

        # Filter objects for the current frame
        frame_objects = data[data[:, 0] == frame_id]

        for obj in frame_objects:
            track_id = int(obj[1])
            bb_left = float(obj[2])
            bb_top = float(obj[3])
            bb_width = float(obj[4])
            bb_height = float(obj[5])
            confidence = 1 #round(float(obj[6]),4)
            cl = int(obj[7])

            # Convert bounding box to xyxy format in normalized coordinates
            x1 = round(bb_left / frame_width,4)
            y1 = round(bb_top / frame_height,4)
            x2 = round((bb_left + bb_width) / frame_width,4)
            y2 = round((bb_top + bb_height) / frame_height,4)

            #1 Pedestrian
            #2 Person on vehicle
            #3 Car
            #4 Bicycle
            #5 Motorbike
            #6 Non motorized vehicle
            #7 Static person
            #8 Distractor
            #9 Occluder
            #10 Occluder on the ground
            #11 Occluder full
            #12 Reflection
            if cl==1 or cl==7 or cl==2:
                out_cl=0
            elif cl==3 or cl==4 or cl==5 or cl==6:
                out_cl=1
            else:
                out_cl=2

            objects[track_id] = {"box": [x1, y1, x2, y2], "class":out_cl, "conf":confidence}

        ts.frames.append({
            "frame_id": frame_id,
            "frame_time": frame_time,
            "image_path": os.path.join(image_dir, f"{(frame_id):06d}{image_ext}"),
            "objects": objects
        })

    ts.frame_times=[]
    for f in ts.frames:
        ts.frame_times.append(f["frame_time"])
