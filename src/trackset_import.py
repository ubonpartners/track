"""Dataset importers and batch converters for TrackSet.

Each dataset gets a TrackSet.import_<name> method (collected here in
TrackSetImportersMixin) that fills one TrackSet from that dataset's native
annotation format, plus a convert_<name>() batch function that walks the
downloaded dataset under /mldata/downloaded_datasets and writes
MOT-equivalent trackset JSON (+ mp4 where applicable) pairs under
/mldata/tracking/<name>/. The canonical trackset formats themselves
(UBTRK2, yaml/json) live on TrackSet in trackset.py.
"""

import configparser
import csv
import cv2
import json
import os
import numpy as np
import xml.etree.ElementTree as ET
import yaml
from scipy.io import loadmat

import stuff

# libyaml-backed loader when available: ~10x faster on MEVA's multi-MB KPF
# files, identical output to yaml.safe_load
_YAML_SAFE_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

def _yaml_safe_load(stream):
    return yaml.load(stream, Loader=_YAML_SAFE_LOADER)


class TrackSetImportersMixin:
    """Dataset-specific import_* methods mixed into TrackSet."""

    def import_caltech_pedestrian(self, path):
        pass

    def import_personpath22(self, sample_uid, sample_dict, video_path):
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

        self.metadata = {
            "frame_rate": fps,
            "width": width,
            "height": height,
            "classes": ["person", "vehicle", "other"],
            "original_video": video_path,
            "box_convention": "visible",
        }
        self.frames = []
        self.frame_times = []
        for frame_id in sorted(by_frame.keys()):
            frame_time = (frame_id - 1) / fps
            self.frames.append({
                "frame_id": frame_id,
                "frame_time": frame_time,
                "objects": by_frame[frame_id],
            })
            self.frame_times.append(frame_time)

    def import_meva(self, geom_path, types_path=None, video_path=None,
                    width=None, height=None, frame_rate=30.0):
        """Import a fully-annotated MEVA (KF1) clip from its KPF geometry.

        MEVA ships per-clip KPF YAML: `<clip>.geom.yml` = per-keyframe boxes
        (`g0: "x1 y1 x2 y2"` in pixels, `id1` = track id, `ts0` = 0-based frame
        index), `<clip>.types.yml` = per-track object class. The annotations are
        keyframed (not every frame); TrackSet.objects_at_time interpolates
        between bracketing frames by track id, so a sparse ground-truth track
        renders continuously — no per-frame filling here.

        The MEVA dataset lives at /mldata/downloaded_datasets/other/MEVA
        (annotations/ + videos/ + krtd/ + map/, alongside MOT17/MOT20/JAAD).

        geom_path:  path to `<clip>.geom.yml`.
        types_path: `<clip>.types.yml` (defaults to the sibling of geom_path).
        video_path: the source video (mp4/avi). When given, its exact
                    width/height/fps are read (MEVA mixes 1920x1072 and
                    1920x1080); otherwise width/height/frame_rate are used.
        """
        if types_path is None:
            cand = geom_path.replace(".geom.yml", ".types.yml").replace(".geom.yaml", ".types.yaml")
            types_path = cand if os.path.exists(cand) else None

        # Auto-locate the source video: MEVA names the clip identically across
        # annotation and video (`<clip>.geom.yml` vs `<clip>.r13.avi`). Look
        # beside the geom and in a sibling videos/ dir.
        if video_path is None:
            import glob as _glob
            stem = os.path.basename(geom_path)
            for suf in (".geom.yml", ".geom.yaml"):
                if stem.endswith(suf):
                    stem = stem[: -len(suf)]
                    break
            gd = os.path.dirname(os.path.abspath(geom_path))
            for d in (gd, os.path.join(gd, "..", "videos"), os.path.join(gd, "..", "video"),
                      os.path.join(os.path.dirname(gd), "videos")):
                hits = _glob.glob(os.path.join(d, stem + "*.avi")) + _glob.glob(os.path.join(d, stem + "*.mp4"))
                if hits:
                    video_path = hits[0]
                    break

        # Exact frame geometry/rate from the video when available (the 1072 vs
        # 1080 difference shifts every box's normalization).
        if video_path is not None and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
            vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
            vf = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
            cap.release()
            if vw: width = vw
            if vh: height = vh
            if vf > 0: frame_rate = vf
        if width is None or height is None:
            raise ValueError("import_meva: give a video_path or explicit width/height "
                             "(MEVA clips are 1920x1072 or 1920x1080)")

        # types: track id -> output class. MEVA cset3 object types map to the
        # shared [person, vehicle, other] scheme so downstream eval/training is
        # identical to the MOT/PersonPath importers.
        def out_class(cset):
            if not cset:
                return 2
            top = max(cset, key=cset.get)
            if top == "person":
                return 0
            if top in ("vehicle", "bike"):
                return 1
            return 2

        cls_of = {}
        if types_path is not None:
            for row in (_yaml_safe_load(open(types_path)) or []):
                t = row.get("types")
                if not t:
                    continue
                cls_of[t.get("id1")] = out_class(t.get("cset3") or t.get("cset2") or {})

        by_frame = {}
        for row in (_yaml_safe_load(open(geom_path)) or []):
            g = row.get("geom")
            if not g:
                continue
            g0 = g.get("g0")
            ts0 = g.get("ts0")
            tid = g.get("id1")
            if g0 is None or ts0 is None or tid is None:
                continue
            try:
                x1, y1, x2, y2 = (float(v) for v in str(g0).split())
            except ValueError:
                continue
            frame_id = int(ts0) + 1  # 1-based, matching the MOT/PersonPath importers
            by_frame.setdefault(frame_id, {})[int(tid)] = {
                "box": [round(x1 / width, 4), round(y1 / height, 4),
                        round(x2 / width, 4), round(y2 / height, 4)],
                "class": cls_of.get(tid, 2),
                "conf": 1.0,
            }

        self.metadata = {
            "frame_rate": frame_rate,
            "width": int(width),
            "height": int(height),
            "classes": ["person", "vehicle", "other"],
        }
        if video_path is not None:
            self.metadata["original_video"] = video_path
        self.frames = []
        self.frame_times = []
        for frame_id in sorted(by_frame.keys()):
            frame_time = (frame_id - 1) / frame_rate
            self.frames.append({
                "frame_id": frame_id,
                "frame_time": frame_time,
                "objects": by_frame[frame_id],
            })
            self.frame_times.append(frame_time)

    def import_otw(self, video_id, rows, video_path):
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

        self.metadata = {
            "frame_rate": fps,
            "width": width,
            "height": height,
            "classes": ["person", "vehicle", "other"],
            "original_video": video_path,
        }
        self.frames = []
        self.frame_times = []
        for frame_id in sorted(by_frame.keys()):
            frame_time = (frame_id - 1) / fps
            self.frames.append({
                "frame_id": frame_id,
                "frame_time": frame_time,
                "objects": by_frame[frame_id],
            })
            self.frame_times.append(frame_time)

    def import_egohumans(self, video_name, images, annotations, image_root,
                         frame_rate=20.0):
        """Import one EgoHumans benchmark video (COCO-video/TAO style).

        EgoHumans (github.com/rawalkhirodkar/egohumans) is multi-view: 4
        head-mounted Aria cameras (ego, 1408x1408 fisheye rgb) + up to 15
        static GoPros (exo, 3840x2160) capturing the same scene at 20 FPS.
        The tracking benchmark ships one COCO-video JSON per camera family
        (coco_track/tracking_{exo,ego_rgb,ego_slam}.json) over the test
        sequences 01_tagging/02_legoassemble/03_fencing, with a "videos"
        entry per (subsequence, camera), "images" carrying file_name +
        video_id, and person-only "annotations" carrying pixel xywh bbox +
        instance_id.

        instance_id is the identity across all cameras of a subsequence, so
        track ids stay comparable between views. Image file names encode the
        capture-clock frame number (00017.jpg = frame 17, 1-based); a few
        capture frames are dropped, so frame_time is derived from the file
        name, not the dense COCO frame_id. Frames are kept as image_path
        references into image_root; convert_egohumans then encodes a
        constant-rate mp4 (repeating the previous image over capture gaps
        so video time stays aligned with frame_time).

        video_name: "videos" entry name, e.g. "tagging_001_tagging_aria01_rgb"
        images: this video's image dicts, annotations: its annotation dicts.
        image_root: directory that file_name paths are relative to (the
        dataset's data/ dir).
        """
        def capture_frame_of(image):
            stem = os.path.splitext(os.path.basename(image["file_name"]))[0]
            return int(stem)

        images = sorted(images, key=capture_frame_of)
        if len(images) == 0:
            raise ValueError(f"EgoHumans video {video_name} has no images")
        width = int(images[0]["width"])
        height = int(images[0]["height"])

        anns_by_image = {}
        for ann in annotations:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        self.metadata = {
            "frame_rate": frame_rate,
            "width": width,
            "height": height,
            "classes": ["person", "vehicle", "other"],
        }
        self.frames = []
        self.frame_times = []
        for image in images:
            frame_id = capture_frame_of(image)
            frame_time = (frame_id - 1) / frame_rate
            objects = {}
            for ann in anns_by_image.get(image["id"], []):
                x, y, w, h = (float(v) for v in ann["bbox"])
                x1 = round(max(0.0, min(1.0, x / width)), 4)
                y1 = round(max(0.0, min(1.0, y / height)), 4)
                x2 = round(max(0.0, min(1.0, (x + w) / width)), 4)
                y2 = round(max(0.0, min(1.0, (y + h) / height)), 4)
                if x2 <= x1 or y2 <= y1:
                    continue
                # person-only benchmark; crowd/ignore boxes -> "other"
                out_cl = 2 if (ann.get("iscrowd") or ann.get("ignore")) else 0
                objects[int(ann["instance_id"])] = {
                    "box": [x1, y1, x2, y2],
                    "class": out_cl,
                    "conf": 1.0,
                }
            self.frames.append({
                "frame_id": frame_id,
                "frame_time": frame_time,
                "image_path": os.path.join(image_root, image["file_name"]),
                "objects": objects,
            })
            self.frame_times.append(frame_time)

    def import_masa(self, json_path, min_score=0.0):
        """Import a MASA auto-label run (ubonpartners/masa fork,
        demo/video_to_trackset_json.py output) into TrackSet format.

        The JSON carries per-frame pixel-xyxy boxes with MASA track ids,
        scores and a label index into its "classes" list (the text-prompt
        phrases). Class names are mapped onto the corpus scheme
        ["person", "vehicle", "other"]: person->0, known vehicle names->1,
        anything else is DROPPED — "other" is reserved for ignore regions
        pre-annotated in source datasets, never for auto-detected objects.
        Scores are preserved as confidence so downstream eval can
        threshold.

        min_score: drop boxes below this confidence at import time (the dump
        itself uses a low floor so thresholds can be tuned downstream).
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        fps = float(data["fps"])
        width = int(data["width"])
        height = int(data["height"])
        class_names = [str(c).strip().lower() for c in data.get("classes", [])]
        vehicle_names = {"car", "truck", "bus", "motorcycle", "scooter",
                         "bicycle", "vehicle", "van"}

        def out_class(label):
            name = class_names[label] if 0 <= label < len(class_names) else ""
            if name == "person":
                return 0
            if name in vehicle_names:
                return 1
            return None  # not a class we use -> drop the box

        self.metadata = {
            "frame_rate": fps,
            "width": width,
            "height": height,
            "classes": ["person", "vehicle", "other"],
            "original_video": data.get("video"),
            "autolabel": {
                "source": "masa",
                "texts": data.get("texts"),
                "masa_config": data.get("masa_config"),
                "masa_checkpoint": data.get("masa_checkpoint"),
            },
        }
        self.frames = []
        self.frame_times = []
        for frame_idx, frame in enumerate(data["frames"]):
            objects = {}
            for obj in frame.get("objects", []):
                score = float(obj.get("score", 1.0))
                if score < min_score:
                    continue
                cl = out_class(int(obj.get("label", -1)))
                if cl is None:
                    continue
                x1, y1, x2, y2 = (float(v) for v in obj["bbox"])
                x1 = round(max(0.0, min(1.0, x1 / width)), 4)
                y1 = round(max(0.0, min(1.0, y1 / height)), 4)
                x2 = round(max(0.0, min(1.0, x2 / width)), 4)
                y2 = round(max(0.0, min(1.0, y2 / height)), 4)
                if x2 <= x1 or y2 <= y1:
                    continue
                objects[int(obj["track_id"])] = {
                    "box": [x1, y1, x2, y2],
                    "class": cl,
                    "conf": round(score, 4),
                }
            self.frames.append({
                "frame_id": frame_idx + 1,
                "frame_time": frame_idx / fps,
                "objects": objects,
            })
            self.frame_times.append(frame_idx / fps)

    def import_chirla(self, annotation_json_path, video_path):
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

        self.metadata = {
            "frame_rate": fps,
            "width": width,
            "height": height,
            "classes": ["person", "vehicle", "other"],
            "original_video": video_path,
            "box_convention": "visible",
        }
        self.frames = []
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
            self.frames.append({"frame_id": i,
                                "frame_time": i / fps,
                                "objects": objects})

    def import_roundabouthd(self, sct_path, video_path):
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
        self.metadata = {
            "frame_rate": fps,
            "width": width,
            "height": height,
            "classes": ["person", "vehicle", "other"],
            "original_video": video_path,
            "box_convention": "visible",
        }
        self.frames = [{"frame_id": i,
                        "frame_time": i / fps,
                        "objects": by_frame.get(i, {})}
                       for i in range(n)]

    def import_uvg_vcm(self, annotation_json_path, video_path,
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

        self.metadata = {
            "frame_rate": float(fps),
            "width": int(width),
            "height": int(height),
            "classes": ["person", "vehicle", "other"],
            "original_video": video_path,
            "box_convention": "visible",
        }
        self.frames = []
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
            self.frames.append({"frame_id": i,
                                "frame_time": i / float(fps),
                                "objects": objects})

    def import_jaad(self, annotation_xml_path, video_path):
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

        self.metadata = {
            "frame_rate": fps,
            "width": frame_width,
            "height": frame_height,
            "classes": ["person", "vehicle", "other"],
            "original_video": video_path,
            "box_convention": "fullbody",
        }
        self.frames = []
        self.frame_times = []
        for frame_id in range(1, seq_length + 1):
            frame_time = (frame_id - 1) / fps
            self.frames.append({
                "frame_id": frame_id,
                "frame_time": frame_time,
                "objects": by_frame.get(frame_id, {}),
            })
            self.frame_times.append(frame_time)

    def import_mot(self, seqinfo_path):
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

        self.frames = []
        self.metadata={
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

            self.frames.append({
                "frame_id": frame_id,
                "frame_time": frame_time,
                "image_path": os.path.join(image_dir, f"{(frame_id):06d}{image_ext}"),
                "objects": objects
            })

        self.frame_times=[]
        for f in self.frames:
            self.frame_times.append(f["frame_time"])


def extract_frames_from_seq(seq_file, output_video):
    cap = cv2.VideoCapture(seq_file)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 FPS if unspecified

    # Video writer to save MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, frame_rate, (frame_width, frame_height))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        out.write(frame)

    cap.release()
    out.release()
    return frames, frame_rate


def process_caltech_sequence(vbb_file, frames, frame_rate, output_yaml, output_video):
    print("processing",vbb_file)
    data = loadmat(vbb_file)
    annotations = data['A'][0][0]
    obj_lists = annotations['objLists'][0]

    img_width, img_height = frames[0].shape[1], frames[0].shape[0]

    yaml_data = {'frames': []}

    objLbl = [str(v[0]) for v in data['A'][0][0][4][0]]
    labels=list(set(objLbl))

    classes=["person"]
    for l in labels:
        if not l in classes:
            classes.append(l)

    metadata={"frame_rate": frame_rate,
              "width": img_width,
              "height": img_height,
              "original_video": output_video,
              "classes": classes}
    yaml_data['metadata']=metadata

    for frame_id, obj_list in enumerate(obj_lists):
        frame_data = {
            'frame_id': frame_id,
            'frame_time': frame_id / frame_rate,
            'objects': {}
        }

        if obj_list.shape[1] > 0:
            for id, pos, occl, lock, posv in zip(obj_list['id'][0],
                                                 obj_list['pos'][0],
                                                 obj_list['occl'][0],
                                                 obj_list['lock'][0],
                                                 obj_list['posv'][0]):
                x, y, w, h = pos[0].tolist()
                id=int(id[0][0])-1
                label=str(objLbl[id])
                assert label in classes

                x_min = round(float((x-1) / img_width),4)
                y_min = round(float((y-1) / img_height),4)
                x_max = round(float((x-1 + w) / img_width),4)
                y_max = round(float((y-1 + h) / img_height),4)
                frame_data['objects'][id] = {
                    'box': [x_min, y_min, x_max, y_max],
                    'class': classes.index(label),  # Assuming "person" class ID is 0
                    'conf': 1.0
                }



        yaml_data['frames'].append(frame_data)
    print(f"{len(yaml_data['frames'])} frames")
    # Write YAML file
    with open(output_yaml, 'w') as yaml_file:
        yaml.dump(yaml_data, yaml_file, default_flow_style=False)


def convert_caltech_pedestrian():

    base="/mldata/downloaded_datasets/other/caltech_pedestrian"

    for s in range (0,12):
        for i in range(0,40):
            vbb_file=base+f"/annotations/annotations/set{s:02d}/V{i:03d}.vbb"
            if s>=6:
                seq_file=base+f"/Test/set{s:02d}/set{s:02d}/V{i:03d}.seq"
            else:
                seq_file=base+f"/Train/set{s:02d}/set{s:02d}/V{i:03d}.seq"
            output_yaml=f"/mldata/tracking/caltech_pedestrian/annotation/set{s:02d}_V{i:03d}.yaml"
            output_video=f"/mldata/tracking/caltech_pedestrian/video/set{s:02d}_V{i:03d}.mp4"

            if os.path.isfile(vbb_file) is False:
                continue

            print(f" Set {s} index {i} : Processing .seq file and creating MP4 video...")
            frames, frame_rate = extract_frames_from_seq(seq_file, output_video)

            print("Processing .vbb file and creating YAML annotations...")
            process_caltech_sequence(vbb_file, frames, frame_rate, output_yaml, output_video)

            print(f"Output video saved to: {output_video}")
            print(f"Output YAML saved to: {output_yaml}")


def convert_mot():
    output_folder="/mldata/tracking_original/mot"
    folders=["/mldata/downloaded_datasets/other/MOT20/train",
             "/mldata/downloaded_datasets/other/MOT17/train"]
    stuff.makedir(output_folder+"/annotation/")
    stuff.makedir(output_folder+"/video/")
    for f in folders:
        seqs=os.listdir(f)
        for s in seqs:
            input_path=f+"/"+s+"/seqinfo.ini"
            output_path=output_folder+"/annotation/"+s+".json"
            output_video_path=output_folder+"/video/"+s+".mp4"
            print("Processing",f,s,"....")
            ts=trackset.TrackSet(input_path)
            ts.export_yaml(output_path, output_video_path)


def convert_personpath22(src_root="/mldata/downloaded_datasets/other/personpath22/dataset/personpath22",
                         output_folder="/mldata/tracking_original/personpath22",
                         anno_variant="visible"):
    """Convert PersonPath22 (gluoncv-motion format) into MOT-equivalent
    JSON+mp4 pairs under output_folder/{annotation,video}/.

    anno_variant: "visible" or "amodal" — picks anno_visible_2022 or anno_amodal_2022.
    The src_root default matches the layout produced by download.py
    (which nests under <root>/dataset/personpath22/{annotation,raw_data}).
    """
    variant_stem = {
        "visible": "anno_visible_2022",
        "amodal":  "anno_amodal_2022",
    }[anno_variant]
    anno_index_path = os.path.join(src_root, "annotation", variant_stem + ".json")
    anno_per_sample_dir = os.path.join(src_root, "annotation", variant_stem)
    videos_root = os.path.join(src_root, "raw_data")

    stuff.makedir(output_folder + "/annotation/")
    stuff.makedir(output_folder + "/video/")

    print(f"Loading {anno_index_path} ...")
    with open(anno_index_path, 'r') as f:
        all_annos = json.load(f)
    samples = all_annos.get("samples", {})
    print(f"Found {len(samples)} samples in PersonPath22 ({anno_variant})")

    skipped_missing = 0
    for uid, index_sample in samples.items():
        stem = uid[:-4] if uid.endswith(".mp4") else uid
        video_path = os.path.join(videos_root, stem + ".mp4")
        if not os.path.isfile(video_path):
            print(f"  skipping {uid}: video not found at {video_path}")
            skipped_missing += 1
            continue
        out_anno = output_folder + "/annotation/" + stem + ".json"
        out_video = output_folder + "/video/" + stem + ".mp4"
        if os.path.isfile(out_anno) and os.path.isfile(out_video):
            continue

        # Per-sample annotations live in a sibling directory; the index file
        # only carries metadata + sample_file=True for samples that defer.
        if index_sample.get("sample_file"):
            per_sample_path = os.path.join(anno_per_sample_dir, uid + ".json")
            with open(per_sample_path, 'r') as f:
                sample = json.load(f)
        else:
            sample = index_sample

        print(f"Processing {uid}...")
        ts = trackset.TrackSet()
        ts.import_personpath22(uid, sample, video_path)
        if os.path.isfile(out_video):
            # Skip the mp4 copy; just rewrite the JSON pointing at the existing copy.
            ts.metadata["original_video"] = out_video
            ts.export_yaml(out_anno, output_video=None)
        else:
            ts.export_yaml(out_anno, out_video)

    if skipped_missing:
        print(f"Done. Skipped {skipped_missing} samples with no source video.")


def convert_jaad(src_root="/mldata/downloaded_datasets/other/JAAD",
                 output_folder="/mldata/tracking_original/jaad"):
    """Convert JAAD XML + mp4 clips into MOT-like JSON+mp4 tracksets."""
    annotations_dir = os.path.join(src_root, "annotations")
    videos_dir = os.path.join(src_root, "JAAD_clips")

    stuff.makedir(output_folder + "/annotation/")
    stuff.makedir(output_folder + "/video/")

    xml_files = sorted([f for f in os.listdir(annotations_dir) if f.endswith(".xml")])
    print(f"Found {len(xml_files)} JAAD annotation files")

    skipped_missing_video = 0
    skipped_failed = 0
    for xml_name in xml_files:
        stem = xml_name[:-4]
        annotation_xml_path = os.path.join(annotations_dir, xml_name)
        video_path = os.path.join(videos_dir, stem + ".mp4")
        if not os.path.isfile(video_path):
            print(f"  skipping {xml_name}: video not found at {video_path}")
            skipped_missing_video += 1
            continue

        out_anno = output_folder + "/annotation/" + stem + ".json"
        out_video = output_folder + "/video/" + stem + ".mp4"
        if os.path.isfile(out_anno) and os.path.isfile(out_video):
            continue

        print(f"Processing {xml_name}...")
        ts = trackset.TrackSet()
        try:
            ts.import_jaad(annotation_xml_path, video_path)
        except Exception as e:
            print(f"  failed {xml_name}: {e}")
            skipped_failed += 1
            continue

        if os.path.isfile(out_video):
            # Keep existing copied mp4; just rewrite annotation JSON.
            ts.metadata["original_video"] = out_video
            ts.export_yaml(out_anno, output_video=None)
        else:
            ts.export_yaml(out_anno, out_video)

    print(f"Done. missing_video={skipped_missing_video} failed={skipped_failed}")


def _frame_pts_monotonic(path, fps):
    """True if decoded frame pts sit on a ~uniform 1/fps grid. Catches
    sources whose container timestamps are broken (MEVA r13 AVIs stamp
    pts=dts with B-frames): a -c copy remux of one plays with visible
    glitches even though the bitstream decodes cleanly."""
    import subprocess
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts_time", "-of", "csv=p=0", path],
        capture_output=True, text=True).stdout.split()
    ts = [float(x.rstrip(",")) for x in out if x.rstrip(",")]
    lo, hi = 0.5 / fps, 1.5 / fps
    return all(lo <= b - a <= hi for a, b in zip(ts, ts[1:]))


def _remux_to_mp4(src, dst):
    """Repackage a video into an mp4 container without transcoding
    (ffmpeg -c copy): bit-identical encoded frames, identical timing.
    Falls back to an exact 1:1 x264 transcode (same fps, same frame
    count, near-lossless crf 18) if the source codec can't be stored in
    mp4 OR the copied stream's decoded frame pts are non-monotonic
    (broken container stamps; the decoder restores true display order
    and setpts stamps a clean CFR grid). Writes via a temp name so
    interrupted runs can't leave a plausible-looking partial mp4."""
    import subprocess
    tmp = dst + ".part.mp4"
    fps = _native_fps(src)
    copy_args = ["-c", "copy"]
    transcode_args = ["-vf", f"setpts=N/{fps}/TB", "-r", f"{fps}",
                      "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                      "-an"]
    r = None
    for args in (copy_args, transcode_args):
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", src] + args + ["-movflags", "+faststart", tmp]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            if args is copy_args and not _frame_pts_monotonic(tmp, fps):
                continue
            os.replace(tmp, dst)
            return
    if os.path.exists(tmp):
        os.remove(tmp)
    raise RuntimeError(f"ffmpeg could not convert {src} to mp4: {r.stderr[-300:]}")

def _convert_meva_clip(args):
    """Convert one MEVA clip (worker for convert_meva's pool).
    Returns (status, stem) with status in done/missing/failed/empty."""
    geom_path, output_folder = args
    stem = os.path.basename(geom_path)
    for suf in (".geom.yml", ".geom.yaml"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    out_anno = output_folder + "/annotation/" + stem + ".json"
    out_video = output_folder + "/video/" + stem + ".mp4"
    ts = trackset.TrackSet()
    try:
        ts.import_meva(geom_path)
        if "original_video" not in ts.metadata:
            return ("missing", stem)
        if len(ts.frames) == 0:
            return ("empty", stem)
        if not os.path.isfile(out_video):
            _remux_to_mp4(ts.metadata["original_video"], out_video)
        ts.metadata["original_video"] = out_video
        ts.export_yaml(out_anno)
    except Exception as e:
        print(f"  failed {stem}: {e}")
        return ("failed", stem)
    return ("done", stem)

def convert_chirla(src_root="/mldata/downloaded_datasets/other/chirla",
                   output_folder="/mldata/tracking_original/chirla"):
    """Convert CHIRLA camera clips into trackset JSON + mp4 pairs.

    Output stems: chirla_<seq>_<camera>_<timestamp> with ':' -> '-'
    (colons break too many downstream tools). mpeg4-in-avi sources are
    remuxed to mp4 (transcode fallback on broken pts, see
    _remux_to_mp4). Existing outputs are not redone.
    """
    stuff.makedir(output_folder + "/annotation/")
    stuff.makedir(output_folder + "/video/")

    pairs = []
    anno_root = os.path.join(src_root, "annotations")
    for seq in sorted(os.listdir(anno_root)):
        seq_dir = os.path.join(anno_root, seq)
        if not os.path.isdir(seq_dir):
            continue
        for f in sorted(os.listdir(seq_dir)):
            if not f.endswith(".json"):
                continue
            video = os.path.join(src_root, "videos", seq, f[:-5] + ".avi")
            pairs.append((seq, os.path.join(seq_dir, f), video))
    print(f"Found {len(pairs)} CHIRLA annotation files")

    counts = {"done": 0, "already": 0, "missing": 0, "failed": 0,
              "empty_gt": 0}
    for seq, anno_path, video_path in pairs:
        stem = ("chirla_" + seq + "_" +
                os.path.basename(anno_path)[:-5].replace(":", "-"))
        out_anno = output_folder + "/annotation/" + stem + ".json"
        out_video = output_folder + "/video/" + stem + ".mp4"
        if os.path.isfile(out_anno) and os.path.isfile(out_video):
            counts["already"] += 1
            continue
        if not os.path.isfile(video_path):
            print(f"  missing video for {anno_path}")
            counts["missing"] += 1
            continue
        ts = trackset.TrackSet()
        try:
            ts.import_chirla(anno_path, video_path)
            if not any(f["objects"] for f in ts.frames):
                # 12/70 cameras were never visited; keep them out of the
                # corpus rather than shipping all-empty "GT"
                counts["empty_gt"] += 1
                continue
            if not os.path.isfile(out_video):
                _remux_to_mp4(video_path, out_video)
            ts.metadata["original_video"] = out_video
            ts.export_yaml(out_anno)
        except Exception as e:
            print(f"  failed {stem}: {e}")
            counts["failed"] += 1
            continue
        counts["done"] += 1
        if counts["done"] % 10 == 0:
            print(f"  {counts}")
    print(f"chirla done. {counts}")


def convert_roundabouthd(
        src_root="/mldata/downloaded_datasets/other/bath_1574/RoundaboutHD",
        output_folder="/mldata/tracking_original/roundabouthd"):
    """Convert the 4 RoundaboutHD cameras into trackset JSON + mp4.

    Source videos are 4K MPEG-4 Part 2 (Simple Profile) — no desktop
    hardware decode at that resolution, unplayable in practice — so
    they are transcoded to h264 (NVENC when available, else x264,
    crf/cq 22, 2s keyframes) rather than copied.
    """
    import subprocess
    stuff.makedir(output_folder + "/annotation/")
    stuff.makedir(output_folder + "/video/")

    def _transcode(src, dst):
        tmp = dst + ".part.mp4"
        for codec_args in ((["-c:v", "h264_nvenc", "-preset", "p5",
                             "-rc", "vbr", "-cq", "22", "-b:v", "0"]),
                           (["-c:v", "libx264", "-preset", "medium",
                             "-crf", "22"])):
            cmd = (["ffmpeg", "-y", "-v", "error", "-i", src] + codec_args +
                   ["-g", "30", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", "-an", tmp])
            if subprocess.run(cmd, capture_output=True).returncode == 0:
                os.replace(tmp, dst)
                return
        raise RuntimeError(f"transcode failed for {src}")

    for cam in ("c001", "c002", "c003", "c004"):
        stem = "roundabouthd_" + cam
        out_anno = output_folder + "/annotation/" + stem + ".json"
        out_video = output_folder + "/video/" + stem + ".mp4"
        if os.path.isfile(out_anno) and os.path.isfile(out_video):
            print(f"  {stem} already converted")
            continue
        cam_dir = os.path.join(src_root, "images" + cam)
        video = os.path.join(cam_dir, "video.mp4")
        sct = os.path.join(cam_dir, "SCT", "SCT_GT.txt")
        if not (os.path.isfile(video) and os.path.isfile(sct)):
            print(f"  {stem}: missing {video} or {sct}")
            continue
        ts = trackset.TrackSet()
        ts.import_roundabouthd(sct, video)
        if not os.path.isfile(out_video):
            _transcode(video, out_video)
        ts.metadata["original_video"] = out_video
        ts.export_yaml(out_anno)
        nb = sum(len(f["objects"]) for f in ts.frames)
        print(f"  {stem}: {len(ts.frames)} frames, {nb} boxes")
    print("roundabouthd done.")


def convert_uvg_vcm(src_root="/mldata/downloaded_datasets/other/uvg_vcm",
                    output_folder="/mldata/tracking_original/uvg_vcm"):
    """Convert downloaded UVG-VCM sequences into trackset JSON + mp4.

    Only sequences with BOTH a raw YUV on disk and a tracking-schema
    annotation JSON are converted (license-plate sequences use an
    incompatible schema and are skipped). The one-time transcode is
    x264 crf 18 yuv420p at native fps — raw source is yuv444p16le
    parsed from the filename `<Seq>_<W>x<H>_<fps>fps_..._<frames>.yuv`.
    """
    import re
    import subprocess
    stuff.makedir(output_folder + "/annotation/")
    stuff.makedir(output_folder + "/video/")

    for seq in sorted(os.listdir(src_root)):
        seq_dir = os.path.join(src_root, seq)
        if not os.path.isdir(seq_dir):
            continue
        yuvs = [f for f in os.listdir(seq_dir) if f.endswith((".yuv", "_yuv"))]
        annos = [f for f in os.listdir(seq_dir)
                 if f.endswith(".json") and "annotation" in f.lower()]
        if not yuvs or not annos:
            continue
        anno_path = os.path.join(seq_dir, annos[0])
        with open(anno_path) as fh:
            head = json.load(fh)
        objs = next((v for k, v in head.items() if k.isdigit() and v), None)
        if not objs or not isinstance(objs[0], dict) \
                or "track_id" not in objs[0]:
            print(f"  {seq}: not tracking-schema annotation, skipped")
            continue
        m = re.search(r"_(\d+)x(\d+)_(\d+)fps", yuvs[0])
        if not m:
            print(f"  {seq}: cannot parse geometry from {yuvs[0]}, skipped")
            continue
        w, h, fps = int(m.group(1)), int(m.group(2)), int(m.group(3))

        stem = "uvgvcm_" + seq
        out_anno = output_folder + "/annotation/" + stem + ".json"
        out_video = output_folder + "/video/" + stem + ".mp4"
        if os.path.isfile(out_anno) and os.path.isfile(out_video):
            print(f"  {stem} already converted")
            continue
        if not os.path.isfile(out_video):
            tmp = out_video + ".part.mp4"
            cmd = ["ffmpeg", "-y", "-v", "error",
                   "-f", "rawvideo", "-pix_fmt", "yuv444p16le",
                   "-s", f"{w}x{h}", "-r", str(fps),
                   "-i", os.path.join(seq_dir, yuvs[0]),
                   "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  {stem}: transcode failed: {r.stderr[-200:]}")
                continue
            os.replace(tmp, out_video)
        ts = trackset.TrackSet()
        ts.import_uvg_vcm(anno_path, out_video, w, h, fps)
        ts.export_yaml(out_anno)
        nb = sum(len(f["objects"]) for f in ts.frames)
        print(f"  {stem}: {len(ts.frames)} frames, {nb} boxes")
    print("uvg_vcm done.")


def convert_meva(src_root="/mldata/downloaded_datasets/other/MEVA",
                 output_folder="/mldata/tracking_original/meva",
                 workers=8, augment=True, augment_limit=0, lite=True):
    """Convert MEVA (KF1) KPF clips into trackset JSON + video pairs under
    output_folder/{annotation,video}/.

    Walks src_root/annotations recursively for `<clip>.geom.yml`;
    import_meva auto-locates each clip's video (needed for exact
    width/height/fps) and the sibling `.types.yml`. Clips whose video is
    missing, or that contain no boxes, are skipped and counted. The local
    MEVA drop may be partial (the current one is a single 5-minute
    two-camera-set fixture) — re-run after adding clips; existing outputs
    are not redone. MEVA ships h264-in-avi; output videos are losslessly
    remuxed to mp4 (see _remux_to_mp4).
    """
    stuff.makedir(output_folder + "/annotation/")
    stuff.makedir(output_folder + "/video/")

    geom_paths = []
    for root, _dirs, files in os.walk(os.path.join(src_root, "annotations")):
        for f in files:
            if f.endswith(".geom.yml") or f.endswith(".geom.yaml"):
                geom_paths.append(os.path.join(root, f))
    print(f"Found {len(geom_paths)} MEVA geom files")

    # cheap pre-pass: skip clips whose outputs already exist, before paying
    # the (slow) KPF yaml parse in a worker
    todo = []
    already = 0
    for geom_path in sorted(geom_paths):
        stem = os.path.basename(geom_path)
        for suf in (".geom.yml", ".geom.yaml"):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                break
        if (os.path.isfile(output_folder + "/annotation/" + stem + ".json")
                and os.path.isfile(output_folder + "/video/" + stem + ".mp4")):
            already += 1
            continue
        todo.append((geom_path, output_folder))
    print(f"{already} already converted, {len(todo)} to process with {workers} workers")

    counts = {}
    if len(todo) > 0:
        import multiprocessing
        with multiprocessing.Pool(workers) as pool:
            for i, (status, stem) in enumerate(pool.imap_unordered(_convert_meva_clip, todo)):
                counts[status] = counts.get(status, 0) + 1
                if (i + 1) % 100 == 0 or i + 1 == len(todo):
                    print(f"  {i+1}/{len(todo)} {counts}")

    print(f"meva done. already={already} " +
          " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if augment:
        # MEVA labels only scripted-activity actors; add missing
        # high-confidence bystander tracks so the dataset is fully
        # annotated (GPU-serial second pass; resume-safe).
        from src.autolabel_bridge import augment_dataset
        augment_dataset(output_folder, limit=augment_limit)
    if lite:
        # static mounted cameras; decimate to the analytics grid AFTER
        # augmentation (autolabel needs native framerate)
        from src.dataset_lite import process_dataset
        process_dataset(output_folder, hint="static", max_seconds=120)

def convert_otw(src_root="/mldata/downloaded_datasets/other/otw/otw",
                output_folder="/mldata/tracking_original/otw",
                augment=True, augment_limit=0, lite=True):
    """Convert Out the Window (OTW) into MOT-equivalent JSON+mp4 pairs
    under output_folder/{annotation,video}/.

    Expects src_root to contain the extracted otw.tar.gz layout:
    {homes,lots}/video/*.mp4 + {homes,lots}/annotations.csv. Output stems
    are prefixed with the collection name so the two collections cannot
    collide. Videos whose annotations carry no object tracks (all of lots/
    — it only has actor-less activity regions) are skipped.
    """
    import csv as _csv

    stuff.makedir(output_folder + "/annotation/")
    stuff.makedir(output_folder + "/video/")

    for collection in ("homes", "lots"):
        anno_path = os.path.join(src_root, collection, "annotations.csv")
        videos_dir = os.path.join(src_root, collection, "video")
        if not os.path.isfile(anno_path):
            print(f"  skipping collection {collection}: no {anno_path}")
            continue

        by_video = {}
        with open(anno_path, newline='') as f:
            for row in _csv.reader(f):
                if len(row) < 9:
                    continue
                video_id = row[0].strip()
                if video_id == "" or video_id.startswith("#"):
                    continue  # header row
                by_video.setdefault(video_id, []).append(row)
        print(f"{collection}: {len(by_video)} annotated videos")

        skipped_missing = 0
        skipped_failed = 0
        skipped_empty = 0
        for video_id in sorted(by_video.keys()):
            stem = video_id[:-4] if video_id.endswith(".mp4") else video_id
            video_path = os.path.join(videos_dir, stem + ".mp4")
            if not os.path.isfile(video_path):
                print(f"  skipping {video_id}: video not found at {video_path}")
                skipped_missing += 1
                continue
            out_stem = collection + "_" + stem
            out_anno = output_folder + "/annotation/" + out_stem + ".json"
            out_video = output_folder + "/video/" + out_stem + ".mp4"
            if os.path.isfile(out_anno) and os.path.isfile(out_video):
                continue

            print(f"Processing {collection}/{stem}...")
            ts = trackset.TrackSet()
            try:
                ts.import_otw(video_id, by_video[video_id], video_path)
            except Exception as e:
                print(f"  failed {video_id}: {e}")
                skipped_failed += 1
                continue
            if len(ts.frames) == 0:
                skipped_empty += 1
                continue

            if os.path.isfile(out_video):
                ts.metadata["original_video"] = out_video
                ts.export_yaml(out_anno, output_video=None)
            else:
                ts.export_yaml(out_anno, out_video)

        print(f"{collection} done. missing_video={skipped_missing} failed={skipped_failed} empty={skipped_empty}")

    if augment:
        # OTW annotates only activity actors; add missing high-confidence
        # tracks so the dataset is fully annotated (GPU-serial pass).
        from src.autolabel_bridge import augment_dataset
        augment_dataset(output_folder, limit=augment_limit)
    if lite:
        # static window cameras; doorbell clips with backward-PTS jitter
        # are quarantined. Runs AFTER augmentation (native-fps invariant).
        from src.dataset_lite import process_dataset
        process_dataset(output_folder, hint="static", drop_jitter=True)


def _write_gap_filled_video(ts, out_video):
    """Mux a trackset's image_path jpegs into a constant-rate MJPEG mp4
    without transcoding (ffmpeg concat demuxer + -c copy: the jpeg bits
    are copied verbatim).

    ts.frames' frame_ids are capture-clock ticks with occasional drops;
    each missing tick repeats the previous jpeg in the concat list so
    that video time (tick-1)/fps stays exactly aligned with annotation
    frame_times. Writes via a temp name so an interrupted run can't
    leave a plausible-looking partial mp4 behind.
    """
    import subprocess, tempfile
    fps = ts.metadata["frame_rate"]
    entries = []
    path = None
    fi = 0
    last_tick = ts.frames[-1]["frame_id"]
    for tick in range(1, last_tick + 1):
        if fi < len(ts.frames) and ts.frames[fi]["frame_id"] == tick:
            path = ts.frames[fi]["image_path"]
            fi += 1
        if path is None:
            # ticks before the first available image: repeat the first one
            path = ts.frames[0]["image_path"]
        entries.append(path)
    with tempfile.NamedTemporaryFile("w", suffix=".ffconcat", delete=False) as f:
        f.write("ffconcat version 1.0\n")
        for path in entries:
            f.write(f"file '{path}'\nduration {1.0/fps:.6f}\n")
        list_path = f.name
    tmp = out_video + ".part.mp4"
    try:
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
               "-i", list_path, "-c", "copy", "-movflags", "+faststart", tmp]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise RuntimeError(f"ffmpeg concat mux failed: {r.stderr[-300:]}")
        os.replace(tmp, out_video)
    finally:
        os.remove(list_path)

def convert_egohumans(src_root="/mldata/downloaded_datasets/other/egohumans/data",
                      output_folder="/mldata/tracking_original/egohumans",
                      variants=("exo", "ego_rgb")):
    """Convert the EgoHumans tracking benchmark into MOT-equivalent
    JSON+mp4 pairs under output_folder/{annotation,video}/.

    The source is an image sequence on a capture clock with ~4% dropped
    frames, so the mp4 is re-encoded gap-filled (see
    _write_gap_filled_video) rather than copied; annotations reference the
    mp4 and are self-contained (no image_path into the dataset).

    variants: which camera families to convert out of
    ("exo", "ego_rgb", "ego_slam"). Videos whose subsequence tarball is not
    yet downloaded/extracted are skipped and counted, so this can be re-run
    as more of the dataset lands.
    """
    benchmark_dir = os.path.join(
        src_root, "benchmark", "01_tagging:02_legoassemble:03_fencing", "all", "coco_track")
    stuff.makedir(output_folder + "/annotation/")
    stuff.makedir(output_folder + "/video/")

    for variant in variants:
        anno_path = os.path.join(benchmark_dir, f"tracking_{variant}.json")
        if not os.path.isfile(anno_path):
            print(f"  skipping variant {variant}: no {anno_path}")
            continue
        print(f"Loading {anno_path} ...")
        with open(anno_path, 'r') as f:
            data = json.load(f)

        images_by_video = {}
        for image in data["images"]:
            images_by_video.setdefault(image["video_id"], []).append(image)
        anns_by_video = {}
        for ann in data["annotations"]:
            anns_by_video.setdefault(ann["video_id"], []).append(ann)

        skipped_missing = 0
        skipped_failed = 0
        for video in sorted(data["videos"], key=lambda v: v["name"]):
            images = images_by_video.get(video["id"], [])
            if len(images) == 0:
                continue
            if not os.path.isfile(os.path.join(src_root, images[0]["file_name"])):
                skipped_missing += 1
                continue
            out_anno = output_folder + "/annotation/" + variant + "_" + video["name"] + ".json"
            out_video = output_folder + "/video/" + variant + "_" + video["name"] + ".mp4"
            if os.path.isfile(out_anno) and os.path.isfile(out_video):
                continue
            print(f"Processing {variant}/{video['name']}...")
            ts = trackset.TrackSet()
            try:
                ts.import_egohumans(video["name"], images,
                                    anns_by_video.get(video["id"], []), src_root)
                if not os.path.isfile(out_video):
                    _write_gap_filled_video(ts, out_video)
            except Exception as e:
                print(f"  failed {video['name']}: {e}")
                skipped_failed += 1
                continue
            for frame in ts.frames:
                frame.pop("image_path", None)
            ts.metadata["original_video"] = out_video
            ts.export_yaml(out_anno)

        print(f"{variant} done. missing_images={skipped_missing} failed={skipped_failed}")


def convert_cevo():
    output_folder="/mldata/tracking_original/cevo_april25"
    folder="/mldata/downloaded_datasets/IndiaOfficeFrontDoor"
    stuff.makedir(output_folder+"/annotation/")
    stuff.makedir(output_folder+"/video/")
    seqs=os.listdir(folder)
    for s in seqs:
        if not s.endswith(".mp4"):
            continue
        if not os.path.isfile(folder+"/"+s):
            continue
        input=folder+"/"+s
        output_path=output_folder+"/annotation/"+s[:-4]+".json"
        output_video_path=output_folder+"/video/"+s
        print("Processing",folder,s,"....")
        ts=trackset.TrackSet(input)
        # cevo GT draws visible/partial-extent boxes (e.g. seated people
        # boxed minimally) — declare it so consumers can match conventions
        ts.metadata["box_convention"]="visible"
        ts.export_yaml(output_path, output_video_path)
    fix_cevo25_vfr_times()


def convert_bdd100k_kaggle(
        src_root="/mldata/downloaded_datasets/other/BDD100k_kaggle",
        output_folder="/mldata/tracking_original/bdd100k_mot",
        limit=0):
    """Convert the 50-video Kaggle BDD100K MOT subset (original 30fps
    .mov clips + flattened scalabel CSV) into JSON+mp4 tracksets.

    Frame mapping MEASURED 2026-07-22: label frameIndex k corresponds to
    video time (k-1)/5 s, NOT k/5 — detector-vs-GT IoU sweep over 40
    sampled pedestrian boxes peaks at -1.0 interval (mean best-IoU 0.592
    vs 0.307 at k/5; offsets -2..0 swept). Likely 1-based jpg numbering
    in the original 5 fps extraction. frameIndex 0 clamps to t=0.

    Classes: pedestrian/rider -> person; car/truck/bus/train/motorcycle/
    bicycle/trailer/"other vehicle" -> vehicle; crowd-attribute boxes and
    "other person" -> other (ignore regions). box_convention "fullbody":
    occluded objects carry estimated full extents (visually verified on
    overlapping pedestrians/vehicles), matching MOT/JAAD semantics.
    """
    import csv as _csv
    import subprocess
    person_cats = {"pedestrian", "rider"}
    vehicle_cats = {"car", "truck", "bus", "train", "motorcycle",
                    "bicycle", "trailer", "other vehicle"}
    videos_dir = os.path.join(src_root, "bdd100k", "videos", "train")
    stuff.makedir(output_folder + "/annotation/")
    stuff.makedir(output_folder + "/video/")

    by_video = {}
    with open(os.path.join(src_root, "mot_labels.csv")) as fh:
        for r in _csv.DictReader(fh):
            if r.get("haveVideo") != "True" or not r.get("category"):
                continue
            by_video.setdefault(r["videoName"], []).append(r)

    done = 0
    for vid in sorted(by_video):
        src_video = os.path.join(videos_dir, vid + ".mov")
        if not os.path.isfile(src_video):
            print(f"  skip {vid}: no video")
            continue
        out_anno = output_folder + "/annotation/bdd_" + vid + ".json"
        out_video = output_folder + "/video/bdd_" + vid + ".mp4"
        if os.path.isfile(out_anno) and os.path.isfile(out_video):
            continue
        cap = cv2.VideoCapture(src_video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        cap.release()
        if not os.path.isfile(out_video):
            # RE-ENCODE, not stream copy: the source iPhone movs are
            # portrait streams with +-90 rotation metadata; consumers
            # that read the raw stream (NVDEC/C paths) would see them
            # sideways. Re-encoding applies the rotation to pixels and
            # strips the metadata (ffmpeg autorotate default).
            rc = subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", src_video,
                 "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 out_video])
            if rc.returncode != 0:
                print(f"  FAIL remux {vid}")
                continue
        track_id_map = {}
        by_frame = {}
        for r in by_video[vid]:
            cat = r["category"]
            if r.get("attributes.crowd") == "True" or cat == "other person":
                cl = 2
            elif cat in person_cats:
                cl = 0
            elif cat in vehicle_cats:
                cl = 1
            else:
                continue
            x1 = round(max(0.0, min(1.0, float(r["box2d.x1"]) / width)), 4)
            y1 = round(max(0.0, min(1.0, float(r["box2d.y1"]) / height)), 4)
            x2 = round(max(0.0, min(1.0, float(r["box2d.x2"]) / width)), 4)
            y2 = round(max(0.0, min(1.0, float(r["box2d.y2"]) / height)), 4)
            if x2 <= x1 or y2 <= y1:
                continue
            tid = r["id"]
            if tid not in track_id_map:
                track_id_map[tid] = len(track_id_map) + 1
            k = int(r["frameIndex"])
            by_frame.setdefault(k, {})[track_id_map[tid]] = {
                "box": [x1, y1, x2, y2], "class": cl, "conf": 1.0}
        frames = []
        for k in sorted(by_frame):
            t = max(0.0, (k - 1) / 5.0)
            frames.append({"frame_id": k, "frame_time": round(t, 6),
                           "objects": by_frame[k]})
        doc = {"metadata": {
                   "frame_rate": fps, "width": width, "height": height,
                   "classes": ["person", "vehicle", "other"],
                   "box_convention": "fullbody",
                   "original_video": out_video},
               "frames": frames}
        with open(out_anno, "w") as fh:
            json.dump(doc, fh, indent=4)
        done += 1
    print(f"convert_bdd100k_kaggle: {done} sequences -> {output_folder}")


def _video_codec(path):
    import subprocess
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    return r.stdout.strip()


def _native_fps(path):
    import subprocess
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    try:
        num, den = r.stdout.strip().split("/")
        return float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        import cv2
        return float(cv2.VideoCapture(path).get(cv2.CAP_PROP_FPS))


def _transcode_h264(src, dst):
    """Exact 1:1 near-lossless x264 transcode (same fps, same frame count,
    crf 18) for source codecs autolabel's rfdetr worker env cannot decode
    (AV1). Temp-name write so interrupted runs can't leave a partial mp4."""
    import subprocess
    tmp = f"{dst}.part{os.getpid()}.mp4"
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", src,
         "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-an",
         "-movflags", "+faststart", tmp],
        capture_output=True, text=True)
    if r.returncode != 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"transcode failed for {src}: {r.stderr[-300:]}")
    os.replace(tmp, dst)


def convert_autolabel_folder(src_folder, output_folder, shard="",
                             convention="fullbody", cuts=False):
    """Generic importer: fully autolabel every mp4 in src_folder into a
    dataset at output_folder/{annotation,video} suitable for utrack
    optimization. Requires the autolabel checkout (see
    src/autolabel_bridge.py — helpful error if missing).

    Resume-by-skip per video; shard="i/N" runs every N-th video (launch
    N processes to overlap GPU/decode). The autolabel export is already
    in track's annotation JSON format; videos are copied in unchanged.
    cuts=True enables autolabel's scene-cut detection (edited multi-shot
    sources, e.g. movies).
    """
    from src.autolabel_bridge import autolabel_video
    import shutil
    stuff.makedir(output_folder + "/annotation/")
    stuff.makedir(output_folder + "/video/")
    vids = sorted(f for f in os.listdir(src_folder)
                  if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")))
    si, sn = 0, 1
    if shard:
        si, sn = (int(x) for x in shard.split("/"))
    done = 0
    for k, v in enumerate(vids):
        if k % sn != si:
            continue
        stem = os.path.splitext(v)[0]
        out_anno = output_folder + "/annotation/" + stem + ".json"
        out_video = output_folder + "/video/" + stem + ".mp4"
        src = os.path.join(src_folder, v)
        try:
            if not (os.path.isfile(out_anno) and os.path.isfile(out_video)):
                # AV1 sources: rfdetr's worker env can't decode them, so
                # transcode to h264 as the dataset-local copy up front and
                # autolabel that file instead of the original
                if _video_codec(src) == "av1":
                    if not os.path.isfile(out_video):
                        _transcode_h264(src, out_video)
                    src = out_video
                autolabel_video(src, out_anno, convention=convention,
                                cuts=cuts)
            elif (json.load(open(out_anno)).get("metadata", {})
                    .get("min_annotated_height")):
                continue
            # else: annotation exists but was written by an external
            # labeller run — fall through to stamp it
            if not os.path.isfile(out_video):
                shutil.copy(src, out_video)
            # point the annotation at the dataset-local video copy
            d = json.load(open(out_anno))
            d.setdefault("metadata", {})["original_video"] = out_video
            # autolabel stamps its processing rate (native/stride) as
            # frame_rate; track consumers treat frame_rate as the video's
            # native clock (the tracker times decoded frames as index/fps,
            # so a halved rate plays detections at 2x their true time)
            d["metadata"]["frame_rate"] = _native_fps(out_video)
            # fully-autolabelled: annotation completeness only above the
            # detector-reliability knee (normalized height, per-class)
            d["metadata"]["min_annotated_height"] = {"person": 0.045}
            with open(out_anno, "w") as fh:
                json.dump(d, fh, indent=4)
            done += 1
            print(f"autolabelled {stem} ({done})", flush=True)
        except Exception as e:
            # one poisoned video must not kill the batch; resume-by-skip
            # re-attempts it on the next run
            print(f"FAIL {stem}: {type(e).__name__}: {e}", flush=True)
    print(f"convert_autolabel_folder: {done} videos -> {output_folder}")


def convert_raw_movies(src_folder="/mldata/video/youtube",
                       output_folder="/mldata/tracking_original/raw_movies",
                       shard="", lite=True):
    """Raw movie/trailer mp4s, fully autolabelled. Edited multi-shot
    content, so autolabel's scene-cut detection (TransNetV2) is enabled:
    tracks must not survive or merge across cuts. Moving camera: the
    final lite pass decimates to the analytics grid (hint:bodycam)."""
    convert_autolabel_folder(src_folder, output_folder, shard=shard,
                             cuts=True)
    if lite:
        from src.dataset_lite import process_dataset
        process_dataset(output_folder, hint="bodycam")


def convert_bwc_videotext(
        src_folder="/mldata/downloaded_datasets/other/BWC-VideoText-359/"
                   "eval_videos",
        output_folder="/mldata/tracking_original/bwc-videotext", shard="",
        lite=True):
    """Body-worn-camera eval videos, fully autolabelled (use case 2)."""
    convert_autolabel_folder(src_folder, output_folder, shard=shard)
    if lite:
        from src.dataset_lite import process_dataset
        process_dataset(output_folder, hint="bodycam")


def reduce_dataset(folder, n=50, group_fn=None, manifest=None):
    """Offline heuristic selection of the top-N clips of a converted
    dataset (no GPU): rich in annotations, diverse across scenes.

    Scoring per clip (from the annotation JSON alone):
      score = person_boxes + 2*person_tracks + 0.5*vehicle_boxes
              + 10*mean_objects_per_frame
    (annotation volume dominates; density bonus favours busy scenes).
    Diversity: clips are grouped by scene/camera (group_fn(stem); default
    strips date/time+digits — MEVA "<date>.<t0>.<t1>.<site>.<cam>" groups
    to "site.cam", OTW "homes_00012" to "homes_") and selection is a
    round-robin over groups in descending score, so one prolific camera
    cannot fill the whole budget.

    Writes <folder>/reduced_<n>.json (list of stems) and returns it.
    Consumers: autolabel_bridge.augment_dataset(names=...), or any
    importer that wants a "reduced" mode.
    """
    anno_dir = os.path.join(folder, "annotation")

    def default_group(stem):
        parts = stem.split(".")
        if len(parts) >= 5:               # MEVA-style dotted stems
            return ".".join(parts[3:])[:40]
        return "".join(c for c in stem if not c.isdigit())[:40]

    gf = group_fn or default_group
    scored = []
    for f in sorted(os.listdir(anno_dir)):
        if not f.endswith(".json"):
            continue
        try:
            d = json.load(open(os.path.join(anno_dir, f)))
        except Exception:
            continue
        frames = d.get("frames", [])
        if not frames:
            continue
        pb = vb = 0
        ptr, vtr = set(), set()
        for fr in frames:
            for tid, o in fr["objects"].items():
                if o.get("class") == 0:
                    pb += 1
                    ptr.add(tid)
                elif o.get("class") == 1:
                    vb += 1
                    vtr.add(tid)
        dens = (pb + vb) / max(len(frames), 1)
        score = pb + 2 * len(ptr) + 0.5 * vb + 10 * dens
        if score <= 0:
            continue
        stem = f[:-5]
        scored.append((score, stem, gf(stem)))

    by_group = {}
    for sc, stem, g in sorted(scored, reverse=True):
        by_group.setdefault(g, []).append((sc, stem))
    picked = []
    while len(picked) < n and any(by_group.values()):
        # round-robin: best remaining clip of each group, richest group
        # first, until the budget is filled
        for g in sorted(by_group,
                        key=lambda g: -(by_group[g][0][0]
                                        if by_group[g] else -1)):
            if by_group[g]:
                picked.append(by_group[g].pop(0)[1])
                if len(picked) >= n:
                    break
    out = manifest or os.path.join(folder, f"reduced_{n}.json")
    with open(out, "w") as fh:
        json.dump(sorted(picked), fh, indent=1)
    print(f"reduce_dataset: {len(picked)}/{len(scored)} clips "
          f"({len(by_group)} scene groups) -> {out}")
    return out


def apply_reduction(folder, manifest=None, n=50):
    """Prune a converted dataset to its reduced_N selection: clips NOT in
    the manifest move to <folder>_unreduced/{annotation,video} (same-fs
    rename — instant and reversible; delete that dir to reclaim space).
    Runs reduce_dataset first if the manifest doesn't exist yet."""
    import shutil
    manifest = manifest or os.path.join(folder, f"reduced_{n}.json")
    if not os.path.isfile(manifest):
        reduce_dataset(folder, n=n, manifest=manifest)
    keep = set(json.load(open(manifest)))
    arch = folder.rstrip("/") + "_unreduced"
    stuff.makedir(arch + "/annotation/")
    stuff.makedir(arch + "/video/")
    moved = 0
    for sub, ext in (("annotation", ".json"), ("video", ".mp4")):
        d = os.path.join(folder, sub)
        for f in sorted(os.listdir(d)):
            stem, e = os.path.splitext(f)
            if e != ext or stem in keep:
                continue
            shutil.move(os.path.join(d, f), os.path.join(arch, sub, f))
            moved += 1
    # keep the manifest with the reduced set
    print(f"apply_reduction: kept {len(keep)}, moved {moved} files "
          f"-> {arch}")


def estimate_bdd_time_offsets(detector_cache_root=None):
    """Per-clip 5fps-extraction time offset for bdd100k_mot GT (run AFTER
    detection caches exist). The Kaggle subset's label frameIndex maps to
    video time with a PER-CLIP integer-interval offset (measured
    2026-07-22: 33 clips at -1, 12 at 0, 2 at -2, 1 at +1 — a global
    mapping cost up to 0.4 detector recall on misaligned clips and
    corrupted failure analyses). Chooses among 4 discrete hypotheses by
    detector agreement (person+vehicle best-IoU@0.5 rate, det_fill
    caches) and restamps frame_time in place; idempotent via
    metadata.time_offset_intervals. Alignment recovery of an acquisition
    artifact — boxes are never modified."""
    import sys as _sys
    _sys.path.insert(0, os.path.expanduser("~/stuff/ubonpartners/autolabel"))
    from autolabel.pipeline import cache_dir_for
    from autolabel.detect import load_detections
    folder = "/mldata/tracking/bdd100k_mot"
    anno = sorted(f for f in os.listdir(folder + "/annotation")
                  if f.endswith(".json"))
    for name in anno:
        ap = os.path.join(folder, "annotation", name)
        vp = os.path.join(folder, "video", name[:-5] + ".mp4")
        c = cache_dir_for(vp)
        p = os.path.join(c, "det_fill.npz")
        if not os.path.isfile(p):
            print(f"  skip {name}: no det cache")
            continue
        meta, det = load_detections(p)
        d = json.load(open(ap))
        H, W = d["metadata"]["height"], d["metadata"]["width"]
        pfps = meta["fps"]
        scores = {}
        for off in (-2, -1, 0, 1):
            tot = hit = 0
            for f in d["frames"][::3]:
                k = f["frame_id"]
                vfi = int(round(max(0.0, (k + off) / 5.0) * pfps))
                if vfi >= len(det):
                    continue
                b, s, l = det[vfi]
                for o in f["objects"].values():
                    cl = o.get("class")
                    if cl not in (0, 1):
                        continue
                    g = o["box"]
                    if (g[3] - g[1]) < 0.03:
                        continue
                    import numpy as np
                    gpx = np.array(g, float) * [W, H, W, H]
                    db = b[l == cl]
                    if not len(db):
                        continue
                    tot += 1
                    ix = np.clip(np.minimum(gpx[2], db[:, 2])
                                 - np.maximum(gpx[0], db[:, 0]), 0, None)
                    iy = np.clip(np.minimum(gpx[3], db[:, 3])
                                 - np.maximum(gpx[1], db[:, 1]), 0, None)
                    i = ix * iy
                    iou = i / np.maximum(
                        (gpx[2]-gpx[0])*(gpx[3]-gpx[1])
                        + (db[:, 2]-db[:, 0])*(db[:, 3]-db[:, 1]) - i, 1e-9)
                    if iou.max() >= 0.5:
                        hit += 1
            if tot >= 50:
                scores[off] = hit / tot
        if not scores:
            continue
        bo = max(scores, key=scores.get)
        if d["metadata"].get("time_offset_intervals") == bo:
            continue
        for f in d["frames"]:
            f["frame_time"] = round(max(0.0, (f["frame_id"] + bo) / 5.0), 6)
        d["metadata"]["time_offset_intervals"] = bo
        with open(ap + ".tmp", "w") as fh:
            json.dump(d, fh, indent=4)
        os.replace(ap + ".tmp", ap)
        print(f"  {name}: offset {bo} (agreement {max(scores.values()):.2f})")


def fix_cevo25_vfr_times(folder="/mldata/tracking/cevo_april25"):
    """Restamp cevo_april25 GT frame_times from the video's real decoded
    PTS. Most of these cameras record variable frame rate (intervals
    0.03-0.13s) but the annotations carried synthetic times from a
    ROUNDED integer frame_rate, drifting seconds off video time by end of
    clip (time-based scoring then matches tracker output against stale
    GT). Frame ids are 0-based and dense, one per decoded frame, so frame
    k's true time is the k-th display-ordered frame's PTS.

    B-frame caveat: two clips were muxed with pts=dts (decode-order
    stamps), so the decoder — which emits display order via picture-
    order-count — yields locally swapped stamps. The stamp multiset is
    still the display timeline (one frame per coded picture); sorting
    reconstructs it exactly. Idempotent; safe to re-run."""
    import av
    import numpy as np
    for name in sorted(os.listdir(folder + "/annotation")):
        if not name.endswith(".json"):
            continue
        ap = folder + "/annotation/" + name
        vp = folder + "/video/" + name[:-5] + ".mp4"
        if not os.path.isfile(vp):
            print("fix_cevo25_vfr_times: no video for", name)
            continue
        d = json.load(open(ap))
        times = []
        with av.open(vp) as c:
            stream = c.streams.video[0]
            tb = float(stream.time_base)
            for fr in c.decode(stream):
                t = fr.pts if fr.pts is not None else fr.dts
                times.append(float(t) * tb if t is not None else np.nan)
        pts = np.asarray(times, np.float64)
        if len(pts) > 1 and np.any(np.diff(pts) < 0):
            pts = np.sort(pts)  # broken pts=dts muxing (see docstring)
        pts = pts - pts[0]
        if len(pts) != len(d["frames"]):
            print(f"fix_cevo25_vfr_times: SKIP {name} "
                  f"(pts {len(pts)} != frames {len(d['frames'])})")
            continue
        dt = np.diff(pts)
        vfr = bool(len(dt) and np.median(dt) > 0
                   and np.max(np.abs(dt - np.median(dt)))
                   > 0.02 * np.median(dt))
        for f in d["frames"]:
            f["frame_time"] = round(float(pts[f["frame_id"]]), 6)
        d["metadata"]["frame_rate"] = round(
            float((len(pts) - 1) / max(pts[-1] - pts[0], 1e-9)), 6)
        d["metadata"]["vfr"] = vfr
        with open(ap, "w") as fh:
            json.dump(d, fh, indent=4)
        print("fix_cevo25_vfr_times: fixed", name, "vfr", vfr,
              "frame_rate ->", d["metadata"]["frame_rate"])


def dofix():
    dr="/mldata/tracking/cevo/annotation"
    seqs=os.listdir(dr)
    for s in seqs:
        d=stuff.load_dictionary(dr+"/"+s)
        x=d["metadata"]["original_video"]
        x=x.replace("/tracking/video", "/tracking/cevo/video")
        d["metadata"]["original_video"]=x
        on=dr+"/"+s
        on=on.replace(".yaml",".json")
        with open(on, 'w') as json_file:
                json.dump(d, json_file, indent=4)


# Imported at module bottom so the circular dependency with trackset.py is
# safe in either import order: by the time either module needs the other's
# attributes, both class definitions above exist.
import src.trackset as trackset
