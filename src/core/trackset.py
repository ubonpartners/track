"""TrackSet: track's own annotation/result container (json/yaml + UBTRK2
storage, frame records, time interpolation).

Moved verbatim from src/trackset.py (repo_cleanup.md stage 4c); the
tracker-driving import_create method became src/tracker/run.py.
"""
import base64
import bisect
import json
import shutil
import time

import cv2
import stuff
import yaml

import src.core.objects as tu


class TrackSet:
    def __init__(self, path=None, decode_payloads=True, analysis_mode=False):
        self.name="No name"
        self.source_name=None
        self.frame_times=[]
        self.frames=[]
        self.metadata={}
        self.videoreader=None
        if path is not None:
            self.name=f"Import {path}"
            self.source_name=path
            if path.endswith(".ubtrk2") or stuff.is_ubtrk2_file(path):
                self.import_track_file(
                    path,
                    decode_payloads=decode_payloads,
                    analysis_mode=analysis_mode,
                )
                return
            if path.endswith(".ini") or path.endswith((".geom.yml", ".geom.yaml")):
                # native dataset formats moved to src/formats (stage 3);
                # TrackSet only reads its own formats now. Checked before
                # the generic yaml branch: a .geom.yml also ends in .yml.
                raise ValueError(f"{path}: use src.formats.load() for MOT/MEVA sources")
            if path.endswith(".yml") or path.endswith(".yaml") or path.endswith(".json"):
                self.import_yaml(path)
                return

    def _encode_frame_objects_for_storage(self, objects):
        if objects is None:
            return None
        out = {}
        for track_id, record in objects.items():
            out[str(int(track_id))] = record
        return out

    def _decode_frame_objects_from_storage(self, objects):
        if objects is None:
            return None
        out = {}
        for track_id, record in objects.items():
            out[int(track_id)] = record
        return out

    def frame_index_at_time(self, t, nearest=False):
        if len(self.frame_times)==0:
            return None
        index = bisect.bisect_left(self.frame_times, t+1e-7) - 1
        index = max(0, min(index, len(self.frame_times)-1))

        if nearest and index+1<len(self.frame_times):
            if abs(t-self.frame_times[index+1])<abs(t-self.frame_times[index]):
                return index+1

        return index

    def frame_time_after(self, t):
        index=self.frame_index_at_time(t)
        index=min(index+1, len(self.frame_times)-1)
        return self.frame_times[index]

    def frame_time_before(self, t):
        index=self.frame_index_at_time(t)
        index=max(0, index-1)
        return self.frame_times[index]

    def duration_seconds(self):
        return self.frame_times[-1] if len(self.frame_times)!=0 else 0

    def first_frame_time(self):
        return self.frame_times[0] if len(self.frame_times)!=0 else 0

    def last_frame_time(self):
        return self.frame_times[-1] if len(self.frame_times)!=0 else 0

    def trim(self, start_time, end_time):
        new_frame_times=[]
        new_frames=[]
        for i,t in enumerate(self.frame_times):
            if t>=start_time and t<=end_time:
                new_frame_times.append(t)
                new_frames.append(self.frames[i])
        self.frame_times=new_frame_times
        self.frames=new_frames

    def _decode_jpeg_blob(self, blob):
        if blob is None:
            return None
        data = blob.get("data")
        if isinstance(data, str):
            data = base64.b64decode(data)
        return {
            "time": blob.get("time"),
            "quality": blob.get("quality"),
            "data": data,
        }

    def _normalise_object_record(self, obj):
        if obj is None:
            return None
        out = {
            "box": obj["box"],
            "class": obj["class"],
            "confidence": obj.get("confidence", obj.get("conf")),
        }
        if "pose_points" in obj:
            out["pose_points"] = obj["pose_points"]
        elif "pose_pos" in obj and "pose_conf" in obj:
            pose_points = []
            for pos, conf in zip(obj["pose_pos"], obj["pose_conf"]):
                pose_points.extend([pos[0], pos[1], conf])
            out["pose_points"] = pose_points
        if "face_points" in obj:
            out["face_points"] = obj["face_points"]
        elif "face_pos" in obj and "face_conf" in obj:
            face_points = []
            for pos, conf in zip(obj["face_pos"], obj["face_conf"]):
                face_points.extend([pos[0], pos[1], conf])
            out["face_points"] = face_points
        if "attrs" in obj:
            out["attrs"] = obj["attrs"]
        elif "attr" in obj:
            out["attrs"] = obj["attr"]
        for key in [
            "subbox",
            "subbox_conf",
            "reid_vector",
            "face_embedding",
            "clip_embedding",
            "fiqa_score",
        ]:
            if key in obj:
                out[key] = obj[key]
        if "face_jpeg" in obj:
            out["face_jpeg"] = self._decode_jpeg_blob(obj["face_jpeg"])
        if "clip_jpeg" in obj:
            out["clip_jpeg"] = self._decode_jpeg_blob(obj["clip_jpeg"])
        return out

    def _object_to_storage_dict(self, obj):
        record = {
            "box": obj.box,
            "class": obj.cl,
            "confidence": obj.confidence,
            "attrs": obj.attr,
            "subbox": obj.subbox,
            "subbox_conf": obj.subbox_conf,
        }
        if obj.num_pose > 0:
            pose_points = []
            for pos, conf in zip(obj.pose_pos, obj.pose_conf):
                pose_points.extend([pos[0], pos[1], conf])
            record["pose_points"] = pose_points
        if obj.num_face_points > 0:
            face_points = []
            for pos, conf in zip(obj.face_pos, obj.face_conf):
                face_points.extend([pos[0], pos[1], conf])
            record["face_points"] = face_points
        if obj.reid_vector is not None:
            record["reid_vector"] = obj.reid_vector
        if obj.face_jpeg is not None:
            record["face_jpeg"] = obj.face_jpeg
        if obj.clip_jpeg is not None:
            record["clip_jpeg"] = obj.clip_jpeg
        if obj.face_embedding is not None:
            record["face_embedding"] = obj.face_embedding
        if obj.clip_embedding is not None:
            record["clip_embedding"] = obj.clip_embedding
        if hasattr(obj, "fiqa_score") and obj.fiqa_score is not None:
            record["fiqa_score"] = obj.fiqa_score
        return {k: v for k, v in record.items() if v is not None}

    def _frame_overlay_debug(self, frame):
        debug = {}
        for key, value in (frame.get("debug") or {}).items():
            if isinstance(value, dict) and "type" in value and "data" in value:
                debug[key] = value
        if frame.get("inference_roi") is not None:
            debug.setdefault("inference_roi", {"type": "roi", "data": {"roi": frame["inference_roi"]}})
        if frame.get("motion_roi") is not None:
            debug.setdefault("motion_roi", {"type": "roi", "data": {"roi": frame["motion_roi"]}})
        return debug if len(debug) > 0 else None

    def get_Object(self, index, track_id, class_remap_table=None):
        frame=self.frames[index]
        o=self._normalise_object_record(frame["objects"][track_id])
        cl=o["class"]
        if class_remap_table is not None:
            cl=class_remap_table[cl]
        obj=tu.Object(detection=o, time=frame["frame_time"])
        obj.cl=cl
        obj.track_id=track_id
        if "fiqa_score" in o:
            obj.fiqa_score = o["fiqa_score"]
        return obj

    def object_class_name(self, object):
        classes=self.metadata["classes"]
        return classes[object.cl]

    def objects_at_time(self, t, min_conf=0.0001, class_remap=None, class_remap_table=None):
        if class_remap is not None:
            assert class_remap_table is None
            class_remap_table=stuff.make_class_remap_table(self.metadata["classes"], class_remap)

        index_left=self.frame_index_at_time(t)
        if index_left is None:
            return None

        frame_left=self.frames[index_left]

        # a frame having "objects" as None means tracking was not run
        # so we need to find bracketing frames where it's not None,
        # if such exist

        while(frame_left["objects"] is None and index_left>0):
            index_left-=1
            frame_left=self.frames[index_left]

        index_right=index_left+1 if index_left+1<len(self.frames) else index_left
        frame_right=self.frames[index_right]
        while(frame_right["objects"] is None and index_right+1<len(self.frames)):
            index_right+=1
            frame_right=self.frames[index_right]

        frac=(t-frame_left["frame_time"])/(frame_right["frame_time"]-frame_left["frame_time"]+1e-7)
        frac=min(1.0, max(0.0, frac))

        if frame_left["objects"] is None:
            object_set_left=set()
        else:
            object_set_left=set(frame_left["objects"].keys())

        if frame_right["objects"] is None:
            object_set_right=set()
        else:
            object_set_right=set(frame_right["objects"].keys())

        ret=[]

        # if we are very close to an actual frame time, just return those objects

        if frac>0.99:
            for track_id in object_set_right:
                obj=self.get_Object(index_right, track_id, class_remap_table)
                ret.append(obj)
            return [o for o in ret if o.confidence>=min_conf and o.cl is not None]

        if frac<0.01:
            for track_id in object_set_left:
                obj=self.get_Object(index_left, track_id, class_remap_table)
                ret.append(obj)
            return [o for o in ret if o.confidence>=min_conf and o.cl is not None]

        # ok, we are between two frames
        # we interpolate the objects that are in both frames
        # for the ones that are not we return the ones from the frame we are closes to

        common_obj=list(object_set_left.intersection(object_set_right))
        left_only=list(object_set_left-object_set_right)
        right_only=list(object_set_right-object_set_left)

        for track_id in common_obj:
            obj_left=self.get_Object(index_left, track_id, class_remap_table)
            obj_right=self.get_Object(index_right, track_id, class_remap_table)
            if obj_left.cl is None or obj_right.cl is None:
                continue
            obj=tu.object_interpolate(obj_left, obj_right, frac)
            obj.track_id=track_id
            ret.append(obj)

        if frac<=0.5 and t-frame_left["frame_time"]<0.1:
            for track_id in left_only:
                obj=self.get_Object(index_left, track_id, class_remap_table)
                if obj.cl is not None:
                    ret.append(obj)
        elif frac>=0.5 and frame_right["frame_time"]-t<0.1:
            for track_id in right_only:
                obj=self.get_Object(index_right, track_id, class_remap_table)
                if obj.cl is not None:
                    ret.append(obj)
        ret=[o for o in ret if o.confidence>=min_conf]
        return ret

    def img_path_at_time(self, t, nearest=True):
        index=self.frame_index_at_time(t)
        if index is None:
            return None
        frame=self.frames[index]
        frame_time=frame["frame_time"]
        # pick frame with closest time
        if nearest and index+1<len(self.frames):
            frame_right=self.frames[index+1]
            frame_timep1=frame_right["frame_time"]
            if abs(t-frame_timep1)<abs(t-frame_time):
                frame=frame_right
        if "image_path" in frame:
            path=frame["image_path"]
            return path
        return None

    def img_at_time(self, t):
        if self.videoreader is None and "original_video" in self.metadata:
            self.videoreader=stuff.RandomAccessVideoReader(self.metadata["original_video"])
        if self.videoreader is not None:
            img, _=self.videoreader.get_frame_at_time(t)
            return img
        path=self.img_path_at_time(t)
        if path is not None:
            return cv2.imread(path)
        return None

    def debug_at_time(self,t, nearest=False):
        index=self.frame_index_at_time(t, nearest=nearest)
        if index is None:
            return None, t

        frame=self.frames[index]
        debug = self._frame_overlay_debug(frame)
        if debug is not None:
            return debug, frame["frame_time"]
        return None, frame["frame_time"]

    def skip_at_time(self,t, nearest=False):
        index=self.frame_index_at_time(t, nearest=nearest)
        if index is None:
            return True
        frame=self.frames[index]
        result_type = frame.get("result_type")
        if result_type is not None:
            return str(result_type).startswith("skip")
        return frame["objects"] is None

    def add_frame(self, object_list, time, img_path=None, debug=None,
                  result_type=None, motion_score=None, motion_roi=None, inference_roi=None,
                  inference_dets=None, clip_embedding=None):
        if object_list is None:
            objects=None
        else:
            objects={}
            for o in object_list:
                objects[o.track_id]=self._object_to_storage_dict(o)
        assert len(self.frame_times)==0 or time>self.frame_times[-1]
        self.frame_times.append(time)
        self.frames.append({
                "frame_time": time,
                "result_type": result_type,
                "motion_score": motion_score,
                "motion_roi": motion_roi,
                "inference_roi": inference_roi,
                "inference_dets": inference_dets,
                "clip_embedding": clip_embedding,
                "objects": objects,
                "image_path": img_path,
                "debug": debug
            })

    def add_frame_result(self, frame_result, img_path=None):
        self.add_frame(
            frame_result.get("objects"),
            frame_result["frame_time"],
            img_path=img_path,
            debug=frame_result.get("debug"),
            result_type=frame_result.get("result_type"),
            motion_score=frame_result.get("motion_score"),
            motion_roi=frame_result.get("motion_roi"),
            inference_roi=frame_result.get("inference_roi"),
            inference_dets=frame_result.get("inference_dets"),
            clip_embedding=frame_result.get("clip_embedding"),
        )

    def export_yaml(self, file, output_video=None):
        file=file.replace(",","-")
        file=file.replace(" ","-")

        if output_video is not None:
            if "original_video" in self.metadata:
                shutil.copy(self.metadata["original_video"], output_video)
                self.metadata['original_video']=output_video
            else:
                # Video writer to save MP4
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out = cv2.VideoWriter(output_video,
                                    fourcc,
                                    self.metadata['frame_rate'],
                                    (self.metadata['width'], self.metadata['height']))
                for f in self.frames:
                    img=cv2.imread(f["image_path"])
                    out.write(img)
                    del f["image_path"]
                self.metadata['original_video']=output_video
                out.release()
        dict={"metadata":self.metadata, "frames":self.frames}
        if file.endswith(".json"):
            with open(file, 'w') as json_file:
                json.dump(dict, json_file, indent=4)
        else:
            with open(file, 'w') as outfile:
                yaml.dump(dict, outfile, default_flow_style=False)

    def export_track_file(self, path):
        """Write the canonical UBTRK2 binary track/debug run format.

        The output file contains one metadata box followed by one self-contained
        frame box per processed frame. It is intended for both durable storage
        and network transport of tracker runs.
        """
        metadata = {
            "schema_version": 2,
            "kind": "trackset",
            "container": "UBTRK2",
            "source_video": self.metadata.get("original_video"),
            "frame_rate": self.metadata.get("frame_rate"),
            "width": self.metadata.get("width"),
            "height": self.metadata.get("height"),
            "classes": self.metadata.get("classes", []),
            "payload_encoding": {
                "array_codec": stuff.CODEC_RAW,
                "container": "ubtrk2-value-v1",
            },
        }
        with stuff.UBTRK2Writer(path, metadata) as writer:
            for frame in self.frames:
                record = {
                    "frame_time": frame["frame_time"],
                    "result_type": frame.get("result_type"),
                    "motion_score": frame.get("motion_score"),
                    "motion_roi": frame.get("motion_roi"),
                    "inference_roi": frame.get("inference_roi"),
                    "inference_dets": frame.get("inference_dets"),
                    "clip_embedding": frame.get("clip_embedding"),
                    "objects": self._encode_frame_objects_for_storage(frame.get("objects")),
                    "debug": frame.get("debug"),
                    "image_path": frame.get("image_path"),
                }
                writer.write_frame(record)

    def import_yaml(self, yaml_file):
        config=stuff.load_dictionary(yaml_file)
        self.metadata=config["metadata"]
        self.frames=[]
        self.videoreader=None
        self.frame_times=[]
        for frame in config["frames"]:
            normalised = {
                "frame_time": frame["frame_time"],
                "result_type": frame.get("result_type"),
                "motion_score": frame.get("motion_score"),
                "motion_roi": frame.get("motion_roi"),
                "inference_roi": frame.get("inference_roi"),
                "inference_dets": frame.get("inference_dets"),
                "clip_embedding": frame.get("clip_embedding"),
                "objects": frame.get("objects"),
                "image_path": frame.get("image_path"),
                "debug": frame.get("debug"),
            }
            self.frames.append(normalised)
            self.frame_times.append(normalised["frame_time"])

    def import_track_file(self, path, decode_payloads=True, analysis_mode=False):
        """Read the canonical UBTRK2 binary track/debug run format."""
        reader = stuff.UBTRK2Reader(path)
        metadata = reader.metadata
        self.metadata = {
            "frame_rate": metadata.get("frame_rate"),
            "width": metadata.get("width"),
            "height": metadata.get("height"),
            "classes": metadata.get("classes", []),
        }
        if metadata.get("source_video") is not None:
            self.metadata["original_video"] = metadata["source_video"]
        self.frames = []
        self.frame_times = []
        for frame in reader.iter_frames(
            decode_nested=decode_payloads,
            analysis_mode=analysis_mode,
        ):
            normalised = {
                "frame_time": frame["frame_time"],
                "result_type": frame.get("result_type"),
                "motion_score": frame.get("motion_score"),
                "motion_roi": frame.get("motion_roi"),
                "inference_roi": frame.get("inference_roi"),
                "inference_dets": frame.get("inference_dets"),
                "clip_embedding": frame.get("clip_embedding"),
                "objects": self._decode_frame_objects_from_storage(frame.get("objects")),
                "image_path": frame.get("image_path"),
                "debug": frame.get("debug"),
            }
            self.frames.append(normalised)
            self.frame_times.append(normalised["frame_time"])
