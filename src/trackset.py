
import configparser
import cv2
import os
import numpy as np
import bisect
import src.track_util as tu
import src.trackers as trackers
from tqdm.auto import tqdm
import yaml
import json
import stuff
import math
import shutil
import base64
import time
from scipy.io import loadmat

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
            if path.endswith(".ini"):
                self.import_mot(path)
                return
            if path.endswith(".vbb"):
                self.import_caltech_pedestrian(path)
                return
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

    def import_create(self,
                      video,
                      track_min_interval=0.05,
                      display="Tracking...",
                      pbar=None,
                      config_file=None,
                      params=None,
                      debug=False,
                      debug_enable=False,
                      mpwq_context=None,
                      mpwq_progress_fn=None,
                      start_time=0,
                      end_time=100000):

        assert len(self.frame_times)==0

        target_classes=["person", "face"]

        param_dict={}
        if config_file is not None:
            if isinstance(config_file, str):
                self.name=f"Import-create {stuff.name_from_file(config_file)}"
                config=stuff.load_dictionary(config_file)
                for c in config:
                    param_dict[c]=config[c]
            else:
                self.name=f"Import-create noname"
                for c in config_file:
                    param_dict[c]=config_file[c]
        if params is not None:
            for p in params:
                param_dict[p]=params[p]
        param_dict["original_trackset"]=video
        tracker=trackers.create_tracker(param_dict,
                                        track_min_interval=track_min_interval,
                                        debug_enable=debug_enable,
                                        start_time=start_time,
                                        end_time=end_time,
                                        classes=target_classes)

        frame_times=None
        if hasattr(tracker, 'get_frame_times'):
            frame_times=tracker.get_frame_times()

        needs_frames=True
        if hasattr(tracker, 'needs_frames'):
            needs_frames=tracker.needs_frames()

        cap=None

        if isinstance(video, TrackSet):
            if video.source_name is not None:
                self.name+=f":{stuff.name_from_file(video.source_name)}"
            else:
                self.name+=f" none {video.name}"
            fps=video.metadata["frame_rate"]
            duration=video.duration_seconds()
            width=video.metadata["width"]
            height=video.metadata["height"]
        else:
            self.name+=f" Video={stuff.name_from_file(video)}"
            self.source_name=video
            cap = cv2.VideoCapture(video)
            fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration=fps*frame_count
            if needs_frames is False:
                del cap

        t=0

        self.metadata={
                "frame_rate": fps,
                "width": width,
                "height": height,
                "classes": target_classes,
            }

        if isinstance(video, TrackSet):
            if "original_video" in video.metadata:
                self.metadata["original_video"]=video.metadata["original_video"]

        if frame_times is None:
            frame_times=[]
            t=0
            while t<=duration and t<end_time:
                if (t>=start_time):
                    frame_times.append(t)
                t+=(1.0/fps)

        frame_times=[t for t in frame_times if t>=start_time and t<=end_time]

        if pbar is None and mpwq_progress_fn is None:
            if stuff.platform_stuff.is_jetson():
                tqdm.monitor_interval = 0 # seems to crash here otherwise,
            pbar=tqdm(total=len(frame_times),
                      desc=f"{display:35s}",
                      colour="#ffcc00",
                      leave=False)
        elif mpwq_progress_fn is not None:
            mpwq_progress_fn(mpwq_context, desc=f"{display:35s}", total=len(frame_times))

        if debug:
            display=stuff.Display(width=1280, height=720)

        fn=0
        for t in frame_times:
            frame=None
            if needs_frames:
                if cap is not None:
                    success, frame = cap.read()
                    if success is False:
                        break
                else:
                    frame=video.img_at_time(t)
                if frame is None:
                    break

            frame_result=tracker.track_frame(frame, t, debug_enable=debug_enable)
            objects=frame_result.get("objects")

            if debug:
                display.clear()
                if objects is not None:
                    for o in objects:
                        o.draw(display, clr=(128,255,255,255), thickness=1)
                display.show(frame, title=f"time={t:5.2f}")
                events=display.get_events(0)

            if frame_result is not None:
                img_path=video.img_path_at_time(t) if cap is None else None
                self.add_frame_result(frame_result, img_path=img_path)
            fn+=1
            if pbar is not None:
                pbar.update(1)
            elif mpwq_progress_fn is not None:
                mpwq_progress_fn(mpwq_context, update=1)

        if cap is not None:
            cap.release()

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

def onoff(x):
    if x:
        return "[ON]"
    else:
        return "[OFF]"

def display_trackset(trackset_list=None, trackset_gt=None, frame_events_list=None, cl=["person"], output=None, max_duration=10000):

    if trackset_list is None:
        if trackset_gt is None:
            raise ValueError("display_trackset requires at least one trackset.")
        trackset_list = [trackset_gt]

    tss=[]

    for i, ts in enumerate(trackset_list):
        name=f"Trackset {i}"
        if isinstance(ts, str):
            name+=":"+ts
            ts=TrackSet(ts)
        name+="["+ts.name+"]"
        frame_events=None
        if frame_events_list is not None:
            frame_events=frame_events_list[i]
        tss.append({"name": name,
                    "display":stuff.Display(width=1280, height=720, output=output, name=name),
                    "selected_ids":[],
                    "show":False,
                    "trackset":ts,
                    "frame_events":frame_events})

    if isinstance(trackset_gt, str):
        trackset_gt=TrackSet(trackset_gt)

    trackset_base=trackset_gt if trackset_gt is not None else trackset_list[0]
    duration=min(max_duration, trackset_base.duration_seconds())
    t=0
    paused=True
    show_gts=True
    show_det=True
    show_help=True
    show_stats=True

    debug_overlays_enabled={}
    while(t<duration):
        for ts in tss:
            trackset=ts["trackset"]
            display=ts["display"]
            selected_ids=ts["selected_ids"]
            frame_events=ts["frame_events"]
            display.clear()

            img=trackset_base.img_at_time(t)

            events={}
            stats={}
            if frame_events:
                best_diff=100000
                best_index=0
                for i, e in enumerate(frame_events):
                    diff=abs(e["frame_time"]-t)
                    if diff<best_diff:
                        best_diff=diff
                        best_index=i
                events=frame_events[best_index]["events"]
                stats=frame_events[best_index]["stats"]

            if trackset_gt and show_gts:
                objs_gt=trackset_gt.objects_at_time(t)
                for o in objs_gt:
                    obj_cl=trackset_gt.metadata["classes"][o.cl]
                    # Render "other" GT (crowd / ignore regions) distinctly: faint
                    # gray, no event matching, no track-id selection. Detections
                    # inside these are also ignored by compute_metrics.
                    if obj_cl == "other":
                        o.draw(display, clr=(80, 96, 96, 96), thickness=1,
                               label_prefix="[ign]")
                        continue
                    if not obj_cl in cl:
                        continue
                    a=200 if o.track_id in selected_ids else 48
                    clr=(a,0,0,0)
                    thickness=2
                    prefix="?"
                    #print(o.track_id)
                    for e in events:
                        if math.isnan(events[e]["OId"]):
                            continue
                       # print(f"OID is {events[e]["OId"]}")
                        if int(events[e]["OId"])==int(o.track_id):
                            if events[e]["Type"]=="SWITCH":
                                clr=(a,0,128,128)
                                prefix="[SW]"
                            elif events[e]["Type"]=="MATCH":
                                clr=(a,0,128,0)
                                if show_det:
                                    prefix=None # don't double-label OK if we show detections
                                else:
                                    prefix="[OK]"
                            elif events[e]["Type"]=="MISS":
                                clr=(a,128,0,0)
                                prefix="[MISS]"
                                thickness=4
                            elif events[e]["Type"] in ["TRANSFER","ASCEND","MIGRATE"]:
                                prefix="[TRANS]"
                            else:
                                prefix="[??]"
                                print(f"weird gt event type ", events[e]["Type"])
                    if prefix!=None:
                        o.draw(display, clr=clr, thickness=thickness, label_prefix=prefix)

            if trackset and show_det:
                objs=trackset.objects_at_time(t)
                for o in objs:
                    obj_cl=trackset.metadata["classes"][o.cl]
                    if not obj_cl in cl:
                        continue
                    a=200 if o.track_id in selected_ids else 48
                    clr=(a,255,255,255)
                    thickness=2
                    prefix="?"
                    for e in events:
                        if math.isnan(events[e]["HId"]):
                            continue
                        if int(events[e]["HId"])==o.track_id:
                            if events[e]["Type"]=="SWITCH" or events[e]["Type"]=="TRANSFER":
                                clr=(a,255,255,0)
                                prefix="[SW]"
                            elif events[e]["Type"]=="MATCH":
                                clr=(a,0,255,0)
                                prefix="[OK]"
                            elif events[e]["Type"]=="FP":
                                clr=(a,255,0,0)
                                thickness=4 #4
                                prefix="[FP]"
                            elif events[e]["Type"]=="MIGRATE":
                                clr=(a,255,255,0)
                                prefix="[MIG]"
                            elif events[e]["Type"]=="ASCEND":
                                clr=(a,255,255,0)
                                prefix="[ASC]"
                            else:
                                prefix="[??]"
                                print(f"weird det event type ", events[e]["Type"])
                    o.draw(display, clr=clr, thickness=thickness, label_prefix=prefix)

            debug, debug_time=trackset.debug_at_time(t, nearest=True)

            if trackset.skip_at_time(t, nearest=True):
                display.draw_text(f"Nearest processed frame at t={debug_time:5.2f} SKIPPED", 0.05,0.05)
            else:
                display.draw_text(f"Nearest processed frame at t={debug_time:5.2f} TRACKED", 0.05,0.05)

            if debug is not None:
                for i,d in enumerate(debug):
                    if not d in debug_overlays_enabled:
                        debug_overlays_enabled[d]=False
                    debug_entry=debug[d]
                    debug_entry_type=debug_entry["type"]
                    debug_entry_data=debug_entry["data"]
                    if not d in debug_overlays_enabled or debug_overlays_enabled[d]==False:
                        continue
                    if debug_entry_type=="detections":
                        stuff.draw_boxes(display,
                                        debug_entry_data["detections"],
                                        attributes=debug_entry_data.get("attribute_names"),
                                        highlight_index=None,
                                        class_names=debug_entry_data["class_names"])
                        if ts["show"]:
                            for i,d in enumerate(debug_entry_data["detections"]): #print(debug_entry_data["detections"])
                                print(f"{i} ", d["confidence"])
                            ts["show"]=False
                    if debug_entry_type=="motion_field":
                        flow=stuff.decode_payload(debug_entry_data["flow"]) if "flow" in debug_entry_data else debug_entry_data.get("motion_array")
                        if flow is not None:
                            grid_w=flow.shape[1]
                            grid_h=flow.shape[0]
                            for y in range(grid_h):
                                for x in range(grid_w):
                                    cx=(x+0.5)/grid_w
                                    cy=(y+0.5)/grid_h
                                    vx=flow[y][x][0]
                                    vy=flow[y][x][1]
                                    thr=0.001
                                    if abs(vx)>thr or abs(vy)>thr:
                                        display.draw_line([cx,cy],
                                                        [cx+vx, cy+vy],
                                                        clr=(128,255,255,0), thickness=1)
                            delta_array = None
                            if "delta" in debug_entry_data:
                                delta_array = stuff.decode_payload(debug_entry_data["delta"])
                            elif "delta_array" in debug_entry_data:
                                delta_array = debug_entry_data["delta_array"]
                            if delta_array is not None:
                                for y in range(grid_h):
                                    for x in range(grid_w):
                                        clr=max(0,min(255,int(delta_array[y][x])))
                                        box=[x/grid_w, y/grid_h, (x+1)/grid_w, (y+1)/grid_h]
                    if debug_entry_type=="cost_map":
                        cost_map=stuff.decode_payload(debug_entry_data["cost_map"])
                        scale=debug_entry_data["scale"]
                        if cost_map is not None:
                            grid_w=cost_map.shape[1]
                            grid_h=cost_map.shape[0]
                            for y in range(grid_h):
                                for x in range(grid_w):
                                    clr=max(0,min(255,int(scale*cost_map[y][x])))
                                    box=[x/grid_w, y/grid_h, (x+1)/grid_w, (y+1)/grid_h]
                                    display.draw_box(box, (clr,0,255,0), thickness=-1)
                    if debug_entry_type=="box_prediction":
                        for i in debug_entry_data:
                            display.draw_box(debug_entry_data[i]["from"], clr=(128,255,255,255), thickness=1)
                            display.draw_box(debug_entry_data[i]["to"], clr=(128,255,0,0), thickness=2)
                            if "pose_from" in debug_entry_data[i]:
                                stuff.draw_pose(display,
                                                pose_pos=debug_entry_data[i]["pose_from"],
                                                pose_conf=debug_entry_data[i]["pose_conf"],
                                                thickness=1, clr=(128,255,255,255))
                                stuff.draw_pose(display,
                                                pose_pos=debug_entry_data[i]["pose_to"],
                                                pose_conf=debug_entry_data[i]["pose_conf"],
                                                thickness=2, clr=(128,255,0,0))

                    if debug_entry_type=="roi":
                        box=debug_entry_data["roi"]
                        display.draw_box(box, clr=(16,255,255,0), thickness=-1)
                        display.draw_box(box, clr=(128,255,0,0), thickness=4)
                    if debug_entry_type=="match_cost_trace":
                        # Per-frame (track, det) pair feature trace from utrack.
                        # Each track that survived the spatial pre-filter shows
                        # up at least once per pass. Group by track_id, prefer
                        # the matched record, then render any track that the
                        # tracker did NOT promote into the emitted output —
                        # those are the UNCONFIRMED / LOST internal tracks.
                        from src.analysis import pair_log_schema as _pls
                        try:
                            magic = debug_entry.get("magic")
                            version = debug_entry.get("version")
                            n_records = int(debug_entry.get("n_records", 0))
                        except Exception:
                            magic = version = None; n_records = 0
                        if magic == _pls.PAIR_LOG_MAGIC and version == _pls.PAIR_LOG_VERSION and n_records > 0:
                            arr = _pls.decode_records(debug_entry["data"], n_records)
                            by_track = {}
                            for r in arr:
                                tid = int(r["track_id"])
                                cur = by_track.get(tid)
                                if cur is None or (int(r["was_matched"]) == 1 and int(cur["was_matched"]) == 0):
                                    by_track[tid] = r
                            # Always recompute emitted ids — `objs` from the
                            # show_det branch above isn't in scope when the
                            # detection overlay is toggled off.
                            _emit_objs = trackset.objects_at_time(t) or []
                            emitted_ids = {int(o.track_id) for o in _emit_objs if o.track_id is not None}
                            # Categorise tracks that exist in the trace but did
                            # NOT reach the emitted output:
                            #   was_matched=1  → UNCONFIRMED gathering obs (would-be
                            #                    track being held below the promotion
                            #                    threshold — the "track that should
                            #                    exist but doesn't" case)
                            #   was_matched=0, obs<=1 → UNCONFIRMED killed on first miss
                            #   was_matched=0, obs>=2 → LOST (in track_buffer grace,
                            #                    or about to be removed)
                            n_t = len(emitted_ids)
                            n_unc_pending = n_unc_killed = n_lost = 0
                            for tid, r in by_track.items():
                                if tid in emitted_ids:
                                    continue
                                obs = int(r["observations"])
                                miss = int(r["num_missed"])
                                tsd = float(r["time_since_det"])
                                matched = int(r["was_matched"]) == 1
                                box = [float(r["track_x0"]), float(r["track_y0"]),
                                       float(r["track_x1"]), float(r["track_y1"])]
                                if matched:
                                    clr = (200, 64, 200, 200)  # cyan — UNCONFIRMED matching but not promoted
                                    label = f"[Up:{tid}] obs={obs}"
                                    n_unc_pending += 1
                                elif obs <= 1:
                                    clr = (200, 240, 240, 0)   # yellow — UNCONFIRMED, dying
                                    label = f"[Uk:{tid}] obs={obs} mis={miss}"
                                    n_unc_killed += 1
                                else:
                                    clr = (160, 128, 128, 128) # gray — LOST, in grace period
                                    label = f"[L:{tid}] obs={obs} mis={miss} tsd={tsd:.1f}s"
                                    n_lost += 1
                                display.draw_box(box, clr=clr, thickness=1, select_context=tid)
                                display.draw_text(label, box[0], max(0.0, box[1]-0.012),
                                                  fontScale=0.4, fontColor=clr,
                                                  bgColor=(160,0,0,0))
                            stats_line = (f"tracks: {n_t} TRACKED / {n_unc_pending} UNCONF-pending / "
                                          f"{n_unc_killed} UNCONF-killed / {n_lost} LOST "
                                          f"· trace records: {n_records}")
                            display.draw_text(stats_line, 0.05, 0.07, fontScale=0.5)

            help="HELP\n"
            help+=f"h) toggle this help display {onoff(show_help)}\n"
            help+=f"s) toggle stats display {onoff(show_stats)}\n"
            help+=f"< > advance time, +SHIFT to skip to next tracked frame\n"
            help+=f"D, G toggle tracking Det {onoff(show_det)} GTs {onoff(show_gts)}\n"
            help+=f"<space> toggle continous playback {onoff(not paused)}\n"
            help+=f"Debug overlays-\n"
            for i,e in enumerate(debug_overlays_enabled):
                help+=(f"--- {i+1} Toggle : {e:20s} {onoff(debug_overlays_enabled[e])}\n")

            if show_help:
                display.draw_text(help, 0.05, 0.1)

            if show_stats and len(stats)>0:
                sstats="STATS\n"
                for s in stats:
                    sstats+=f"{s:20}: {stats[s]}\n"
                display.draw_text(sstats, 0.75, 0.1)

            title=ts["name"]+f"time={t:5.2f}"
            display.show(img, title=title)


        #end tss loop

        for i,ts in enumerate(tss):
            display=ts["display"]
            trackset=ts["trackset"]
            events=display.get_events(10)
            for e in events:
                if 'selected' in e:
                    selected_ids=[]
                    for box in e['selected']:
                        selected_ids.append(box['context'])
                    ts["selected_ids"]=selected_ids
                if e['key']=='g':
                    show_gts=not show_gts
                if e['key']=='d':
                    show_det=not show_det
                if e['key']==' ':
                    paused=not paused
                if e['key']=='>':
                    t=trackset.frame_time_after(t)
                if e['key']=='<':
                    t=trackset.frame_time_before(t)
                if e['key']=='.':
                    t+=0.033
                if e['key']==',':
                    t-=0.033
                if e['key']=='s':
                    show_stats=not show_stats
                if e['key']=='x':
                    show=True
                if e['key']=='h':
                    show_help=not show_help
                if e['key'] is not None and e['key']>='1' and e['key']<='9':
                    index=int(e['key'])-1
                    key = list(debug_overlays_enabled.keys())[index]
                    debug_overlays_enabled[key]=not debug_overlays_enabled[key]
        if paused is False:
            t+=0.033
    for ts in tss:
        ts["display"].close()

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
    output_folder="/mldata/tracking/mot"
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
            ts=TrackSet(input_path)
            ts.export_yaml(output_path, output_video_path)

def convert_personpath22(src_root="/mldata/downloaded_datasets/other/personpath22/dataset/personpath22",
                         output_folder="/mldata/tracking/personpath22",
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
        ts = TrackSet()
        ts.import_personpath22(uid, sample, video_path)
        if os.path.isfile(out_video):
            # Skip the mp4 copy; just rewrite the JSON pointing at the existing copy.
            ts.metadata["original_video"] = out_video
            ts.export_yaml(out_anno, output_video=None)
        else:
            ts.export_yaml(out_anno, out_video)

    if skipped_missing:
        print(f"Done. Skipped {skipped_missing} samples with no source video.")

def convert_cevo():
    output_folder="/mldata/tracking/cevo_april25"
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
        ts=TrackSet(input)
        ts.export_yaml(output_path, output_video_path)

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
