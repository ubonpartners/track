
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
import time

class TrackSet:
    def __init__(self, path=None):
        self.name="No name"
        self.source_name=None
        self.frame_times=[]
        self.frames=[]
        self.metadata={}
        self.videoreader=None
        if path is not None:
            self.name=f"Import {path}"
            self.source_name=path
            if path.endswith(".ini"):
                self.import_mot(path)
                return
            if path.endswith(".ini"):
                self.import_caltech_pedestrian(path)
                return
            if path.endswith(".yml") or path.endswith(".yaml") or path.endswith(".json"):
                self.import_yaml(path)
                return
            if path.endswith(".mp4"):
                self.import_cevo(path)
                return

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

    def get_Object(self, index, track_id):
        frame=self.frames[index]
        o=frame["objects"][track_id]
        obj=tu.Object(box=o["box"],
                      cl=o["class"],
                      conf=o["conf"],
                      pose_pos=o["pose_pos"] if "pose_pos" in o else None,
                      pose_conf=o["pose_conf"] if "pose_conf" in o else None,
                      attr=o["attr"] if "attr" in o else None,
                      time=frame["frame_time"])
        obj.track_id=track_id
        return obj

    def object_class_name(self, object):
        classes=self.metadata["classes"]
        return classes[object.cl]

    def objects_at_time(self, t, min_conf=0.0001, class_remap=None):
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
                obj=self.get_Object(index_right, track_id)
                ret.append(obj)
            ret=[o for o in ret if o.confidence>=min_conf]
            return tu.object_class_remap(ret, self.metadata["classes"], class_remap)

        if frac<0.01:
            for track_id in object_set_left:
                obj=self.get_Object(index_left, track_id)
                ret.append(obj)
            ret=[o for o in ret if o.confidence>=min_conf]
            return tu.object_class_remap(ret, self.metadata["classes"], class_remap)

        # ok, we are between two frames
        # we interpolate the objects that are in both frames
        # for the ones that are not we return the ones from the frame we are closes to

        common_obj=list(object_set_left.intersection(object_set_right))
        left_only=list(object_set_left-object_set_right)
        right_only=list(object_set_right-object_set_left)

        for track_id in common_obj:
            obj_left=self.get_Object(index_left, track_id)
            obj_right=self.get_Object(index_right, track_id)
            obj=tu.object_interpolate(obj_left, obj_right, frac)
            obj.track_id=track_id
            ret.append(obj)

        if frac<=0.5 and t-frame_left["frame_time"]<0.1:
            for track_id in left_only:
                obj=self.get_Object(index_left, track_id)
                ret.append(obj)
        elif frac>=0.5 and frame_right["frame_time"]-t<0.1:
            for track_id in right_only:
                obj=self.get_Object(index_right, track_id)
                ret.append(obj)
        ret=[o for o in ret if o.confidence>=min_conf]
        return tu.object_class_remap(ret, self.metadata["classes"], class_remap)

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
        if "tracker_debug" in frame:
            return frame["tracker_debug"], frame["frame_time"]
        return None, frame["frame_time"]

    def skip_at_time(self,t, nearest=False):
        index=self.frame_index_at_time(t, nearest=nearest)
        if index is None:
            return True
        frame=self.frames[index]
        return frame["objects"] is None

    def add_frame(self, object_list, time, img_path=None, tracker_debug=None):
        if object_list is None:
            objects=None
        else:
            objects={}
            for o in object_list:
                objects[o.track_id]={"box":o.box,
                                    "class":o.cl,
                                    "conf":o.confidence,
                                    "pose_pos":o.pose_pos,
                                    "pose_conf":o.pose_conf,
                                    "attr":o.attr}
        assert len(self.frame_times)==0 or time>self.frame_times[-1]
        self.frame_times.append(time)
        self.frames.append({
                "frame_time": time,
                "objects": objects,
                "image_path": img_path,
                "tracker_debug": tracker_debug
            })

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

    def import_yaml(self, yaml_file):
        config=stuff.load_dictionary(yaml_file)
        self.metadata=config["metadata"]
        self.frames=config["frames"]
        self.videoreader=None
        for f in self.frames:
            self.frame_times.append(f["frame_time"])

    def import_cevo(self, mp4file):
        json_path=mp4file[:-4]+" (1).mp4/COCO/annotations.json"
        print(json_path)
        cap = cv2.VideoCapture(mp4file)
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration=frame_count/fps
        annot=stuff.load_dictionary(json_path)
        num_frames=len(annot["images"])

        self.frames = []
        for i in range(num_frames):
            frame={"frame_id": i,
                   "frame_time": i/fps,
                   "objects": {}}
            self.frames.append(frame)

        self.metadata={
                "frame_rate": fps,
                "width": width,
                "height": height,
                "classes": ["person", "face"],
                "original_video": mp4file
            }

        for a in annot["annotations"]:
            x=a["bbox"][0]/width
            y=a["bbox"][1]/height
            w=a["bbox"][2]/width
            h=a["bbox"][3]/height
            f=a["image_id"]
            assert f>0
            f-=1
            id=int(a["attributes"]["ID"])
            c=a["category_id"]
            if c==2:
                id+=1000
            self.frames[f]["objects"][id]={"box": [x, y, x+w, y+h], "class":(c-1), "conf":1}
        self.frame_times=[]
        for f in self.frames:
            self.frame_times.append(f["frame_time"])

    def import_create(self,
                      video,
                      track_min_interval=0.05,
                      display="Tracking...",
                      max_duration=10000,
                      pbar=None,
                      config_file=None,
                      params=None,
                      debug=False,
                      debug_enable=False,
                      mpwq_context=None,
                      mpwq_progress_fn=None):

        assert len(self.frame_times)==0

        param_dict={}
        if config_file is not None:
            self.name=f"Import-create {stuff.name_from_file(config_file)}"
            config=stuff.load_dictionary(config_file)
            for c in config:
                param_dict[c]=config[c]
        if params is not None:
            for p in params:
                param_dict[p]=params[p]
        param_dict["original_trackset"]=video
        tracker=trackers.create_tracker(param_dict, track_min_interval=track_min_interval, debug_enable=debug_enable)

        frame_times=None
        if hasattr(tracker, 'get_frame_times'):
            frame_times=tracker.get_frame_times()

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

        duration=min(duration, max_duration)
        t=0

        self.metadata={
                "frame_rate": fps,
                "width": width,
                "height": height,
                "classes": ["person", "face"],
            }

        if isinstance(video, TrackSet):
            if "original_video" in video.metadata:
                self.metadata["original_video"]=video.metadata["original_video"]

        if frame_times is None:
            frame_times=[]
            t=0
            while t<=duration:
                frame_times.append(t)
                t+=(1.0/fps)

        if pbar is None and mpwq_progress_fn is None:
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
            if cap is not None:
                success, frame = cap.read()
                if success is False:
                    break
            else:
                frame=video.img_at_time(t)
            if frame is None:
                break

            objects, tracker_debug=tracker.track_frame(frame, t, debug_enable=debug_enable)

            if debug:
                display.clear()
                if objects is not None:
                    for o in objects:
                        o.draw(display, clr=(128,255,255,255), thickness=1)
                display.show(frame, title=f"time={t:5.2f}")
                events=display.get_events(0)

            if objects is not None or tracker_debug is not None:
                img_path=video.img_path_at_time(t) if cap is None else None
                self.add_frame(objects, t, img_path=img_path, tracker_debug=tracker_debug)
            #print(t, objects is not None, tracker_debug is not None)
            fn+=1
            if pbar is not None:
                pbar.update(1)
            elif mpwq_progress_fn is not None:
                mpwq_progress_fn(mpwq_context, update=1)

        if cap is not None:
            cap.release()

    def import_caltech_pedestrian(self, path):
        pass

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

def display_trackset(trackset_list=None, trackset_gt=None, frame_events=None, cl=["person"], output=None):

    tss=[]

    for i, ts in enumerate(trackset_list):
        name=f"Trackset {i}"
        if isinstance(ts, str):
            name+=":"+ts
            ts=TrackSet(ts)
        name+="["+ts.name+"]"
        tss.append({"name": name,
                    "display":stuff.Display(width=1280, height=720, output=output, name=name),
                    "selected_ids":[],
                    "show":False,
                    "trackset":ts})

    if isinstance(trackset_gt, str):
        trackset_gt=TrackSet(trackset_gt)

    trackset_base=trackset_gt if trackset_gt is not None else trackset_list[0]
    duration=trackset_base.duration_seconds()
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
                    if debug_entry_type=="yolo_detections":
                        stuff.draw_boxes(display,
                                        debug_entry_data["detections"],
                                        attributes=debug_entry_data["attributes"],
                                        highlight_index=None,
                                        class_names=debug_entry_data["class_names"])
                        if ts["show"]:
                            for i,d in enumerate(debug_entry_data["detections"]): #print(debug_entry_data["detections"])
                                print(f"{i} ", d["confidence"])
                            ts["show"]=False
                    if debug_entry_type=="motion_track":
                        flow=debug_entry_data["motion_array"]
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
                                    clr=max(0,min(255,int(debug_entry_data["delta_array"][y][x])))
                                    box=[x/grid_w, y/grid_h, (x+1)/grid_w, (y+1)/grid_h]
                    if debug_entry_type=="cost_map":
                        cost_map=debug_entry_data["cost_map"]
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
                display.draw_text(sstats, 0.8, 0.1)

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
