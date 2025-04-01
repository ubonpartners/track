
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

class TrackSet:
    def __init__(self, path=None):
        self.frame_times=[]
        self.frames=[]
        self.metadata={}
        self.videoreader=None
        if path is not None:
            if path.endswith(".ini"):
                self.import_mot(path)
                return
            if path.endswith(".ini"):
                self.import_caltech_pedestrian(path)
                return
            if path.endswith(".yml") or path.endswith(".yaml") or path.endswith(".json"):
                self.import_yaml(path)
                return

    def frame_index_at_time(self, t):
        if len(self.frame_times)==0:
            return None
        index = bisect.bisect_left(self.frame_times, t) - 1
        index = max(0, min(index, len(self.frame_times)-1))
        return index
    
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
        
    def objects_at_time(self, t, min_conf=0.0001):
        index=self.frame_index_at_time(t)
        if index is None:
            return None
        
        frame=self.frames[index]
        indexp1=index+1 if index+1<len(self.frames) else index
        framep1=self.frames[index+1] if index+1<len(self.frames) else frame

        s1=set(frame["objects"].keys())
        s2=set(framep1["objects"].keys())

        common_obj=list(s1.intersection(s2))
        f_only=list(s1-s2)
        p1_only=list(s2-s1)
        frac=(t-frame["frame_time"])/(framep1["frame_time"]-frame["frame_time"]+1e-7)
        frac=min(1.0, max(0.0, frac))
        ret=[]
        for track_id in common_obj:
            obj=self.get_Object(index, track_id)
            objp1=self.get_Object(indexp1, track_id)
            obj=tu.object_interpolate(obj, objp1, frac)
            obj.track_id=track_id
            ret.append(obj)

        if frac<0.5:
            for track_id in f_only:
                obj=self.get_Object(index, track_id)
                ret.append(obj)
        else:
            for track_id in p1_only:
                obj=self.get_Object(indexp1, track_id)
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
            framep1=self.frames[index+1]
            frame_timep1=framep1["frame_time"]
            if abs(t-frame_timep1)<abs(t-frame_time):
                frame=framep1
        if "image_path" in frame:
            path=frame["image_path"]
            return path
        return None

    def img_at_time(self, t):
        if self.videoreader is None and "original_video" in self.metadata:
            self.videoreader=stuff.RandomAccessVideoReader(self.metadata["original_video"])
        if self.videoreader is not None:
            return self.videoreader.get_frame_at_time(t)
        path=self.img_path_at_time(t)
        if path is not None:
            return cv2.imread(path)
        return None

    def add_frame(self, object_list, time, img_path=None):
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
                "image_path": img_path
            })

    def export_yaml(self, file, output_video=None):
        file=file.replace(",","-")
        file=file.replace(" ","-")

        if output_video is not None:
            # Video writer to save MP4
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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

    def import_create(self,
                      video,
                      track_min_interval=0.05,
                      display="Tracking...",
                      max_duration=10000,
                      pbar=None,
                      config_file=None,
                      params=None,
                      debug=False):

        assert len(self.frame_times)==0
        
        param_dict={}
        if config_file is not None:
            config=stuff.load_dictionary(config_file)
            for c in config:
                param_dict[c]=config[c]
        if params is not None:
            for p in params:
                param_dict[p]=params[p]

        tracker=trackers.create_tracker(param_dict, track_min_interval=track_min_interval)

        cap=None

        if isinstance(video, TrackSet):
            fps=video.metadata["frame_rate"]
            duration=video.duration_seconds()
            width=video.metadata["width"]
            height=video.metadata["height"]
        else:
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
                "classes": ["person"],
            }
  
        if isinstance(video, TrackSet):
            if "original_video" in video.metadata:
                self.metadata["original_video"]=video.metadata["original_video"]

        if pbar is None:
            pbar=tqdm(total=int(duration*fps)+1,
                      desc=f"{display:35s}",
                      colour="#ffcc00")
        else:
            pbar.reset(total=int(duration*fps))
            pbar.set_description(f"{display:35s}")
            pbar.refresh()
        if debug:
            display=stuff.Display(width=1280, height=720)
            
        while t<=duration:
            if cap is not None:
                success, frame = cap.read()
                if success is False:
                    break
            else:
                frame=video.img_at_time(t)
            if frame is None:
                break

            objects=tracker.track_frame(frame, t)

            #if t>6.4 and t<7.5:
            #    if objects is not None:
            #        for o in objects:
            #            print(f"{t} {o.track_id} {o.box}")
            if debug:
                display.clear()
                if objects is not None:
                    for o in objects:
                        o.draw(display, clr=(128,255,255,255), thickness=1)
                display.show(frame, title=f"time={t:5.2f}")
                events=display.get_events(0)

            if objects is not None:
                img_path=video.img_path_at_time(t) if cap is None else None
                self.add_frame(objects, t, img_path=img_path)

            t+=(1.0/fps)
            pbar.update(1)

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
                confidence = round(float(obj[6]),4)
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
                if cl==1 or cl==7:
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

def display_trackset(trackset=None, trackset_gt=None, frame_events=None, cl=["person"], output=None):
    if isinstance(trackset, str):
        trackset=ts.TrackSet(trackset)
    if isinstance(trackset_gt, str):
        trackset_gt=ts.TrackSet(trackset_gt)

    trackset_base=trackset_gt if trackset_gt is not None else trackset
    duration=trackset_base.duration_seconds()
    t=0
    paused=True
    show_gts=True
    show_det=True

    display=stuff.Display(width=1280, height=720, output=output)
    selected_ids=[]

    while(t<duration):
        display.clear()

        img=trackset_base.img_at_time(t)

        events={}
        if frame_events:
            best_diff=100000
            best_index=0
            for i, e in enumerate(frame_events):
                diff=abs(e["frame_time"]-t)
                if diff<best_diff:
                    best_diff=diff
                    best_index=i
            events=frame_events[best_index]["events"]

        if trackset_gt and show_gts:
            objs_gt=trackset_gt.objects_at_time(t)
            for o in objs_gt:
                obj_cl=trackset_gt.metadata["classes"][o.cl]
                if not obj_cl in cl:
                    continue
                a=255 if o.track_id in selected_ids else 48
                clr=(a,0,0,0)
                thickness=2
                for e in events:
                    if math.isnan(events[e]["OId"]):
                        continue
                    if int(events[e]["OId"])==o.track_id:
                        if events[e]["Type"]=="SWITCH":
                            clr=(a,0,128,128)
                        elif events[e]["Type"]=="MATCH":
                            clr=(a,0,128,0)
                        elif events[e]["Type"]=="MISS":
                            clr=(a,0,0,128)
                            thickness=4
                o.draw(display, clr=clr, thickness=thickness)

        if trackset and show_det:
            objs=trackset.objects_at_time(t)
            for o in objs:
                obj_cl=trackset.metadata["classes"][o.cl]
                if not obj_cl in cl:
                    continue
                a=255 if o.track_id in selected_ids else 48
                clr=(a,255,255,255)
                thickness=2
                for e in events:
                    if math.isnan(events[e]["HId"]):
                        continue
                    if int(events[e]["HId"])==o.track_id:
                        if events[e]["Type"]=="SWITCH":
                            clr=(a,0,255,255)
                        elif events[e]["Type"]=="MATCH":
                            clr=(a,0,255,0)
                        elif events[e]["Type"]=="FP":
                            clr=(a,0,0,255)
                            thickness=4
                o.draw(display, clr=clr, thickness=thickness)

        display.show(img, title=f"time={t:5.2f}")
        events=display.get_events(10)
        for e in events:
            if 'selected' in e:
                selected_ids=[]
                for box in e['selected']:
                    selected_ids.append(box['context'])
            if e['key']=='g':
                show_gts=not show_gts
            if e['key']=='d':
                show_det=not show_det
            if e['key']==' ':
                paused=not paused
            if e['key']=='.':
                t+=0.033
            if e['key']==',':
                t-=0.033
        if paused is False:
            t+=0.033
    display.close()

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
    folders=["/mldata/downloaded_datasets/other/MOT20/train",
             "/mldata/downloaded_datasets/other/MOT17/train"]
    
    for f in folders:
        seqs=os.listdir(f)
        for s in seqs:
            input_path=f+"/"+s+"/seqinfo.ini"
            output_path="/mldata/tracking/mot/annotation/"+s+".json"
            output_video_path="/mldata/tracking/mot/video/"+s+".mp4"
            print("Processing",f,s,"....")
            ts=TrackSet(input_path)
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