
import configparser
import cv2
import os
import numpy as np
import bisect
import src.track_util as tu
import src.trackers as trackers
import motmetrics as mm
import ultralytics
import pickle
import time
from datetime import datetime
import copy
from tqdm.auto import tqdm
import yaml
import json
import traceback
import stuff
import xlsxwriter
from multiprocessing import Process, Queue

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
        
    def objects_at_time(self, t):
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

        assert "tracker_type" in param_dict, "tracker type must be specified"
        
        if not "model" in param_dict:
            param_dict["model"]="/mldata/weights/yolo11l-dpa-131224.pt"
            print(f"WARNING: Model not specified in config; using default model {param_dict['model']}")

        tracker_type=param_dict["tracker_type"]
        if tracker_type=="bytetrack" or tracker_type=="botsort":
            tracker=trackers.ultralytics_tracker(param_dict, track_min_interval=track_min_interval)
        elif tracker_type=="nvof":
            tracker=trackers.nvof_tracker(param_dict, track_min_interval=track_min_interval)
        elif tracker_type=="cevo":
            tracker=trackers.cevo_tracker(param_dict, track_min_interval=track_min_interval)
        else:
            print(f"Unkown tracker type {tracker_type}")
            exit()

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
                "classes": ["person", "vehicle"],
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

                if cl==1 or cl==7:
                    out_cl=0
                else:
                    out_cl=1

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


def mot_obj(obj, w, h):
    ol=int(obj.box[0]*w)
    ot=int(obj.box[1]*h)
    ow=int((obj.box[2]-obj.box[0])*w)
    oh=int((obj.box[3]-obj.box[1])*h)
    return [obj.track_id, ol, ot, ow, oh]

def compute_metrics(gt, test, max_duration=1000, frame_metrics=False, match_iou=0.5):
    duration=min(max_duration, max(gt.duration_seconds(), test.duration_seconds()))
    t=0
    img_w=gt.metadata["width"]
    img_h=gt.metadata["height"]
    # run evaluation at the framerate of the original video
    time_incr=1.0/gt.metadata["frame_rate"]
    acc = mm.MOTAccumulator(auto_id=True)
    while t<duration:
        # get GT and Test objects at time
        # this interpolates objects if there is no frame at that time
        gt_obj=gt.objects_at_time(t)
        gt_obj=[o for o in gt_obj if o.cl==0] # FIXME: person only for now
        test_obj=test.objects_at_time(t)
        if test_obj is None or gt_obj is None:
            break
        gt_dets=[mot_obj(g, img_w, img_h) for g in gt_obj]
        t_dets=[mot_obj(t, img_w, img_h) for t in test_obj]
        gt_dets=np.array(gt_dets)
        t_dets=np.array(t_dets)
        C=[[]]
        if len(gt_dets)>0 and len(t_dets)>0:
            C = mm.distances.iou_matrix(gt_dets[:,1:], t_dets[:,1:], \
                                max_iou=match_iou) # format: gt, t

        acc.update(gt_dets[:,0].astype('int').tolist() if len(gt_dets)>0 else [], \
                   t_dets[:,0].astype('int').tolist() if len(t_dets)>0 else [], C)
        t+=time_incr

    mh = mm.metrics.create()

    summary = mh.compute(acc, metrics=['num_frames', 'idf1', 'idp', 'idr', \
                                    'recall', 'precision', 'num_objects', \
                                    'mostly_tracked', 'partially_tracked', \
                                    'mostly_lost', 'num_false_positives', \
                                    'num_misses', 'num_switches', \
                                    'num_fragmentations', 'mota', 'motp', \
                                    'num_unique_objects', 'num_matches', \
                                    'idfp', 'idfn', 'idtp'], \
                        name='acc')
    
    metrics_dict=summary.loc['acc'].to_dict()

    # add some extra metrics like
    # 'fp_tracks' - number of detected track IDs that correspond to no GT
    # some _frac metric which is the fraction of the corresponding metric of all objects

    df = acc.mot_events  # This is a typical name for the DataFrame of match events
    # Filter rows that are actual matches (i.e. 'Type' == 'MATCH')
    matches_df = df[df['Type'] == 'MATCH']
    # Get the predicted IDs that *ever* matched
    matched_hids = matches_df['HId'].unique()
    # Get *all* predicted IDs that appeared in the results
    all_hids = df['HId'].dropna().unique()  # Drop NaNs since some rows might not have an HId
    # The set of false-positive track IDs are those not in `matched_hids`
    false_positive_track_ids = set(all_hids) - set(matched_hids)
    # Finally
    num_false_positive_tracks = len(false_positive_track_ids)
    metrics_dict["fp_tracks"]=num_false_positive_tracks

    all_gt_ids = df['OId'].dropna().unique()
    matched_gt_ids = df.loc[df['Type'] == 'MATCH', 'OId'].unique()
    completely_lost_gt_ids = set(all_gt_ids) - set(matched_gt_ids)
    num_never_detected=len(completely_lost_gt_ids)
    metrics_dict["match_iou"]=match_iou
    metrics_dict["missed"]=num_never_detected
    metrics_dict["mostly_lost2"]=metrics_dict["mostly_lost"]-num_never_detected
    metrics_dict["mostly_tracked_frac"]=metrics_dict
    for m in ["mostly_tracked", "partially_tracked", "mostly_lost2", "missed", "fp_tracks"]:
        metrics_dict[m+"_frac"]=metrics_dict[m]/(metrics_dict["num_unique_objects"]+1e-7)
    
    # optionally extract per-frame MOT metrics
    if frame_metrics:
        t=0
        frame_index=0
        frame_events=[]
        while t<duration:
            if frame_index in acc.mot_events.index.get_level_values(0).unique():
                frame=acc.mot_events.xs(frame_index, level=0) #acc.mot_events.loc[frame_index]
                frame_events.append({"frame_time":t, "events":frame.to_dict(orient='index')})
            else:
                # no events for this frame
                frame_events.append({"frame_time":t, "events":{}})
            t+=time_incr
            frame_index+=1
    del mh
    del acc
    if frame_metrics:
        return metrics_dict, frame_events
    return metrics_dict

def result_string(result, columns):
    rh=""
    rs=""
    for c in columns:
        cs=c.split(",")
        key=cs[0]
        hd=cs[1]
        fmt=cs[2]
        if key in result:
            rs+=(f"{fmt.format(result[key])}")
            rh+=hd
        else:
            print(f"{key}: Key not found in dictionary")
    return rs,rh

def get_avg_scores(results, test, param, group=None):
    t=0.0
    n=0
    for r in results:
        if r["params"]["test_key"]==test:
            if group is None or ("group" in r and r["group"]==group):
                if param in r["result"]:
                    if isinstance(r["result"][param], int) or isinstance(r["result"][param], float):
                        t+=r["result"][param]
                        n=n+1
    if n>0:
        t=t/n
        return t
    else:
        return 0
    
def display_results(results, columns, sort_key):
    out_sort=[]
    out_txt=[]
    datasets=[result["params"]["ds_key"] for result in results]
    tests=[result["params"]["test_key"] for result in results]
    groups=[result["group"] if "group" in result else None for result in results]
    groups.append(None)
    datasets=list(set(datasets))
    tests=list(set(tests))
    groups=list(set(groups))
    paramset=set([])
    for r in results:
        paramset=paramset.union(set(r["result"].keys()))
    params=list(paramset)

    results2=[]
    if len(datasets)>1:
        for g in groups:
            name="_overall" if g is None else f"__ovr{g}"
            for t in tests:
                filtered=[]
                for r in results:
                    if r["params"]["test_key"]==t:
                        if g is None or ("group" in r and r["group"]==g):
                            filtered.append(r)
                e={"result":{}, "params":{}}
                e["params"]["ds_key"]=name
                e["params"]["test_key"]=t
                er=e["result"]
                for p in params:
                    er[p]=sum([r["result"][p] for r in filtered])
                weighted_motp_sum=0
                for r in filtered:
                    weighted_motp_sum += r['result']['motp']*r['result']['idtp']
                er["idf1"]= (2 * er["idtp"]) / (2 * er["idtp"] + er["idfp"] + er["idfn"]+1e-7)
                er['mota']= 1 - (er['num_false_positives'] + er['num_misses'] + er['num_switches']) / er['num_objects']
                er['motp']=weighted_motp_sum/er['idtp']
                er['mostly_tracked_frac']/=len(filtered)
                er['partially_tracked_frac']/=len(filtered)
                er['mostly_lost2_frac']/=len(filtered)
                er['missed_frac']/=len(filtered)
                er['fp_tracks_frac']/=len(filtered)
                #er['mota']=er['mota']/er['num_objects']
                #er['motp']=er['motp']/er['num_objects']
                results2.append(e)
            datasets.append(name)

        for g in groups:
            if g is None:
                continue
            n=f"__mean({g})"
            for t in tests:
                e={"result":{}, "params":{}}
                e["params"]["ds_key"]=n
                e["params"]["test_key"]=t
                for p in params:
                    e["result"][p]=get_avg_scores(results, t, p, g)
                results2.append(e)
            datasets.append(n)
        for t in tests:
            e={"result":{}, "params":{}}
            e["params"]["ds_key"]="_arithmean"
            e["params"]["test_key"]=t
            for p in params:
                e["result"][p]=get_avg_scores(results, t, p)
            results2.append(e)
        datasets.append("_arithmean")

    datasets.sort()

    for result in results+results2:
        ds_index=datasets.index(result["params"]["ds_key"])
        rs,rh=result_string(result["result"], columns)
        rh=" "*63+rh
        rs=f"{result["params"]["ds_key"]:30s} {result["params"]["test_key"]:32}"+rs
        out_txt.append(rs)
        out_sort.append(result["result"][sort_key]+ds_index*1000)
    print(rh)
    Z = [x for _,x in sorted(zip(out_sort, out_txt), reverse = True)]
    for z in Z:
        print(z)

    result_location="/mldata/results/track"
    directory=os.path.join(result_location, datetime.today().strftime('%Y-%m-%d'))
    stuff.makedir(directory)
    out_file=os.path.join(directory, "results_spreadsheet.xlsx")
    workbook = xlsxwriter.Workbook(out_file)
    worksheet = workbook.add_worksheet()
    worksheet.set_column(0, 0, 20)
    worksheet.set_column(1, 1, 30)
    for i,c in enumerate(columns):
        cs=c.split(",")
        worksheet.write(0, i+2,  cs[1])
    for i,result in enumerate(results+results2):
        worksheet.write(i+1, 0, result["params"]["ds_key"])
        worksheet.write(i+1, 1, result["params"]["test_key"])
        for j,c in enumerate(columns):
            cs=c.split(",")
            worksheet.write(i+1, j+2,  round(result["result"][cs[0]],3))
    workbook.close()

    return results2

def run_one_test(params, pbar=None):
    trackset=TrackSet()
    print(params)
    trackset_gt=TrackSet(params["ds_path"])
    trackset.import_create(trackset_gt,
                           track_min_interval=params["min_interval"],
                           display=params["display"],
                           max_duration=params["max_duration"],
                           config_file=params["config"],
                           params=params,
                           pbar=pbar)

    match_iou=0.5
    if params is not None and "match_iou" in params:
        match_iou=params["match_iou"]
    result=compute_metrics(trackset_gt, trackset, max_duration=params["max_duration"], match_iou=match_iou)
    del trackset
    del trackset_gt

    entry={#"test":params["test_key"],
           #"dataset":params["ds_key"],
           #"config":params["config"],
           "params":params, 
           "result":result}
    return entry

def worker_fn(work_queue, result_queue, quit_queue, progress_position):
    pbar=tqdm(total=100,
              desc=f"{progress_position:02d}: {'Starting....':31s}",
              colour="#cc"+f"{1010*(progress_position+1)}",
              position=progress_position+1,
              leave=True)
    while True:
        try:
            # Get a job from the work queue
            work_item = work_queue.get(timeout=15)  # Adjust timeout as needed
            if work_item is None:
                # None indicates no more jobs
                break
            # Perform the task
            result = run_one_test(work_item, pbar=pbar)
            # Put the result in the results queue
            result_queue.put(result)
        except Exception as e:
            error_info = traceback.format_exc()
            result_queue.put({"exception":error_info})
            # Break the loop if queue is empty and timeout occurs
            break

    pbar.set_description(f"{progress_position:02d}: {'Done':31s}")
    pbar.refresh()
    # wait to be told to exit (stops pbar getting messed up)
    _ = quit_queue.get(timeout=600)

def test_track(config, split=None):
    
    if isinstance(config, str):
        config=stuff.load_dictionary(config)
    
    resultfile=None
    if "results_cache_file" in config:
        resultfile=config["results_cache_file"]
    num_workers=config["num_workers"]
    cached_results=[]
    if resultfile is not None and os.path.isfile(resultfile):
        with open(resultfile, 'rb') as handle:
            cached_results = pickle.load(handle)
    
    datasets=config["datasets"]
    tests=config["tests"]
    columns=config["columns"]
    output_results=[]

    tests_to_run=[]

    for _,ds_key in enumerate(datasets):
        dataset=datasets[ds_key]
        if split is not None:
            if "split" in dataset:
                if dataset["split"]!=split:
                    continue
        for test_key in tests:
            result=None
            for r in cached_results:
                if r["params"]["test_key"]==test_key and r["params"]["ds_key"]==ds_key:
                    if "regenerate" in datasets[ds_key] and datasets[ds_key]["regenerate"]==True:
                        continue
                    if "regenerate" in tests[test_key] and tests[test_key]["regenerate"]==True:
                        continue
                    result=r
            if result is None:
                
                test=tests[test_key]
                params={}
                for p in test:
                    params[p]=test[p]
                if not "max_duration" in params:
                    params["max_duration"]=1000

                params["ds_path"]=dataset["path"]
                params["display"]=f"{len(tests_to_run):02d}: "+ds_key+"/"+test_key
                params["ds_key"]=ds_key
                params["test_key"]=test_key

                #params={"ds_path":dataset["path"],
                #        "test_type":test["type"],
                #        "test_model":test["model"],
                #        "min_interval":test["min_interval"],
                #        "display":f"{len(tests_to_run):02d}: "+ds_key+"/"+test_key,
                #        "max_duration":max_duration,
                #        "config":test["config"],
                #        "ds_key":ds_key,
                #        "params":test["params"] if "params" in test else None,
                #        "test_key":test_key}
                
                tests_to_run.append(params)
            else:
                output_results.append(result)
    
    print(f"Running {len(tests_to_run)} tests...")
    with tqdm(total=len(tests_to_run),
              desc="search",
              colour="#0000ff",
              position=0,
              leave=True) as pbar:
        start_time=time.time()
        work_queue = Queue()
        result_queue = Queue()
        quit_queue = Queue()
        for item in tests_to_run:
            work_queue.put(item)
        for _ in range(num_workers):
            work_queue.put(None)

        workers = []
        for i in range(num_workers):
            p = Process(target=worker_fn, args=(work_queue, result_queue, quit_queue, i))
            p.start()
            workers.append(p)

        num_results_got=0
        while num_results_got<len(tests_to_run):
            entry=result_queue.get(timeout=600)
            num_results_got+=1
            pbar.update(1)
            #print(f"!!Completed {num_results_got} tests of {len(tests_to_run)}")
            if "exception" in entry:
                print(f"Process exception {entry['exception']}")
                exit()

            cache=True
            ds_key=entry["params"]["ds_key"]
            if "no_cache" in config["datasets"][ds_key]:
                if config["datasets"][ds_key]["no_cache"]==True:
                    cache=False
            if cache is True and resultfile is not None:
                cached_results.append(entry)
                stuff.save_atomic_pickle(cached_results, resultfile)
            output_results.append(entry)

        for _ in range(num_workers):
            quit_queue.put(None)

        for p in workers:
            p.join()
    
    for o in output_results:
        if "group" in config["datasets"][o["params"]["ds_key"]]:
            o["group"]=config["datasets"][o["params"]["ds_key"]]["group"]

    results2=display_results(output_results, columns, config["sort_key"])
    elapsed=time.time()-start_time
    print(f"All done: Evaluated {len(tests_to_run)} tests in {stuff.timestr(elapsed)}")
    return results2

def search_test(config, params, param_vec, param_min, param_max, all_results, split="train"):
    param_vec_clipped=max(min(param_vec, param_max), param_min)
    if split=="train" and tuple(param_vec_clipped) in all_results:
        return all_results[tuple(param_vec_clipped)]["score"], None
    
    result_test_opt_key=config["result_test_opt_key"]
    result_dataset_opt_key=config["result_dataset_opt_key"]
    result_dataset_opt_param=config["result_dataset_opt_param"]
  
    for i,p in enumerate(params):
        config["tests"][result_test_opt_key][p]=param_vec_clipped[i]

    results=test_track(config, split=split)
    val=None
    full_result=None
    for r in results:
        if r["params"]["test_key"]==result_test_opt_key and r["params"]["ds_key"]==result_dataset_opt_key:
            val=r["result"][result_dataset_opt_param]
            full_result=r["result"]
    if split=="train":
        all_results[tuple(param_vec_clipped)]={"score":val, "param_vec":param_vec_clipped}
    for r in full_result:
        full_result[r]=round(full_result[r],3)
    return val,full_result

def search_log(logfile, x):
    logfile.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+": ")
    logfile.write(x+"\n")
    logfile.flush()

def search_track(yaml_file):
    config=stuff.load_dictionary(yaml_file)
    result_log_file=config["result_log_file"]
    logfile=open(result_log_file, "w")
    param_names=[]
    param_initial=[]
    param_step=[]
    param_min=[]
    param_max=[]

    results={}

    for p in config["search_params"]:
        param_names.append(p)
        param_initial.append(None)
    
    test_dict=stuff.load_dictionary(config["tests"]["search_config"]["config"])
    for p in test_dict:
        if p in param_names:
            param_initial[param_names.index(p)]=test_dict[p]
            search_log(logfile, f"Setting parameter {p} initial value to {test_dict[p]} from base config")

    for i,p in enumerate(config["search_params"]):
        if "initial" in config["search_params"][p]:
            param_initial[i]=float(config["search_params"][p]["initial"])
            search_log(logfile, f"Setting parameter {p} initial value to {param_initial[i]} from search config")
        assert param_initial[i] is not None, f"Parameter {p} missing intial value"
        param_step.append(float(config["search_params"][p]["step"]))
        param_min.append(float(config["search_params"][p]["min"]))
        param_max.append(float(config["search_params"][p]["max"]))

    search_log(logfile, "Search params:"+str(param_names))

    val, best_full_result=search_test(config, param_names, param_initial, param_min, param_max, results, split="train")
    vec_best=copy.copy(param_initial)
    val_best=val
    step_multiplier=4
    final_multiplier=0.5
    if "initial_mult" in config:
        step_multiplier=config["initial_mult"]
    if "final_mult" in config:
        final_multiplier=config["final_mult"]
    search_log(logfile, f"Starting step multiplier set to {step_multiplier}")
    iter_count=0
    param_index=0
    last_improvement_iter=0
    improvements_since_validate=0
    last_validate_iter=0
    successive_improvements=0
    search_log(logfile, f"Iter {iter_count:04d} intial {val_best:0.4f} with vector {vec_best}")
    search_log(logfile, f"... best full result {best_full_result}\n")

    while True:
        index = param_index % len(param_names)

        do_val=improvements_since_validate>0 and iter_count>=last_validate_iter+4
        if do_val or iter_count==0:
            valval, full_result_val=search_test(config, param_names, vec_best, param_min, 
                                                param_max, results, split="val")
            search_log(logfile, "======================================================")
            search_log(logfile, f"Iter {iter_count:04d}  **VALIDATE** {valval:0.4f} with vector {vec_best}")
            search_log(logfile, f"... best full result {full_result_val}\n")
            for i,_ in enumerate(vec_best):
                search_log(logfile, f"    {param_names[i]}: {vec_best[i]}")
            search_log(logfile, "======================================================")
            improvements_since_validate=0
            last_validate_iter=iter_count

        vec_up=copy.copy(vec_best)
        vec_down=copy.copy(vec_best)
        vec_up[index]+=step_multiplier*param_step[index]
        vec_up=[round(v,3) for v in vec_up]
        vec_down[index]-=step_multiplier*param_step[index]
        vec_down=[round(v,3) for v in vec_down]
        val_up,full_result_up=search_test(config, param_names, vec_up, param_min, param_max, results, split="train")
        val_down,full_result_down=search_test(config, param_names, vec_down, param_min, param_max, results, split="train")
        if val_up>val_best:
            val_best=val_up
            vec_best=vec_up
            best_full_result=full_result_up
            last_improvement_iter=iter_count
        if val_down>val_best:
            val_best=val_down
            vec_best=vec_down
            best_full_result=full_result_down
            last_improvement_iter=iter_count
        if last_improvement_iter==iter_count:
            search_log(logfile, f"Iter {iter_count:04d} mult: {step_multiplier} param {param_names[index]} new best {val_best:0.4f} with vector {vec_best} total {len(results)} results")
            search_log(logfile, f"... best full result {best_full_result}\n\n")
            successive_improvements+=1
            improvements_since_validate+=1
            if successive_improvements>=2:
                successive_improvements=0
                param_index+=1
        else:
            search_log(logfile, f"...param {param_names[index]} no improvement")
            successive_improvements=0
            param_index+=1

        iter_count+=1
        if iter_count>last_improvement_iter+len(param_names)+1:
            step_multiplier*=0.5
            last_improvement_iter=iter_count
            search_log(logfile, f"Iter {iter_count:04d} ---- reducing multiplier to {step_multiplier}----")
            if step_multiplier<final_multiplier:
                print("All done!")
                exit()

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