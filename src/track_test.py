
import os
import copy
import math
import numpy as np
import motmetrics as mm
import pickle
import time
from tqdm.auto import tqdm
import traceback
import stuff
import xlsxwriter
import datetime
import multiprocessing as mp
from multiprocessing import Process, Queue
import src.trackset as ts

def mot_obj(obj, w, h):
    ol=int(obj.box[0]*w)
    ot=int(obj.box[1]*h)
    ow=int((obj.box[2]-obj.box[0])*w)
    oh=int((obj.box[3]-obj.box[1])*h)
    return [obj.track_id, ol, ot, ow, oh]

def fitness_score(r):
    return r["mota"]-0.0005*r["fp_tracks"]-0.0*r["fp_tracks_frac"]-0.002*r["fp_per_frame"]

def summary_string(r):
    s=f" MOTA:{r['mota']:6.5f}"
    s+=f" Fit:{r['fitness']:6.5f}"
    s+=f" FPpf:{r['fp_per_frame']:5.2f}"
    s+=f" FPTf:{r['fp_tracks_frac']:5.3f}"
    s+=f" FNPo:{r['fn_per_obj']:5.3f}"
    s+=f" FPTr:{r['fp_tracks']}"
    s+=f" SWPo:{r['switch_per_obj']:5.3f}"
    s+=f" FRPo:{r['frag_per_obj']:5.3f}"
    s+=f" Skip:{r['tracked_frames_skipped_frac']:0.2f}"
    return s

def compute_metrics(gt, test, 
                    max_duration=1000, 
                    frame_metrics=False, 
                    match_iou=0.45, 
                    classes_to_test=["person"],
                    eval_rate_divisor=1):
    assert match_iou<0.9 and match_iou>0.1, f"stupid match_iou {match_iou}"
    duration=min(max_duration, max(gt.duration_seconds(), test.duration_seconds()))
    t=0
    img_w=gt.metadata["width"]
    img_h=gt.metadata["height"]
    cl=gt.metadata["classes"]

    # run evaluation at the framerate of the original video
    time_incr=(1.0/gt.metadata["frame_rate"])*eval_rate_divisor
    acc = mm.MOTAccumulator(auto_id=True)
    tot=0
    while t<duration:
        # get GT and Test objects at time
        # this interpolates objects if there is no frame at that time
        gt_obj=gt.objects_at_time(t)
        tot+=len(gt_obj)
        gt_obj=[o for o in gt_obj if gt.metadata["classes"][o.cl] in classes_to_test]
        test_obj=test.objects_at_time(t)
        assert test_obj is not None
        test_obj=[o for o in test_obj if test.metadata["classes"][o.cl] in cl]
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
    metrics_dict["tracked_frames"]=len(test.frames)

    skipped=0
    for i in range(len(test.frames)):
        if test.frames[i]["objects"] is None:
            skipped+=1
    metrics_dict["tracked_frames_skipped"]=skipped
    metrics_dict["tracked_frames_skipped_frac"]=skipped/len(test.frames)

    metrics_dict["tracked_time"]=duration
    metrics_dict["tracked_fps"]=len(test.frames)/duration
    metrics_dict["match_iou"]=match_iou
    metrics_dict["missed"]=num_never_detected
    metrics_dict["mostly_lost2"]=metrics_dict["mostly_lost"]-num_never_detected
    metrics_dict["mostly_tracked_frac"]=metrics_dict
    for m in ["mostly_tracked", "partially_tracked", "mostly_lost2", "missed", "fp_tracks"]:
        metrics_dict[m+"_frac"]=metrics_dict[m]/(metrics_dict["num_unique_objects"]+1e-7)
    metrics_dict["fp_per_frame"]=metrics_dict["num_false_positives"]/(metrics_dict["num_frames"]+1e-7) # false positive dets per frame
    metrics_dict["fn_per_obj"]=metrics_dict["num_misses"]/(metrics_dict["num_objects"]+1e-7) # num false negative dets per real object GT det
    metrics_dict["switch_per_obj"]=metrics_dict["num_switches"]/(metrics_dict["num_unique_objects"]+1e-7) # num switches per unique object
    metrics_dict["frag_per_obj"]=metrics_dict["num_fragmentations"]/(metrics_dict["num_unique_objects"]+1e-7)
    metrics_dict["fitness"]=fitness_score(metrics_dict)
    
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
        #else:
        #    print(f"{key}: Key not found in dictionary")
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
                    if p not in ["fitness","fp_per_frame","fn_per_obj","switch_per_obj","frag_per_obj"]:
                        er[p]=sum([r["result"][p] for r in filtered if "result" in r and p in r["result"]])
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
                er["fp_per_frame"]=er["num_false_positives"]/(er["num_frames"]+1e-7) # false positive dets per frame
                er["fn_per_obj"]=er["num_misses"]/(er["num_objects"]+1e-7) # num false negative dets per real object GT det
                er["switch_per_obj"]=er["num_switches"]/(er["num_unique_objects"]+1e-7) # num switches per unique object
                er["frag_per_obj"]=er["num_fragmentations"]/(er["num_unique_objects"]+1e-7)
                er['tracked_frames']/=len(filtered)
                er['tracked_time']/=len(filtered)
                er['tracked_fps']/=len(filtered)
                er['tracked_frames_skipped_frac']=er['tracked_frames_skipped_frac']/len(filtered)
                er['fitness']=fitness_score(er)
                
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
        rs=f"{result['params']['ds_key']:30s} {result['params']['test_key']:32}"+rs
        out_txt.append(rs)
        out_sort.append(result["result"][sort_key]+ds_index*1000)
    print(rh)
    Z = [x for _,x in sorted(zip(out_sort, out_txt), reverse = True)]
    for z in Z:
        print(z)

    result_location="/mldata/results/track"
    directory=os.path.join(result_location, datetime.date.today().strftime('%Y-%m-%d'))
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
            val = result["result"][cs[0]] if cs[0] in result["result"] else 0
            if math.isinf(val):
                worksheet.write(i+1, j+2, "INF")
            elif math.isnan(val):
                worksheet.write(i+1, j+2, "NAN")
            else:
                worksheet.write(i+1, j+2, round(val, 3))
    workbook.close()

    return results2

def run_one_test(params, pbar=None):
    trackset=ts.TrackSet()
    trackset_gt=ts.TrackSet(params["ds_path"])
    trackset.import_create(trackset_gt,
                           track_min_interval=params["min_interval"],
                           display=params["display"],
                           max_duration=params["max_duration"],
                           config_file=params["config"],
                           params=params,
                           pbar=pbar)
    match_iou=0.45
    if "match_iou" in params:
        match_iou=params["match_iou"]
    eval_rate_divisor=1
    if "eval_rate_divisor" in params:
        eval_rate_divisor=params["eval_rate_divisor"]
    result=compute_metrics(trackset_gt, trackset, 
                           max_duration=params["max_duration"], 
                           match_iou=match_iou,
                           eval_rate_divisor=eval_rate_divisor)
    del trackset
    del trackset_gt

    entry={"params":params, 
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

def track_test(config, split=None):
    
    if isinstance(config, str):
        config=stuff.load_dictionary(config)

    if "framerates" in config:
        expanded_tests={}
        for t in config["tests"]:
            c=config["tests"][t]
            if "min_interval" in c:
                expanded_tests[t]=c
                continue
            for f in config["framerates"]:
                t_fr=copy.deepcopy(c)
                if f<0:
                    t_fr["min_interval"]=f
                else:
                    t_fr["min_interval"]=1/(f+0.01)
                expanded_tests[t+f", {f}fps"]=t_fr
        config["tests"]=expanded_tests
    
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
                
                tests_to_run.append(params)
            else:
                output_results.append(result)

    mp.set_start_method('spawn', force=True)
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
                dname = os.path.dirname(resultfile)
                os.makedirs(dname, exist_ok=True)
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
