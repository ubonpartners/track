
import os
import sys
import copy
import json
import numpy as np
import motmetrics as mm
import pickle
import time
from tqdm.auto import tqdm
import stuff
from stuff import coord
import datetime
import src.trackset as ts
import threading
import logging
import src.track_util as tu

tqdm.set_lock(threading.RLock())

def mot_obj(obj, w, h):
    ol=int(obj.box[0]*w)
    ot=int(obj.box[1]*h)
    ow=int((obj.box[2]-obj.box[0])*w)
    oh=int((obj.box[3]-obj.box[1])*h)
    return [obj.track_id, ol, ot, ow, oh]

def _box_in_ignore(box, ignore_boxes, frac_thresh):
    """True if a fraction >= frac_thresh of `box` overlaps any ignore box."""
    dx1, dy1, dx2, dy2 = box
    det_area = max(1e-7, (dx2 - dx1) * (dy2 - dy1))
    for ix1, iy1, ix2, iy2 in ignore_boxes:
        ow = min(dx2, ix2) - max(dx1, ix1)
        oh = min(dy2, iy2) - max(dy1, iy1)
        if ow > 0 and oh > 0 and (ow * oh) / det_area >= frac_thresh:
            return True
    return False

# fitness penalises the honest FP-track count (de-gamed: IoU==0 run-count,
# GT-grounded, parameter-free), length-normalised by VIDEO DURATION so
# the penalty is sequence-length AND frame-rate invariant — honest_v2 is
# an EPISODE count that scales with duration (a 2s phantom is 1 run at
# 10 or 30 fps), not with sampling density. Aggregate rows sum
# Σhonest/Σduration. K calibrated so the aggregate honest penalty
# matches the old 5e-4*fp_tracks penalty on the shipped networks. Old
# fp_tracks is still reported alongside for comparison.
_FP_TRACK_COEF = 0.35   # per honest-FP episode per second of video
def fitness_score(r):
    h = r.get("fp_tracks_honest_v2")
    if h is None:                       # legacy rows w/o the honest field
        h = r["fp_tracks"] * 10.0       # scale-match so fitness stays sane
    dur = r.get("duration", 0) or 0
    h_rate = h / dur if dur > 0 else 0.0
    return r["mota"] - _FP_TRACK_COEF * h_rate - 0.002 * r["fp_per_frame"]

def gt_class_box_counts(gt):
    """GT box count per class NAME. Drives the multi-class skip rule
    (multi_class_and_hints.md §2/§4): an extra class with ZERO GT boxes
    in a clip is not scored there — its metrics would be free points
    (or, matched against nothing, all-FP noise). Person is exempt from
    the skip: zero-GT person clips are deliberate FP probes."""
    classes = gt.metadata.get("classes", [])
    counts = {c: 0 for c in classes}
    for frame in gt.frames:
        objs = frame.get("objects")
        if not objs:
            continue
        for rec in objs.values():
            cl = rec.get("class", 0)
            if 0 <= cl < len(classes):
                counts[classes[cl]] += 1
    return counts


def fitness_multi_score(r, weights):
    """Config-driven combined fitness (multi_class_and_hints.md §3):
    fitness_multi = sum_c w_c * fitness_c over classes PRESENT in the
    row. A class a clip has no GT for contributes nothing (weight 0 for
    that row, not fitness 0) — the zero-GT guard from plan §14."""
    weights = weights or {"person": 1.0}
    total = 0.0
    present = False
    for cls, w in weights.items():
        key = "fitness" if cls == "person" else f"fitness_{cls}"
        if key in r:
            total += float(w) * r[key]
            present = True
    return total if present else None


def summary_string(r):
    s=f" MOTA:{r['mota']:6.5f}"
    if 'idf1' in r:
        s+=f" IDF1:{r['idf1']:6.5f}"
    if 'fitness' in r:
        s+=f" Fit:{r['fitness']:6.5f}"
    if 'fp_per_frame' in r:
        s+=f" FPpf:{r['fp_per_frame']:5.2f}"
    if 'fp_tracks_frac' in r:
        s+=f" FPTf:{r['fp_tracks_frac']:5.3f}"
    if 'fn_per_obj' in r:
        s+=f" FNPo:{r['fn_per_obj']:5.3f}"
    if 'fp_tracks' in r:
        s+=f" FPTr:{r['fp_tracks']}"
    if 'fp_tracks_honest_v2' in r:
        s+=f" FPh2:{int(r['fp_tracks_honest_v2'])}"
    if 'switch_per_obj' in r:
        s+=f" SWPo:{r['switch_per_obj']:5.3f}"
    if 'frag_per_obj' in r:
        s+=f" FRPo:{r['frag_per_obj']:5.3f}"
    if 'tracked_frames_skipped_frac' in r:
        s+=f" Skip:{r['tracked_frames_skipped_frac']:0.2f}"
    if 'average_detection_roi_area' in r:
        s+=f" dROI:{r['average_detection_roi_area']:0.2f}"
    if 'det_ap_person' in r:
        s+=f" PmAP:{r['det_ap_person']:0.3f}"
    if 'det_ap_face' in r:
        s+=f" FmAP:{r['det_ap_face']:0.3f}"
    if 'det_ap_vehicle' in r:
        s+=f" VmAP:{r['det_ap_vehicle']:0.3f}"
    if 'mota_vehicle' in r:
        s+=f" vMOTA:{r['mota_vehicle']:6.5f}"
    if 'fp_per_frame_vehicle' in r:
        s+=f" vFPpf:{r['fp_per_frame_vehicle']:5.2f}"
    if 'fitness_multi' in r:
        s+=f" FITm:{r['fitness_multi']:6.5f}"
    return s

def annotation_floors(gt_metadata):
    """Per-class annotation completeness floors from GT metadata:
    min_annotated_height: {class_name: normalized_height}. The legacy
    scalar min_annotated_person_height is read as {"person": v}.
    Absent class = fully annotated (no floor)."""
    floors = dict(gt_metadata.get("min_annotated_height") or {})
    legacy = gt_metadata.get("min_annotated_person_height")
    if legacy and "person" not in floors:
        floors["person"] = float(legacy)
    return {k: float(v) for k, v in floors.items()}


def compute_detection_metrics(gt, test,
                              metrics_dict,
                              classes_for_det_map=["person","face"]):
    # compute detection mAP
    # this works if the debug has the detections
    target_class=[]
    conf=[]
    tp=[]
    pred_class=[]

    det_class_remap=stuff.make_class_remap_table(test.metadata["classes"], classes_for_det_map)
    gt_class_remap=stuff.make_class_remap_table(gt.metadata["classes"], classes_for_det_map)

    for i,frame in enumerate(test.frames):
        if "tracker_debug" in frame:
            t=frame["frame_time"]
            debug=frame["tracker_debug"]
            if debug is not None and "detections" in debug:
                det=debug["detections"]
                # get the GT objects from the trackset
                # and the detected objects, do the mAP matching
                gt_obj=gt.objects_at_time(t, class_remap_table=gt_class_remap)
                iou_thr=0.5
                det_obj=[]
                for d in det["data"]["detections"]:
                    cl=d["class"]
                    if cl<0 or cl>=len(classes_for_det_map):
                        logging.error(f"detection class {cl} outside range (classes {test.metadata['classes']})")
                        continue
                    cl=det_class_remap[d["class"]]
                    if cl is not None:
                        det_obj.append(tu.Object(box=d["box"],cl=d["class"],conf=d["confidence"]))

                floors = annotation_floors(gt.metadata)
                if floors:
                    def _keep(o):
                        fl = floors.get(classes_for_det_map[o.cl]
                                        if 0 <= o.cl < len(classes_for_det_map)
                                        else "", 0.0)
                        return not (fl > 0 and (o.box[3] - o.box[1]) < fl)
                    gt_obj = [o for o in gt_obj if _keep(o)]
                    det_obj = [o for o in det_obj if _keep(o)]
                gts = sorted(gt_obj,key=lambda x: x.confidence,reverse=True)
                dets = sorted(det_obj,key=lambda x: x.confidence,reverse=True)
                gt_matched=[-1]*len(gts)
                det_matched=[-1]*len(dets)

                for j,_ in enumerate(dets):
                    for i,_ in enumerate(gts):
                        if gt_matched[i]==-1 and gts[i].cl==dets[j].cl and stuff.box_iou(gts[i].box, dets[j].box)>iou_thr:
                            gt_matched[i]=j
                            det_matched[j]=i
                            break

                for j,_ in enumerate(gts):
                    target_class.append(gts[j].cl)

                for j,_ in enumerate(dets):
                    pred_class.append(dets[j].cl)
                    conf.append(dets[j].confidence)
                    tp.append(0 if det_matched[j]==-1 else 1)

    if len(conf)>5:
        ap, p, r, p_curve, r_curve = stuff.ap_calc(conf, tp, pred_class, target_class, len(classes_for_det_map), min_gt=5, pr_curves=True)
        interesting_thr=[0.25,0.3]
        for cl,cl_name in enumerate(classes_for_det_map):
            metrics_dict["det_ap_"+cl_name]=ap[cl]
            metrics_dict["det_p_"+cl_name]=p[cl]
            metrics_dict["det_r_"+cl_name]=r[cl]
            for thr in interesting_thr:
                s=f"th{int(thr*100):2d}"
                index=int(len(p_curve[cl])*thr)
                metrics_dict[f"det_p_{cl_name}_{s}"]=p_curve[cl][index]
                metrics_dict[f"det_r_{cl_name}_{s}"]=r_curve[cl][index]


# tcode: 0 = FP (unmatched), 1 = matched-to-a-GT (MATCH/SWITCH/transfer
# family). Rows that are neither are not emitted for an HId.
_MATCHED_TYPES = {"MATCH", "SWITCH", "TRANSFER", "MIGRATE", "ASCEND"}


def _events_by_hid_from_df(df):
    """Compact per-HId event sequence from a motmetrics events frame:
    {hid:int -> [(frameid:int, tcode:int), ...] sorted by frameid}."""
    sub = df[df["HId"].notna() & (df["Type"] != "RAW")]
    out = {}
    for hid, g in sub.groupby("HId"):
        g = g.sort_index(level=0)
        seq = []
        for fid, typ in zip(g.index.get_level_values(0), g["Type"]):
            if typ == "FP":
                seq.append((int(fid), 0))
            elif typ in _MATCHED_TYPES:
                seq.append((int(fid), 1))
        if seq:
            out[int(hid)] = seq
    return out


def _honest_fp_runs_core(events_by_hid, cd_frames, gt_cd_frames):
    """De-gamed honest FP-track ruler.

    honest_fp_tracks = number of contiguous runs of FP frames whose box
    overlaps NO real GT object (IoU == 0 with every GT box that frame, or
    no GT that frame), where a matched frame ends a run. Parameter-free.

    Rationale: "no overlap with any real object" is the simplest
    definition of a false detection. Run-COUNT (not frame-SUM): a short
    spurious blip is ONE unique FP; an FP run welded onto a
    GT-matched track is still counted (the gaming the old fp_tracks
    hid). GT-grounded so box-jitter / occlusion-coast on a real track
    stays on/over its real object (IoU>0) and is NOT counted.

    cd_frames[frameid][hid]    = (cx,cy,diag,l,t,w,h) for the hyp box
    gt_cd_frames[frameid][gid] = same for the GT box
    """

    def _iou(a, b):
        ix = min(a[3] + a[5], b[3] + b[5]) - max(a[3], b[3])
        iy = min(a[4] + a[6], b[4] + b[6]) - max(a[4], b[4])
        if ix <= 0 or iy <= 0:
            return 0.0
        inter = ix * iy
        ua = a[5] * a[6] + b[5] * b[6] - inter
        return inter / ua if ua > 0 else 0.0

    def contaminating(frameid, hid):
        c = cd_frames[frameid].get(hid) if (
            cd_frames is not None and frameid < len(cd_frames)) else None
        if c is None or len(c) < 7:
            return True                       # no/old geom -> conservative
        gts = (gt_cd_frames[frameid] if (gt_cd_frames is not None
               and frameid < len(gt_cd_frames)) else None)
        if not gts:
            return True                       # zero real objects -> spurious
        return all(_iou(c, v) <= 0.0 for v in gts.values())

    total = nm = inrun = 0
    for hid, seq in events_by_hid.items():
        matched = any(tc == 1 for _, tc in seq)
        runs = 0
        cur = False
        for fid, tc in seq:
            if tc == 1:                       # matched frame ends a run
                cur = False
                continue
            if contaminating(fid, hid):
                if not cur:
                    runs += 1
                    cur = True
            else:
                cur = False
        total += runs
        if matched:
            inrun += runs
        else:
            nm += runs
    return {"honest_fp_tracks_v2": int(total),
            "nm": int(nm), "inrun": int(inrun)}


def compute_metrics(gt, test,
                    max_duration=1000,
                    frame_metrics=False,
                    match_iou=0.45,
                    classes_to_test=["person"],
                    classes_for_det_map=["person","face"],
                    eval_rate_divisor=1,
                    eval_min_framerate=30.0,
                    min_person_height=0.0,
                    show_pbar=False):
    assert match_iou<0.9 and match_iou>0.1, f"stupid match_iou {match_iou}"
    start_time=min(gt.first_frame_time(), test.first_frame_time())
    t=start_time
    last_time=max(gt.last_frame_time(), test.last_frame_time())
    last_time=min(last_time, t+max_duration)
    duration=last_time-t

    img_w=gt.metadata["width"]
    img_h=gt.metadata["height"]
    cl=gt.metadata["classes"]

    logging.debug(f"metrics main loop")

    # divide the evaluation framerate down until it's the lowest value >= min
    while(gt.metadata["frame_rate"]/(eval_rate_divisor+1) >= eval_min_framerate):
        eval_rate_divisor+=1

    # run evaluation at the framerate of the original video, potentially divided down
    time_incr=(1.0/gt.metadata["frame_rate"])*eval_rate_divisor

    acc = mm.MOTAccumulator(auto_id=True)

    frame_events=[]
    frame_index=0
    # Per-acc.update() hyp/gt box geometry for the honest-FP ruler. One
    # dict per accumulator frame, keyed by track_id == motmetrics HId/OId
    # (auto_id order == update-call order == this list's index).
    # {tid:(cx,cy,diag,l,t,w,h)} in pixels.
    hyp_cd_frames=[]
    gt_cd_frames=[]

    if show_pbar:
        pbar=tqdm(total=int(duration/time_incr),
              desc=f"Computing metrics...",
              colour="#ffcc00",
              leave=False)

    gt_class_remap_table=stuff.make_class_remap_table(gt.metadata["classes"], classes_to_test)
    det_class_remap_table=stuff.make_class_remap_table(test.metadata["classes"], classes_to_test)

    # MOTChallenge-style ignore regions: GT class "other" (e.g. PersonPath22 crowd
    # boxes, MOT distractor/occluder/reflection) marks regions where individuals
    # are not annotated. Test detections whose box is mostly inside such a region
    # should not be counted as false positives.
    ignore_cl_idx = (gt.metadata["classes"].index("other")
                     if "other" in gt.metadata["classes"] else None)
    # Dataset-declared annotation floors: below the per-class normalized
    # height the GT makes no completeness claim (autolabelled/augmented
    # datasets). GT below it is ignored AND unmatched sub-floor
    # predictions are not charged as FP (nobody annotated that zone).
    # The caller's min_person_height combines with the person floor.
    floors = annotation_floors(gt.metadata)
    if min_person_height > 0:
        floors["person"] = max(floors.get("person", 0.0),
                               min_person_height)
    eval_floors = {c: floors[c] for c in floors if c in classes_to_test}
    # Threshold: drop a test detection if >=50% of its area falls inside any
    # ignore region. Matches the standard "don't care" behaviour in MOTChallenge.
    ignore_overlap_frac = 0.5

    while t<last_time:
        # get GT and Test objects at time
        # this interpolates objects if there is no frame at that time

        dbg=False #abs(t-5.125)<0.05
        gt_obj=gt.objects_at_time(t, class_remap_table=gt_class_remap_table)
        test_obj=test.objects_at_time(t, class_remap_table=det_class_remap_table)

        assert test_obj is not None
        test_obj=[o for o in test_obj if test.metadata["classes"][o.cl] in cl]
        if gt_obj is None:
            break

        # Sub-minimum persons (normalized height < min_person_height,
        # i.e. [0-1]^2 units; 0.045 ~= the measured detector-recall knee
        # ~= 58/1280 in 1280-square units) become ignore regions: not FN
        # when missed, and dets matching them are not FP. Same don't-care
        # semantics as crowd boxes.
        small_ignore = []
        if eval_floors:
            kept_gt = []
            for g in gt_obj:
                fl = eval_floors.get(classes_to_test[g.cl], 0.0)
                if fl > 0 and (g.box[3] - g.box[1]) < fl:
                    small_ignore.append(g.box)
                else:
                    kept_gt.append(g)
            gt_obj = kept_gt

        if eval_floors:
            # detections below the floor are ignored symmetrically with
            # GT (both the dataset-declared annotation floors and the
            # caller/search-config min size). A sub-floor det that
            # genuinely matches a remaining above-floor GT still counts
            # (dropping it would fabricate an FN at the boundary).
            gt_boxes_all = [g.box for g in gt_obj]
            kept_t = []
            for det in test_obj:
                fl = eval_floors.get(classes_to_test[det.cl]
                                     if det.cl < len(classes_to_test)
                                     else "", 0.0)
                if (fl > 0 and (det.box[3] - det.box[1]) < fl
                        and not any(coord.box_iou(det.box, gb) >= match_iou
                                    for gb in gt_boxes_all)):
                    continue
                kept_t.append(det)
            test_obj = kept_t

        if ignore_cl_idx is not None or small_ignore:
            ignore_boxes = ([o.box for o in gt.objects_at_time(t) or []
                             if o.cl == ignore_cl_idx]
                            if ignore_cl_idx is not None else [])
            ignore_boxes = ignore_boxes + small_ignore
            if ignore_boxes:
                gt_person_boxes = [g.box for g in gt_obj]
                kept = []
                for det in test_obj:
                    if _box_in_ignore(det.box, ignore_boxes,
                                      ignore_overlap_frac):
                        # If this detection clearly matches a real GT person we
                        # keep it so the matcher can score the match; otherwise
                        # treat it as a "don't care" detection.
                        if not any(coord.box_iou(det.box, gb) >= match_iou
                                   for gb in gt_person_boxes):
                            continue
                    kept.append(det)
                test_obj = kept

        gt_dets=[mot_obj(g, img_w, img_h) for g in gt_obj]
        t_dets=[mot_obj(t, img_w, img_h) for t in test_obj]
        gt_dets=np.array(gt_dets)
        t_dets=np.array(t_dets)
        stats={"num_gt_tracks":len(gt_dets),
            "num_tracks":len(t_dets)}
        frame_events.append({"frame_time":t, "events":{}, "stats":stats})

        C=[[]]
        if len(gt_dets)>0 and len(t_dets)>0:
            C = mm.distances.iou_matrix(gt_dets[:,1:], t_dets[:,1:], \
                                max_iou=match_iou) # format: gt, t

        acc.update(gt_dets[:,0].astype('int').tolist() if len(gt_dets)>0 else [], \
                t_dets[:,0].astype('int').tolist() if len(t_dets)>0 else [], C)
        # capture hyp/gt box geometry for this accumulator frame
        # (mot_obj rows are [track_id, l, t, w, h] in pixels)
        cd={}
        for r in t_dets:
            tid=int(r[0]); l,tp,w,h=float(r[1]),float(r[2]),float(r[3]),float(r[4])
            cd[tid]=(l+w*0.5, tp+h*0.5, (w*w+h*h)**0.5, l, tp, w, h)
        hyp_cd_frames.append(cd)
        gd={}
        for r in gt_dets:
            gid=int(r[0]); l,tp,w,h=float(r[1]),float(r[2]),float(r[3]),float(r[4])
            gd[gid]=(l+w*0.5, tp+h*0.5, (w*w+h*h)**0.5, l, tp, w, h)
        gt_cd_frames.append(gd)
        t+=time_incr
        if show_pbar:
            pbar.update(1)
        frame_index=0

    logging.debug(f"metrics processing")

    if show_pbar:
        pbar.set_description("PyMOT processing...")

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
    # video duration in seconds; fitness uses it to length-normalise
    # the honest FP-track episode count (frame-rate invariant).
    metrics_dict["duration"]=duration

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

    # De-gamed honest FP-track count (IoU==0 run-count, GT-grounded,
    # parameter-free). THIS drives `fitness`. Old fp_tracks is kept
    # for reporting/comparison only.
    try:
        ev_by_hid = _events_by_hid_from_df(df)
        hrn = _honest_fp_runs_core(ev_by_hid, hyp_cd_frames, gt_cd_frames)
        metrics_dict["fp_tracks_honest_v2"] = hrn["honest_fp_tracks_v2"]
        metrics_dict["fp_h2_nm"]            = hrn["nm"]
        metrics_dict["fp_h2_inrun"]         = hrn["inrun"]
    except Exception:
        # logging.warning has no handler in spawn workers — write to
        # fd 2 which the eval captures.
        import traceback as _tb
        sys.stderr.write("honest-fp compute FAILED: "
                         + _tb.format_exc() + "\n"); sys.stderr.flush()
        # Fall back to the old gamed count so fitness degrades
        # gracefully (failure is already loud on stderr).
        metrics_dict["fp_tracks_honest_v2"] = num_false_positive_tracks
        metrics_dict["fp_h2_nm"]            = num_false_positive_tracks
        metrics_dict["fp_h2_inrun"]         = 0

    all_gt_ids = df['OId'].dropna().unique()
    matched_gt_ids = df.loc[df['Type'] == 'MATCH', 'OId'].unique()
    completely_lost_gt_ids = set(all_gt_ids) - set(matched_gt_ids)
    metrics_dict["missed"]=len(completely_lost_gt_ids)

    logging.debug(f"metrics aux")

    if show_pbar:
        pbar.close()

    skipped=0
    total_detection_roi_area=0
    metrics_dict["tracked_frames"]=len(test.frames)
    for i in range(len(test.frames)):
        if test.frames[i]["objects"] is None:
            skipped+=1
        else:
            detection_roi_area=1.0
            if "tracker_debug" in test.frames[i] and test.frames[i]["tracker_debug"] is not None:
                if "detection_roi" in test.frames[i]["tracker_debug"]:
                    detection_roi_area=stuff.box_a(test.frames[i]["tracker_debug"]["detection_roi"]["data"]["roi"])
            total_detection_roi_area+=detection_roi_area

    average_detection_roi_area=total_detection_roi_area/(len(test.frames)-skipped+1e-7)
    metrics_dict["average_detection_roi_area"]=average_detection_roi_area

    metrics_dict["tracked_frames_skipped"]=skipped
    metrics_dict["tracked_frames_skipped_frac"]=skipped/len(test.frames)

    metrics_dict["tracked_time"]=duration
    metrics_dict["tracked_fps"]=len(test.frames)/duration
    metrics_dict["match_iou"]=match_iou

    metrics_dict["mostly_lost2"]=metrics_dict["mostly_lost"]-metrics_dict["missed"]
    for m in ["mostly_tracked", "partially_tracked", "mostly_lost2", "missed", "fp_tracks"]:
        metrics_dict[m+"_frac"]=metrics_dict[m]/(metrics_dict["num_unique_objects"]+1e-7)
    metrics_dict["fp_per_frame"]=metrics_dict["num_false_positives"]/(metrics_dict["num_frames"]+1e-7) # false positive dets per frame
    metrics_dict["fn_per_obj"]=metrics_dict["num_misses"]/(metrics_dict["num_objects"]+1e-7) # num false negative dets per real object GT det
    metrics_dict["switch_per_obj"]=metrics_dict["num_switches"]/(metrics_dict["num_unique_objects"]+1e-7) # num switches per unique object
    metrics_dict["frag_per_obj"]=metrics_dict["num_fragmentations"]/(metrics_dict["num_unique_objects"]+1e-7)
    metrics_dict["fitness"]=fitness_score(metrics_dict)

    # optionally extract per-frame MOT metrics
    if frame_metrics:
        logging.debug(f"metrics per frame")
        t=start_time
        frame_index=0

        while t<last_time:
            assert frame_events[frame_index]["frame_time"]==t
            if frame_index in acc.mot_events.index.get_level_values(0).unique():
                frame=acc.mot_events.xs(frame_index, level=0) #acc.mot_events.loc[frame_index]
                events=frame.to_dict(orient='index')
                frame_events[frame_index]["events"]=events
                num_match=0
                num_miss=0
                num_switch=0
                num_fp=0
                for e in events:
                    if events[e]["Type"]=="MATCH":
                        num_match+=1
                    if events[e]["Type"]=="MISS":
                        num_miss+=1
                    if events[e]["Type"]=="SWITCH":
                        num_switch+=1
                    if events[e]["Type"]=="FP":
                        num_fp+=1
                num_gt=frame_events[frame_index]["stats"]["num_gt_tracks"]
                denom = num_gt if num_gt > 0 else 1e-7  # avoid division by zero
                mota = 1.0 - (num_miss + num_fp + num_switch) / denom

                frame_events[frame_index]["stats"]["mota"] = mota
                frame_events[frame_index]["stats"]|={"frame":frame_index,
                                                     "num_match":num_match,
                                                     "num_miss":num_miss,
                                                     "num_fp":num_fp,
                                                     "num_switch":num_switch,
                                                     "mota":mota }
            t+=time_incr
            frame_index+=1
    del mh
    del acc

    logging.debug(f"detection metrics")

    if classes_for_det_map:      # None/[] = skip (the per-extra-class calls; det AP is computed once, on the person pass)
        compute_detection_metrics(gt, test, metrics_dict, classes_for_det_map)

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
            if fmt=="seconds_ago":
                rs+=(f"{stuff.format_seconds_ago(result[key]):>6s}")
            else:
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

def display_results(config, results, columns, sort_key):
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
                # Per-class namespacing (multi_class_and_hints.md §2): the
                # SAME derived-metric recompute runs once per class suffix
                # ("" = person/base, "_vehicle", ...) from that suffix's own
                # summed counts. Suffixes are detected from the idtp_* keys.
                nonsum_bases=["fitness","fp_per_frame","fn_per_obj","switch_per_obj","frag_per_obj"]
                suffixes=[""]+sorted({p[len("idtp"):] for p in params
                                      if p.startswith("idtp") and p!="idtp"})
                nonsum={b+sfx for b in nonsum_bases for sfx in suffixes} | {"fitness_multi"}
                # Per-clip volume cap (search_review.md §1.3): with
                # clip_weight_cap_pctl set, a clip whose GT volume
                # (num_objects, per class suffix) exceeds that percentile of
                # its peers contributes scaled-down COUNTS — so one crowd
                # monster (MOT20-05 = 75.6% of v11's val boxes) cannot own a
                # rollup. Rates/derived metrics recompute from the scaled
                # counts, so everything stays self-consistent.
                cap_pctl=config.get("clip_weight_cap_pctl")
                row_w={}   # id(row) -> {suffix: weight}
                if cap_pctl:
                    for sfx in suffixes:
                        vk="num_objects"+sfx
                        vols=[r["result"][vk] for r in filtered
                              if vk in r.get("result",{}) and r["result"][vk]>0]
                        if len(vols)>=2:
                            cap=float(np.percentile(vols, float(cap_pctl)))
                            for r in filtered:
                                v=r.get("result",{}).get(vk, 0)
                                if v>cap>0:
                                    row_w.setdefault(id(r),{})[sfx]=cap/v
                capped_bases={"num_frames","num_objects","num_false_positives",
                              "num_misses","num_switches","num_unique_objects",
                              "num_fragmentations","num_matches","idtp","idfp",
                              "idfn","fp_tracks","fp_tracks_honest_v2",
                              "fp_h2_nm","fp_h2_inrun","duration","missed",
                              "mostly_tracked","partially_tracked",
                              "mostly_lost","mostly_lost2"}
                def _split_suffix(key):
                    for sfx in sorted(suffixes, key=len, reverse=True):
                        if sfx and key.endswith(sfx):
                            return key[:-len(sfx)], sfx
                    return key, ""
                def _w(r, key):
                    base, sfx=_split_suffix(key)
                    if base not in capped_bases:
                        return 1.0
                    return row_w.get(id(r), {}).get(sfx, 1.0)
                for p in params:
                    if p not in nonsum:
                        vals=[_w(r, p)*r["result"][p] for r in filtered
                              if "result" in r and p in r["result"]]
                        # A key NO row in this group has stays absent — an
                        # empty sum would fabricate zero counts, and the
                        # derived recompute would then report e.g.
                        # mota_vehicle=1.0 for a group with no vehicle GT.
                        if vals:
                            er[p]=sum(vals)
                for sfx in suffixes:
                    if ("idtp"+sfx) not in er:
                        continue
                    weighted_motp_sum=0
                    for r in filtered:
                        rr=r.get("result", {})
                        if ("motp"+sfx) in rr and ("idtp"+sfx) in rr:
                            weighted_motp_sum += rr["motp"+sfx]*rr["idtp"+sfx]*_w(r, "idtp"+sfx)
                    er["idf1"+sfx]= (2 * er["idtp"+sfx]) / (2 * er["idtp"+sfx] + er["idfp"+sfx] + er["idfn"+sfx]+1e-7)
                    er["mota"+sfx]= 1 - (er["num_false_positives"+sfx] + er["num_misses"+sfx] + er["num_switches"+sfx]) / (er["num_objects"+sfx]+1e-7)
                    er["motp"+sfx]=weighted_motp_sum/(er["idtp"+sfx]+1e-7)
                    er["fp_per_frame"+sfx]=er["num_false_positives"+sfx]/(er["num_frames"+sfx]+1e-7) # false positive dets per frame
                    er["fn_per_obj"+sfx]=er["num_misses"+sfx]/(er["num_objects"+sfx]+1e-7) # num false negative dets per real object GT det
                    er["switch_per_obj"+sfx]=er["num_switches"+sfx]/(er["num_unique_objects"+sfx]+1e-7) # num switches per unique object
                    er["frag_per_obj"+sfx]=er["num_fragmentations"+sfx]/(er["num_unique_objects"+sfx]+1e-7)

                stats_to_avg=['mostly_tracked_frac','partially_tracked_frac','mostly_lost2_frac',
                              'missed_frac','fp_tracks_frac', 'time',
                              'tracked_frames','tracked_time','tracked_fps','tracked_frames_skipped_frac',
                              'average_detection_roi_area','det_ap_person', 'det_ap_face', 'det_ap_vehicle']

                for x in stats_to_avg:
                    if x in er:
                        er[x]=er[x]/len(filtered)
                # Suffixed fraction averages divide by the number of rows
                # that HAVE the class — clips without it must not dilute.
                for sfx in suffixes:
                    if sfx=="":
                        continue
                    for b in ['mostly_tracked_frac','partially_tracked_frac','mostly_lost2_frac',
                              'missed_frac','fp_tracks_frac','tracked_frames_skipped_frac']:
                        x=b+sfx
                        if x in er:
                            n=len([r for r in filtered if x in r.get("result",{})])
                            er[x]=er[x]/max(1,n)

                for sfx in suffixes:
                    if ("idtp"+sfx) in er:
                        view={b: er[b+sfx] for b in
                              ["mota","fp_per_frame","fp_tracks","fp_tracks_honest_v2","duration"]
                              if (b+sfx) in er}
                        er["fitness"+sfx]=fitness_score(view)
                fm=fitness_multi_score(er, config.get("fitness_weights"))
                if fm is not None:
                    er["fitness_multi"]=fm

                results2.append(e)
            datasets.append(name)

        # _groupmean (search_review.md §1.1): the BALANCED objective row —
        # weighted mean across the per-group __ovr rollups, so no group can
        # buy influence with GT density (micro-average within a group,
        # macro-average across groups). group_weights: {group: w} in the
        # yaml reweights; weights normalise over the groups PRESENT for
        # each key, so a metric one group lacks (e.g. vehicle keys in a
        # person-only group) averages over the groups that have it.
        real_groups=[g for g in groups if g is not None]
        if real_groups:
            gw_cfg=config.get("group_weights") or {}
            for t in tests:
                grows=[]
                for r2 in results2:
                    ds=r2["params"]["ds_key"]
                    if r2["params"]["test_key"]==t and ds.startswith("__ovr"):
                        grows.append((ds[len("__ovr"):], r2["result"]))
                if not grows:
                    continue
                er={}
                allkeys=set()
                for _, gr in grows:
                    allkeys |= set(gr.keys())
                for k in allkeys:
                    num=0.0; den=0.0
                    for gname, gr in grows:
                        if k in gr and isinstance(gr[k], (int, float)):
                            w=float(gw_cfg.get(gname, 1.0))
                            num+=w*gr[k]; den+=w
                    if den>0:
                        er[k]=num/den
                results2.append({"params":{"ds_key":"_groupmean","test_key":t},
                                 "result":er})
            datasets.append("_groupmean")

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

    if True:
        all_results=results+results2

        column_text=[]
        column_keys=[]
        for c in columns:
            column_text.append(c.split(",")[1])
            column_keys.append(c.split(",")[0])

        result=[]
        for r in all_results:
            rc=copy.deepcopy(r["result"])
            ds=r["params"]["ds_key"]
            ds=ds[:2]+ds[2:].replace("_", "")
            rc["dataset"]=ds
            rc["test"]=r['params']['test_key']
            result.append(rc)

        unique_datasets = list(dict.fromkeys(r["dataset"] for r in result))
        unique_datasets.sort()

        for r in result:
            assert "dataset" in r

        def sort_fn(r):
            return unique_datasets.index(r["dataset"]) *1000 + r[sort_key]

        data_out = stuff.show_data(result, ["dataset","test"]+column_keys,
                        ["dataset","test"]+column_text, sort_fn)
        #result["params"]["ds_key"]
        cur_time=datetime.datetime.now().strftime('%Y%m%d-%H%M')
        if "results_location" in config:
            result_location=config["results_location"]
            stuff.makedir(result_location)
            out_file=result_location+"/results-"+ \
                cur_time+".txt"
            with open(out_file, "w") as f:
                f.write(data_out)
                f.write("\n")

    return results2

def track_test_work_fn(params, mpwq_context, mpwq_progress_fn):
    logging.debug("Running here")
    trackset=ts.TrackSet()
    trackset_gt=ts.TrackSet(params["ds_path"])
    logging.debug(f"import create")
    trackset.import_create(trackset_gt,
                           track_min_interval=params["min_interval"],
                           display=params["display"],
                           config_file=params["config"],
                           params=params,
                           mpwq_context=mpwq_context,
                           mpwq_progress_fn=mpwq_progress_fn,
                           # max_duration is a REAL compute cap since
                           # 2026-07-23: upyc_tracker truncates the h264 at
                           # extraction (duration-suffixed cache entry, so
                           # capped and full evals never share files) — the
                           # C pipeline only ever sees the window. Scoring
                           # is capped by the same value in compute_metrics.
                           end_time=params.get("max_duration", 100000))
    match_iou=0.45
    if "match_iou" in params:
        match_iou=params["match_iou"]
    eval_rate_divisor=params.get("eval_rate_divisor", 1)
    eval_min_framerate=params.get("eval_min_framerate", 30.0)

    # Multi-class scoring (multi_class_and_hints.md §2): person is scored
    # exactly as before (un-suffixed keys — nothing downstream changes);
    # every EXTRA class in `classes` gets its own compute_metrics pass with
    # every result key namespaced `<key>_<class>`. A clip whose GT has zero
    # boxes of an extra class skips that class (§4: absent GT must never
    # touch a metric); person is exempt — zero-GT person clips are the
    # deliberate FP probes v11 already carries.
    eval_classes=params.get("classes", ["person"])
    if isinstance(eval_classes, str):
        eval_classes=[c.strip() for c in eval_classes.split(",") if c.strip()]
    det_map=params.get("classes_for_det_map")
    if det_map is None:
        det_map=["person","face"]
        if "vehicle" in eval_classes:
            det_map=det_map+["vehicle"]

    logging.debug(f"compute metrics")
    result=compute_metrics(trackset_gt, trackset,
                           min_person_height=params.get("min_person_height", 0.0),
                           max_duration=params["max_duration"],
                           match_iou=match_iou,
                           classes_for_det_map=det_map,
                           eval_rate_divisor=eval_rate_divisor,
                           eval_min_framerate=eval_min_framerate)

    gt_counts=gt_class_box_counts(trackset_gt)
    for cls in eval_classes:
        if cls == "person":
            continue
        if gt_counts.get(cls, 0) == 0:
            continue
        # Per-class pass: no min_person_height (the person floor must not
        # gate wide-short vehicles — plan §14) and no det-AP recompute.
        r_cls=compute_metrics(trackset_gt, trackset,
                              max_duration=params["max_duration"],
                              match_iou=match_iou,
                              classes_to_test=[cls],
                              classes_for_det_map=None,
                              eval_rate_divisor=eval_rate_divisor,
                              eval_min_framerate=eval_min_framerate)
        for k, v in r_cls.items():
            result[f"{k}_{cls}"]=v

    fm=fitness_multi_score(result, params.get("fitness_weights"))
    if fm is not None:
        result["fitness_multi"]=fm

    del trackset
    del trackset_gt
    logging.debug(f"set entry")
    entry={"params":params,
           "result":result,
           "time":datetime.datetime.now()}

    logging.debug(f"done")
    return entry

def on_result_callback(mpwq_context, result):
    cache=True
    ds_key=result["params"]["ds_key"]
    if "no_cache" in mpwq_context["config"]["datasets"][ds_key]:
        if mpwq_context["config"]["datasets"][ds_key]["no_cache"]==True:
            cache=False
    if cache is True and mpwq_context["resultfile"] is not None:
        mpwq_context["cached_results"].append(result)
        stuff.save_atomic_pickle(mpwq_context["cached_results"], mpwq_context["resultfile"])
        #logging.info(f"Saved {len(mpwq_context["cached_results"])} cached results")

def track_test(config, split=None, desc="track test"):
    start_time=time.time()
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
    num_workers=stuff.resolve_num_workers(config["num_workers"])
    cached_results=[]
    if resultfile is not None and os.path.isfile(resultfile):
        with open(resultfile, 'rb') as handle:
            cached_results = pickle.load(handle)

    datasets=config["datasets"]
    tests=config["tests"]
    columns=config["columns"]
    output_results=[]

    # Optional family allow-list. Absent/empty => use all families.
    include_families=config.get("include_families")
    if isinstance(include_families,str):
        include_families=[f.strip() for f in include_families.split(",") if f.strip()]
    if include_families:
        include_families=set(include_families)

    tests_to_run=[]

    for _,ds_key in enumerate(datasets):
        dataset=datasets[ds_key]
        if include_families and dataset.get("family") not in include_families:
            continue
        if split is not None:
            if "split" in dataset:
                if dataset["split"]!=split:
                    continue
        for test_key in tests:
            result=None
            for r in cached_results:
                if r["params"]["test_key"]==test_key and r["params"]["ds_key"]==ds_key:
                    if "regenerate" in datasets[ds_key] and datasets[ds_key]["regenerate"]==True:
                        r["params"]["need_regenerate"]=True
                        continue
                    if "regenerate" in tests[test_key] and tests[test_key]["regenerate"]==True:
                        r["params"]["need_regenerate"]=True
                        continue
                    result=r
            if result is None:

                test=tests[test_key]
                params={}
                for p in test:
                    params[p]=test[p]
                if not "max_duration" in params:
                    params["max_duration"]=1000
                # copy some parameters from top level to each test config
                params_to_copy=["eval_rate_divisor", "eval_min_framerate", "min_person_height",
                                "classes", "classes_for_det_map", "fitness_weights"]
                for p in params_to_copy:
                    if p in config:
                        params[p]=config[p]
                # Dataset extras (multi_class_and_hints.md §5): every
                # per-dataset key beyond the harness-reserved set rides
                # into params → import_create's param_dict → the tracker
                # config yaml verbatim. This is how `stream_hint: bodycam`
                # reaches track_stream_create (the C side resolves the
                # (hint:x) variant axis from it).
                reserved={"path","split","group","family","regenerate","no_cache"}
                for k,v in dataset.items():
                    if k not in reserved:
                        params[k]=v
                params["ds_path"]=dataset["path"]
                params["display"]=f"{len(tests_to_run):02d}: "+ds_key+"/"+test_key
                params["ds_key"]=ds_key
                params["test_key"]=test_key

                tests_to_run.append(params)
            else:
                output_results.append(result)


    cached_results_new=[r for r in cached_results if "need_regenerate" not in r["params"]]
    logging.info(f"cached results {len(cached_results)}; deleting {len(cached_results)-len(cached_results_new)} need to run {len(tests_to_run)} tests")
    cached_results=cached_results_new

    on_result_context={"cached_results": cached_results,
                       "config":config,
                       "resultfile": resultfile}

    results = stuff.mp_workqueue_run(tests_to_run,
                                     track_test_work_fn,
                                     num_workers=num_workers,
                                     desc=desc,
                                     result_callback_context=on_result_context,
                                     result_callback=on_result_callback)

    for entry in results:
        output_results.append(entry)

    for o in output_results:
        if "time" in o:
            o["result"]["time"]=(datetime.datetime.now()-o["time"]).total_seconds()
        if "group" in config["datasets"][o["params"]["ds_key"]]:
            o["group"]=config["datasets"][o["params"]["ds_key"]]["group"]

    results2=display_results(config, output_results, columns, config["sort_key"])
    elapsed=time.time()-start_time
    _write_eval_summary_json(config, output_results, results2, elapsed)
    print(f"All done: Evaluated {len(tests_to_run)} tests in {stuff.timestr(elapsed)}")
    return results2


def _summary_metric_keys():
    """Float keys exposed in the per-test summary JSON sidecar. Per-class
    namespaced variants (multi_class_and_hints.md §2) ride along for every
    extra class, plus the combined objective."""
    base = [
        "fitness", "mota", "idf1", "fp_tracks",
        "fp_tracks_honest_v2", "fp_h2_nm", "fp_h2_inrun",
        "fp_per_frame", "fn_per_obj", "switch_per_obj", "frag_per_obj",
        "motp", "duration",
        "num_frames", "num_objects", "num_false_positives",
        "num_misses", "num_switches",
    ]
    out = list(base)
    for cls in ("vehicle", "animal"):
        out += [k + "_" + cls for k in base]
    out.append("fitness_multi")
    return out


def _result_subset(result, keys):
    out = {}
    for k in keys:
        if k in result:
            v = result[k]
            if isinstance(v, (np.floating, np.integer)):
                v = v.item()
            out[k] = v
    return out


def _write_eval_summary_json(config, output_results, rollups, elapsed):
    """Sidecar JSON next to the text results report. No-op without
    `results_location` in the config.

    Structure:
        {
          "elapsed_seconds": float,
          "num_clips": int,
          "tests": {
            "<test_key>": {
              "overall":   {fitness, mota, fp_tracks, ...},  # __ovr<group> if 1 group, else _arithmean
              "groups":    {"<group>": {...}, ...},          # one per __ovr<group>; "<group>_mean" for __mean(<group>)
              "arithmean": {...},                             # _arithmean rollup
              "clips":     {"<ds_key>": {...}, ...}           # raw per-clip metrics
            }
          }
        }
    """
    if "results_location" not in config:
        return
    location = config["results_location"]
    stuff.makedir(location)
    keys = _summary_metric_keys()

    tests_by_key = {}
    for entry in rollups:
        test_key = entry["params"]["test_key"]
        ds_key = entry["params"]["ds_key"]
        bucket = tests_by_key.setdefault(test_key, {
            "overall": None, "groups": {}, "arithmean": None, "clips": {},
        })
        metrics = _result_subset(entry["result"], keys)
        if ds_key.startswith("__ovr"):
            group = ds_key[len("__ovr"):]
            bucket["groups"][group] = metrics
        elif ds_key.startswith("__mean("):
            group = ds_key[len("__mean("):-1]
            bucket["groups"].setdefault(group, {})
            bucket["groups"][group + "_mean"] = metrics
        elif ds_key == "_arithmean":
            bucket["arithmean"] = metrics
        elif ds_key == "_groupmean":
            bucket["groupmean"] = metrics

    for entry in output_results:
        test_key = entry["params"]["test_key"]
        ds_key = entry["params"]["ds_key"]
        bucket = tests_by_key.setdefault(test_key, {
            "overall": None, "groups": {}, "arithmean": None, "clips": {},
        })
        bucket["clips"][ds_key] = _result_subset(entry["result"], keys)

    # "overall" is the single-group __ovr rollup when there is exactly one
    # group, else the arithmetic mean across all clips.
    for bucket in tests_by_key.values():
        groups = bucket["groups"]
        non_mean_groups = [k for k in groups if not k.endswith("_mean")]
        if len(non_mean_groups) == 1:
            bucket["overall"] = groups[non_mean_groups[0]]
        else:
            bucket["overall"] = bucket["arithmean"]

    summary = {
        "elapsed_seconds": elapsed,
        "num_clips": len({e["params"]["ds_key"] for e in output_results}),
        "tests": tests_by_key,
    }

    cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    out_path = os.path.join(location, f"results-{cur_time}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    try:
        _write_eval_summary_html(
            os.path.join(location, f"results-{cur_time}.html"), summary)
    except Exception:
        import traceback
        sys.stderr.write("eval html report failed: "
                         + traceback.format_exc() + "\n")


def _write_eval_summary_html(path, summary):
    """Self-contained sortable eval report (search_review.md §4.3):
    per-test rollup cards + a click-to-sort per-clip table, worst clips
    surfaced. Inline data, vanilla JS, no server, negatives visible."""
    payload = json.dumps(summary)
    html = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>eval results</title><style>
body{font-family:system-ui,sans-serif;margin:20px;background:#111;color:#ddd}
h1{font-size:18px} h2{font-size:14px;color:#aaa;margin-top:22px}
table{border-collapse:collapse;font-size:12px;margin-top:6px}
td,th{padding:2px 9px;border-bottom:1px solid #2a2a2a;text-align:right;white-space:nowrap}
th{color:#8ab;cursor:pointer;user-select:none} td:first-child,th:first-child{text-align:left}
.neg{color:#ff6b6b} .rollup td{color:#ffd43b}
</style></head><body><h1>eval results</h1><div id="root"></div>
<script>
const D = __PAYLOAD__;
const COLS = ["fitness","fitness_multi","mota","idf1","mota_vehicle","idf1_vehicle",
              "fp_per_frame","fp_tracks_honest_v2","num_objects","duration"];
function fmt(v){ if(v===undefined||v===null) return "";
  if(typeof v!=="number") return String(v);
  const s = Math.abs(v)>=1000 ? v.toFixed(0) : v.toFixed(3);
  return v<0 ? '<span class="neg">'+s+"</span>" : s; }
function table(rows, id){
  let sortk=COLS[0], asc=false;
  const el=document.createElement("table"); el.id=id;
  function render(){
    const sorted=[...rows].sort((a,b)=>{
      const x=a[1][sortk], y=b[1][sortk];
      return ((x===undefined)-(y===undefined)) || (asc?x-y:y-x) || 0; });
    el.innerHTML="<tr><th>clip</th>"+COLS.map(c=>
      '<th data-k="'+c+'">'+c+(c===sortk?(asc?" \u25b2":" \u25bc"):"")+"</th>").join("")+"</tr>"+
      sorted.map(([k,r])=>"<tr"+(k.startsWith("_")?' class="rollup"':"")+"><td>"+k+"</td>"+
        COLS.map(c=>"<td>"+fmt(r[c])+"</td>").join("")+"</tr>").join("");
    el.querySelectorAll("th[data-k]").forEach(th=>th.onclick=()=>{
      if(sortk===th.dataset.k) asc=!asc; else {sortk=th.dataset.k; asc=false;}
      render(); });
  }
  render(); return el;
}
const root=document.getElementById("root");
for(const [tk, t] of Object.entries(D.tests||{})){
  const h=document.createElement("h2"); h.textContent="test: "+tk; root.appendChild(h);
  const rows=[];
  if(t.overall) rows.push(["_overall", t.overall]);
  if(t.groupmean) rows.push(["_groupmean", t.groupmean]);
  for(const [g,m] of Object.entries(t.groups||{})) rows.push(["_"+g, m]);
  for(const [c,m] of Object.entries(t.clips||{})) rows.push([c, m]);
  root.appendChild(table(rows, "t_"+tk));
}
</script></body></html>"""
    with open(path, "w") as f:
        f.write(html.replace("__PAYLOAD__", payload))
