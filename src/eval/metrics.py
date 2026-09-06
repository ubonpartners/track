"""Per-clip scoring: MOT metrics via motmetrics, the honest-FP ruler,
detection metrics, fitness / fitness_multi, score_tracksets.

Moved verbatim from src/track_test.py (repo_cleanup.md stage 4a).
"""
import logging
import sys
import motmetrics as mm
import numpy as np
from tqdm.auto import tqdm
import stuff
from stuff import coord
import src.track_util as tu

from src.eval.matching import _box_in_ignore, mot_obj, permissive_iou, permissive_iou_matrix



# fitness penalises the honest FP-track count (de-gamed: IoU==0 run-count,
# GT-grounded, parameter-free), length-normalised by VIDEO DURATION so
# the penalty is sequence-length AND frame-rate invariant — honest_v2 is
# an EPISODE count that scales with duration (a 2s phantom is 1 run at
# 10 or 30 fps), not with sampling density. Aggregate rows sum
# Σhonest/Σduration. K calibrated so the aggregate honest penalty
# matches the old 5e-4*fp_tracks penalty on the shipped networks. Old
# fp_tracks is still reported alongside for comparison.
_FP_TRACK_COEF = 0.35   # per honest-FP episode per second of video


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
                    convention_permissive=None,
                    show_pbar=False):
    assert match_iou<0.9 and match_iou>0.1, f"stupid match_iou {match_iou}"
    # Convention-permissive box matching: None means AUTO — enabled iff
    # the GT annotation declares its box convention (tier-2 derived
    # annotations carry 'box_convention' from the corpus registry; legacy
    # ones don't). Explicit True/False overrides.
    if convention_permissive is None:
        convention_permissive = bool(gt.metadata.get("box_convention"))
    # directionality: forgive only the mismatch the other legitimate
    # convention produces (see permissive_iou_matrix)
    gt_convention = gt.metadata.get("box_convention")
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

    # Pair scorer for the "does this det genuinely match a GT" rescue
    # checks below — permissive when enabled, plain IoU otherwise, so the
    # ignore-region / annotation-floor filters agree with the matcher and
    # never drop a convention-mismatched but correct detection.
    _fwh = (gt.metadata.get("width"), gt.metadata.get("height"))
    _fwh = _fwh if all(_fwh) else None
    if convention_permissive:
        def pair_iou(a, b):
            return permissive_iou(a, b, gt_convention=gt_convention,
                                  frame_wh=_fwh)
    else:
        pair_iou = coord.box_iou

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
                        and not any(pair_iou(det.box, gb) >= match_iou
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
                        if not any(pair_iou(det.box, gb) >= match_iou
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
            if convention_permissive:
                # Same semantics as mm.distances.iou_matrix (dist=1-iou,
                # NaN above the max distance) but with the convention-
                # permissive score, computed on the normalized boxes
                # (IoU is invariant to per-axis scaling so this matches
                # the pixel-space plain-IoU values up to mot_obj's int
                # truncation). One vectorized (n_gt, n_test) pass.
                pdist = 1.0 - permissive_iou_matrix(
                    [g.box for g in gt_obj], [d.box for d in test_obj],
                    gt_convention=gt_convention, frame_wh=_fwh)
                C = np.where(pdist > match_iou, np.nan, pdist)
            else:
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



def score_tracksets(trackset_gt, trackset, params):
    """The per-clip scoring block (person + per-class passes + fitness_multi),
    shared by the mp work-fn and the single-shared-state path."""
    match_iou=0.45
    if "match_iou" in params:
        match_iou=params["match_iou"]
    eval_rate_divisor=params.get("eval_rate_divisor", 1)
    eval_min_framerate=params.get("eval_min_framerate", 30.0)
    # None = AUTO (enabled iff the GT declares 'box_convention');
    # a search yaml can force it with an explicit true/false.
    convention_permissive=params.get("convention_permissive")

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
                           eval_min_framerate=eval_min_framerate,
                           convention_permissive=convention_permissive)

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
                              eval_min_framerate=eval_min_framerate,
                              convention_permissive=convention_permissive)
        for k, v in r_cls.items():
            result[f"{k}_{cls}"]=v

    fm=fitness_multi_score(result, params.get("fitness_weights"))
    if fm is not None:
        result["fitness_multi"]=fm
    return result
