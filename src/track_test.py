
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
import ubon_pycstuff.ubon_pycstuff as upyc

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

# fitness penalises the HONEST FP-track count (exp 20260515-honest-fp-iou0:
# IoU==0 run-count, GT-grounded, parameter-free — the de-gamed replacement
# for the gameable `fp_tracks`), LENGTH-NORMALISED: honest_v2 is a raw
# count so it grows with sequence length, unlike mota & fp_per_frame which
# are rates. Penalise the honest-FP-episode RATE per second of VIDEO
# (honest_v2 / duration), NOT per num_frames: num_frames = duration *
# frame_rate / eval_rate_divisor conflates duration with sampling fps,
# and the honest count is an EPISODE count that scales with duration
# (a 2s phantom is 1 run at 10 or 30 fps), not with sampling density.
# (fp_per_frame legitimately uses num_frames — its numerator IS a
# per-frame event count.) duration-based ⇒ all fitness terms are
# sequence-length AND frame-rate invariant; works per-clip and on summed
# aggregate rows (Σhonest/Σduration). K calibrated so the aggregate
# honest penalty ≈ the OLD 5e-4*fp_tracks penalty on the current
# networks — see RESEARCH_OUT/.../honest-fp-iou0/scale_rate*.log.
# Reported fp_tracks_honest_v2 stays a raw count; mota, idf1, old
# fp_tracks all still reported.
_FP_TRACK_COEF = 0.35   # per honest-FP episode per second of video;
# calibrated: ship full176 honest 795 / 6798s = 0.117 ep/s; 0.35*0.117
# = 0.0409 ≈ old 5e-4*81 = 0.0405 (ratio 1.01). calib_dur.log.
def fitness_score(r):
    h = r.get("fp_tracks_honest_v2")
    if h is None:                       # legacy rows w/o the honest field
        h = r["fp_tracks"] * 10.0       # ~scale-match so fitness stays sane
    dur = r.get("duration", 0) or 0
    h_rate = h / dur if dur > 0 else 0.0
    return (r["mota"] - _FP_TRACK_COEF * h_rate
            - 0.0 * r["fp_tracks_frac"] - 0.002 * r["fp_per_frame"])

def summary_string(r):
    s=f" MOTA:{r['mota']:6.5f}"
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
        s+=f" FPh2:{r['fp_tracks_honest_v2']}"
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
    return s

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

def _honest_fp_tracks(df, hyp_cd_frames=None,
                       l_lead=5, l_lag=5, g_max=10, theta=2.0):
    """Segment-based unique-FP count over a motmetrics events frame, with
    a spatial-excursion gate.

    Robust-unique-FP definition (experiments
    20260515-honest-fp-track-metric-definition / -spatial-gate). Per
    hypothesis track HId, order its events by frame and split into
    matched vs FP frames (Type=='FP' = unmatched; MATCH/SWITCH/transfer
    family = associated-to-a-GT). Charge a unique FP for each unmatched
    segment that is BOTH temporally material AND spatially excursive:
      - lead-in : FP run BEFORE first match, len >= l_lead
      - lag-out : FP run AFTER  last  match, len >= l_lag
      - bridge  : FP run BETWEEN two matches, len >= g_max
      - never-matched track: always 1 FP (== the gamed metric).

    Spatial gate (crit-3 showed temporal-only false-alarms 31x on a
    clean tracker because gaming and *legitimate occlusion* share the
    same temporal signature — only spatial info separates them):
    a segment counts only if the hyp box during the run deviates from
    the expected continuation by > theta box-diagonals —
      - bridge : expected = linear interp between the bounding matched
                 frames' hyp centroids (a legit occlusion bridge coasts
                 along that line; a stitched/teleport FP excursions off);
      - lead/lag: expected = the adjacent matched frame's hyp centroid
                 (a benign short coast stays near it; latching a
                 different FP object moves far).
    `hyp_cd_frames[frameid][hid] = (cx, cy, diag)` (pixels), frameid ==
    motmetrics auto-id == that list's index. If centroids are missing
    the segment falls back to temporal-only (conservative: charged if
    temporally material), which also gives the old behaviour when
    hyp_cd_frames is None.

    Invariant: honest_fp_tracks >= gamed fp_tracks.
    """
    ev = _events_by_hid_from_df(df)
    return _honest_fp_core(ev, hyp_cd_frames, l_lead, l_lag, g_max, theta)


# tcode: 0 = FP (unmatched), 1 = matched-to-a-GT (MATCH/SWITCH/transfer
# family). Rows that are neither are not emitted for an HId.
_MATCHED_TYPES = {"MATCH", "SWITCH", "TRANSFER", "MIGRATE", "ASCEND"}


def _events_by_hid_from_df(df):
    """Compact, picklable per-HId event sequence from a motmetrics events
    frame: {hid:int -> [(frameid:int, tcode:int), ...] sorted by frameid}.
    The single source of truth consumed by both the live metric and the
    offline threshold sweep (identical logic, no reimplementation)."""
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


def _honest_fp_core(events_by_hid, cd_frames,
                    l_lead=5, l_lag=5, g_max=10, theta=2.0):
    """Pure honest-FP counter over the compact event structure. Used
    identically by `_honest_fp_tracks` (live) and the offline
    threshold-sweep, so a sweep result is exactly what the live metric
    would have produced. `cd_frames[frameid][hid]=(cx,cy,diag)` or None."""

    def cd(frameid, hid):
        if cd_frames is None or frameid >= len(cd_frames):
            return None
        return cd_frames[frameid].get(hid)

    def excursive(run_fids, hid, ref_a, ref_b):
        if ref_a is None or ref_a[1] is None:
            return True                       # no geometry -> conservative
        fa, ca = ref_a
        cb = ref_b[1] if ref_b else None
        fb = ref_b[0] if ref_b else None
        worst = 0.0
        for fid in run_fids:
            c = cd(fid, hid)
            if c is None:
                return True
            if cb is not None and fb != fa:
                w = (fid - fa) / (fb - fa)
                ex, ey = ca[0] + w * (cb[0] - ca[0]), ca[1] + w * (cb[1] - ca[1])
                diag = 0.5 * (ca[2] + cb[2]) or 1.0
            else:
                ex, ey, diag = ca[0], ca[1], (ca[2] or 1.0)
            d = ((c[0] - ex) ** 2 + (c[1] - ey) ** 2) ** 0.5 / diag
            worst = max(worst, d)
        return worst > theta

    honest = fully = lead = lag = bridge = 0
    for hid, seq in events_by_hid.items():
        fids = [f for f, _ in seq]
        is_fp = [tc == 0 for _, tc in seq]
        matched_pos = [i for i, (_, tc) in enumerate(seq) if tc == 1]
        if not matched_pos:
            honest += 1; fully += 1
            continue
        first_m, last_m = matched_pos[0], matched_pos[-1]

        lead_fids = [fids[i] for i in range(0, first_m) if is_fp[i]]
        if len(lead_fids) >= l_lead and excursive(
                lead_fids, hid, (fids[first_m], cd(fids[first_m], hid)), None):
            honest += 1; lead += 1

        lag_fids = [fids[i] for i in range(last_m + 1, len(seq)) if is_fp[i]]
        if len(lag_fids) >= l_lag and excursive(
                lag_fids, hid, (fids[last_m], cd(fids[last_m], hid)), None):
            honest += 1; lag += 1

        run_fids = []
        anchor_a = (fids[first_m], cd(fids[first_m], hid))
        for i in range(first_m + 1, last_m + 1):
            if is_fp[i]:
                run_fids.append(fids[i])
            else:                              # matched anchor
                if len(run_fids) >= g_max:
                    anchor_b = (fids[i], cd(fids[i], hid))
                    if excursive(run_fids, hid, anchor_a, anchor_b):
                        honest += 1; bridge += 1
                anchor_a = (fids[i], cd(fids[i], hid))
                run_fids = []
    return {
        "honest_fp_tracks": int(honest),
        "fully_unmatched": int(fully),
        "leadin": int(lead), "lagout": int(lag), "bridge": int(bridge),
        "thresholds": {"l_lead": l_lead, "l_lag": l_lag,
                        "g_max": g_max, "theta": theta},
    }


def _honest_fp_frames_core(events_by_hid, cd_frames,
                           theta=2.0, nm_policy="track"):
    """Frame-resolution honest-FP measure (experiment
    20260515-honest-fp-frame-metric). The fp_tracks gaming exploit
    *merges* an unrelated FP run onto a matched track: the unique-track
    count drops but the wrong (FP) *frames* stay. So count surviving FP
    frames, gated PER FRAME by spatial excursion, in exactly the
    lead/lag/bridge regions of a matched track where merging hides them.

    The segment-COUNT formulation was FALSIFIED (exp#5): a length
    threshold L fights theta — as theta rises clean drops but crit-1
    decoupling drops with it, monotone-opposite, no joint band. The
    frame metric removes the length threshold entirely: a legit
    occlusion coast stays near the local anchor model every frame
    (non-excursive, length irrelevant) so it is never charged however
    long; a gamed/teleport merge sits far from the model so its frames
    are charged. The theta-response is therefore hypothesised to be
    qualitatively different (a clean/decoupling crossover can exist).

    Per hyp track HId (events sorted by frame):
      - never-matched track: NOT the gaming vector (the gamed metric
        already counts it 1, honestly); nm_policy 'track' charges 1
        (mirror gamed -> keeps crit-3 on a track-count scale),
        'frames' charges its FP-frame count (sensitivity check).
      - else: FP frames before first match = lead (model = first
        matched centroid, held); after last match = lag (model = last
        matched centroid, held); between two matched anchors = bridge
        (model = linear interp of the two bounding matched centroids).
      - charge each FP frame whose hyp centroid deviates from its
        local model by > theta box-diagonals. Missing geometry ->
        conservative (charge the frame; theta-only when cd_frames None).

    Pure: identical inputs/logic for the live path and the offline
    sweep (no reimplementation), same dump contract as
    `_honest_fp_core`. Side-channel only; does not touch `fitness`.
    """

    def cd(frameid, hid):
        if cd_frames is None or frameid >= len(cd_frames):
            return None
        return cd_frames[frameid].get(hid)

    def exc(c, ex, ey, diag):
        if c is None:
            return True                       # no geometry -> conservative
        diag = diag or 1.0
        d = ((c[0] - ex) ** 2 + (c[1] - ey) ** 2) ** 0.5 / diag
        return d > theta

    total = nm = lead = lag = bridge = 0
    for hid, seq in events_by_hid.items():
        fids = [f for f, _ in seq]
        is_fp = [tc == 0 for _, tc in seq]
        matched_pos = [i for i, (_, tc) in enumerate(seq) if tc == 1]
        if not matched_pos:
            c = (len([1 for f in is_fp if f]) if nm_policy == "frames"
                 else 1)
            total += c; nm += c
            continue
        first_m, last_m = matched_pos[0], matched_pos[-1]

        ca = cd(fids[first_m], hid)
        for i in range(0, first_m):
            if is_fp[i] and (ca is None or
                             exc(cd(fids[i], hid), ca[0], ca[1], ca[2])):
                total += 1; lead += 1

        cz = cd(fids[last_m], hid)
        for i in range(last_m + 1, len(seq)):
            if is_fp[i] and (cz is None or
                             exc(cd(fids[i], hid), cz[0], cz[1], cz[2])):
                total += 1; lag += 1

        fa, anch_a = fids[first_m], cd(fids[first_m], hid)
        run = []
        for i in range(first_m + 1, last_m + 1):
            if is_fp[i]:
                run.append(i)
            else:
                fb, anch_b = fids[i], cd(fids[i], hid)
                for j in run:
                    if anch_a is None or anch_b is None or fb == fa:
                        total += 1; bridge += 1
                        continue
                    w = (fids[j] - fa) / (fb - fa)
                    ex = anch_a[0] + w * (anch_b[0] - anch_a[0])
                    ey = anch_a[1] + w * (anch_b[1] - anch_a[1])
                    dg = 0.5 * (anch_a[2] + anch_b[2])
                    if exc(cd(fids[j], hid), ex, ey, dg):
                        total += 1; bridge += 1
                fa, anch_a = fb, anch_b
                run = []
    return {
        "honest_fp_frames": int(total),
        "nm": int(nm), "leadin": int(lead),
        "lagout": int(lag), "bridge": int(bridge),
        "thresholds": {"theta": theta, "nm_policy": nm_policy},
    }


def _honest_fp_episodes_core(events_by_hid, cd_frames,
                             theta=0.5, g_gap=0):
    """Episode-COUNT honest-FP (experiment 20260515-honest-fp-episode-
    count). Restores the user-facing FRAGMENTATION semantics `fp_tracks`
    exists for — tracks are what a user is presented with, so the same
    FP volume shown as 2 distinct wrong stretches is worse than 1 — while
    inheriting the exp#6-validated PER-FRAME spatial gate.

        honest = (# standalone phantom tracks, never matched)
               + (every connected run of spatially-excursive FP frames
                  hidden inside a matched track's lead/lag/bridge,
                  each counted ONCE as a track-equivalent)

    i.e. "a contaminated stretch of an otherwise-real track is, for
    scoring, its own FP track." This is a FIX of fp_tracks (count units,
    fragmentation-sensitive), not the frame-volume surrogate
    `_honest_fp_frames_core`:
      - gaming-resistant: merging an FP run onto a matched track no
        longer hides it — it still scores 1 episode, exactly as the
        standalone FP track would have;
      - no false alarm: a benign occlusion coast is non-excursive every
        frame (exp#6 property) -> 0 episodes regardless of LENGTH, so the
        exp#5 length-threshold pathology cannot recur (no length knob).

    `g_gap` = max consecutive non-excursive FP frames tolerated inside
    one episode (0 = strict; a MATCHED frame always ends an episode = the
    real object was re-acquired, so later excursive FP is a NEW distinct
    wrong stretch the user sees). Same pure inputs/logic for the live
    path and the offline sweep (no reimplementation); same dump contract
    as `_honest_fp_core`. Side-channel only; does not touch `fitness`.
    """

    def cd(frameid, hid):
        if cd_frames is None or frameid >= len(cd_frames):
            return None
        return cd_frames[frameid].get(hid)

    def exc(c, ex, ey, diag):
        if c is None:
            return True                       # no geometry -> conservative
        diag = diag or 1.0
        d = ((c[0] - ex) ** 2 + (c[1] - ey) ** 2) ** 0.5 / diag
        return d > theta

    def episodes(flags):
        """# maximal True-runs in `flags`, bridging <= g_gap Falses."""
        n = 0
        in_ep = False
        gap = 0
        for f in flags:
            if f:
                if not in_ep:
                    n += 1
                    in_ep = True
                gap = 0
            elif in_ep:
                gap += 1
                if gap > g_gap:
                    in_ep = False
        return n

    total = nm = lead = lag = bridge = 0
    for hid, seq in events_by_hid.items():
        fids = [f for f, _ in seq]
        is_fp = [tc == 0 for _, tc in seq]
        matched_pos = [i for i, (_, tc) in enumerate(seq) if tc == 1]
        if not matched_pos:
            total += 1; nm += 1            # standalone phantom = 1 result
            continue
        first_m, last_m = matched_pos[0], matched_pos[-1]

        ca = cd(fids[first_m], hid)
        lf = [(ca is None or exc(cd(fids[i], hid), ca[0], ca[1], ca[2]))
              for i in range(0, first_m) if is_fp[i]]
        e = episodes(lf); total += e; lead += e

        cz = cd(fids[last_m], hid)
        gf = [(cz is None or exc(cd(fids[i], hid), cz[0], cz[1], cz[2]))
              for i in range(last_m + 1, len(seq)) if is_fp[i]]
        e = episodes(gf); total += e; lag += e

        fa, anch_a = fids[first_m], cd(fids[first_m], hid)
        run = []
        for i in range(first_m + 1, last_m + 1):
            if is_fp[i]:
                run.append(i)
            else:
                fb, anch_b = fids[i], cd(fids[i], hid)
                bf = []
                for j in run:
                    if anch_a is None or anch_b is None or fb == fa:
                        bf.append(True)
                        continue
                    w = (fids[j] - fa) / (fb - fa)
                    ex = anch_a[0] + w * (anch_b[0] - anch_a[0])
                    ey = anch_a[1] + w * (anch_b[1] - anch_a[1])
                    dg = 0.5 * (anch_a[2] + anch_b[2])
                    bf.append(exc(cd(fids[j], hid), ex, ey, dg))
                e = episodes(bf); total += e; bridge += e
                fa, anch_a = fb, anch_b
                run = []
    return {
        "honest_fp_episodes": int(total),
        "nm": int(nm), "leadin": int(lead),
        "lagout": int(lag), "bridge": int(bridge),
        "thresholds": {"theta": theta, "g_gap": g_gap},
    }


def _honest_fp_iou_gt_core(events_by_hid, cd_frames, gt_cd_frames,
                           theta_gt=1.0, g_gap=0):
    """GT-GROUNDED honest-FP (experiment 20260515-honest-fp-iou-gt).

    exp#5/#7 trilemma: from self-consistency geometry (hyp vs its own
    linear-interp motion) you get <=2 of {count semantics, gaming-
    resistance, clean-robustness}. The blind spot is the REFERENCE: a
    crude self-interp mislabels legit non-linear/occluded motion as
    excursive, forcing a persistence threshold that kills counts.

    Fix: ground the gate in GROUND TRUTH. An FP hyp frame is
    *contaminating* iff its box is far from EVERY real GT object that
    frame (min normalised centroid distance to any GT > theta_gt), OR
    there is no GT object at all that frame (a detection with zero real
    objects nearby is a pure phantom), OR hyp geometry is missing
    (conservative). GT knows where the real object is even through a
    detector-missed occlusion, so a benign coast sits near a GT (not
    charged) and a phantom/teleport is far from all GT (charged) —
    SHORT OR LONG, no persistence threshold needed.

    Per the user's harm ruling (a contaminated stretch of an otherwise-
    real track is a DISTINCT defect, as bad as a separate phantom), the
    primary output is a COUNT: standalone phantom tracks (never matched
    = 1 each, == gamed) + every connected run of contaminating FP frames
    inside a matched track = 1 track-equivalent defect. `g_gap` bridges
    <= that many non-contaminating FP frames within one run; a MATCHED
    frame always ends a run (real object re-acquired -> a later
    contaminating run is a NEW distinct defect the user sees). Also
    reports the contaminating-frame SUM (volume view).

    Pure: identical inputs/logic for the live path and the offline
    sweep. `gt_cd_frames[frameid][gid]=(cx,cy,diag,l,t,w,h)`. Side-
    channel only; does not touch `fitness`.
    """

    def hd(frameid, hid):
        if cd_frames is None or frameid >= len(cd_frames):
            return None
        return cd_frames[frameid].get(hid)

    def contaminating(frameid, hid):
        c = hd(frameid, hid)
        if c is None:
            return True                       # no geometry -> conservative
        if gt_cd_frames is None or frameid >= len(gt_cd_frames):
            return True
        gts = gt_cd_frames[frameid]
        if not gts:
            return True                       # detection, zero real objects
        best = min(
            (((c[0] - g[0]) ** 2 + (c[1] - g[1]) ** 2) ** 0.5
             / (0.5 * (c[2] + g[2]) or 1.0))
            for g in gts.values())
        return best > theta_gt                # far from EVERY GT object

    def episodes(flags):
        n = 0
        in_ep = False
        gap = 0
        for f in flags:
            if f:
                if not in_ep:
                    n += 1
                    in_ep = True
                gap = 0
            elif in_ep:
                gap += 1
                if gap > g_gap:
                    in_ep = False
        return n

    total = nm = inrun = fsum = 0
    for hid, seq in events_by_hid.items():
        is_fp = [tc == 0 for _, tc in seq]
        matched_pos = [i for i, (_, tc) in enumerate(seq) if tc == 1]
        if not matched_pos:
            total += 1; nm += 1
            continue
        flags = []
        for i, (fid, _tc) in enumerate(seq):
            if is_fp[i]:
                con = contaminating(fid, hid)
                flags.append(con)
                if con:
                    fsum += 1
            else:
                flags.append(False)           # matched frame ends a run
        e = episodes(flags)
        total += e; inrun += e
    return {
        "honest_fp_iou_gt": int(total),
        "nm": int(nm), "inrun_episodes": int(inrun),
        "contam_frames": int(fsum),
        "thresholds": {"theta_gt": theta_gt, "g_gap": g_gap},
    }


def _honest_fp_runs_core(events_by_hid, cd_frames, gt_cd_frames):
    """RESOLVED honest FP-track ruler (experiment 20260515-honest-fp-iou0;
    converged with the user through exp#5-#9 + the criterion correction).

    honest_fp_tracks = number of contiguous runs of FP frames whose box
    OVERLAPS NO REAL GT OBJECT (IoU == 0 with every GT box that frame, or
    no GT that frame), where a MATCHED frame ends a run (the real object
    was (re)acquired ⇒ a later spurious run is a NEW distinct defect).

    PARAMETER-FREE. Rationale (settled with the user):
      - "no overlap with any real object" is the simplest, least-arguable
        definition of a false detection; replaces the tunable θ_gt proxy.
      - run-COUNT (not frame-SUM): a brief 2-3 frame spurious blip is ONE
        unique FP; merging FP+FP into one run stays 1 (never penalised);
        an FP run welded onto a GT-matched track is still counted (the
        gaming the old fp_tracks hid). Fragmentation-aware (the property
        fp_tracks exists for: tracks are what users see).
      - GT-grounded ⇒ box-jitter / occlusion-coast on a real track stays
        on/over its real object (IoU>0) ⇒ NOT counted (no false alarm);
        GT interpolation is valid corpus-wide (verified: 'no GT in frame'
        = 0.5%), so no Lmin / persistence threshold is needed or used.
      - NO crit-3 'clean≈gamed' constraint: gamed is the broken baseline
        that misses in-track FP runs; honest is EXPECTED to exceed it
        (clean ship ≈9.6× ≈797 vs 83) — that gap IS the de-gamed signal.

    Invariant (verified 615/615 clips): honest ≤ #FP frames ==
    motmetrics num_false_positives. Gaming-resistant: under the iter1→
    iter2 exploit gamed −65% but honest −28% ≈ real FP-volume −18%
    (signature 0.27, honest 0.63 — decisively NOT the illusion).

    Pure; same dump contract; needs the full hyp box (cd tuple
    (cx,cy,diag,l,t,w,h)). Side-channel only; does not touch `fitness`.
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
    return {
        "honest_fp_tracks_v2": int(total),
        "nm": int(nm), "inrun": int(inrun),
        "thresholds": {"gate": "iou0", "param_free": True},
    }


def _honest_fp_gt_runlen_core(events_by_hid, cd_frames, gt_cd_frames,
                              theta_gt=1.0, l_min=8, g_gap=2):
    """GT-grounded honest-FP COUNT with a minimum contiguous-run length
    (experiment 20260515-honest-fp-gt-runlen). The LAST post-hoc-ruler
    attempt (pre-committed).

    exp#5/#7/#8 falsified the COUNT family. Sharpened diagnosis: the
    blocker is COUNT DISCRETISATION over a noisy per-frame signal — a
    clean tracker emits MANY SHORT far-from-everything blips, each
    counted as 1. exp#5 tried a min-run-length but on SELF-INTERP
    *unmatched* runs, where legit occlusions also look like long runs,
    so L fought theta. exp#8 used the correct GT-grounded "contaminating"
    definition but NO min length, so blips exploded clean. This is the
    untested combination: GT-grounded contaminating runs **with** L_min.

    Mechanism: under GT grounding a legit occlusion coast is near the
    (GT-known) object => NOT contaminating => never a long contaminating
    run. Clean detector noise = SHORT contaminating runs. A gamed
    phantom-merge = a SUSTAINED contaminating run. So L_min separates
    clean(short) from gamed(long) and no longer fights theta_gt because
    the legit-occlusion confound that coupled them in exp#5 is removed by
    GT grounding.

        honest = # of contiguous contaminating-FP runs whose contaminating
                 length >= l_min (g_gap non-contaminating FP frames
                 bridged within a run; a MATCHED frame hard-ends a run =
                 real object re-acquired, a later run is a NEW defect).
        never-matched track: its FP run is scored by the SAME rule (a
        real phantom is far from GT for >= l_min frames => 1; a GT-missed
        real object stays near GT => 0, correctly not a phantom).

    Fragmentation-aware COUNT (satisfies the user's harm ruling: a
    contaminated stretch is a distinct defect). Pure; same dump contract.
    Side-channel only; does not touch `fitness`.
    """

    def hd(frameid, hid):
        if cd_frames is None or frameid >= len(cd_frames):
            return None
        return cd_frames[frameid].get(hid)

    def contaminating(frameid, hid):
        c = hd(frameid, hid)
        if c is None:
            return True
        if gt_cd_frames is None or frameid >= len(gt_cd_frames):
            return True
        gts = gt_cd_frames[frameid]
        if not gts:
            return True
        best = min(
            (((c[0] - g[0]) ** 2 + (c[1] - g[1]) ** 2) ** 0.5
             / (0.5 * (c[2] + g[2]) or 1.0))
            for g in gts.values())
        return best > theta_gt

    total = nm = inrun = 0
    for hid, seq in events_by_hid.items():
        matched = any(tc == 1 for _, tc in seq)
        clen = 0          # contaminating frames in the current run
        gap = 0           # consecutive non-contaminating FP frames
        active = False

        def close_run(n):
            return 1 if n >= l_min else 0

        runs = 0
        for i, (fid, tc) in enumerate(seq):
            if tc == 1:                       # matched -> hard end of run
                if active:
                    runs += close_run(clen)
                active = False; clen = 0; gap = 0
                continue
            # tc == 0 (FP frame)
            if contaminating(fid, hid):
                if not active:
                    active = True; clen = 0
                clen += 1; gap = 0
            elif active:
                gap += 1
                if gap > g_gap:
                    runs += close_run(clen)
                    active = False; clen = 0; gap = 0
        if active:
            runs += close_run(clen)

        total += runs
        if matched:
            inrun += runs
        else:
            nm += runs
    return {
        "honest_fp_gt_runlen": int(total),
        "nm": int(nm), "inrun_episodes": int(inrun),
        "thresholds": {"theta_gt": theta_gt, "l_min": l_min,
                        "g_gap": g_gap},
    }


# ===== FROZEN honest ruler (exp#6 20260515-honest-fp-frame-metric) =====
# The FRAME formulation is the first honest-FP measure to PASS the joint
# gate (segment-COUNT was FALSIFIED, exp#5). These two constants ARE the
# freeze: nm_policy='track' joint-OK at theta in [0.3,0.6] (clean_frac
# 0.16/0.06, crit-1 0.99/0.86); theta=0.5 is the centre with margin.
# Changing either value re-opens the ruler and MUST be a logged
# Experiment-Log event (RESEARCH_LOG.md), never a silent edit. fitness
# (fitness_score) still does NOT read this — side-channel until a clean
# full-pipeline re-pin promotes it (§3/§4.5).
HONEST_FP_FRAME_THETA = 0.5
HONEST_FP_FRAME_NM = "track"


def compute_metrics(gt, test,
                    max_duration=1000,
                    frame_metrics=False,
                    match_iou=0.45,
                    classes_to_test=["person"],
                    classes_for_det_map=["person","face"],
                    eval_rate_divisor=1,
                    eval_min_framerate=30.0,
                    show_pbar=False,
                    metrics="python",
                    honest_dump_tag=None):
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

    if metrics=="python":
        use_c_metrics=False
    elif metrics=="c":
        use_c_metrics=True
    else:
        assert False, f"Unknown metrics {metrics}"

    if use_c_metrics:
        c_mota=upyc.c_mota_metrics()
    else:
        acc = mm.MOTAccumulator(auto_id=True)

    frame_events=[]
    frame_index=0
    # Per-acc.update() hyp centroids for the honest-FP spatial gate
    # (exp 20260515-honest-fp-spatial-gate). One dict per accumulator
    # frame, keyed by hyp track_id == motmetrics HId (auto_id order ==
    # update-call order == this list's index). Cheap; only read post-loop.
    hyp_cd_frames=[]
    # Parallel per-acc.update() GT box geometry for the GT-grounded
    # honest-FP gate (exp 20260515-honest-fp-iou-gt). Same index == same
    # accumulator frame as hyp_cd_frames. {gt_id:(cx,cy,diag,l,t,w,h)}
    # — full box so the offline gate can use IoU or centroid distance.
    # Side-channel only; dumped, never read by fitness.
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

        if ignore_cl_idx is not None:
            ignore_boxes = [o.box for o in gt.objects_at_time(t) or []
                            if o.cl == ignore_cl_idx]
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

        if use_c_metrics:
            udets=[o.to_det() for o in test_obj]
            ugt=[o.to_det() for o in gt_obj]
            c_mota.add_frame(udets, ugt)
        else:
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
            # capture hyp centroids for this accumulator frame (mot_obj
            # rows are [track_id, l, t, w, h] in pixels)
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

    if use_c_metrics:
        metrics_dict=c_mota.get_results()
        metrics_dict["duration"]=duration
    else:
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
        # video DURATION in seconds (last_time - first t), NOT num_frames:
        # num_frames = duration * frame_rate / eval_rate_divisor conflates
        # duration with sampling fps. fitness length-normalises the honest
        # FP-track *episode* count by duration (frame-rate invariant).
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

        # --- honest (segment-based) unique-FP, side-channel only --------------
        # Experiment 20260515-honest-fp-track-metric-definition. Does NOT
        # alter fitness (fitness_score reads fp_tracks/fp_per_frame/mota) —
        # this is an extra diagnostic field so the frozen metric (§3) is
        # untouched. A partially-matched track currently scores 0 unique FP
        # however much FP it carries before/after/between its matched
        # section; that is the exploit. Here we additionally charge a unique
        # FP for each *material* unmatched segment.
        # Store only summable scalars — display_results sum()s every result
        # key across clips, so a nested dict here would crash aggregation.
        ev_by_hid = None
        try:
            ev_by_hid = _events_by_hid_from_df(df)
            hfp = _honest_fp_core(ev_by_hid, hyp_cd_frames)
            metrics_dict["fp_tracks_honest"]  = hfp["honest_fp_tracks"]
            metrics_dict["fp_honest_fully"]   = hfp["fully_unmatched"]
            metrics_dict["fp_honest_leadin"]  = hfp["leadin"]
            metrics_dict["fp_honest_lagout"]  = hfp["lagout"]
            metrics_dict["fp_honest_bridge"]  = hfp["bridge"]
            # FROZEN frame ruler (exp#6) — same in-memory ev/cd inputs the
            # offline sweep was correctness-gated on, so live==offline by
            # construction. Side-channel; fitness untouched.
            hff = _honest_fp_frames_core(ev_by_hid, hyp_cd_frames,
                                         theta=HONEST_FP_FRAME_THETA,
                                         nm_policy=HONEST_FP_FRAME_NM)
            metrics_dict["fp_frames_honest"]   = hff["honest_fp_frames"]
            metrics_dict["fp_fhonest_nm"]      = hff["nm"]
            metrics_dict["fp_fhonest_leadin"]  = hff["leadin"]
            metrics_dict["fp_fhonest_lagout"]  = hff["lagout"]
            metrics_dict["fp_fhonest_bridge"]  = hff["bridge"]
            # RESOLVED honest FP-track ruler (exp 20260515-honest-fp-iou0):
            # IoU==0 run-count, GT-grounded, parameter-free. THIS now
            # drives `fitness` (scaled coef, see fitness_score). Old
            # fp_tracks kept for reporting/comparison only.
            hrn = _honest_fp_runs_core(ev_by_hid, hyp_cd_frames,
                                       gt_cd_frames)
            metrics_dict["fp_tracks_honest_v2"] = hrn["honest_fp_tracks_v2"]
            metrics_dict["fp_h2_nm"]            = hrn["nm"]
            metrics_dict["fp_h2_inrun"]         = hrn["inrun"]
        except Exception as _e:  # never let instrumentation break the eval
            # logging.warning has no handler in spawn workers -> invisible;
            # write to fd 2 which the eval captures.
            import traceback as _tb
            sys.stderr.write("honest-fp compute FAILED: "
                             + _tb.format_exc() + "\n"); sys.stderr.flush()
            metrics_dict["fp_tracks_honest"]  = num_false_positive_tracks
            metrics_dict["fp_honest_fully"]   = num_false_positive_tracks
            metrics_dict["fp_honest_leadin"]  = 0
            metrics_dict["fp_honest_lagout"]  = 0
            metrics_dict["fp_honest_bridge"]  = 0
            metrics_dict["fp_frames_honest"]  = 0
            metrics_dict["fp_fhonest_nm"]     = 0
            metrics_dict["fp_fhonest_leadin"] = 0
            metrics_dict["fp_fhonest_lagout"] = 0
            metrics_dict["fp_fhonest_bridge"] = 0
            # fitness reads fp_tracks_honest_v2 — on failure fall back to
            # the old gamed count so fitness degrades gracefully (and the
            # failure is already loud on stderr + the completeness assert).
            metrics_dict["fp_tracks_honest_v2"] = num_false_positive_tracks
            metrics_dict["fp_h2_nm"]            = num_false_positive_tracks
            metrics_dict["fp_h2_inrun"]         = 0
        # Threshold-sweep cache (exp 20260515-honest-fp-threshold-sweep):
        # SEPARATE from the honest try so a dump failure is LOUD (the sweep
        # corpus must be complete; a silent gap would bias the verdict). A
        # post-eval assert (dump count == clip count) backstops this.
        _dd = os.environ.get("HONEST_FP_DUMP_DIR")
        if _dd and honest_dump_tag and ev_by_hid is not None:
            import gzip
            os.makedirs(_dd, exist_ok=True)
            tmp = os.path.join(_dd, f".{honest_dump_tag}.tmp")
            fin = os.path.join(_dd, f"{honest_dump_tag}.pkl.gz")
            try:
                with gzip.open(tmp, "wb") as _f:
                    pickle.dump({"ds_key": honest_dump_tag,
                                 "events_by_hid": ev_by_hid,
                                 "cd_frames": hyp_cd_frames,
                                 "gt_cd_frames": gt_cd_frames,
                                 "gamed_fp_tracks": int(num_false_positive_tracks)},
                                _f, protocol=5)
                os.replace(tmp, fin)   # atomic; never a half-written pkl
            except Exception:
                import traceback as _tb
                sys.stderr.write(f"honest-fp DUMP FAILED [{honest_dump_tag}]: "
                                 + _tb.format_exc() + "\n"); sys.stderr.flush()
                raise   # loud: a missing dump invalidates the sweep

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
    if frame_metrics and not use_c_metrics:
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
    if not use_c_metrics:
        del mh
        del acc

    logging.debug(f"detection metrics")

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
                for p in params:
                    if p not in ["fitness","fp_per_frame","fn_per_obj","switch_per_obj","frag_per_obj"]:
                        er[p]=sum([r["result"][p] for r in filtered if "result" in r and p in r["result"]])
                weighted_motp_sum=0

                for r in filtered:
                    weighted_motp_sum += r['result']['motp']*r['result']['idtp']
                er["idf1"]= (2 * er["idtp"]) / (2 * er["idtp"] + er["idfp"] + er["idfn"]+1e-7)
                er['mota']= 1 - (er['num_false_positives'] + er['num_misses'] + er['num_switches']) / (er['num_objects']+1e-7)
                er['motp']=weighted_motp_sum/(er['idtp']+1e-7)
                er["fp_per_frame"]=er["num_false_positives"]/(er["num_frames"]+1e-7) # false positive dets per frame
                er["fn_per_obj"]=er["num_misses"]/(er["num_objects"]+1e-7) # num false negative dets per real object GT det
                er["switch_per_obj"]=er["num_switches"]/(er["num_unique_objects"]+1e-7) # num switches per unique object
                er["frag_per_obj"]=er["num_fragmentations"]/(er["num_unique_objects"]+1e-7)

                stats_to_avg=['mostly_tracked_frac','partially_tracked_frac','mostly_lost2_frac',
                              'missed_frac','fp_tracks_frac', 'time',
                              'tracked_frames','tracked_time','tracked_fps','tracked_frames_skipped_frac',
                              'average_detection_roi_area','det_ap_person', 'det_ap_face']

                for x in stats_to_avg:
                    if x in er:
                        er[x]=er[x]/len(filtered)

                er['fitness']=fitness_score(er)

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

# C-runtime log patterns that mean a NN bin was silently disabled —
# tracker continued but the head we tried to evaluate wasn't actually
# loaded. Eval results in that state would be invalid (they'd reflect
# a no-NN tracker), so the eval must abort loudly. Patterns mirror what
# ubon_cstuff's nn.c / nn_state.c / utrack.c emit at log_error level.
_NN_LOAD_FAIL_PATTERNS = (
    "failed to load",
    "in_dim mismatch",
    "in_dim out of range",
    "short header",
)


def _capture_stderr_around(callable_):
    """Run `callable_` while capturing fd-2 output (C-side log_error)
    into a string. Returns (return_value, captured_text)."""
    import tempfile
    saved_fd = os.dup(2)
    try:
        with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".stderr", prefix="trackwf_",
                delete=False) as ef:
            err_path = ef.name
        try:
            ef_fd = os.open(err_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            os.dup2(ef_fd, 2)
            os.close(ef_fd)
            try:
                rv = callable_()
            finally:
                try: os.fsync(2)
                except OSError: pass
                os.dup2(saved_fd, 2)
            try:
                with open(err_path, "r", errors="replace") as f:
                    captured = f.read()
            except OSError:
                captured = ""
            return rv, captured
        finally:
            try: os.unlink(err_path)
            except OSError: pass
    finally:
        try: os.close(saved_fd)
        except OSError: pass


def track_test_work_fn(params, mpwq_context, mpwq_progress_fn):
    logging.debug("Running here")
    trackset=ts.TrackSet()
    trackset_gt=ts.TrackSet(params["ds_path"])
    logging.debug(f"import create")
    def _do_import():
        trackset.import_create(trackset_gt,
                               track_min_interval=params["min_interval"],
                               display=params["display"],
                               config_file=params["config"],
                               params=params,
                               mpwq_context=mpwq_context,
                               mpwq_progress_fn=mpwq_progress_fn)
    _, captured_stderr = _capture_stderr_around(_do_import)
    # Forward captured stderr so the main process still sees it.
    if captured_stderr:
        sys.stderr.write(captured_stderr)
        sys.stderr.flush()
    nn_load_fail = next(
        (pat for pat in _NN_LOAD_FAIL_PATTERNS if pat in captured_stderr),
        None,
    )
    match_iou=0.45
    if "match_iou" in params:
        match_iou=params["match_iou"]
    eval_rate_divisor=params.get("eval_rate_divisor", 1)
    eval_min_framerate=params.get("eval_min_framerate", 30.0)
    logging.debug(f"compute metrics")
    result=compute_metrics(trackset_gt, trackset,
                           max_duration=params["max_duration"],
                           match_iou=match_iou,
                           eval_rate_divisor=eval_rate_divisor,
                           eval_min_framerate=eval_min_framerate,
                           honest_dump_tag=f'{params.get("test_key","t")}__{params.get("ds_key","ds")}')

    del trackset
    del trackset_gt
    logging.debug(f"set entry")
    entry={"params":params,
           "result":result,
           "time":datetime.datetime.now()}
    if nn_load_fail is not None:
        # Surface to the parent — parent aggregator will abort.
        entry["nn_load_fail"] = {
            "pattern": nn_load_fail,
            "captured_excerpt": captured_stderr[-1000:],
        }

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
                params_to_copy=["eval_rate_divisor", "eval_min_framerate"]
                for p in params_to_copy:
                    if p in config:
                        params[p]=config[p]
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

    nn_failures = [o for o in output_results if "nn_load_fail" in o]
    if nn_failures:
        # Loud, immediate abort. Silently producing a "no head" bench when
        # the test was meant to evaluate a specific NN is the exact failure
        # mode we don't want to ship past.
        msgs = []
        for f in nn_failures[:5]:
            p = f["params"]
            msgs.append(
                f"  test={p['test_key']} clip={p['ds_key']} "
                f"pattern={f['nn_load_fail']['pattern']!r}\n"
                f"  excerpt: {f['nn_load_fail']['captured_excerpt'][:300]}"
            )
        more = "" if len(nn_failures) <= 5 else f"\n  ... and {len(nn_failures) - 5} more"
        raise RuntimeError(
            "NN bin failed to load in one or more workers — results would "
            "silently reflect a NN-disabled tracker. Aborting eval.\n"
            + "\n".join(msgs) + more
        )

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
    # Float keys exposed in the per-test summary. Aligns with what
    # `eval_head_fitness` historically wrote so downstream JSON consumers
    # (run_pipeline.sh, notebooks) keep working unchanged.
    return [
        "fitness", "mota", "idf1", "fp_tracks", "fp_per_frame",
        "fn_per_obj", "switch_per_obj", "frag_per_obj", "motp",
        "num_frames", "num_objects", "num_false_positives",
        "num_misses", "num_switches",
        "fp_tracks_honest", "fp_honest_fully", "fp_honest_leadin",
        "fp_honest_lagout", "fp_honest_bridge",   # exp 20260515-honest-fp-metric-def
        "fp_frames_honest", "fp_fhonest_nm", "fp_fhonest_leadin",
        "fp_fhonest_lagout", "fp_fhonest_bridge",  # exp#6 FROZEN frame ruler
        "fp_tracks_honest_v2", "fp_h2_nm", "fp_h2_inrun",  # exp#10 RESOLVED
        # ruler — drives `fitness` (IoU=0 run-count). Must be a summary key
        # so fitness_score finds it on aggregate rows too.
        "duration",   # video seconds; fitness length-normalises honest_v2
                      # by duration (frame-rate invariant). Summable.
    ]


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
    """Sidecar JSON next to the text results report.

    Structure:
        {
          "elapsed_seconds": float,
          "num_clips": int,
          "tests": {
            "<test_key>": {
              "overall":   {fitness, mota, fp_tracks, ...},   # __ovr<group> when 1 group, else _arithmean
              "groups":    {"<group>": {...}, ...},           # one per __ovr<group> rollup
              "arithmean": {...},                              # _arithmean rollup
              "clips":     {"<ds_key>": {...}, ...}            # raw per-clip metrics
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
            # per-group arithmetic mean — keep alongside the overall.
            group = ds_key[len("__mean("):-1]
            bucket["groups"].setdefault(group, {})
            bucket["groups"][group + "_mean"] = metrics
        elif ds_key == "_arithmean":
            bucket["arithmean"] = metrics

    # Per-clip raw rows.
    for entry in output_results:
        test_key = entry["params"]["test_key"]
        ds_key = entry["params"]["ds_key"]
        bucket = tests_by_key.setdefault(test_key, {
            "overall": None, "groups": {}, "arithmean": None, "clips": {},
        })
        bucket["clips"][ds_key] = _result_subset(entry["result"], keys)

    # "overall" is the single-group __ovr rollup when there is exactly one
    # group, else the arithmetic mean across all clips. This mirrors what
    # `eval_head_fitness` returned as `overall`.
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
    logging.info(f"Wrote eval summary JSON: {out_path}")
