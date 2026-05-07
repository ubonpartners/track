"""
Build the training corpus for the *creation head* — the NN that
replaces `param_new_track_thr` and `param_immediate_confirm_thr` in
utrack.c.

Approach: walk the runtime's emitted tracks (low-thr run, so many
weak-confidence tracks were created). For each track, its first-frame
det features form the *example*. The label comes from GT lookahead
on the track's emitted lineage:

  - `discard` (0):     track stayed aligned with no single GT id
                       for ≥ K_useful consecutive frames → spurious.
  - `unconfirmed` (1): track was useful but frame-1 quality / second-
                       frame alignment didn't justify immediate confirm.
  - `tracked` (2):     track aligned with the same GT id from frame 1
                       at high IoU AND held into frame 2 → safe to
                       confirm immediately.

Each example: 7-dim features
  det_conf, det_w, det_h, det_aspect, pose_kp_visible,
  scene_density (log dets/frame), det_y_norm (frame edge proxy)

Run prerequisites:
  - pair_log v8 with `new_track_thr=0.3, immediate_confirm_thr=2.0`
    (i.e., the C runtime created many low-conf tracks, no shortcut to
    TRACKED). See bench/pair_log_config_v8_lowthr.yaml.

Usage:
  python -m bench.build_creation_corpus \\
      --pair-log-dir runs/track_analysis/pair_log_v8_lowthr \\
      --gt-config bench/pair_log_config_v8_lowthr.yaml \\
      --out bench/data/creation_corpus
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
import src.trackset as ts


CREATION_FEATURE_NAMES = (
    # base 7 (v1)
    "det_conf",
    "det_w",
    "det_h",
    "det_aspect",
    "pose_kp_visible",
    "scene_density",     # log(n_dets in this frame)
    "det_y_norm",
    # v2: face / FIQA — face gives a much stronger "is this real?"
    # signal than det.conf alone in cluttered scenes.
    "subbox_conf",       # face confidence (0 if no face on det)
    "fiqa_score",        # face image quality
    # v2: "Bayesian" frame-level context — model learns priors from these
    "frame_mean_conf",   # avg det.conf in this frame
    "frame_max_conf",    # max det.conf in this frame
    "frame_face_frac",   # fraction of dets with face (subbox_conf>0)
    "rel_conf_to_mean",  # det.conf - frame_mean_conf (how confident this
                         # det is *relative to* the typical det in scene)
)
N_FEATURES = len(CREATION_FEATURE_NAMES)

EXAMPLE_DTYPE = np.dtype([
    ("sequence",      "U64"),
    ("track_id",      "<u8"),
    ("frame_idx",     "<u4"),
    ("features",      "<f4", (N_FEATURES,)),
    ("label",         "u1"),
    ("alignment_run", "<u4"),
    ("first_iou",     "<f4"),
    ("split",         "U16"),
])


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    iw = max(0.0, ix1 - ix0); ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    a_area = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    b_area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = a_area + b_area - inter
    return float(inter / union) if union > 1e-9 else 0.0


def _iou_matrix(track_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Vectorised IoU between two arrays of xyxy boxes.

    track_boxes: (T, 4), gt_boxes: (G, 4) → returns (T, G).
    """
    if track_boxes.size == 0 or gt_boxes.size == 0:
        return np.zeros((track_boxes.shape[0], gt_boxes.shape[0]),
                        dtype=np.float32)
    t = track_boxes
    g = gt_boxes
    ix0 = np.maximum(t[:, None, 0], g[None, :, 0])
    iy0 = np.maximum(t[:, None, 1], g[None, :, 1])
    ix1 = np.minimum(t[:, None, 2], g[None, :, 2])
    iy1 = np.minimum(t[:, None, 3], g[None, :, 3])
    iw = np.maximum(0.0, ix1 - ix0)
    ih = np.maximum(0.0, iy1 - iy0)
    inter = iw * ih
    a_area = np.maximum(0.0, t[:, 2] - t[:, 0]) * np.maximum(0.0, t[:, 3] - t[:, 1])
    g_area = np.maximum(0.0, g[:, 2] - g[:, 0]) * np.maximum(0.0, g[:, 3] - g[:, 1])
    union = a_area[:, None] + g_area[None, :] - inter
    return np.where(union > 1e-9, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def _gt_objects_at_time(gt: ts.TrackSet, t: float,
                        class_name: str = "person") -> List[Tuple[int, np.ndarray]]:
    """Return [(track_id, xyxy)] for GT objects at time t.

    The trackset returns Object instances (not dicts) with `box` and
    `track_id` attributes. Class filter omitted — GT files are
    person-only (or class is implicit). If a multi-class scenario is
    needed, route through getattr(o, 'cl', 'person').
    """
    out: List[Tuple[int, np.ndarray]] = []
    for o in gt.objects_at_time(t):
        box = getattr(o, "box", None)
        if box is None:
            continue
        out.append((int(getattr(o, "track_id", -1)),
                    np.asarray(box, dtype=np.float32)))
    return out


def _best_gt(box: np.ndarray, gt_list) -> Tuple[Optional[int], float]:
    best_id, best_iou = None, 0.0
    for tid, gtb in gt_list:
        iou = _iou_xyxy(box, gtb)
        if iou > best_iou:
            best_iou = iou; best_id = tid
    return best_id, best_iou


def _pose_visible_count(pose_points, conf_thr: float = 0.3) -> int:
    if pose_points is None:
        return 0
    arr = np.asarray(pose_points, dtype=np.float32).reshape(-1, 3)
    return int((arr[:, 2] > conf_thr).sum())


def _norm_box_dims(box, frame_h: float, frame_w: float):
    box = np.asarray(box, dtype=np.float32)
    w = max(0.0, float(box[2] - box[0]))
    h = max(0.0, float(box[3] - box[1]))
    cy = 0.5 * (float(box[1]) + float(box[3]))
    aspect = w / max(h, 1e-3)
    return w, h, aspect, cy


def extract_sequence(seq_name: str, ubtrk2_path: str, gt_path: str,
                     split: str, k_useful: int = 3,
                     iou_match: float = 0.5,
                     iou_immediate: float = 0.6) -> Optional[np.ndarray]:
    # Lazy-load ubtrk2 (don't decode reid/clip embeddings; they're huge
    # and we only need box + confidence + pose_points). MOT20-05 is 1.7G
    # and full decode is ~5+ minutes.
    run = ts.TrackSet(ubtrk2_path, decode_payloads=False, analysis_mode=True)
    gt = ts.TrackSet(gt_path)
    if not run.frames:
        return None

    # PASS 1: walk frames in order. For each frame, get GT once. For
    # each emitted track in that frame, compute its GT alignment and
    # store. This is O(n_frames * n_tracks_per_frame * n_gt_per_frame),
    # but the outer pass over GT is once per frame instead of once per
    # (track, frame_in_track_history).
    n_dets_per_frame: List[int] = []
    # Frame-level "context" stats — used as Bayesian priors at inference.
    # Computed from inference_dets (the raw detector output, before
    # matching) so they reflect "what scene am I in" not "what tracks
    # exist".
    frame_mean_conf: List[float] = []
    frame_max_conf: List[float] = []
    frame_face_frac: List[float] = []
    # alignment_history[track_id] = list of (frame_idx, gtid_or_-1, iou)
    alignment_history: Dict[int, List[Tuple[int, int, float]]] = {}
    track_first_appearance: Dict[int, Tuple[int, dict]] = {}

    for i, frame in enumerate(run.frames):
        objs = frame.get("objects") or {}
        inference = frame.get("inference_dets") or []
        n_dets_per_frame.append(len(inference))
        # Frame-level context. Walk inference_dets once.
        if inference:
            confs = []; n_face = 0
            for d in inference:
                if not isinstance(d, dict): continue
                c = float(d.get("confidence", 0.0))
                confs.append(c)
                if float(d.get("subbox_conf", 0.0)) > 0.0:
                    n_face += 1
            if confs:
                frame_mean_conf.append(sum(confs) / len(confs))
                frame_max_conf.append(max(confs))
                frame_face_frac.append(n_face / len(confs))
            else:
                frame_mean_conf.append(0.0); frame_max_conf.append(0.0); frame_face_frac.append(0.0)
        else:
            frame_mean_conf.append(0.0); frame_max_conf.append(0.0); frame_face_frac.append(0.0)
        if not isinstance(objs, dict) or not objs:
            continue
        t = float(frame.get("frame_time", 0.0))
        gt_list = _gt_objects_at_time(gt, t)
        if not gt_list:
            # No GT this frame — record alignment as miss for every track
            for tid_raw, obj in objs.items():
                if not isinstance(obj, dict):
                    continue
                tid = int(tid_raw)
                if tid not in track_first_appearance and (
                        obj.get("box") or obj.get("bbox")) is not None:
                    track_first_appearance[tid] = (i, obj)
                alignment_history.setdefault(tid, []).append((i, -1, 0.0))
            continue
        # Vectorised IoU for the whole frame.
        tids: List[int] = []
        track_box_list: List[np.ndarray] = []
        objs_kept: List[dict] = []
        for tid_raw, obj in objs.items():
            if not isinstance(obj, dict):
                continue
            box = obj.get("box") or obj.get("bbox")
            if box is None:
                continue
            tids.append(int(tid_raw))
            track_box_list.append(np.asarray(box, dtype=np.float32))
            objs_kept.append(obj)
        if not tids:
            continue
        track_boxes = np.stack(track_box_list, axis=0)
        gt_ids_arr = np.asarray([gid for gid, _ in gt_list], dtype=np.int64)
        gt_boxes = np.stack([gb for _, gb in gt_list], axis=0)
        ious = _iou_matrix(track_boxes, gt_boxes)  # (T, G)
        best_g = ious.argmax(axis=1)  # (T,)
        best_iou = ious[np.arange(ious.shape[0]), best_g]
        for k, tid in enumerate(tids):
            if tid not in track_first_appearance:
                track_first_appearance[tid] = (i, objs_kept[k])
            alignment_history.setdefault(tid, []).append(
                (i, int(gt_ids_arr[best_g[k]]), float(best_iou[k])))

    # PASS 2: per-track labelling.
    out_rows: List[tuple] = []
    for tid, hist in alignment_history.items():
        first_idx, first_obj = track_first_appearance[tid]
        bbox = first_obj.get("box") or first_obj.get("bbox")
        if bbox is None:
            continue
        bbox = np.asarray(bbox, dtype=np.float32)
        det_conf = float(first_obj.get("confidence", 0.0))
        pose_kp = float(_pose_visible_count(first_obj.get("pose_points")))
        w, h, aspect, cy = _norm_box_dims(bbox, 1.0, 1.0)
        scene_density = float(np.log1p(n_dets_per_frame[first_idx]))
        # v2 features
        subbox_conf = float(first_obj.get("subbox_conf", 0.0))
        fiqa_score = float(first_obj.get("fiqa_score", 0.0))
        f_mean = (frame_mean_conf[first_idx] if first_idx < len(frame_mean_conf) else 0.0)
        f_max  = (frame_max_conf[first_idx]  if first_idx < len(frame_max_conf)  else 0.0)
        f_face = (frame_face_frac[first_idx] if first_idx < len(frame_face_frac) else 0.0)
        rel_conf = det_conf - f_mean
        feats = np.array([det_conf, w, h, aspect, pose_kp, scene_density,
                          cy, subbox_conf, fiqa_score, f_mean, f_max,
                          f_face, rel_conf], dtype=np.float32)
        np.nan_to_num(feats, copy=False)

        # walk hist (already in frame order because frames iterated in order)
        first_aligned_id = -1
        first_iou = 0.0
        run_len_max = 0
        cur_id = -1
        cur_run = 0
        second_aligned = False
        for j, (fi, gtid, iou) in enumerate(hist):
            aligned = (gtid >= 0 and iou >= iou_match)
            if j == 0:
                first_iou = iou
                first_aligned_id = gtid if aligned else -1
            elif j == 1:
                second_aligned = (aligned and gtid == first_aligned_id
                                  and first_aligned_id >= 0)
            if aligned:
                if cur_id == gtid:
                    cur_run += 1
                else:
                    cur_id, cur_run = gtid, 1
                if cur_run > run_len_max:
                    run_len_max = cur_run
            else:
                cur_id, cur_run = -1, 0

        useful = run_len_max >= k_useful
        if not useful:
            label = 0
        elif first_aligned_id >= 0 and first_iou >= iou_immediate \
             and second_aligned:
            label = 2
        else:
            label = 1

        out_rows.append((seq_name, np.uint64(tid),
                         np.uint32(first_idx), feats,
                         np.uint8(label), np.uint32(run_len_max),
                         np.float32(first_iou), split))

    if not out_rows:
        return None
    return np.array(out_rows, dtype=EXAMPLE_DTYPE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair-log-dir", required=True)
    p.add_argument("--gt-config", required=True)
    p.add_argument("--out", default="bench/data/creation_corpus")
    p.add_argument("--k-useful", type=int, default=3)
    p.add_argument("--iou-match", type=float, default=0.5)
    p.add_argument("--iou-immediate", type=float, default=0.6)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.gt_config))
    by_split: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}
    ubtrk2_root = os.path.join(args.pair_log_dir, "ubtrk2")
    n_seq_ok = 0
    for seq_name, seq in cfg["dataset"].items():
        split = seq.get("split", "train")
        gt_path = seq["trackset"]
        ubtrk2_path = os.path.join(ubtrk2_root, f"{seq_name}.ubtrk2")
        if not os.path.exists(ubtrk2_path):
            continue
        ex = extract_sequence(seq_name, ubtrk2_path, gt_path, split,
                              k_useful=args.k_useful,
                              iou_match=args.iou_match,
                              iou_immediate=args.iou_immediate)
        if ex is None or len(ex) == 0:
            continue
        n_disc = int((ex["label"] == 0).sum())
        n_unc  = int((ex["label"] == 1).sum())
        n_trk  = int((ex["label"] == 2).sum())
        print(f"  [{split:5s}] {seq_name:38s}  n={len(ex):5d}  "
              f"disc={n_disc:4d} unc={n_unc:4d} trk={n_trk:4d}",
              flush=True)
        by_split[split].append(ex)
        n_seq_ok += 1

    print(f"\nProcessed {n_seq_ok} sequences")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for split, chunks in by_split.items():
        if not chunks:
            continue
        arr = np.concatenate(chunks)
        out_path = f"{args.out}_{split}.npz"
        np.savez_compressed(out_path, records=arr,
                            feature_names=np.array(CREATION_FEATURE_NAMES,
                                                    dtype=object))
        n = len(arr)
        n_disc = int((arr["label"] == 0).sum())
        n_unc  = int((arr["label"] == 1).sum())
        n_trk  = int((arr["label"] == 2).sum())
        print(f"  Wrote {out_path}: {n} examples  "
              f"discard={n_disc} ({100*n_disc/n:.1f}%)  "
              f"unconfirmed={n_unc} ({100*n_unc/n:.1f}%)  "
              f"tracked={n_trk} ({100*n_trk/n:.1f}%)")


if __name__ == "__main__":
    main()
