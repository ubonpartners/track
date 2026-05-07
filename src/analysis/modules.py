from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import stuff

import src.track_test as track_test
from src.analysis.variants import VariantImplementation

try:
    import ubon_pycstuff.ubon_pycstuff as upyc
except Exception:  # pragma: no cover
    upyc = None


def _to_float_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        out = float(v)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    return float(stuff.box_iou(list(a), list(b)))


def _box_iou_vec(a: Sequence[float], bs: np.ndarray) -> np.ndarray:
    if bs.size == 0:
        return np.zeros((0,), dtype=np.float32)
    aa = np.asarray(a, dtype=np.float32).reshape(4)
    bb = np.asarray(bs, dtype=np.float32)
    if bb.ndim != 2 or bb.shape[1] != 4:
        return np.zeros((0,), dtype=np.float32)

    inter_x0 = np.maximum(aa[0], bb[:, 0])
    inter_y0 = np.maximum(aa[1], bb[:, 1])
    inter_x1 = np.minimum(aa[2], bb[:, 2])
    inter_y1 = np.minimum(aa[3], bb[:, 3])
    inter_w = np.maximum(0.0, inter_x1 - inter_x0)
    inter_h = np.maximum(0.0, inter_y1 - inter_y0)
    inter = inter_w * inter_h

    area_a = max(0.0, float(aa[2] - aa[0])) * max(0.0, float(aa[3] - aa[1]))
    area_b = np.maximum(0.0, bb[:, 2] - bb[:, 0]) * np.maximum(0.0, bb[:, 3] - bb[:, 1])
    denom = area_a + area_b - inter
    return np.where(denom > 1e-12, inter / denom, 0.0).astype(np.float32, copy=False)


def _intersect_mask_vec(a: Sequence[float], bs: np.ndarray) -> np.ndarray:
    if bs.size == 0:
        return np.zeros((0,), dtype=bool)
    aa = np.asarray(a, dtype=np.float32).reshape(4)
    bb = np.asarray(bs, dtype=np.float32)
    if bb.ndim != 2 or bb.shape[1] != 4:
        return np.zeros((0,), dtype=bool)
    x_left = np.maximum(aa[0], bb[:, 0])
    y_top = np.maximum(aa[1], bb[:, 1])
    x_right = np.minimum(aa[2], bb[:, 2])
    y_bottom = np.minimum(aa[3], bb[:, 3])
    return (x_right > x_left) & (y_bottom > y_top)


def _build_vbox(
    curr_box: Sequence[float],
    of_pred_box: Sequence[float],
    kf_pred_box: Sequence[float],
    expand: float,
) -> List[float]:
    x0 = min(float(curr_box[0]), float(of_pred_box[0]), float(kf_pred_box[0]))
    y0 = min(float(curr_box[1]), float(of_pred_box[1]), float(kf_pred_box[1]))
    x1 = max(float(curr_box[2]), float(of_pred_box[2]), float(kf_pred_box[2]))
    y1 = max(float(curr_box[3]), float(of_pred_box[3]), float(kf_pred_box[3]))
    w = x1 - x0
    h = y1 - y0
    e = float(expand)
    return [
        x0 - (0.5 * e * w),
        y0 - (0.5 * e * h),
        x1 + (0.5 * e * w),
        y1 + (0.5 * e * h),
    ]


def _box_center(box: Sequence[float]) -> Tuple[float, float]:
    return (0.5 * (float(box[0]) + float(box[2])), 0.5 * (float(box[1]) + float(box[3])))


def _box_wh(box: Sequence[float]) -> Tuple[float, float]:
    return (max(0.0, float(box[2]) - float(box[0])), max(0.0, float(box[3]) - float(box[1])))


def _best_iou_match(box: Sequence[float], gt_objects: Sequence[Any]) -> Tuple[Optional[Any], float]:
    best = None
    best_iou = 0.0
    for g in gt_objects:
        iou = _box_iou(box, g.box)
        if iou > best_iou:
            best_iou = iou
            best = g
    return best, best_iou


def _class_name(class_names: Sequence[str], class_id: Any) -> Optional[str]:
    try:
        idx = int(class_id)
    except Exception:
        return None
    if idx < 0 or idx >= len(class_names):
        return None
    return str(class_names[idx])


def _decode_vec(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        v = stuff.decode_payload(value)
    except Exception:
        v = value
    try:
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _extract_flow(frame: Dict[str, Any]) -> Optional[np.ndarray]:
    debug = frame.get("debug")
    if not isinstance(debug, dict):
        return None
    motion_field = debug.get("motion_field")
    if not isinstance(motion_field, dict):
        return None
    motion_data = motion_field.get("data")
    if not isinstance(motion_data, dict):
        return None
    flow = motion_data.get("flow")
    if flow is None:
        flow = motion_data.get("motion_array")
    if flow is None:
        return None
    try:
        flow = stuff.decode_payload(flow)
    except Exception:
        pass
    try:
        arr = np.asarray(flow, dtype=np.float32)
    except Exception:
        return None
    if arr.ndim != 3 or arr.shape[2] < 2:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _det_box(det: Dict[str, Any]) -> Optional[List[float]]:
    box = det.get("box")
    if not isinstance(box, list) or len(box) != 4:
        return None
    vals = [_to_float_or_none(v) for v in box]
    if any(v is None for v in vals):
        return None
    return [float(v) for v in vals]


def _cache_map(sequence_ctx: Dict[str, Any], name: str) -> Dict[Any, Any]:
    cache = sequence_ctx.get("_analysis_cache")
    if not isinstance(cache, dict):
        cache = {}
        sequence_ctx["_analysis_cache"] = cache
    slot = cache.get(name)
    if not isinstance(slot, dict):
        slot = {}
        cache[name] = slot
    return slot


def _gt_objects_at_time(sequence_ctx: Dict[str, Any], t: float) -> List[Any]:
    cache = _cache_map(sequence_ctx, "gt_objects_at_time")
    key = float(t)
    if key not in cache:
        gt = sequence_ctx["gt_trackset"]
        cache[key] = gt.objects_at_time(key) or []
    return cache[key]


def _gt_objects_at_time_class(sequence_ctx: Dict[str, Any], t: float, class_name: str) -> List[Any]:
    cache = _cache_map(sequence_ctx, "gt_objects_at_time_class")
    key = (float(t), str(class_name))
    if key not in cache:
        gt = sequence_ctx["gt_trackset"]
        gt_classes = gt.metadata.get("classes", [])
        cache[key] = [
            g
            for g in _gt_objects_at_time(sequence_ctx, float(t))
            if _class_name(gt_classes, g.cl) == class_name
        ]
    return cache[key]


def _flow_at_frame_idx(sequence_ctx: Dict[str, Any], frame_idx: int, frame: Dict[str, Any]) -> Optional[np.ndarray]:
    cache = _cache_map(sequence_ctx, "flow_by_frame_idx")
    if frame_idx not in cache:
        cache[frame_idx] = _extract_flow(frame)
    return cache[frame_idx]


def _run_objects_for_frame_class(
    sequence_ctx: Dict[str, Any],
    frame_idx: int,
    frame: Dict[str, Any],
    class_name: str,
) -> List[Tuple[Any, Dict[str, Any], List[float]]]:
    cache = _cache_map(sequence_ctx, "run_objects_for_frame_class")
    key = (int(frame_idx), str(class_name))
    if key in cache:
        return cache[key]
    out: List[Tuple[Any, Dict[str, Any], List[float]]] = []
    run_classes = sequence_ctx["run_trackset"].metadata.get("classes", [])
    frame_objects = frame.get("objects")
    if isinstance(frame_objects, dict):
        for track_id, obj in frame_objects.items():
            if not isinstance(obj, dict):
                continue
            if _class_name(run_classes, obj.get("class")) != class_name:
                continue
            box = _det_box(obj)
            if box is None:
                continue
            out.append((track_id, obj, box))
    cache[key] = out
    return out


def _run_reid_vectors_for_frame_class(
    sequence_ctx: Dict[str, Any],
    frame_idx: int,
    frame: Dict[str, Any],
    class_name: str,
) -> List[Tuple[Any, np.ndarray]]:
    cache = _cache_map(sequence_ctx, "run_reid_vectors_for_frame_class")
    key = (int(frame_idx), str(class_name))
    if key in cache:
        return cache[key]
    out: List[Tuple[Any, np.ndarray]] = []
    for track_id, obj, _box in _run_objects_for_frame_class(sequence_ctx, frame_idx, frame, class_name):
        vec = _decode_vec(obj.get("reid_vector"))
        if vec is None:
            continue
        out.append((track_id, vec))
    cache[key] = out
    return out


def _next_det_candidates_for_frame_class(
    sequence_ctx: Dict[str, Any],
    frame_idx: int,
    frame: Dict[str, Any],
    class_name: str,
    require_reid: bool,
) -> List[Dict[str, Any]]:
    cache = _cache_map(sequence_ctx, "next_det_candidates")
    key = (int(frame_idx), str(class_name), bool(require_reid))
    if key in cache:
        return cache[key]
    out: List[Dict[str, Any]] = []
    run_classes = sequence_ctx["run_trackset"].metadata.get("classes", [])
    next_dets = frame.get("inference_dets")
    if isinstance(next_dets, list):
        for det in next_dets:
            if not isinstance(det, dict):
                continue
            if _class_name(run_classes, det.get("class")) != class_name:
                continue
            box = _det_box(det)
            if box is None:
                continue
            entry: Dict[str, Any] = {"det": det, "box": box}
            if require_reid:
                vec = _decode_vec(det.get("reid_vector"))
                if vec is None:
                    continue
                entry["reid_vec"] = vec
            out.append(entry)
    cache[key] = out
    return out


def _candidate_buckets_for_frame_class(
    sequence_ctx: Dict[str, Any],
    frame_idx: int,
    frame: Dict[str, Any],
    class_name: str,
    candidate_limit: int,
) -> Dict[int, Dict[str, Any]]:
    cache = _cache_map(sequence_ctx, "candidate_buckets")
    key = (int(frame_idx), str(class_name), int(candidate_limit))
    if key in cache:
        return cache[key]
    candidates = _next_det_candidates_for_frame_class(
        sequence_ctx,
        frame_idx,
        frame,
        class_name,
        require_reid=True,
    )
    if candidate_limit > 0 and len(candidates) > candidate_limit:
        candidates = sorted(
            candidates,
            key=lambda c: _to_float_or_none(c["det"].get("confidence")) or -1e12,
            reverse=True,
        )[:candidate_limit]
    buckets = _build_candidate_dim_buckets(candidates)
    cache[key] = buckets
    return buckets


def _l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1)
    return np.divide(
        arr,
        norms[:, None],
        out=np.zeros_like(arr),
        where=(norms[:, None] >= 1e-9),
    )


def _build_utrack_global_reid_norm(
    run_reid_vectors: Sequence[Tuple[Any, np.ndarray]],
    candidate_buckets: Dict[int, Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    q_by_dim: Dict[int, List[Tuple[Any, np.ndarray]]] = {}
    for track_id, q in run_reid_vectors:
        dim = int(q.shape[0])
        q_by_dim.setdefault(dim, []).append((track_id, np.asarray(q, dtype=np.float32).reshape(-1)))

    for dim, bucket in candidate_buckets.items():
        c = np.asarray(bucket["vecs"], dtype=np.float32)
        q_list = q_by_dim.get(dim, [])
        if c.ndim != 2 or c.shape[0] == 0:
            continue
        sum_vec = np.sum(c, axis=0)
        for _track_id, qv in q_list:
            sum_vec = sum_vec + qv
        denom = float(c.shape[0] + len(q_list)) + 1e-7
        mean = sum_vec / denom
        c_norm = _l2_normalize_rows(c - mean[None, :]).astype(np.float32, copy=False)
        q_norm_by_tid: Dict[Any, np.ndarray] = {}
        for track_id, qv in q_list:
            centered = qv - mean
            n = float(np.linalg.norm(centered))
            if n < 1e-9:
                q_norm_by_tid[track_id] = np.zeros_like(centered)
            else:
                q_norm_by_tid[track_id] = (centered / n).astype(np.float32, copy=False)
        out[dim] = {"candidate_norm": c_norm, "query_norm_by_tid": q_norm_by_tid}
    return out


def _candidate_rank(target_index: int, scored_candidates: List[Tuple[int, float]]) -> Optional[int]:
    for rank_index, (idx, _) in enumerate(scored_candidates):
        if idx == target_index:
            return rank_index + 1
    return None


def _use_default_reid_ops(variant_impl: Any) -> bool:
    cls = variant_impl.__class__
    return (
        cls.normalize_reid is VariantImplementation.normalize_reid
        and cls.reid_similarity is VariantImplementation.reid_similarity
    )


def _use_default_match_combine(variant_impl: Any) -> bool:
    cls = variant_impl.__class__
    return cls.combine_match_score is VariantImplementation.combine_match_score


def _default_reid_similarity_vectorized(
    query_vec: np.ndarray,
    candidate_vecs: np.ndarray,
    *,
    mean_normalize: bool,
) -> np.ndarray:
    q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    c = np.asarray(candidate_vecs, dtype=np.float32)
    if c.ndim != 2 or c.shape[0] == 0 or c.shape[1] != q.shape[0]:
        return np.zeros((0,), dtype=np.float32)

    if mean_normalize:
        mean = (np.sum(c, axis=0) + q) / float(c.shape[0] + 1)
        q_center = q - mean
        c_center = c - mean
    else:
        q_center = q
        c_center = c

    q_norm = float(np.linalg.norm(q_center))
    if q_norm < 1e-9:
        qn = np.zeros_like(q_center)
    else:
        qn = q_center / q_norm

    c_norms = np.linalg.norm(c_center, axis=1)
    c_normed = np.divide(
        c_center,
        c_norms[:, None],
        out=np.zeros_like(c_center),
        where=(c_norms[:, None] >= 1e-9),
    )
    return np.dot(c_normed, qn).astype(np.float32, copy=False)


def _size_ratio_vec(box: Sequence[float], candidate_boxes: np.ndarray) -> np.ndarray:
    b = np.asarray(box, dtype=np.float32).reshape(4)
    c = np.asarray(candidate_boxes, dtype=np.float32)
    if c.ndim != 2 or c.shape[0] == 0 or c.shape[1] != 4:
        return np.zeros((0,), dtype=np.float32)
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    area_c = np.maximum(0.0, c[:, 2] - c[:, 0]) * np.maximum(0.0, c[:, 3] - c[:, 1])
    min_area = np.minimum(area_b, area_c)
    max_area = np.maximum(area_b, area_c)
    ratio = np.full(max_area.shape, 1e9, dtype=np.float32)
    valid = min_area > 1e-9
    ratio[valid] = (max_area[valid] / min_area[valid]).astype(np.float32, copy=False)
    return ratio


def _build_candidate_dim_buckets(candidates: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    buckets: Dict[int, Dict[str, Any]] = {}
    for cand in candidates:
        vec = np.asarray(cand["reid_vec"], dtype=np.float32).reshape(-1)
        dim = int(vec.shape[0])
        box = np.asarray(cand["box"], dtype=np.float32).reshape(4)
        conf = _to_float_or_none(cand["det"].get("confidence"))
        slot = buckets.setdefault(
            dim,
            {
                "dets": [],
                "vecs_list": [],
                "boxes_list": [],
                "conf_list": [],
            },
        )
        slot["dets"].append(cand["det"])
        slot["vecs_list"].append(vec)
        slot["boxes_list"].append(box)
        slot["conf_list"].append(0.0 if conf is None else float(conf))

    for dim, slot in buckets.items():
        vecs_list = slot["vecs_list"]
        boxes_list = slot["boxes_list"]
        slot["vecs"] = np.stack(vecs_list, axis=0).astype(np.float32, copy=False)
        slot["boxes"] = np.stack(boxes_list, axis=0).astype(np.float32, copy=False)
        slot["conf"] = np.asarray(slot["conf_list"], dtype=np.float32)
    return buckets


def _apply_query_ema(
    ema_state: Dict[Any, np.ndarray],
    track_id: Any,
    query_vec: np.ndarray,
    ema_weight: Optional[float],
) -> np.ndarray:
    if ema_weight is None:
        return query_vec
    w = float(max(0.0, min(1.0, ema_weight)))
    if w <= 0.0:
        return query_vec
    q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    prev = ema_state.get(track_id)
    if prev is None or int(prev.shape[0]) != int(q.shape[0]):
        ema = q.copy()
    else:
        ema = w * prev + (1.0 - w) * q
    ema_state[track_id] = ema
    return ema


def _update_query_history(
    history_state: Dict[Any, List[np.ndarray]],
    track_id: Any,
    query_vec: np.ndarray,
    history_len: int,
) -> List[np.ndarray]:
    q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    if history_len <= 1:
        history_state[track_id] = [q]
        return [q]
    prev = history_state.get(track_id) or []
    if len(prev) > 0 and int(prev[-1].shape[0]) != int(q.shape[0]):
        prev = []
    prev.append(q)
    if len(prev) > history_len:
        prev = prev[-history_len:]
    history_state[track_id] = prev
    return prev


def _normalize_reid(
    variant_impl: Any, query_vec: np.ndarray, candidate_vecs: Sequence[np.ndarray]
) -> Tuple[np.ndarray, List[np.ndarray]]:
    normalize_fn = getattr(variant_impl, "normalize_reid", None)
    if callable(normalize_fn):
        return normalize_fn(query_vec, candidate_vecs)

    vectors = [np.asarray(query_vec, dtype=np.float32).reshape(-1)] + [
        np.asarray(v, dtype=np.float32).reshape(-1) for v in candidate_vecs
    ]
    mean = np.mean(np.stack(vectors, axis=0), axis=0)

    def mean_l2(v: np.ndarray) -> np.ndarray:
        centered = v - mean
        n = float(np.linalg.norm(centered))
        if n < 1e-9:
            return np.zeros_like(centered)
        return centered / n

    return mean_l2(vectors[0]), [mean_l2(v) for v in vectors[1:]]


def _reid_similarity(variant_impl: Any, a: np.ndarray, b: np.ndarray) -> float:
    sim_fn = getattr(variant_impl, "reid_similarity", None)
    if callable(sim_fn):
        return float(sim_fn(a, b))
    return float(np.dot(a, b))


def _combine_match_score(
    variant_impl: Any,
    of_score: float,
    reid_similarity: float,
    detection_confidence: float,
    module_params: Dict[str, Any],
) -> float:
    combine_fn = getattr(variant_impl, "combine_match_score", None)
    if callable(combine_fn):
        return float(combine_fn(of_score, reid_similarity, detection_confidence, module_params))
    sim_weight = float(module_params.get("sim_weight", 0.2))
    fuse_scores = float(module_params.get("fuse_scores", 0.94))
    match_thr = float(module_params.get("match_thr", 0.0))
    score = float(of_score) + sim_weight * float(reid_similarity)
    if score < match_thr:
        return 0.0
    return score * (max(0.0, float(detection_confidence)) ** fuse_scores)


def _combine_match_scores_vectorized(
    of_scores: np.ndarray,
    reid_sims: np.ndarray,
    det_conf: np.ndarray,
    module_params: Dict[str, Any],
) -> np.ndarray:
    mode = str(module_params.get("combine_mode", "add")).strip().lower()
    fuse_scores = float(module_params.get("fuse_scores", 0.94))
    match_thr = float(module_params.get("match_thr", 0.0))
    sim_weight = float(module_params.get("sim_weight", 0.2))

    of_clip = np.clip(np.asarray(of_scores, dtype=np.float32), 0.0, 1.0)
    reid = np.asarray(reid_sims, dtype=np.float32)
    conf = np.maximum(0.0, np.asarray(det_conf, dtype=np.float32))
    sim01 = np.clip(0.5 * (reid + 1.0), 0.0, 1.0)

    if mode == "add":
        raw = of_clip + sim_weight * reid
    elif mode == "hinge":
        reid_thr = float(module_params.get("reid_hinge_thr", 0.0))
        raw = of_clip + sim_weight * np.maximum(reid - reid_thr, 0.0)
    elif mode == "gate":
        alpha = float(module_params.get("reid_gate_alpha", 0.5))
        alpha = max(0.0, min(1.0, alpha))
        gate = (1.0 - alpha) + alpha * sim01
        raw = of_clip * gate
    elif mode == "adaptive_gate":
        alpha = float(module_params.get("reid_gate_alpha", 0.5))
        alpha_scale = float(module_params.get("adaptive_alpha_motion_scale", 0.35))
        alpha_min = float(module_params.get("adaptive_alpha_min", 0.0))
        alpha_max = float(module_params.get("adaptive_alpha_max", 0.95))
        alpha_eff = alpha + alpha_scale * (1.0 - of_clip)
        alpha_eff = np.clip(alpha_eff, alpha_min, alpha_max)
        gate = (1.0 - alpha_eff) + alpha_eff * sim01
        raw = of_clip * gate
    elif mode == "geom":
        of_w = float(module_params.get("geom_of_weight", 1.0))
        reid_w = float(module_params.get("geom_reid_weight", 1.0))
        raw = np.power(of_clip, of_w) * np.power(sim01, reid_w)
    elif mode == "logit_blend":
        logit_of_w = float(module_params.get("logit_of_weight", 1.0))
        logit_reid_w = float(module_params.get("logit_reid_weight", 1.0))
        logit_bias = float(module_params.get("logit_bias", 0.0))
        eps = 1e-6
        of_p = np.clip(of_clip, eps, 1.0 - eps)
        reid_p = np.clip(sim01, eps, 1.0 - eps)
        of_logit = np.log(of_p / (1.0 - of_p))
        reid_logit = np.log(reid_p / (1.0 - reid_p))
        mix = logit_of_w * of_logit + logit_reid_w * reid_logit + logit_bias
        raw = 1.0 / (1.0 + np.exp(-mix))
    elif mode == "centered_add":
        reid_center = float(np.mean(reid)) if reid.size > 0 else 0.0
        raw = of_clip + sim_weight * (reid - reid_center)
    elif mode == "harmonic":
        beta = float(module_params.get("harmonic_beta", 0.5))
        beta = max(1e-6, min(1.0 - 1e-6, beta))
        eps = 1e-6
        raw = 1.0 / (beta / (of_clip + eps) + (1.0 - beta) / (sim01 + eps))
    else:
        raw = of_clip + sim_weight * reid

    scores = np.zeros_like(raw, dtype=np.float32)
    valid = raw >= match_thr
    scores[valid] = raw[valid] * np.power(conf[valid], fuse_scores)
    return scores


def _module_result(
    metrics: Dict[str, Optional[float]],
    counts: Dict[str, int],
    metric_counts: Dict[str, int],
) -> Dict[str, Any]:
    out_metrics: Dict[str, Optional[float]] = {}
    for k, v in metrics.items():
        if v is None:
            out_metrics[k] = None
        else:
            out_metrics[k] = float(v)
    return {
        "metrics": out_metrics,
        "counts": {k: int(v) for k, v in counts.items()},
        "metric_counts": {k: int(v) for k, v in metric_counts.items()},
    }


def evaluate_detection(
    sequence_ctx: Dict[str, Any], variant_impl: Any, module_params: Dict[str, Any]
) -> Dict[str, Any]:
    _ = variant_impl
    gt = sequence_ctx["gt_trackset"]
    run = sequence_ctx["run_trackset"]
    classes_for_det_map = list(module_params.get("classes_for_det_map", ["person", "face"]))
    metrics: Dict[str, Any] = {}
    track_test.compute_detection_metrics(
        gt=gt,
        test=run,
        metrics_dict=metrics,
        classes_for_det_map=classes_for_det_map,
    )

    frames_with_detector_output = 0
    for frame in run.frames:
        debug = frame.get("debug") or {}
        if isinstance(debug, dict) and "detector_output" in debug:
            frames_with_detector_output += 1

    clean_metrics: Dict[str, Optional[float]] = {}
    metric_counts: Dict[str, int] = {}
    for k, v in metrics.items():
        fv = _to_float_or_none(v)
        if fv is None:
            continue
        clean_metrics[k] = fv
        metric_counts[k] = frames_with_detector_output

    return _module_result(
        metrics=clean_metrics,
        counts={"frames_with_detector_output": frames_with_detector_output},
        metric_counts=metric_counts,
    )


def evaluate_optical_flow(
    sequence_ctx: Dict[str, Any], variant_impl: Any, module_params: Dict[str, Any]
) -> Dict[str, Any]:
    gt = sequence_ctx["gt_trackset"]
    run = sequence_ctx["run_trackset"]
    class_name = str(module_params.get("class_name", "person"))
    seed_match_iou = float(module_params.get("seed_match_iou", 0.5))

    eligible = 0
    samples = 0
    missing_flow = 0
    ious: List[float] = []
    top1_hits = 0
    ranks: List[int] = []

    frames = run.frames
    for i in range(len(frames) - 1):
        frame = frames[i]
        next_frame = frames[i + 1]
        run_objects = _run_objects_for_frame_class(sequence_ctx, i, frame, class_name)
        if len(run_objects) == 0:
            continue
        t = float(frame["frame_time"])
        t_next = float(next_frame["frame_time"])
        gt_curr = _gt_objects_at_time_class(sequence_ctx, t, class_name)
        gt_next = _gt_objects_at_time_class(sequence_ctx, t_next, class_name)
        if len(gt_curr) == 0 or len(gt_next) == 0:
            continue
        gt_next_by_id = {int(g.track_id): g for g in gt_next}
        flow = _flow_at_frame_idx(sequence_ctx, i, frame)

        for _track_id, obj, box in run_objects:
            best_gt, best_iou = _best_iou_match(box, gt_curr)
            if best_gt is None or best_iou < seed_match_iou:
                continue
            target_gt_id = int(best_gt.track_id)
            next_target = gt_next_by_id.get(target_gt_id)
            if next_target is None:
                continue
            eligible += 1
            if flow is None:
                missing_flow += 1
                continue
            pred_fn = getattr(variant_impl, "predict_of_for_object", None)
            pred = pred_fn(obj, box, flow) if callable(pred_fn) else variant_impl.predict_of(box, flow)
            if pred is None or len(pred) != 4:
                continue
            samples += 1
            iou_same = _box_iou(pred, next_target.box)
            ious.append(iou_same)

            candidate_scores = [(int(g.track_id), _box_iou(pred, g.box)) for g in gt_next]
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            if len(candidate_scores) > 0 and candidate_scores[0][0] == target_gt_id:
                top1_hits += 1
            rank = _candidate_rank(
                target_index=target_gt_id,
                scored_candidates=[(gtid, score) for gtid, score in candidate_scores],
            )
            if rank is not None:
                ranks.append(rank)

    mean_iou = float(np.mean(ious)) if len(ious) > 0 else None
    median_iou = float(np.median(ious)) if len(ious) > 0 else None
    top1_hit_rate = (top1_hits / samples) if samples > 0 else None
    mean_rank = float(np.mean(ranks)) if len(ranks) > 0 else None
    flow_coverage = (samples / eligible) if eligible > 0 else None

    metric_counts = {
        "mean_iou": samples,
        "median_iou": samples,
        "top1_hit_rate": samples,
        "mean_rank": len(ranks),
        "flow_coverage": eligible,
    }
    return _module_result(
        metrics={
            "mean_iou": mean_iou,
            "median_iou": median_iou,
            "top1_hit_rate": top1_hit_rate,
            "mean_rank": mean_rank,
            "flow_coverage": flow_coverage,
        },
        counts={
            "eligible": eligible,
            "samples": samples,
            "missing_flow": missing_flow,
        },
        metric_counts=metric_counts,
    )


def _predict_kalman_one_step(
    prev_box: Sequence[float],
    curr_box: Sequence[float],
    t_prev: float,
    t_curr: float,
    t_next: float,
    use_upyc_kalman: bool,
    kf_fps_scale: Optional[float] = None,
    kf_std_weight_pos: Optional[float] = None,
    kf_std_weight_vel: Optional[float] = None,
    kf_init_pos_mult: Optional[float] = None,
    kf_init_vel_mult: Optional[float] = None,
    kf_noise_use_wh_axes: Optional[bool] = None,
    kf_noise_w_floor_h_ratio: Optional[float] = None,
) -> Optional[List[float]]:
    dt = t_curr - t_prev
    dt2 = t_next - t_curr
    if dt <= 1e-9 or dt2 <= 0:
        return None

    if use_upyc_kalman and upyc is not None and hasattr(upyc, "c_kalmanboxtracker"):
        try:
            kf_kwargs: Dict[str, Any] = {}
            if kf_fps_scale is not None:
                kf_kwargs["fps_scale"] = float(kf_fps_scale)
            if kf_std_weight_pos is not None:
                kf_kwargs["std_weight_pos"] = float(kf_std_weight_pos)
            if kf_std_weight_vel is not None:
                kf_kwargs["std_weight_vel"] = float(kf_std_weight_vel)
            if kf_init_pos_mult is not None:
                kf_kwargs["init_pos_mult"] = float(kf_init_pos_mult)
            if kf_init_vel_mult is not None:
                kf_kwargs["init_vel_mult"] = float(kf_init_vel_mult)
            if kf_noise_use_wh_axes is not None:
                kf_kwargs["noise_use_wh_axes"] = bool(kf_noise_use_wh_axes)
            if kf_noise_w_floor_h_ratio is not None:
                kf_kwargs["noise_w_floor_h_ratio"] = float(kf_noise_w_floor_h_ratio)
            kf = upyc.c_kalmanboxtracker(
                (float(prev_box[0]), float(prev_box[1]), float(prev_box[2]), float(prev_box[3])),
                float(t_prev),
                **kf_kwargs,
            )
            _ = kf.predict(float(t_curr))
            kf.update(
                (float(curr_box[0]), float(curr_box[1]), float(curr_box[2]), float(curr_box[3])),
                float(t_curr),
            )
            pred = kf.predict(float(t_next))
            out = [float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])]
            return [_clip01(v) for v in out]
        except Exception:
            pass

    prev = np.asarray(prev_box, dtype=np.float32)
    curr = np.asarray(curr_box, dtype=np.float32)
    vel = (curr - prev) / float(dt)
    pred = curr + vel * float(dt2)
    return [_clip01(float(v)) for v in pred]


def _build_kf_tracker_prev_curr(
    prev_box: Sequence[float],
    curr_box: Sequence[float],
    t_prev: float,
    t_curr: float,
    use_upyc_kalman: bool,
    kf_fps_scale: Optional[float] = None,
    kf_std_weight_pos: Optional[float] = None,
    kf_std_weight_vel: Optional[float] = None,
    kf_init_pos_mult: Optional[float] = None,
    kf_init_vel_mult: Optional[float] = None,
    kf_noise_use_wh_axes: Optional[bool] = None,
    kf_noise_w_floor_h_ratio: Optional[float] = None,
) -> Any:
    if not (use_upyc_kalman and upyc is not None and hasattr(upyc, "c_kalmanboxtracker")):
        return None
    try:
        kf_kwargs: Dict[str, Any] = {}
        if kf_fps_scale is not None:
            kf_kwargs["fps_scale"] = float(kf_fps_scale)
        if kf_std_weight_pos is not None:
            kf_kwargs["std_weight_pos"] = float(kf_std_weight_pos)
        if kf_std_weight_vel is not None:
            kf_kwargs["std_weight_vel"] = float(kf_std_weight_vel)
        if kf_init_pos_mult is not None:
            kf_kwargs["init_pos_mult"] = float(kf_init_pos_mult)
        if kf_init_vel_mult is not None:
            kf_kwargs["init_vel_mult"] = float(kf_init_vel_mult)
        if kf_noise_use_wh_axes is not None:
            kf_kwargs["noise_use_wh_axes"] = bool(kf_noise_use_wh_axes)
        if kf_noise_w_floor_h_ratio is not None:
            kf_kwargs["noise_w_floor_h_ratio"] = float(kf_noise_w_floor_h_ratio)
        kf = upyc.c_kalmanboxtracker(
            (float(prev_box[0]), float(prev_box[1]), float(prev_box[2]), float(prev_box[3])),
            float(t_prev),
            **kf_kwargs,
        )
        _ = kf.predict(float(t_curr))
        kf.update(
            (float(curr_box[0]), float(curr_box[1]), float(curr_box[2]), float(curr_box[3])),
            float(t_curr),
        )
        return kf
    except Exception:
        return None


def evaluate_kalman(
    sequence_ctx: Dict[str, Any], variant_impl: Any, module_params: Dict[str, Any]
) -> Dict[str, Any]:
    _ = variant_impl
    gt = sequence_ctx["gt_trackset"]
    run = sequence_ctx["run_trackset"]
    class_name = str(module_params.get("class_name", "person"))
    seed_match_iou = float(module_params.get("seed_match_iou", 0.5))
    use_upyc_kalman = bool(module_params.get("use_upyc_kalman", True))
    kf_fps_scale = _to_float_or_none(module_params.get("kf_fps_scale"))
    kf_std_weight_pos = _to_float_or_none(module_params.get("kf_std_weight_pos"))
    kf_std_weight_vel = _to_float_or_none(module_params.get("kf_std_weight_vel"))
    kf_init_pos_mult = _to_float_or_none(module_params.get("kf_init_pos_mult"))
    kf_init_vel_mult = _to_float_or_none(module_params.get("kf_init_vel_mult"))
    kf_noise_use_wh_axes = bool(module_params.get("kf_noise_use_wh_axes", False))
    kf_noise_w_floor_h_ratio = _to_float_or_none(module_params.get("kf_noise_w_floor_h_ratio"))

    eligible = 0
    samples = 0
    missing_prev_track = 0
    ious: List[float] = []
    center_errors: List[float] = []
    wh_errors: List[float] = []
    top1_hits = 0

    frames = run.frames
    for i in range(1, len(frames) - 1):
        prev_frame = frames[i - 1]
        frame = frames[i]
        next_frame = frames[i + 1]
        prev_objects = prev_frame.get("objects")
        if not isinstance(prev_objects, dict):
            continue
        run_objects = _run_objects_for_frame_class(sequence_ctx, i, frame, class_name)
        if len(run_objects) == 0:
            continue
        t_prev = float(prev_frame["frame_time"])
        t = float(frame["frame_time"])
        t_next = float(next_frame["frame_time"])

        gt_curr = _gt_objects_at_time_class(sequence_ctx, t, class_name)
        gt_next = _gt_objects_at_time_class(sequence_ctx, t_next, class_name)
        if len(gt_curr) == 0 or len(gt_next) == 0:
            continue
        gt_next_by_id = {int(g.track_id): g for g in gt_next}

        for track_id, _obj, curr_box in run_objects:
            best_gt, best_iou = _best_iou_match(curr_box, gt_curr)
            if best_gt is None or best_iou < seed_match_iou:
                continue
            target_gt_id = int(best_gt.track_id)
            target_next = gt_next_by_id.get(target_gt_id)
            if target_next is None:
                continue
            eligible += 1

            prev_obj = prev_objects.get(track_id)
            if not isinstance(prev_obj, dict):
                missing_prev_track += 1
                continue
            prev_box = _det_box(prev_obj)
            if prev_box is None:
                missing_prev_track += 1
                continue

            pred = _predict_kalman_one_step(
                prev_box=prev_box,
                curr_box=curr_box,
                t_prev=t_prev,
                t_curr=t,
                t_next=t_next,
                use_upyc_kalman=use_upyc_kalman,
                kf_fps_scale=kf_fps_scale,
                kf_std_weight_pos=kf_std_weight_pos,
                kf_std_weight_vel=kf_std_weight_vel,
                kf_init_pos_mult=kf_init_pos_mult,
                kf_init_vel_mult=kf_init_vel_mult,
                kf_noise_use_wh_axes=kf_noise_use_wh_axes,
                kf_noise_w_floor_h_ratio=kf_noise_w_floor_h_ratio,
            )
            if pred is None:
                continue
            samples += 1
            ious.append(_box_iou(pred, target_next.box))

            cx_p, cy_p = _box_center(pred)
            cx_g, cy_g = _box_center(target_next.box)
            center_errors.append(float(np.hypot(cx_p - cx_g, cy_p - cy_g)))
            pw, ph = _box_wh(pred)
            gw, gh = _box_wh(target_next.box)
            wh_errors.append(0.5 * (abs(pw - gw) + abs(ph - gh)))

            candidate_scores = [(int(g.track_id), _box_iou(pred, g.box)) for g in gt_next]
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            if len(candidate_scores) > 0 and candidate_scores[0][0] == target_gt_id:
                top1_hits += 1

    metric_counts = {
        "mean_iou": samples,
        "median_iou": samples,
        "mean_center_error": len(center_errors),
        "mean_wh_error": len(wh_errors),
        "top1_hit_rate": samples,
        "coverage": eligible,
    }
    return _module_result(
        metrics={
            "mean_iou": float(np.mean(ious)) if len(ious) > 0 else None,
            "median_iou": float(np.median(ious)) if len(ious) > 0 else None,
            "mean_center_error": float(np.mean(center_errors)) if len(center_errors) > 0 else None,
            "mean_wh_error": float(np.mean(wh_errors)) if len(wh_errors) > 0 else None,
            "top1_hit_rate": (top1_hits / samples) if samples > 0 else None,
            "coverage": (samples / eligible) if eligible > 0 else None,
        },
        counts={
            "eligible": eligible,
            "samples": samples,
            "missing_prev_track": missing_prev_track,
        },
        metric_counts=metric_counts,
    )


def evaluate_reid(
    sequence_ctx: Dict[str, Any], variant_impl: Any, module_params: Dict[str, Any]
) -> Dict[str, Any]:
    gt = sequence_ctx["gt_trackset"]
    run = sequence_ctx["run_trackset"]
    class_name = str(module_params.get("class_name", "person"))
    seed_match_iou = float(module_params.get("seed_match_iou", 0.5))
    target_iou_thr = float(module_params.get("target_iou_thr", 0.4))
    candidate_limit = int(module_params.get("candidate_limit", 0) or 0)
    object_sample_cap = int(module_params.get("object_sample_cap", 0) or 0)
    query_ema_weight = _to_float_or_none(module_params.get("query_ema_weight"))
    query_history_len = max(1, int(module_params.get("query_history_len", 1) or 1))
    query_history_mode = str(module_params.get("query_history_mode", "latest")).strip().lower()
    if query_history_mode not in {"latest", "best", "mean"}:
        query_history_mode = "latest"
    mean_normalize = bool(module_params.get("reid_mean_normalize", True))
    reid_max_size_ratio = float(module_params.get("reid_max_size_ratio", 0.0) or 0.0)
    compute_of_gated = bool(module_params.get("compute_of_gated", True))
    gated_of_iou_thr = float(module_params.get("gated_of_iou_thr", 0.02))
    gated_of_topk = int(module_params.get("gated_of_topk", 64) or 0)
    use_default_reid = _use_default_reid_ops(variant_impl)
    ema_state: Dict[Any, np.ndarray] = {}
    history_state: Dict[Any, List[np.ndarray]] = {}

    eligible = 0
    samples = 0
    missing_query_reid = 0
    missing_candidates = 0
    no_correct_candidate = 0
    size_filtered_candidates = 0

    top1_hits = 0
    top3_hits = 0
    ranks: List[int] = []
    margins: List[float] = []

    gated_samples = 0
    gated_top1_hits = 0
    gated_top3_hits = 0
    gated_ranks: List[int] = []
    gated_margins: List[float] = []
    gated_missing_flow = 0
    gated_missing_candidates = 0
    gated_no_correct_candidate = 0

    frames = run.frames
    for i in range(len(frames) - 1):
        if object_sample_cap > 0 and samples >= object_sample_cap:
            break
        frame = frames[i]
        next_frame = frames[i + 1]
        run_objects = _run_objects_for_frame_class(sequence_ctx, i, frame, class_name)
        if len(run_objects) == 0:
            continue

        t = float(frame["frame_time"])
        t_next = float(next_frame["frame_time"])
        gt_curr = _gt_objects_at_time_class(sequence_ctx, t, class_name)
        gt_next = _gt_objects_at_time_class(sequence_ctx, t_next, class_name)
        if len(gt_curr) == 0 or len(gt_next) == 0:
            continue
        gt_next_by_id = {int(g.track_id): g for g in gt_next}

        candidate_buckets = _candidate_buckets_for_frame_class(
            sequence_ctx, i + 1, next_frame, class_name, candidate_limit
        )
        flow = _flow_at_frame_idx(sequence_ctx, i, frame) if compute_of_gated else None

        for _track_id, obj, box in run_objects:
            if object_sample_cap > 0 and samples >= object_sample_cap:
                break
            observations_state[_track_id] = int(observations_state.get(_track_id, 0)) + 1
            best_gt, best_iou = _best_iou_match(box, gt_curr)
            if best_gt is None or best_iou < seed_match_iou:
                continue
            target_next = gt_next_by_id.get(int(best_gt.track_id))
            if target_next is None:
                continue
            eligible += 1

            qv = _decode_vec(obj.get("reid_vector"))
            if qv is None:
                missing_query_reid += 1
                continue
            qv = _apply_query_ema(ema_state, _track_id, qv, query_ema_weight)
            query_history = _update_query_history(history_state, _track_id, qv, query_history_len)
            if len(candidate_buckets) == 0:
                missing_candidates += 1
                continue

            dim = int(qv.shape[0])
            bucket = candidate_buckets.get(dim)
            if bucket is None:
                missing_candidates += 1
                continue

            cand_idx = np.arange(int(bucket["boxes"].shape[0]), dtype=np.int32)
            if reid_max_size_ratio > 1.0:
                ratios = _size_ratio_vec(box, bucket["boxes"])
                keep = ratios <= reid_max_size_ratio
                size_filtered_candidates += int(np.count_nonzero(~keep))
                cand_idx = cand_idx[keep]
            if cand_idx.size == 0:
                missing_candidates += 1
                continue

            cand_boxes = bucket["boxes"][cand_idx]
            target_ious = _box_iou_vec(target_next.box, cand_boxes)
            if target_ious.size == 0:
                missing_candidates += 1
                continue
            correct_local = int(np.argmax(target_ious))
            correct_iou = float(target_ious[correct_local])
            if correct_iou < target_iou_thr:
                no_correct_candidate += 1
                continue

            cand_vecs_arr = bucket["vecs"][cand_idx]
            if use_default_reid:
                sim_rows = [
                    _default_reid_similarity_vectorized(
                        q_hist,
                        cand_vecs_arr,
                        mean_normalize=mean_normalize,
                    )
                    for q_hist in query_history
                ]
                sims_mat = np.stack(sim_rows, axis=0)
                if query_history_mode == "best":
                    sims = np.max(sims_mat, axis=0)
                elif query_history_mode == "mean":
                    sims = np.mean(sims_mat, axis=0)
                else:
                    sims = sims_mat[-1]
            else:
                cand_vecs_list = [cand_vecs_arr[j] for j in range(int(cand_vecs_arr.shape[0]))]
                sim_rows = []
                for q_hist in query_history:
                    qn, cns = _normalize_reid(variant_impl, q_hist, cand_vecs_list)
                    sim_rows.append(
                        np.asarray(
                            [_reid_similarity(variant_impl, qn, cn) for cn in cns],
                            dtype=np.float32,
                        )
                    )
                sims_mat = np.stack(sim_rows, axis=0)
                if query_history_mode == "best":
                    sims = np.max(sims_mat, axis=0)
                elif query_history_mode == "mean":
                    sims = np.mean(sims_mat, axis=0)
                else:
                    sims = sims_mat[-1]
            if sims.size == 0:
                missing_candidates += 1
                continue
            order = np.argsort(-sims, kind="mergesort")
            samples += 1

            if int(order[0]) == correct_local:
                top1_hits += 1
            if np.any(order[:3] == correct_local):
                top3_hits += 1
            rank = int(np.where(order == correct_local)[0][0]) + 1
            ranks.append(rank)
            correct_score = float(sims[correct_local])
            best_incorrect = None
            for idx in order:
                if int(idx) != correct_local:
                    best_incorrect = float(sims[int(idx)])
                    break
            margins.append(float(correct_score - (best_incorrect if best_incorrect is not None else 0.0)))

            if compute_of_gated:
                pred_fn = getattr(variant_impl, "predict_of_for_object", None)
                pred = (
                    pred_fn(obj, box, flow) if callable(pred_fn) else variant_impl.predict_of(box, flow)
                ) if flow is not None else None
                if pred is None or len(pred) != 4:
                    gated_missing_flow += 1
                    continue
                of_ious = _box_iou_vec(pred, cand_boxes)
                if of_ious.size == 0:
                    gated_missing_candidates += 1
                    continue
                gated_idx = np.where(of_ious >= gated_of_iou_thr)[0]
                if gated_idx.size == 0:
                    gated_missing_candidates += 1
                    continue
                if gated_of_topk > 0 and gated_idx.size > gated_of_topk:
                    order_gate = np.argsort(-of_ious[gated_idx], kind="mergesort")
                    gated_idx = gated_idx[order_gate[:gated_of_topk]]
                if not np.any(gated_idx == correct_local):
                    gated_no_correct_candidate += 1
                    continue

                gated_sims = sims[gated_idx]
                if gated_sims.size == 0:
                    gated_missing_candidates += 1
                    continue
                gated_order = np.argsort(-gated_sims, kind="mergesort")
                gated_samples += 1

                correct_pos = int(np.where(gated_idx == correct_local)[0][0])
                if int(gated_order[0]) == correct_pos:
                    gated_top1_hits += 1
                if np.any(gated_order[:3] == correct_pos):
                    gated_top3_hits += 1
                gated_rank = int(np.where(gated_order == correct_pos)[0][0]) + 1
                gated_ranks.append(gated_rank)
                gated_correct_score = float(gated_sims[correct_pos])
                gated_best_incorrect = None
                for jj in gated_order:
                    if int(jj) != correct_pos:
                        gated_best_incorrect = float(gated_sims[int(jj)])
                        break
                gated_margins.append(
                    float(
                        gated_correct_score
                        - (gated_best_incorrect if gated_best_incorrect is not None else 0.0)
                    )
                )

    metric_counts = {
        "top1_hit_rate": samples,
        "top3_hit_rate": samples,
        "mean_rank": len(ranks),
        "mean_margin": len(margins),
        "candidate_coverage": eligible,
        "of_gated_top1_hit_rate": gated_samples,
        "of_gated_top3_hit_rate": gated_samples,
        "of_gated_mean_rank": len(gated_ranks),
        "of_gated_mean_margin": len(gated_margins),
        "of_gated_candidate_coverage": eligible,
    }
    return _module_result(
        metrics={
            "top1_hit_rate": (top1_hits / samples) if samples > 0 else None,
            "top3_hit_rate": (top3_hits / samples) if samples > 0 else None,
            "mean_rank": float(np.mean(ranks)) if len(ranks) > 0 else None,
            "mean_margin": float(np.mean(margins)) if len(margins) > 0 else None,
            "candidate_coverage": (samples / eligible) if eligible > 0 else None,
            "of_gated_top1_hit_rate": (gated_top1_hits / gated_samples) if gated_samples > 0 else None,
            "of_gated_top3_hit_rate": (gated_top3_hits / gated_samples) if gated_samples > 0 else None,
            "of_gated_mean_rank": float(np.mean(gated_ranks)) if len(gated_ranks) > 0 else None,
            "of_gated_mean_margin": (
                float(np.mean(gated_margins)) if len(gated_margins) > 0 else None
            ),
            "of_gated_candidate_coverage": (gated_samples / eligible) if eligible > 0 else None,
        },
        counts={
            "eligible": eligible,
            "samples": samples,
            "missing_query_reid": missing_query_reid,
            "missing_candidates": missing_candidates,
            "no_correct_candidate": no_correct_candidate,
            "size_filtered_candidates": size_filtered_candidates,
            "of_gated_samples": gated_samples,
            "of_gated_missing_flow": gated_missing_flow,
            "of_gated_missing_candidates": gated_missing_candidates,
            "of_gated_no_correct_candidate": gated_no_correct_candidate,
        },
        metric_counts=metric_counts,
    )


def evaluate_match_cost(
    sequence_ctx: Dict[str, Any], variant_impl: Any, module_params: Dict[str, Any]
) -> Dict[str, Any]:
    gt = sequence_ctx["gt_trackset"]
    run = sequence_ctx["run_trackset"]
    class_name = str(module_params.get("class_name", "person"))
    seed_match_iou = float(module_params.get("seed_match_iou", 0.5))
    target_iou_thr = float(module_params.get("target_iou_thr", 0.4))
    candidate_limit = int(module_params.get("candidate_limit", 0) or 0)
    object_sample_cap = int(module_params.get("object_sample_cap", 0) or 0)
    query_ema_weight = _to_float_or_none(module_params.get("query_ema_weight"))
    query_history_len = max(1, int(module_params.get("query_history_len", 1) or 1))
    query_history_mode = str(module_params.get("query_history_mode", "latest")).strip().lower()
    if query_history_mode not in {"latest", "best", "mean"}:
        query_history_mode = "latest"
    mean_normalize = bool(module_params.get("reid_mean_normalize", True))
    reid_norm_mode = str(module_params.get("reid_norm_mode", "utrack_global_mean")).strip().lower()
    if reid_norm_mode not in {"pair_mean", "utrack_global_mean"}:
        reid_norm_mode = "utrack_global_mean"
    reid_max_size_ratio = float(module_params.get("reid_max_size_ratio", 0.0) or 0.0)
    vbox_expand = float(module_params.get("vbox_expand", 0.1))
    use_kf_score = bool(module_params.get("use_kf_score", True))
    kf_weight_cfg = float(module_params.get("kf_weight", 1.0))
    kf_warmup = float(module_params.get("kf_warmup", 1.9))
    kf_d2_enabled = bool(module_params.get("kf_d2_enabled", True))
    kf_d2_weight = float(module_params.get("kf_d2_weight", 2.0))
    of_weight = float(module_params.get("of_weight", 1.0))
    use_upyc_kalman = bool(module_params.get("use_upyc_kalman", True))
    kf_fps_scale = _to_float_or_none(module_params.get("kf_fps_scale"))
    kf_std_weight_pos = _to_float_or_none(module_params.get("kf_std_weight_pos"))
    kf_std_weight_vel = _to_float_or_none(module_params.get("kf_std_weight_vel"))
    kf_init_pos_mult = _to_float_or_none(module_params.get("kf_init_pos_mult"))
    kf_init_vel_mult = _to_float_or_none(module_params.get("kf_init_vel_mult"))
    kf_noise_use_wh_axes = bool(module_params.get("kf_noise_use_wh_axes", False))
    kf_noise_w_floor_h_ratio = _to_float_or_none(module_params.get("kf_noise_w_floor_h_ratio"))
    use_default_reid = _use_default_reid_ops(variant_impl)
    use_default_combine = _use_default_match_combine(variant_impl)
    ema_state: Dict[Any, np.ndarray] = {}
    history_state: Dict[Any, List[np.ndarray]] = {}
    observations_state: Dict[Any, int] = {}

    eligible = 0
    samples = 0
    missing_flow = 0
    missing_query_reid = 0
    missing_candidates = 0
    no_correct_candidate = 0
    size_filtered_candidates = 0
    vbox_filtered_candidates = 0

    top1_hits = 0
    correct_passes = 0
    correct_scores: List[float] = []
    best_scores: List[float] = []
    margins: List[float] = []

    frames = run.frames
    for i in range(len(frames) - 1):
        if object_sample_cap > 0 and samples >= object_sample_cap:
            break
        frame = frames[i]
        next_frame = frames[i + 1]
        run_objects = _run_objects_for_frame_class(sequence_ctx, i, frame, class_name)
        if len(run_objects) == 0:
            continue

        t = float(frame["frame_time"])
        t_next = float(next_frame["frame_time"])
        t_prev = float(frames[i - 1]["frame_time"]) if i > 0 else t
        gt_curr = _gt_objects_at_time_class(sequence_ctx, t, class_name)
        gt_next = _gt_objects_at_time_class(sequence_ctx, t_next, class_name)
        if len(gt_curr) == 0 or len(gt_next) == 0:
            continue
        gt_next_by_id = {int(g.track_id): g for g in gt_next}

        flow = _flow_at_frame_idx(sequence_ctx, i, frame)
        if flow is None:
            continue

        candidate_buckets = _candidate_buckets_for_frame_class(
            sequence_ctx, i + 1, next_frame, class_name, candidate_limit
        )
        run_reid_vectors = _run_reid_vectors_for_frame_class(sequence_ctx, i, frame, class_name)
        utrack_norm = (
            _build_utrack_global_reid_norm(run_reid_vectors, candidate_buckets)
            if reid_norm_mode == "utrack_global_mean"
            else {}
        )
        prev_box_by_track: Dict[Any, List[float]] = {}
        if i > 0:
            prev_frame = frames[i - 1]
            for tid, _obj_prev, bprev in _run_objects_for_frame_class(
                sequence_ctx, i - 1, prev_frame, class_name
            ):
                prev_box_by_track[tid] = bprev

        for _track_id, obj, box in run_objects:
            if object_sample_cap > 0 and samples >= object_sample_cap:
                break
            observations_state[_track_id] = int(observations_state.get(_track_id, 0)) + 1
            best_gt, best_iou = _best_iou_match(box, gt_curr)
            if best_gt is None or best_iou < seed_match_iou:
                continue
            target_next = gt_next_by_id.get(int(best_gt.track_id))
            if target_next is None:
                continue
            eligible += 1

            qv = _decode_vec(obj.get("reid_vector"))
            if qv is None:
                missing_query_reid += 1
                continue
            qv = _apply_query_ema(ema_state, _track_id, qv, query_ema_weight)
            query_history = _update_query_history(history_state, _track_id, qv, query_history_len)
            if len(candidate_buckets) == 0:
                missing_candidates += 1
                continue
            dim = int(qv.shape[0])
            bucket = candidate_buckets.get(dim)
            if bucket is None:
                missing_candidates += 1
                continue

            pred_fn = getattr(variant_impl, "predict_of_for_object", None)
            pred = pred_fn(obj, box, flow) if callable(pred_fn) else variant_impl.predict_of(box, flow)
            if pred is None or len(pred) != 4:
                missing_flow += 1
                continue

            kf_pred_box = list(box)
            kf_scores_all = None
            kf_weight_eff = 0.0
            if use_kf_score:
                obs = int(observations_state[_track_id])
                if obs >= 2:
                    f = float(np.power(max(0.1, 1.0 / float(obs)), kf_warmup))
                    kf_weight_eff = max(0.0, kf_weight_cfg * (1.0 - f))
                    prev_box = prev_box_by_track.get(_track_id)
                    if prev_box is not None and i > 0:
                        kf_tracker = _build_kf_tracker_prev_curr(
                            prev_box=prev_box,
                            curr_box=box,
                            t_prev=t_prev,
                            t_curr=t,
                            use_upyc_kalman=use_upyc_kalman,
                            kf_fps_scale=kf_fps_scale,
                            kf_std_weight_pos=kf_std_weight_pos,
                            kf_std_weight_vel=kf_std_weight_vel,
                            kf_init_pos_mult=kf_init_pos_mult,
                            kf_init_vel_mult=kf_init_vel_mult,
                            kf_noise_use_wh_axes=kf_noise_use_wh_axes,
                            kf_noise_w_floor_h_ratio=kf_noise_w_floor_h_ratio,
                        )
                        if kf_tracker is not None:
                            try:
                                pred_kf = kf_tracker.predict(float(t_next))
                                kf_pred_box = [_clip01(float(pred_kf[j])) for j in range(4)]
                            except Exception:
                                kf_pred_box = list(box)
                            if kf_d2_enabled and hasattr(kf_tracker, "measurement_mahalanobis2"):
                                try:
                                    d2 = np.asarray(
                                        [
                                            float(
                                                kf_tracker.measurement_mahalanobis2(
                                                    (
                                                        float(bb[0]),
                                                        float(bb[1]),
                                                        float(bb[2]),
                                                        float(bb[3]),
                                                    )
                                                )
                                            )
                                            for bb in bucket["boxes"]
                                        ],
                                        dtype=np.float32,
                                    )
                                    kf_scores_all = np.exp(
                                        -0.5 * d2 / max(1e-6, float(kf_d2_weight))
                                    ).astype(np.float32, copy=False)
                                except Exception:
                                    kf_scores_all = _box_iou_vec(kf_pred_box, bucket["boxes"])
                            else:
                                kf_scores_all = _box_iou_vec(kf_pred_box, bucket["boxes"])
                        else:
                            kf_pred = _predict_kalman_one_step(
                                prev_box=prev_box,
                                curr_box=box,
                                t_prev=t_prev,
                                t_curr=t,
                                t_next=t_next,
                                use_upyc_kalman=use_upyc_kalman,
                                kf_fps_scale=kf_fps_scale,
                                kf_std_weight_pos=kf_std_weight_pos,
                                kf_std_weight_vel=kf_std_weight_vel,
                                kf_init_pos_mult=kf_init_pos_mult,
                                kf_init_vel_mult=kf_init_vel_mult,
                                kf_noise_use_wh_axes=kf_noise_use_wh_axes,
                                kf_noise_w_floor_h_ratio=kf_noise_w_floor_h_ratio,
                            )
                            if kf_pred is not None:
                                kf_pred_box = kf_pred
                                kf_scores_all = _box_iou_vec(kf_pred_box, bucket["boxes"])
                            else:
                                kf_weight_eff = 0.0
                    else:
                        kf_weight_eff = 0.0

            cand_idx = np.arange(int(bucket["boxes"].shape[0]), dtype=np.int32)
            if reid_max_size_ratio > 1.0:
                ratios = _size_ratio_vec(pred, bucket["boxes"])
                keep = ratios <= reid_max_size_ratio
                size_filtered_candidates += int(np.count_nonzero(~keep))
                cand_idx = cand_idx[keep]
            if vbox_expand > 0.0 and cand_idx.size > 0:
                vbox = _build_vbox(box, pred, kf_pred_box, vbox_expand)
                keep = _intersect_mask_vec(vbox, bucket["boxes"][cand_idx])
                vbox_filtered_candidates += int(np.count_nonzero(~keep))
                cand_idx = cand_idx[keep]
            if cand_idx.size == 0:
                missing_candidates += 1
                continue

            cand_boxes = bucket["boxes"][cand_idx]
            target_ious = _box_iou_vec(target_next.box, cand_boxes)
            if target_ious.size == 0:
                missing_candidates += 1
                continue
            correct_local = int(np.argmax(target_ious))
            correct_iou = float(target_ious[correct_local])
            if correct_iou < target_iou_thr:
                no_correct_candidate += 1
                continue

            cand_vecs_arr = bucket["vecs"][cand_idx]
            if use_default_reid:
                if (
                    reid_norm_mode == "utrack_global_mean"
                    and query_history_len == 1
                    and query_history_mode == "latest"
                    and (query_ema_weight is None or query_ema_weight <= 0.0)
                ):
                    norm_slot = utrack_norm.get(dim)
                    qn = None if norm_slot is None else norm_slot["query_norm_by_tid"].get(_track_id)
                    if qn is not None:
                        cand_norm = norm_slot["candidate_norm"][cand_idx]
                        reid_sims = np.dot(cand_norm, qn).astype(np.float32, copy=False)
                    else:
                        reid_sims = np.zeros((0,), dtype=np.float32)
                else:
                    sim_rows = [
                        _default_reid_similarity_vectorized(
                            q_hist,
                            cand_vecs_arr,
                            mean_normalize=mean_normalize,
                        )
                        for q_hist in query_history
                    ]
                    sims_mat = np.stack(sim_rows, axis=0)
                    if query_history_mode == "best":
                        reid_sims = np.max(sims_mat, axis=0)
                    elif query_history_mode == "mean":
                        reid_sims = np.mean(sims_mat, axis=0)
                    else:
                        reid_sims = sims_mat[-1]
            else:
                cand_vecs_list = [cand_vecs_arr[j] for j in range(int(cand_vecs_arr.shape[0]))]
                sim_rows = []
                for q_hist in query_history:
                    qn, cns = _normalize_reid(variant_impl, q_hist, cand_vecs_list)
                    sim_rows.append(
                        np.asarray(
                            [_reid_similarity(variant_impl, qn, cn) for cn in cns],
                            dtype=np.float32,
                        )
                    )
                sims_mat = np.stack(sim_rows, axis=0)
                if query_history_mode == "best":
                    reid_sims = np.max(sims_mat, axis=0)
                elif query_history_mode == "mean":
                    reid_sims = np.mean(sims_mat, axis=0)
                else:
                    reid_sims = sims_mat[-1]
            if reid_sims.size == 0:
                missing_candidates += 1
                continue

            of_scores = _box_iou_vec(pred, cand_boxes)
            kf_scores = np.zeros_like(of_scores, dtype=np.float32)
            if kf_weight_eff > 0.0 and kf_scores_all is not None:
                kf_scores = np.asarray(kf_scores_all[cand_idx], dtype=np.float32)
            motion_scores = of_scores
            if kf_weight_eff > 0.0:
                motion_scores = (
                    of_scores * of_weight + kf_scores * kf_weight_eff
                ) / max(1e-9, of_weight + kf_weight_eff)

            if use_default_combine:
                scores = _combine_match_scores_vectorized(
                    of_scores=motion_scores,
                    reid_sims=reid_sims,
                    det_conf=bucket["conf"][cand_idx],
                    module_params=module_params,
                )
            else:
                scores = np.asarray(
                    [
                        _combine_match_score(
                            variant_impl=variant_impl,
                            of_score=float(motion_scores[j]),
                            reid_similarity=float(reid_sims[j]),
                            detection_confidence=float(bucket["conf"][cand_idx][j]),
                            module_params=module_params,
                        )
                        for j in range(int(reid_sims.shape[0]))
                    ],
                    dtype=np.float32,
                )
            if scores.size == 0:
                continue
            order = np.argsort(-scores, kind="mergesort")
            samples += 1
            best_scores.append(float(scores[int(order[0])]))

            if int(order[0]) == correct_local:
                top1_hits += 1
            correct_score = float(scores[correct_local])
            best_incorrect = None
            for idx in order:
                if int(idx) != correct_local:
                    best_incorrect = float(scores[int(idx)])
                    break
            correct_scores.append(correct_score)
            if correct_score > 0:
                correct_passes += 1
            margins.append(float(correct_score - (best_incorrect if best_incorrect is not None else 0.0)))

    metric_counts = {
        "top1_hit_rate": samples,
        "correct_pass_rate": len(correct_scores),
        "mean_correct_score": len(correct_scores),
        "mean_best_score": len(best_scores),
        "mean_margin": len(margins),
        "candidate_coverage": eligible,
    }
    return _module_result(
        metrics={
            "top1_hit_rate": (top1_hits / samples) if samples > 0 else None,
            "correct_pass_rate": (correct_passes / len(correct_scores)) if len(correct_scores) > 0 else None,
            "mean_correct_score": float(np.mean(correct_scores)) if len(correct_scores) > 0 else None,
            "mean_best_score": float(np.mean(best_scores)) if len(best_scores) > 0 else None,
            "mean_margin": float(np.mean(margins)) if len(margins) > 0 else None,
            "candidate_coverage": (samples / eligible) if eligible > 0 else None,
        },
        counts={
            "eligible": eligible,
            "samples": samples,
            "missing_flow": missing_flow,
            "missing_query_reid": missing_query_reid,
            "missing_candidates": missing_candidates,
            "no_correct_candidate": no_correct_candidate,
            "size_filtered_candidates": size_filtered_candidates,
            "vbox_filtered_candidates": vbox_filtered_candidates,
        },
        metric_counts=metric_counts,
    )


def _box_inside(inner: Sequence[float], outer: Sequence[float]) -> bool:
    return (
        float(inner[0]) >= float(outer[0])
        and float(inner[1]) >= float(outer[1])
        and float(inner[2]) <= float(outer[2])
        and float(inner[3]) <= float(outer[3])
    )


def evaluate_roi_skip(
    sequence_ctx: Dict[str, Any], variant_impl: Any, module_params: Dict[str, Any]
) -> Dict[str, Any]:
    _ = variant_impl
    _ = module_params
    gt = sequence_ctx["gt_trackset"]
    run = sequence_ctx["run_trackset"]
    class_name = str(module_params.get("class_name", "person"))

    total_frames = len(run.frames)
    skipped = 0
    tracked = 0
    roi_areas: List[float] = []
    gt_eval = 0
    gt_outside = 0
    gt_not_fully_inside = 0

    for frame in run.frames:
        result_type = str(frame.get("result_type") or "")
        objects = frame.get("objects")
        is_skip = result_type.startswith("skip") or objects is None
        if is_skip:
            skipped += 1
        else:
            tracked += 1
            roi = frame.get("inference_roi")
            if isinstance(roi, list) and len(roi) == 4:
                roi_areas.append(float(stuff.box_a(roi)))

        roi = frame.get("inference_roi")
        if not (isinstance(roi, list) and len(roi) == 4):
            continue
        gt_objects = _gt_objects_at_time_class(sequence_ctx, float(frame["frame_time"]), class_name)
        for g in gt_objects:
            gt_eval += 1
            if _box_iou(g.box, roi) <= 1e-7:
                gt_outside += 1
            if not _box_inside(g.box, roi):
                gt_not_fully_inside += 1

    metric_counts = {
        "tracked_frames_skipped_frac": total_frames,
        "tracked_frames_frac": total_frames,
        "average_inference_roi_area": len(roi_areas),
        "gt_outside_inference_roi_frac": gt_eval,
        "gt_not_fully_inside_inference_roi_frac": gt_eval,
    }
    return _module_result(
        metrics={
            "tracked_frames_skipped_frac": (skipped / total_frames) if total_frames > 0 else None,
            "tracked_frames_frac": (tracked / total_frames) if total_frames > 0 else None,
            "average_inference_roi_area": float(np.mean(roi_areas)) if len(roi_areas) > 0 else None,
            "gt_outside_inference_roi_frac": (gt_outside / gt_eval) if gt_eval > 0 else None,
            "gt_not_fully_inside_inference_roi_frac": (gt_not_fully_inside / gt_eval) if gt_eval > 0 else None,
        },
        counts={
            "total_frames": total_frames,
            "tracked_frames": tracked,
            "skipped_frames": skipped,
            "gt_eval_boxes": gt_eval,
        },
        metric_counts=metric_counts,
    )


def evaluate_pair_logger(
    sequence_ctx: Dict[str, Any], variant_impl: Any, module_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Phase 1 of the NNUE-style learned scoring effort (UTRACK_NN.md).

    Reads `frame["debug"]["match_cost_trace"]` (populated by the C side
    when `utrack.debug_match_cost_trace` is true), labels each (track,
    det) pair via best-IoU GT alignment on both sides, and writes a
    per-sequence NPZ to `module_params.output_dir`.

    Variant-independent: trace contents depend only on the tracker config
    used at generate time, not on Python-side variants.

    Required module_params:
        output_dir: str  — where to write <seq>.npz
    Optional:
        seed_match_iou:    float = 0.5  — track-side GT IoU threshold
        det_match_iou:     float = 0.5  — det-side GT IoU threshold
        class_name:        str   = "person"
        require_any_trace: bool  = True — fail loudly if no frame has a
                                          trace (catches forgotten flag)
    """
    import os
    import re
    from src.analysis.pair_log_schema import (
        PAIR_LOG_DTYPE,
        PAIR_LOG_FEATURE_NAMES,
        PAIR_LOG_MAGIC,
        PAIR_LOG_VERSION,
        decode_records,
        record_size_bytes,
    )

    output_dir = module_params.get("output_dir")
    if not output_dir:
        raise ValueError("pair_logger: module_params.output_dir is required")
    seed_match_iou = float(module_params.get("seed_match_iou", 0.5))
    det_match_iou = float(module_params.get("det_match_iou", 0.5))
    class_name = str(module_params.get("class_name", "person"))
    require_any_trace = bool(module_params.get("require_any_trace", True))

    gt = sequence_ctx["gt_trackset"]
    run = sequence_ctx["run_trackset"]
    frames = run.frames

    # Walk frames in order, building a per-track GT-id history along the
    # way. The GT id of a track is whatever GT object its tracked box
    # most recently matched above seed_match_iou. Tracks that never
    # match GT this run never end up in track_gt_history; their pairs
    # get dropped.
    track_gt_history: Dict[Any, int] = {}

    record_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []

    n_frames_with_trace = 0
    n_frames_with_records = 0
    n_dropped_no_gt_track = 0
    n_dropped_no_gt_det = 0
    n_pos = 0
    n_neg = 0

    for i, frame in enumerate(frames):
        t = float(frame.get("frame_time", 0.0))
        gt_curr = _gt_objects_at_time_class(sequence_ctx, t, class_name)

        # Update track→GT history from this frame's tracked outputs.
        # Only covers TRACKED (emitted) tracks; UNCONFIRMED/LOST refresh
        # happens below from the pair-trace records.
        if frame.get("objects"):
            for track_id, _obj, box in _run_objects_for_frame_class(
                sequence_ctx, i, frame, class_name
            ):
                best_gt, best_iou = _best_iou_match(box, gt_curr)
                if best_gt is not None and best_iou >= seed_match_iou:
                    track_gt_history[track_id] = int(best_gt.track_id)

        # Pull this frame's pair trace, if any.
        debug = frame.get("debug") or {}
        trace_blob = debug.get("match_cost_trace")
        if not isinstance(trace_blob, dict):
            continue

        n_frames_with_trace += 1

        magic = int(trace_blob.get("magic", 0))
        version = int(trace_blob.get("version", 0))
        if magic != PAIR_LOG_MAGIC:
            raise ValueError(
                f"pair_logger: trace magic mismatch in frame {i}: "
                f"got 0x{magic:08x}, expected 0x{PAIR_LOG_MAGIC:08x}"
            )
        if version != PAIR_LOG_VERSION:
            raise ValueError(
                f"pair_logger: trace version mismatch in frame {i}: "
                f"got {version}, expected {PAIR_LOG_VERSION}"
            )

        record_size = int(trace_blob.get("record_size", 0))
        if record_size != record_size_bytes():
            raise ValueError(
                f"pair_logger: trace record_size {record_size} != "
                f"schema {record_size_bytes()} (frame {i})"
            )

        n_records = int(trace_blob.get("n_records", 0))
        data = trace_blob.get("data") or b""
        if n_records == 0 or len(data) == 0:
            continue
        records = decode_records(data, n_records)
        n_frames_with_records += 1

        # Refresh track_gt_history from this frame's pair-trace track
        # boxes BEFORE labelling. Covers tracks the per-frame
        # objects-update missed (UNCONFIRMED/LOST never emit objects)
        # and overwrites stale cache entries when a track has drifted to
        # a different GT identity. Walks distinct track_ids only — most
        # frames have many records per track.
        seen_tids: Dict[int, Tuple[float, float, float, float]] = {}
        for r_idx in range(int(records.shape[0])):
            rec = records[r_idx]
            tid = int(rec["track_id"])
            if tid not in seen_tids:
                seen_tids[tid] = (
                    float(rec["track_x0"]),
                    float(rec["track_y0"]),
                    float(rec["track_x1"]),
                    float(rec["track_y1"]),
                )
        for tid, track_box in seen_tids.items():
            best_gt, best_iou = _best_iou_match(list(track_box), gt_curr)
            if best_gt is not None and best_iou >= seed_match_iou:
                track_gt_history[tid] = int(best_gt.track_id)

        # Vectorised labelling: build numpy arrays of track and det
        # boxes for the whole frame, then loop only to do per-pair GT
        # lookups (which are O(|gt|) and cheap in practice).
        kept_indices: List[int] = []
        kept_labels: List[int] = []
        for r_idx in range(int(records.shape[0])):
            rec = records[r_idx]

            track_id_int: int = int(rec["track_id"])
            track_gt_id = track_gt_history.get(track_id_int)
            if track_gt_id is None:
                n_dropped_no_gt_track += 1
                continue

            det_box = [
                float(rec["det_x0"]),
                float(rec["det_y0"]),
                float(rec["det_x1"]),
                float(rec["det_y1"]),
            ]
            det_best_gt, det_best_iou = _best_iou_match(det_box, gt_curr)
            if det_best_gt is None or det_best_iou < det_match_iou:
                n_dropped_no_gt_det += 1
                continue
            det_gt_id = int(det_best_gt.track_id)

            label = 1 if det_gt_id == track_gt_id else 0
            kept_indices.append(r_idx)
            kept_labels.append(label)
            if label:
                n_pos += 1
            else:
                n_neg += 1

        if kept_indices:
            record_chunks.append(records[np.asarray(kept_indices, dtype=np.int64)])
            label_chunks.append(np.asarray(kept_labels, dtype=np.uint8))

    if require_any_trace and n_frames_with_trace == 0:
        raise RuntimeError(
            "pair_logger: no frame in this sequence carried a "
            "match_cost_trace debug payload. Check that the tracker "
            "config has `utrack.debug_match_cost_trace: true` and that "
            "the run was generated with that config (force_regen: true "
            "may be needed to invalidate the UBTRK2 cache)."
        )

    # Sort records deterministically by (frame_idx, track_id, ...) so
    # repeated runs produce byte-identical NPZ files (Phase-1 gate).
    if record_chunks:
        records_arr = np.concatenate(record_chunks)
        labels_arr = np.concatenate(label_chunks)
        sort_key = np.lexsort((
            records_arr["det_x0"],
            records_arr["track_id"],
            records_arr["pass_id"],
            records_arr["frame_time"],
        ))
        records_arr = records_arr[sort_key]
        labels_arr = labels_arr[sort_key]
    else:
        records_arr = np.zeros(0, dtype=PAIR_LOG_DTYPE)
        labels_arr = np.zeros(0, dtype=np.uint8)

    seq_name = str(sequence_ctx.get("sequence_name", "unknown"))
    seq_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", seq_name).strip("_") or "unnamed"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{seq_safe}.npz")
    tmp_path = out_path + ".tmp"
    np.savez(
        tmp_path,
        records=records_arr,
        labels=labels_arr,
        feature_names=np.array(PAIR_LOG_FEATURE_NAMES, dtype=object),
        schema_version=np.int64(PAIR_LOG_VERSION),
        schema_magic=np.uint32(PAIR_LOG_MAGIC),
    )
    # np.savez writes "<tmp>.npz" if path lacks the suffix; we passed
    # tmp+.tmp so the actual file is tmp+.tmp.npz. Normalise.
    real_tmp = tmp_path if os.path.exists(tmp_path) else tmp_path + ".npz"
    os.replace(real_tmp, out_path)

    n_total = int(records_arr.shape[0])
    return _module_result(
        metrics={
            "n_pairs": float(n_total),
            "n_positives": float(n_pos),
            "n_negatives": float(n_neg),
            "positive_rate": (float(n_pos) / float(n_total)) if n_total > 0 else None,
            "frames_with_trace": float(n_frames_with_trace),
            "frames_with_records": float(n_frames_with_records),
        },
        counts={
            "n_pairs": n_total,
            "n_positives": n_pos,
            "n_negatives": n_neg,
            "n_dropped_no_gt_track": n_dropped_no_gt_track,
            "n_dropped_no_gt_det": n_dropped_no_gt_det,
            "n_frames_with_trace": n_frames_with_trace,
            "n_frames_with_records": n_frames_with_records,
            "n_frames_total": len(frames),
        },
        metric_counts={
            "positive_rate": n_total,
        },
    )


_MODULE_REGISTRY: Dict[str, Callable[[Dict[str, Any], Any, Dict[str, Any]], Dict[str, Any]]] = {
    "detection": evaluate_detection,
    "optical_flow": evaluate_optical_flow,
    "kalman": evaluate_kalman,
    "reid": evaluate_reid,
    "match_cost": evaluate_match_cost,
    "roi_skip": evaluate_roi_skip,
    "pair_logger": evaluate_pair_logger,
}


def list_registered_modules() -> List[str]:
    return sorted(_MODULE_REGISTRY.keys())


def run_module(
    module_name: str,
    sequence_ctx: Dict[str, Any],
    variant_impl: Any,
    module_params: Dict[str, Any],
) -> Dict[str, Any]:
    fn = _MODULE_REGISTRY.get(module_name)
    if fn is None:
        raise ValueError(
            f"Unknown module {module_name!r}; registered modules: {list_registered_modules()}"
        )
    return fn(sequence_ctx, variant_impl, module_params)
