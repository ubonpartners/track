from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, List, Sequence

import numpy as np


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _flow_delta_nearest(flow: np.ndarray, x: float, y: float) -> List[float]:
    grid_h, grid_w = int(flow.shape[0]), int(flow.shape[1])
    ix = min(grid_w - 1, max(0, int(x * grid_w + 0.5)))
    iy = min(grid_h - 1, max(0, int(y * grid_h + 0.5)))
    return [float(flow[iy, ix, 0]), float(flow[iy, ix, 1])]


def _flow_delta_bilinear(flow: np.ndarray, x: float, y: float) -> List[float]:
    grid_h, grid_w = int(flow.shape[0]), int(flow.shape[1])
    if grid_w <= 1 or grid_h <= 1:
        return _flow_delta_nearest(flow, x, y)

    ux = _clip01(x) * (grid_w - 1)
    uy = _clip01(y) * (grid_h - 1)
    x0 = int(np.floor(ux))
    y0 = int(np.floor(uy))
    x1 = min(grid_w - 1, x0 + 1)
    y1 = min(grid_h - 1, y0 + 1)
    tx = ux - x0
    ty = uy - y0

    f00 = flow[y0, x0]
    f01 = flow[y0, x1]
    f10 = flow[y1, x0]
    f11 = flow[y1, x1]

    top = (1.0 - tx) * f00 + tx * f01
    bot = (1.0 - tx) * f10 + tx * f11
    f = (1.0 - ty) * top + ty * bot
    return [float(f[0]), float(f[1])]


def _flow_delta_bilinear_centered(flow: np.ndarray, x: float, y: float) -> List[float]:
    """
    Bilinear flow sampling with cell-centered coordinate mapping.

    Map normalized coords to cell-center index space:
      u = x * grid_w - 0.5, v = y * grid_h - 0.5
    This matches the C-side centered interpretation of OF vectors.
    """
    grid_h, grid_w = int(flow.shape[0]), int(flow.shape[1])
    if grid_w <= 1 or grid_h <= 1:
        return _flow_delta_nearest(flow, x, y)

    fx = _clip01(x) * grid_w - 0.5
    fy = _clip01(y) * grid_h - 0.5
    x0 = int(np.floor(fx))
    y0 = int(np.floor(fy))
    tx = fx - x0
    ty = fy - y0

    if x0 < 0:
        x0 = 0
        tx = 0.0
    if y0 < 0:
        y0 = 0
        ty = 0.0
    if x0 >= grid_w - 1:
        x0 = grid_w - 1
        tx = 0.0
    if y0 >= grid_h - 1:
        y0 = grid_h - 1
        ty = 0.0

    x1 = min(grid_w - 1, x0 + 1)
    y1 = min(grid_h - 1, y0 + 1)

    f00 = flow[y0, x0]
    f01 = flow[y0, x1]
    f10 = flow[y1, x0]
    f11 = flow[y1, x1]

    top = (1.0 - tx) * f00 + tx * f01
    bot = (1.0 - tx) * f10 + tx * f11
    f = (1.0 - ty) * top + ty * bot
    return [float(f[0]), float(f[1])]


def _resolve_bilerp_sampler(
    module_params: Dict[str, Any],
) -> Callable[[np.ndarray, float, float], List[float]]:
    centering = str(module_params.get("bilerp_centering", "edge")).strip().lower()
    if centering in {"cell", "center", "centered", "cell_center", "cell_centered"}:
        return _flow_delta_bilinear_centered
    return _flow_delta_bilinear


def _predict_of_box(
    box: Sequence[float],
    flow: np.ndarray,
    *,
    corrected_sampling: bool,
    flow_sampler: Callable[[np.ndarray, float, float], List[float]],
    samples: Sequence[Sequence[float]] | None = None,
    flow_gain: float = 1.0,
) -> List[float]:
    # Mirrors motion_track_predict_box_inplace in ubon_cstuff.
    x0, y0, x1, y1 = [float(v) for v in box]
    if samples is None:
        samples = [
            (0.5, 0.5, 0.5),
            (0.35, 0.5, 0.125),
            (0.65, 0.5, 0.125),
            (0.5, 0.35, 0.125),
            (0.5, 0.65, 0.125),
        ]
    dx = 0.0
    dy = 0.0
    for xf, yf, w in samples:
        if corrected_sampling:
            px = x0 * (1.0 - xf) + x1 * xf
            py = y0 * (1.0 - yf) + y1 * yf
        else:
            px = x0 * xf + x1 * (1.0 - xf)
            py = y0 * yf + y1 * (1.0 - yf)
        ddx, ddy = flow_sampler(flow, px, py)
        dx += w * ddx
        dy += w * ddy
    return [
        _clip01(x0 - flow_gain * dx),
        _clip01(y0 - flow_gain * dy),
        _clip01(x1 - flow_gain * dx),
        _clip01(y1 - flow_gain * dy),
    ]


_DEFAULT_FIXED5_SAMPLES = [
    (0.5, 0.5, 0.5),
    (0.35, 0.5, 0.125),
    (0.65, 0.5, 0.125),
    (0.5, 0.35, 0.125),
    (0.5, 0.65, 0.125),
]


def _parse_pose_points(obj: Dict[str, Any]) -> Dict[int, Sequence[float]]:
    pose = obj.get("pose_points")
    if not isinstance(pose, list):
        return {}
    out: Dict[int, Sequence[float]] = {}
    n = len(pose) // 3
    for i in range(n):
        try:
            x = float(pose[3 * i + 0])
            y = float(pose[3 * i + 1])
            c = float(pose[3 * i + 2])
        except Exception:
            continue
        out[i] = (x, y, c)
    return out


def _predict_from_pose_points(
    obj: Dict[str, Any],
    box: Sequence[float],
    flow: np.ndarray,
    *,
    required_indices: Sequence[int],
    conf_thr: float,
    flow_gain: float,
    flow_sampler: Callable[[np.ndarray, float, float], List[float]],
) -> List[float]:
    pose_map = _parse_pose_points(obj)
    x0, y0, x1, y1 = [float(v) for v in box]
    selected = []
    for idx in required_indices:
        p = pose_map.get(idx)
        if p is None:
            return []
        px, py, pc = p
        if pc <= conf_thr:
            return []
        selected.append((px, py))
    if len(selected) == 0:
        return []

    dx = 0.0
    dy = 0.0
    w = 1.0 / len(selected)
    for px, py in selected:
        ddx, ddy = flow_sampler(flow, px, py)
        dx += w * ddx
        dy += w * ddy
    return [
        _clip01(x0 - flow_gain * dx),
        _clip01(y0 - flow_gain * dy),
        _clip01(x1 - flow_gain * dx),
        _clip01(y1 - flow_gain * dy),
    ]


@dataclass
class VariantImplementation:
    name: str
    impl: str
    module_params: Dict[str, Any]

    def predict_of(self, box: Sequence[float], flow: np.ndarray) -> List[float]:
        return _predict_of_box(
            box,
            flow,
            corrected_sampling=False,
            flow_sampler=_flow_delta_nearest,
            samples=_DEFAULT_FIXED5_SAMPLES,
        )

    def predict_of_for_object(
        self, obj: Dict[str, Any], box: Sequence[float], flow: np.ndarray
    ) -> List[float]:
        _ = obj
        return self.predict_of(box, flow)

    def normalize_reid(
        self, query_vec: np.ndarray, candidate_vecs: Sequence[np.ndarray]
    ) -> tuple[np.ndarray, List[np.ndarray]]:
        if len(candidate_vecs) == 0:
            q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
            q_norm = np.linalg.norm(q)
            if q_norm < 1e-9:
                return q * 0.0, []
            return q / q_norm, []

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

        qn = mean_l2(vectors[0])
        cn = [mean_l2(v) for v in vectors[1:]]
        return qn, cn

    def reid_similarity(self, query_vec_norm: np.ndarray, candidate_vec_norm: np.ndarray) -> float:
        return float(np.dot(query_vec_norm, candidate_vec_norm))

    def combine_match_score(
        self,
        of_score: float,
        reid_similarity: float,
        detection_confidence: float,
        module_params: Dict[str, Any],
    ) -> float:
        mode = str(module_params.get("combine_mode", "add")).strip().lower()
        sim_weight = float(module_params.get("sim_weight", 0.2))
        fuse_scores = float(module_params.get("fuse_scores", 0.94))
        match_thr = float(module_params.get("match_thr", 0.0))
        det_conf = max(0.0, float(detection_confidence))
        of_clip = _clip01(float(of_score))
        reid = float(reid_similarity)
        sim01 = _clip01(0.5 * (reid + 1.0))

        if mode == "add":
            raw = of_clip + sim_weight * reid
        elif mode == "hinge":
            reid_thr = float(module_params.get("reid_hinge_thr", 0.0))
            raw = of_clip + sim_weight * max(0.0, reid - reid_thr)
        elif mode == "gate":
            alpha = float(module_params.get("reid_gate_alpha", 0.5))
            alpha = max(0.0, min(1.0, alpha))
            raw = of_clip * ((1.0 - alpha) + alpha * sim01)
        elif mode == "adaptive_gate":
            alpha = float(module_params.get("reid_gate_alpha", 0.5))
            alpha_scale = float(module_params.get("adaptive_alpha_motion_scale", 0.35))
            alpha_min = float(module_params.get("adaptive_alpha_min", 0.0))
            alpha_max = float(module_params.get("adaptive_alpha_max", 0.95))
            alpha_eff = alpha + alpha_scale * (1.0 - of_clip)
            alpha_eff = max(alpha_min, min(alpha_max, alpha_eff))
            raw = of_clip * ((1.0 - alpha_eff) + alpha_eff * sim01)
        elif mode == "geom":
            of_w = float(module_params.get("geom_of_weight", 1.0))
            reid_w = float(module_params.get("geom_reid_weight", 1.0))
            raw = (of_clip ** of_w) * (sim01 ** reid_w)
        elif mode == "logit_blend":
            logit_of_w = float(module_params.get("logit_of_weight", 1.0))
            logit_reid_w = float(module_params.get("logit_reid_weight", 1.0))
            logit_bias = float(module_params.get("logit_bias", 0.0))
            eps = 1e-6
            of_p = max(eps, min(1.0 - eps, of_clip))
            reid_p = max(eps, min(1.0 - eps, sim01))
            of_logit = math.log(of_p / (1.0 - of_p))
            reid_logit = math.log(reid_p / (1.0 - reid_p))
            mix = logit_of_w * of_logit + logit_reid_w * reid_logit + logit_bias
            raw = 1.0 / (1.0 + math.exp(-mix))
        elif mode == "centered_add":
            raw = of_clip + sim_weight * reid
        elif mode == "harmonic":
            beta = float(module_params.get("harmonic_beta", 0.5))
            beta = max(1e-6, min(1.0 - 1e-6, beta))
            eps = 1e-6
            raw = 1.0 / (beta / (of_clip + eps) + (1.0 - beta) / (sim01 + eps))
        else:
            raw = of_clip + sim_weight * reid

        if raw < match_thr:
            return 0.0
        return raw * (det_conf ** fuse_scores)


class BaselineVariant(VariantImplementation):
    pass


class OpticalFlowPrototypeV1(VariantImplementation):
    """Example OF prototype variant using bilinear flow sampling."""

    def predict_of(self, box: Sequence[float], flow: np.ndarray) -> List[float]:
        flow_gain = float(self.module_params.get("flow_gain", 1.0))
        proto_samples = self.module_params.get(
            "samples",
            [
                (0.5, 0.5, 0.4),
                (0.25, 0.5, 0.15),
                (0.75, 0.5, 0.15),
                (0.5, 0.25, 0.15),
                (0.5, 0.75, 0.15),
            ],
        )
        return _predict_of_box(
            box,
            flow,
            corrected_sampling=False,
            flow_sampler=_resolve_bilerp_sampler(self.module_params),
            samples=proto_samples,
            flow_gain=flow_gain,
        )


class OpticalFlowPosePointsV1(VariantImplementation):
    """Use pose anchors (nose/shoulders/hips) for OF translation when available."""

    _POSE_REQUIRED_INDICES = (0, 5, 6, 11, 12)  # nose, shoulders, hips

    def predict_of_for_object(
        self, obj: Dict[str, Any], box: Sequence[float], flow: np.ndarray
    ) -> List[float]:
        pred = _predict_from_pose_points(
            obj,
            box,
            flow,
            required_indices=self._POSE_REQUIRED_INDICES,
            conf_thr=float(self.module_params.get("pose_conf_thr", 0.2)),
            flow_gain=float(self.module_params.get("flow_gain", 1.0)),
            flow_sampler=_resolve_bilerp_sampler(self.module_params),
        )
        if len(pred) == 4:
            return pred
        # fallback to baseline behavior when required pose anchors are unavailable
        return _predict_of_box(
            box,
            flow,
            corrected_sampling=False,
            flow_sampler=_flow_delta_nearest,
            samples=_DEFAULT_FIXED5_SAMPLES,
        )


class OrthogonalOFVariant(VariantImplementation):
    """
    Orthogonal OF controls:
    - point_selector: fixed5 | pose5 | pose_or_fixed5 | pose5_strict
    - flow_sampler: nearest | bilerp | bilerp_centered
    - bilerp_centering: edge | cell_centered  (used when flow_sampler=bilerp)
    """

    _POSE_REQUIRED_INDICES = (0, 5, 6, 11, 12)

    def _sampler(self) -> Callable[[np.ndarray, float, float], List[float]]:
        mode = str(self.module_params.get("flow_sampler", "nearest")).strip().lower()
        if mode in {"bilerp_centered", "bilinear_centered", "bilerp_cell", "bilinear_cell"}:
            return _flow_delta_bilinear_centered
        if mode in {"bilerp", "bilinear"}:
            return _resolve_bilerp_sampler(self.module_params)
        return _flow_delta_nearest

    def _fixed_predict(self, box: Sequence[float], flow: np.ndarray) -> List[float]:
        return _predict_of_box(
            box,
            flow,
            corrected_sampling=False,
            flow_sampler=self._sampler(),
            samples=_DEFAULT_FIXED5_SAMPLES,
            flow_gain=float(self.module_params.get("flow_gain", 1.0)),
        )

    def predict_of(self, box: Sequence[float], flow: np.ndarray) -> List[float]:
        return self._fixed_predict(box, flow)

    def predict_of_for_object(
        self, obj: Dict[str, Any], box: Sequence[float], flow: np.ndarray
    ) -> List[float]:
        selector = str(self.module_params.get("point_selector", "fixed5")).strip().lower()
        if selector in {"fixed5", "fixed"}:
            return self._fixed_predict(box, flow)

        pose_pred = _predict_from_pose_points(
            obj,
            box,
            flow,
            required_indices=self._POSE_REQUIRED_INDICES,
            conf_thr=float(self.module_params.get("pose_conf_thr", 0.2)),
            flow_gain=float(self.module_params.get("flow_gain", 1.0)),
            flow_sampler=self._sampler(),
        )
        if len(pose_pred) == 4:
            return pose_pred

        if selector in {
            "pose5",
            "pose",
            "pose_or_fixed5",
            "pose_or_fixed",
            "hybrid",
        }:
            return self._fixed_predict(box, flow)
        # strict pose mode with missing anchors => no prediction (opt-in only)
        return []


_VARIANT_REGISTRY = {
    "baseline": BaselineVariant,
    "of_proto_v1": OpticalFlowPrototypeV1,
    "of_pose_points_v1": OpticalFlowPosePointsV1,
    "of_orthogonal_v1": OrthogonalOFVariant,
}


def list_registered_variants() -> List[str]:
    return sorted(_VARIANT_REGISTRY.keys())


def create_variant(variant_spec: Dict[str, Any]) -> VariantImplementation:
    impl = str(variant_spec.get("impl", "baseline"))
    variant_name = str(variant_spec.get("name", impl))
    cls = _VARIANT_REGISTRY.get(impl)
    if cls is None:
        raise ValueError(
            f"Unknown variant impl={impl!r}; registered variants: {list_registered_variants()}"
        )
    return cls(
        name=variant_name,
        impl=impl,
        module_params=dict(variant_spec.get("module_params") or {}),
    )
