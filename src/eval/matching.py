"""Box matching for eval: MOT object rows, convention-permissive IoU
(visible vs fullbody GT), ignore-region containment.

Moved verbatim from src/track_test.py (repo_cleanup.md stage 4a).
"""
import numpy as np



def mot_obj(obj, w, h):
    ol=int(obj.box[0]*w)
    ot=int(obj.box[1]*h)
    ow=int((obj.box[2]-obj.box[0])*w)
    oh=int((obj.box[3]-obj.box[1])*h)
    return [obj.track_id, ol, ot, ow, oh]



def permissive_iou_matrix(gt_boxes, test_boxes, gt_convention=None,
                          max_aspect=3.5, frame_wh=None):
    """Convention-permissive pairwise IoU between two box lists.

    GT corpora use two box conventions — 'fullbody' (amodal, includes the
    occluded lower body) and 'visible' (visible extent only) — and the
    tracker's own convention is mixed. The difference is almost purely
    VERTICAL extent. DIRECTIONAL (ledger 2026-07-24 Convention-permissive matching): only the mismatch the
    other legitimate convention would produce is forgiven —
      gt_convention == "visible":  max(iou, iou(g, t_vclip))  — a TALLER
        tracker box (legitimate fullbody emission) matches; a shorter one
        stays penalized.
      gt_convention == "fullbody": max(iou, iou(g_vclip, t)) — a SHORTER
        tracker box (legitimate visible emission) matches; over-expansion
        stays penalized.
      gt_convention None/other: symmetric (both directions) — legacy.
    A symmetric version forgave generic vertical sloppiness everywhere
    (fullbody corpora gained +0.02-0.09 with zero convention mismatch in
    the A/B in that ledger entry), which would let the optimizer drift box extents
    without objective pressure; the directional form keeps geometry
    honest on every axis except the one where the labels genuinely
    disagree. x-axis discrimination is untouched in all modes.

    gt_boxes: (n,4), test_boxes: (m,4), boxes [x1,y1,x2,y2] (any
    consistent units — IoU is invariant to per-axis scaling). Returns an
    (n,m) float array. Fully vectorized: the vclip'd box's height equals
    the pair's y-overlap and its intersection with the other box is the
    plain intersection, so no per-pair python work is needed."""
    G = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    T = np.asarray(test_boxes, dtype=float).reshape(-1, 4)
    gx1, gy1, gx2, gy2 = (G[:, None, i] for i in range(4))
    tx1, ty1, tx2, ty2 = (T[None, :, i] for i in range(4))
    ix = np.maximum(0.0, np.minimum(gx2, tx2) - np.maximum(gx1, tx1))
    iy = np.maximum(0.0, np.minimum(gy2, ty2) - np.maximum(gy1, ty1))
    inter = ix * iy
    gw = gx2 - gx1
    tw = tx2 - tx1
    area_g = gw * (gy2 - gy1)
    area_t = tw * (ty2 - ty1)
    # g clipped to t's y-range has area gw*iy (and vice versa); the
    # clipped box's intersection with the other box is still `inter`.
    eps = 1e-12
    iou = inter / np.maximum(area_g + area_t - inter, eps)
    iou_gclip = inter / np.maximum(gw * iy + area_t - inter, eps)
    iou_tclip = inter / np.maximum(area_g + tw * iy - inter, eps)
    # plausibility bound (ledger 2026-07-24 Convention-permissive matching): the forgiven (taller) box must
    # itself stay within a typical standing-human aspect ratio — beyond
    # h <= max_aspect*w the extra height is not a legitimate fullbody
    # interpretation, just a bad box, and plain IoU applies.
    gh = gy2 - gy1
    th = ty2 - ty1
    if frame_wh is not None and max_aspect is not None:
        # boxes are normalized: pixel aspect = (h*H)/(w*W); cap in pixel
        # space so the "typical standing human" bound is frame-shape
        # independent
        scale = float(frame_wh[1]) / max(float(frame_wh[0]), 1.0)
        cap = max_aspect / scale
        t_plaus = th <= cap * np.maximum(tw, eps)
        g_plaus = gh <= cap * np.maximum(gw, eps)
    else:
        t_plaus = np.ones_like(th, dtype=bool)
        g_plaus = np.ones_like(gh, dtype=bool)
    if gt_convention == "visible":
        return np.maximum(iou, np.where(t_plaus, iou_tclip, iou))
    if gt_convention == "fullbody":
        return np.maximum(iou, np.where(g_plaus, iou_gclip, iou))
    both = np.maximum(np.where(g_plaus, iou_gclip, iou),
                      np.where(t_plaus, iou_tclip, iou))
    return np.maximum(iou, both)



def permissive_iou(box_a, box_b, gt_convention=None, max_aspect=3.5,
                   frame_wh=None):
    """Scalar convention-permissive IoU for one [x1,y1,x2,y2] box pair
    (box_a is GT; see permissive_iou_matrix for directionality)."""
    return float(permissive_iou_matrix([box_a], [box_b],
                                       gt_convention=gt_convention,
                                       max_aspect=max_aspect,
                                       frame_wh=frame_wh)[0, 0])



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
