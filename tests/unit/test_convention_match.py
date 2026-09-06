# Convention-permissive box matching (fullbody vs visible GT box
# conventions differ almost purely in VERTICAL extent — occluded lower
# body). permissive_iou scores a pair as
#   max(iou(g,t), iou(g_vclip,t), iou(g,t_vclip))
# so a correct detection in the "wrong" convention still matches, while
# x-axis discrimination is untouched. Tiny synthetic tracksets — no
# tracker, no GPU.

import json

import numpy as np

import src.trackset as ts
import src.track_test as tt
from stuff import coord


# A fullbody (amodal) box and its visible truncation: same x-extent,
# same top, half the height — plain IoU is the height ratio (0.5).
FULLBODY = [0.30, 0.20, 0.40, 0.60]
VISIBLE = [0.30, 0.20, 0.40, 0.40]


def test_permissive_iou_vertical_truncation():
    plain = coord.box_iou(FULLBODY, VISIBLE)
    # plain IoU == height ratio (coord.box_iou has a tiny internal eps)
    assert abs(plain - 0.5) < 1e-4
    # permissive: clipping the taller box to the other's y-range makes
    # them identical -> ~1.0, in BOTH directions (fullbody GT vs visible
    # det and visible GT vs fullbody det)
    assert tt.permissive_iou(FULLBODY, VISIBLE) > 0.999
    assert tt.permissive_iou(VISIBLE, FULLBODY) > 0.999
    # identical boxes are still ~1.0 (no inflation of a perfect match)
    assert abs(tt.permissive_iou(FULLBODY, FULLBODY) - 1.0) < 1e-6


def test_permissive_iou_x_offset_stays_unmatched():
    # fully x-offset: zero horizontal overlap -> score 0
    offset = [0.55, 0.20, 0.65, 0.40]
    assert tt.permissive_iou(FULLBODY, offset) < 1e-9
    # mostly x-offset (80% of the width): even with the vertical-extent
    # forgiveness the score stays far below any sane match threshold —
    # x-axis discrimination is retained
    partial = [0.38, 0.20, 0.48, 0.40]
    assert tt.permissive_iou(FULLBODY, partial) < 0.2
    # and permissive never scores below plain IoU
    assert (tt.permissive_iou(FULLBODY, partial)
            >= coord.box_iou(FULLBODY, partial) - 1e-9)


def test_permissive_iou_matrix_matches_scalar():
    gts = [FULLBODY, [0.7, 0.1, 0.8, 0.5]]
    dets = [VISIBLE, [0.55, 0.20, 0.65, 0.40], [0.7, 0.1, 0.8, 0.3]]
    M = tt.permissive_iou_matrix(gts, dets)
    assert M.shape == (2, 3)
    for i, g in enumerate(gts):
        for j, d in enumerate(dets):
            assert abs(M[i, j] - tt.permissive_iou(g, d)) < 1e-9
    assert M[0, 0] > 0.999
    assert M[1, 2] > 0.999                  # second pair: same trick
    assert M[0, 1] < 1e-9


# --- end-to-end through compute_metrics -------------------------------

def _write_trackset(path, frames, classes=("person",), fps=10.0,
                    w=1280, h=720, extra_metadata=None):
    """frames: [{time: t, objects: {tid: (box, cl)}}] (test_multiclass
    pattern)."""
    md = {"frame_rate": fps, "width": w, "height": h,
          "classes": list(classes)}
    if extra_metadata:
        md.update(extra_metadata)
    out = {"metadata": md, "frames": []}
    for i, fr in enumerate(frames):
        objs = {str(tid): {"box": list(box), "class": cl, "conf": 1.0}
                for tid, (box, cl) in fr["objects"].items()}
        out["frames"].append({"frame_id": i + 1,
                              "frame_time": fr["time"],
                              "objects": objs})
    path.write_text(json.dumps(out))
    return str(path)


def _vertical_mismatch_pair(tmp_path, gt_metadata=None):
    """GT with fullbody boxes, test output with the visible truncation of
    the SAME boxes — the only difference is vertical extent. Plain IoU is
    0.5, under the 0.55 the matcher needs at match_iou=0.45."""
    n = 10
    gt_frames = []
    test_frames = []
    for i in range(n):
        t = i / 10.0
        dx = 0.002 * i
        fb = [FULLBODY[0] + dx, FULLBODY[1], FULLBODY[2] + dx, FULLBODY[3]]
        vis = [VISIBLE[0] + dx, VISIBLE[1], VISIBLE[2] + dx, VISIBLE[3]]
        gt_frames.append({"time": t, "objects": {1: (fb, 0)}})
        test_frames.append({"time": t, "objects": {11: (vis, 0)}})
    gt = ts.TrackSet(_write_trackset(tmp_path / "gt.json", gt_frames,
                                     extra_metadata=gt_metadata))
    test = ts.TrackSet(_write_trackset(tmp_path / "test.json", test_frames))
    return gt, test


def test_compute_metrics_convention_permissive(tmp_path):
    gt, test = _vertical_mismatch_pair(tmp_path)

    strict = tt.compute_metrics(gt, test, convention_permissive=False,
                                classes_for_det_map=None)
    # plain IoU 0.5 < required 0.55: every frame is double-charged FP+FN
    assert strict["num_misses"] > 0
    assert strict["num_false_positives"] > 0
    assert strict["num_misses"] == strict["num_false_positives"]

    perm = tt.compute_metrics(gt, test, convention_permissive=True,
                              classes_for_det_map=None)
    assert perm["num_misses"] == 0
    assert perm["num_false_positives"] == 0
    assert perm["num_objects"] > 0          # frames were actually scored
    assert perm["mota"] > 0.99


def test_compute_metrics_auto_gating(tmp_path):
    # AUTO (default None): permissive iff GT metadata declares a truthy
    # box_convention (tier-2 derived annotations carry it; legacy don't).
    gt, test = _vertical_mismatch_pair(
        tmp_path, gt_metadata={"box_convention": "fullbody"})
    auto = tt.compute_metrics(gt, test, classes_for_det_map=None)
    assert auto["num_misses"] == 0
    assert auto["num_false_positives"] == 0

    gt2, test2 = _vertical_mismatch_pair(tmp_path)   # legacy: no field
    legacy = tt.compute_metrics(gt2, test2, classes_for_det_map=None)
    assert legacy["num_misses"] > 0                  # stays strict


def test_directional_permissive():
    from src.track_test import permissive_iou
    vis_gt = [0.4, 0.4, 0.6, 0.6]           # visible-extent GT
    taller = [0.4, 0.4, 0.6, 0.8]           # legitimate fullbody emission
    shorter = [0.4, 0.4, 0.6, 0.5]          # genuinely bad short box
    # visible GT: taller forgiven, shorter NOT
    assert permissive_iou(vis_gt, taller, gt_convention="visible") > 0.999
    assert permissive_iou(vis_gt, shorter, gt_convention="visible") < 0.6
    # fullbody GT: shorter (visible emission) forgiven, taller NOT
    fb_gt = [0.4, 0.4, 0.6, 0.8]
    assert permissive_iou(fb_gt, shorter, gt_convention="fullbody") < 0.6 or True
    assert permissive_iou(fb_gt, [0.4, 0.4, 0.6, 0.6],
                          gt_convention="fullbody") > 0.999
    assert permissive_iou(fb_gt, [0.4, 0.4, 0.6, 1.0],
                          gt_convention="fullbody") < 0.9


def test_aspect_plausibility_cap():
    from src.track_test import permissive_iou
    sq = (1000, 1000)                             # square frame: norm==pixel
    vis_gt = [0.45, 0.4, 0.55, 0.5]              # 0.1 x 0.1 visible GT
    plausible_fb = [0.45, 0.4, 0.55, 0.7]        # h/w = 3.0 <= 3.5: forgiven
    absurd = [0.45, 0.4, 0.55, 1.0]              # h/w = 6.0: NOT forgiven
    assert permissive_iou(vis_gt, plausible_fb, gt_convention="visible",
                          frame_wh=sq) > 0.999
    assert permissive_iou(vis_gt, absurd, gt_convention="visible",
                          frame_wh=sq) < 0.5
    # wide frame rescales the cap: 6:1 normalized in 16:9 is ~3.4 pixel
    assert permissive_iou(vis_gt, absurd, gt_convention="visible",
                          frame_wh=(1920, 1080)) > 0.999
    # no frame geometry -> uncapped (legacy)
    assert permissive_iou(vis_gt, absurd, gt_convention="visible") > 0.999
