"""MEVA KPF geometry/types YAML -> TrackSet.

Moved verbatim from TrackSetImportersMixin.import_meva (repo_cleanup.md
stage 3): `read_into(ts, ...)` is the original body with `self` renamed
`ts`; `read(...)` builds a fresh TrackSet and fills it.
"""
import os
import cv2

from src.trackset import TrackSet

import yaml

# libyaml-backed loader when available: ~10x faster on MEVA's multi-MB KPF
# files, identical output to yaml.safe_load
_YAML_SAFE_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def _yaml_safe_load(stream):
    return yaml.load(stream, Loader=_YAML_SAFE_LOADER)



def read(geom_path, types_path=None, video_path=None, width=None, height=None, frame_rate=30.0):
    ts = TrackSet()
    read_into(ts, geom_path, types_path, video_path, width, height, frame_rate)
    return ts


def read_into(ts, geom_path, types_path=None, video_path=None,
                width=None, height=None, frame_rate=30.0):
    """Import a fully-annotated MEVA (KF1) clip from its KPF geometry.

    MEVA ships per-clip KPF YAML: `<clip>.geom.yml` = per-keyframe boxes
    (`g0: "x1 y1 x2 y2"` in pixels, `id1` = track id, `ts0` = 0-based frame
    index), `<clip>.types.yml` = per-track object class. The annotations are
    keyframed (not every frame); TrackSet.objects_at_time interpolates
    between bracketing frames by track id, so a sparse ground-truth track
    renders continuously — no per-frame filling here.

    The MEVA dataset lives at /mldata/downloaded_datasets/other/MEVA
    (annotations/ + videos/ + krtd/ + map/, alongside MOT17/MOT20/JAAD).

    geom_path:  path to `<clip>.geom.yml`.
    types_path: `<clip>.types.yml` (defaults to the sibling of geom_path).
    video_path: the source video (mp4/avi). When given, its exact
                width/height/fps are read (MEVA mixes 1920x1072 and
                1920x1080); otherwise width/height/frame_rate are used.
    """
    if types_path is None:
        cand = geom_path.replace(".geom.yml", ".types.yml").replace(".geom.yaml", ".types.yaml")
        types_path = cand if os.path.exists(cand) else None

    # Auto-locate the source video: MEVA names the clip identically across
    # annotation and video (`<clip>.geom.yml` vs `<clip>.r13.avi`). Look
    # beside the geom and in a sibling videos/ dir.
    if video_path is None:
        import glob as _glob
        stem = os.path.basename(geom_path)
        for suf in (".geom.yml", ".geom.yaml"):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                break
        gd = os.path.dirname(os.path.abspath(geom_path))
        for d in (gd, os.path.join(gd, "..", "videos"), os.path.join(gd, "..", "video"),
                  os.path.join(os.path.dirname(gd), "videos")):
            hits = _glob.glob(os.path.join(d, stem + "*.avi")) + _glob.glob(os.path.join(d, stem + "*.mp4"))
            if hits:
                video_path = hits[0]
                break

    # Exact frame geometry/rate from the video when available (the 1072 vs
    # 1080 difference shifts every box's normalization).
    if video_path is not None and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        vf = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        cap.release()
        if vw: width = vw
        if vh: height = vh
        if vf > 0: frame_rate = vf
    if width is None or height is None:
        raise ValueError("import_meva: give a video_path or explicit width/height "
                         "(MEVA clips are 1920x1072 or 1920x1080)")

    # types: track id -> output class. MEVA cset3 object types map to the
    # shared [person, vehicle, other] scheme so downstream eval/training is
    # identical to the MOT/PersonPath importers.
    def out_class(cset):
        if not cset:
            return 2
        top = max(cset, key=cset.get)
        if top == "person":
            return 0
        if top in ("vehicle", "bike"):
            return 1
        return 2

    cls_of = {}
    if types_path is not None:
        for row in (_yaml_safe_load(open(types_path)) or []):
            t = row.get("types")
            if not t:
                continue
            cls_of[t.get("id1")] = out_class(t.get("cset3") or t.get("cset2") or {})

    by_frame = {}
    for row in (_yaml_safe_load(open(geom_path)) or []):
        g = row.get("geom")
        if not g:
            continue
        g0 = g.get("g0")
        ts0 = g.get("ts0")
        tid = g.get("id1")
        if g0 is None or ts0 is None or tid is None:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in str(g0).split())
        except ValueError:
            continue
        frame_id = int(ts0) + 1  # 1-based, matching the MOT/PersonPath importers
        by_frame.setdefault(frame_id, {})[int(tid)] = {
            "box": [round(x1 / width, 4), round(y1 / height, 4),
                    round(x2 / width, 4), round(y2 / height, 4)],
            "class": cls_of.get(tid, 2),
            "conf": 1.0,
        }

    ts.metadata = {
        "frame_rate": frame_rate,
        "width": int(width),
        "height": int(height),
        "classes": ["person", "vehicle", "other"],
        # MEVA KPF boxes track full actor extent through occlusion
        "box_convention": "fullbody",
    }
    if video_path is not None:
        ts.metadata["original_video"] = video_path
    ts.frames = []
    ts.frame_times = []
    for frame_id in sorted(by_frame.keys()):
        frame_time = (frame_id - 1) / frame_rate
        ts.frames.append({
            "frame_id": frame_id,
            "frame_time": frame_time,
            "objects": by_frame[frame_id],
        })
        ts.frame_times.append(frame_time)
