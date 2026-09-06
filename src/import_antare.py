# Antare body-worn labelled clips -> tier-1 corpus (/mldata/tracking_original/
# antare_bwc). Replaces the 2026-07 labelled_videos importer (sparse 1Hz GT
# over extracted jpegs + densification): that drop and its derived clips
# were retired 2026-09-06 — the "individuals - body camera" drop is dense,
# per-frame, human-labelled GT and needs none of it.
#
# Sources (tier 0), two drops with the same per-clip files:
#   flat   "individuals - body camera":   <src>/<cam>.mp4 + <src>/<cam>/gt/
#   nested "multiple views - body camera and fixed":
#                                         <src>/<scene>/<cam>.mp4 + <src>/<scene>/<cam>/gt/
#          several cameras (body-worn antare-bwc-NN and fixed nc/sm/wh-cam-NN)
#          record the same staged incident; they are imported as independent
#          clips named <incident>-<bwc|fixed>-<NN> (e.g. knife-drawn-fixed-06),
#          incident = SCENE_SLUGS[scene id], NN = the source camera number,
#          so clips of one incident sort together and stay traceable.
#   <cam>.mp4                    10 fps CFR h264 (B-framed; 1080p/1440p/4K)
#   <cam>/gt/gt.txt              MOT16 rows: frame,id,x,y,w,h,conf,class,vis
#                                frame is 1-based and IS the video frame
#                                ordinal (frame k = decoded frame k-1;
#                                checked visually on pub-garden f1,
#                                market-halls f300, 640/antare-bwc-04 f1,
#                                640/nc-cam-06 f150). Coordinates are SOURCE
#                                pixels; a few boxes overrun the frame edge
#                                by a pixel or two (clipped here).
#   <cam>/gt/labels.txt          class names, 1-based line order = class ints
#
# Parser: src/formats/antare.py (load_gt, build_annotation, CLASS_MAP).
#
# Import (per clip):
#   - video copied UNCHANGED into tier 1 (the source is the truth; the
#     eval copy is made by `src.corpus.manifest derive`: 1280 long edge, I+P,
#     and — by tier-2 spec — decimated to the tracker's analytics grid);
#   - camera class recorded as metadata.hint: "bodycam" for antare-bwc-*
#     cameras, "static" for the fixed ones. `derive` reads it per clip, so
#     one corpus can mix moving and fixed cameras; register the yaml
#     entries to match (moving + stream_hint bodycam / static, no hint);
#   - annotation: one frame record per video frame 1..N at t=(k-1)/fps,
#     including frames with no labelled objects (dense GT: an empty frame
#     is labelled absence, not missing data); boxes normalised by source
#     resolution and clipped to [0,1]; classes -> [person, vehicle, other]
#     (bicycle -> vehicle, as the detector's vocabulary treats rideables;
#     "other" keeps ignore-region semantics).
#
# Then:
#   python -m src.cli corpus build antare_bwc
#   python -m src.cli corpus derive antare_bwc --hint bodycam [--divisor 1]
#   python -m src.cli corpus check antare_bwc
# and register the clips in the search yaml (group: moving,
# stream_hint: bodycam) by hand.

import argparse
import json
import os
import shutil
import sys

import src.corpus.media as media
import src.paths as paths
from src.formats.antare import (CLASS_MAP, GT_CLASSES, build_annotation,  # noqa: F401 (re-exported for callers/tests)
                                load_gt)



def t1_default():
    return paths.tier1("antare_bwc")


def src_defaults():
    """The two tier-0 drops this importer knows."""
    return [
        paths.downloads("antare", "individuals - body camera-20260902T102854Z-1-001",
                        "individuals - body camera"),
        paths.downloads("antare", "multiple views - body camera and fixed-20260906T050034Z-1-001",
                        "multiple views - body camera and fixed"),
    ]


# scene folder id -> short incident slug (the folders are truncated titles:
# "641-physical-scuffle-in-queue-results-in-eje"). Add a line per new scene;
# discover() refuses an unmapped one rather than inventing a name.
SCENE_SLUGS = {
    "640": "knife-drawn",        # knife drawn after venue entry refusal
    "641": "queue-scuffle",      # physical scuffle in queue results in ejection
    "643": "refused-entry",      # repeat offender refused entry at the door
    "646": "shoplifting",        # customer shoplifting
    "647": "cart-shouting",      # customer pushes cart and shouts profanities
    "651": "intruder-removed",   # unauthorized person removed from restricted area
    "653": "box-dropped",        # heavy equipment box dropped near colleague
}

def probe(video):
    """(width, height, fps, n_frames). Frame count from packet count, not
    the container's nb_frames tag. Refuses VFR sources: frame k -> t=(k-1)/fps
    is only true on a constant-rate stream."""
    i = media.probe_video(video, count=True)
    assert i.fps > 0, f"{video}: ffprobe cannot express the frame rate"
    assert i.fps == i.r_fps, f"{video}: VFR ({i.fps} avg vs {i.r_fps} r) — not importable as-is"
    return i.width, i.height, i.fps, i.n_frames


def camera_hint(cam):
    """Camera class from the antare camera name: body-worn units are
    antare-bwc-NN; everything else (nc-cam, sm-cam, wh-cam) is a fixed
    mount."""
    return "bodycam" if cam.startswith("antare-bwc") else "static"


def clip_stem(scene_dir, cam):
    """<incident>-<bwc|fixed>-<NN>: e.g. ("640-knife-drawn-...", "nc-cam-06")
    -> "knife-drawn-fixed-06". NN is the source camera's own number."""
    sid = scene_dir.split("-", 1)[0]
    assert sid in SCENE_SLUGS, f"{scene_dir}: add its scene id to SCENE_SLUGS"
    kind = "bwc" if camera_hint(cam) == "bodycam" else "fixed"
    num = cam.rsplit("-", 1)[1]
    assert num.isdigit(), f"{cam}: expected a trailing camera number"
    return f"{SCENE_SLUGS[sid]}-{kind}-{num}"


def _has_gt(folder, cam):
    return os.path.isfile(os.path.join(folder, cam, "gt", "gt.txt"))


def discover(src, flat_hint="bodycam"):
    """[(stem, video_path, gt_dir, hint, scene)] for both layouts.
    Flat: <src>/<cam>.mp4 -> stem <cam>, hint = flat_hint (the flat drop
    is all body-worn footage with scene-named files, so the camera name
    says nothing). Nested: <src>/<scene>/<cam>.mp4 -> stem
    clip_stem(scene, cam), hint from the camera name."""
    out = []
    for f in sorted(os.listdir(src)):
        cam = f[:-4]
        if f.endswith(".mp4") and _has_gt(src, cam):
            out.append((cam, os.path.join(src, f), os.path.join(src, cam, "gt"),
                        flat_hint, None))
    for d in sorted(os.listdir(src)):
        folder = os.path.join(src, d)
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            cam = f[:-4]
            if f.endswith(".mp4") and _has_gt(folder, cam):
                out.append((clip_stem(d, cam), os.path.join(folder, f),
                            os.path.join(folder, cam, "gt"), camera_hint(cam), d))
    return out


def import_clip(stem, src_video, gt_dir, hint, scene, out_root):
    w, h, fps, n_frames = probe(src_video)
    rows = load_gt(gt_dir)
    gt_max = max(r[0] for r in rows)
    assert gt_max <= n_frames, \
        f"{stem}: gt frame {gt_max} beyond video ({n_frames} frames)"
    os.makedirs(os.path.join(out_root, "video"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "annotation"), exist_ok=True)
    vpath = os.path.join(out_root, "video", stem + ".mp4")
    jpath = os.path.join(out_root, "annotation", stem + ".json")
    if not os.path.isfile(vpath) or os.path.getsize(vpath) != os.path.getsize(src_video):
        shutil.copyfile(src_video, vpath + ".tmp")
        os.replace(vpath + ".tmp", vpath)
    doc, dropped = build_annotation(rows, n_frames, fps, w, h, vpath,
                                    hint=hint, scene=scene)
    with open(jpath + ".tmp", "w") as f:
        json.dump(doc, f)
    os.replace(jpath + ".tmp", jpath)
    ntracks = len({tid for fr in doc["frames"] for tid in fr["objects"]})
    labelled = sum(1 for fr in doc["frames"] if fr["objects"])
    print(f"  {stem:22s} {hint:7s} {w}x{h}@{fps:g} {n_frames} frames, "
          f"{labelled} with objects, {len(rows)} boxes, {ntracks} tracks"
          f"{f', {dropped} rows beyond video DROPPED' if dropped else ''}",
          flush=True)
    return jpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append",
                    help="source drop(s); default: both known drops")
    ap.add_argument("--out", default=None, help="tier-1 corpus dir (default: <tier1>/antare_bwc)")
    ap.add_argument("clips", nargs="*", help="clip stems (default: all)")
    a = ap.parse_args()
    found = []
    out = a.out or t1_default()
    for src in a.src or src_defaults():
        found += discover(src)
    stems = [f[0] for f in found]
    assert len(stems) == len(set(stems)), "duplicate clip stems across sources"
    for stem, video, gt_dir, hint, scene in found:
        if a.clips and stem not in a.clips:
            continue
        import_clip(stem, video, gt_dir, hint, scene, out)


if __name__ == "__main__":
    sys.exit(main())
