# Antare body-worn labelled videos -> eval dataset (MB spec 2026-07-23).
#
# Source: /mldata/downloaded_datasets/antare/labelled_videos/<name>/
#   <name>            the video (folder name includes .mp4)
#   MOT/gt.txt        MOT16-style rows: frame,id,x,y,w,h,conf,class,vis
#                     FRAME INDEXES THE 1FPS EXTRACTED IMAGES (Raw
#                     images_1fps/), i.e. frame N = t ~= (N-1) seconds.
#                     Coordinates are SOURCE-resolution pixels.
#   MOT/labels.txt    class names, 1-based line order = class ints in gt.txt
#
# Import (per video):
#   - split into 2-minute chunks (a >=30s tail is kept as a short chunk);
#   - transcode each chunk: longest side capped at 1280 (never upscale),
#     framerate HALVED iff source >= 23.9 fps (bodycam analytics gate 0.13
#     then lands on exact retained frames: 30->15 fps, 24.4->12.2), exact
#     PTS preserved (select on GLOBAL frame parity + setpts offset), I+P
#     only, ~2s GOP, NVENC;
#   - annotations: rows in the chunk window -> trackset json frames at
#     integer-second times (chunk-local), boxes normalized by SOURCE
#     resolution, classes mapped to [person, vehicle, other]
#     (bicycle -> vehicle: the detector's vocabulary treats rideables as
#     vehicles; REVISABLE), metadata carries sparse-GT provenance
#     (annotation_cadence_s: 1.0) for the densification pass.
#
# These are body-worn videos: register them group: moving,
# stream_hint: bodycam in the search yaml (done by hand there, not here).
#
# Densification (separate step; src/autolabel_bridge.densify_sparse_gt):
# 1Hz human anchors + autolabel in-betweens. This importer only marks the
# metadata so un-densified files are self-describing.

import argparse
import json
import math
import os
import subprocess
import sys
from fractions import Fraction

# class-name -> our GT scheme. Anything unlisted -> "other" (ignore-region
# semantics: not an FN when missed, matches not charged as FP).
CLASS_MAP = {
    "person": "person",
    "bicycle": "vehicle",     # REVISABLE: rideables as vehicles
    "car": "vehicle",
    "motorbike": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "other": "other",
}
GT_CLASSES = ["person", "vehicle", "other"]

CHUNK_SECONDS = 120.0
MIN_TAIL_SECONDS = 30.0
FPS_HALVE_THRESHOLD = 23.9
MAX_EDGE = 1280


def probe(video):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
         "-of", "json", video])
    j = json.loads(out)
    st = j["streams"][0]
    dur = float(j["format"]["duration"])
    return int(st["width"]), int(st["height"]), float(Fraction(st["avg_frame_rate"])), dur


def scale_dims(w, h, max_edge=MAX_EDGE):
    long_side = max(w, h)
    if long_side <= max_edge:
        return w, h
    s = max_edge / float(long_side)
    return (int(round(w * s / 2)) * 2, int(round(h * s / 2)) * 2)


def chunk_spans(duration):
    """[(t0, t1), ...] — 2-minute spans; a tail >= MIN_TAIL_SECONDS is its
    own (short) chunk, a shorter tail is dropped (reported by caller)."""
    spans = []
    t = 0.0
    while t + CHUNK_SECONDS <= duration + 1e-6:
        spans.append((t, t + CHUNK_SECONDS))
        t += CHUNK_SECONDS
    if duration - t >= MIN_TAIL_SECONDS:
        spans.append((t, duration))
    return spans


def load_gt(mot_dir):
    """gt.txt + labels.txt -> [(t_seconds, track_id, x, y, w, h, cls_name)]."""
    labels = [l.strip() for l in open(os.path.join(mot_dir, "labels.txt"))
              if l.strip()]
    rows = []
    for line in open(os.path.join(mot_dir, "gt.txt")):
        parts = line.strip().split(",")
        if len(parts) < 8:
            continue
        fr, tid = int(parts[0]), int(parts[1])
        x, y, w, h = (float(v) for v in parts[2:6])
        cls_i = int(parts[7])
        name = labels[cls_i - 1] if 1 <= cls_i <= len(labels) else "other"
        rows.append(((fr - 1) * 1.0, tid, x, y, w, h, name))
    return rows


def transcode_chunk(src, dst, t0, t1, dims, halve):
    """Exact-cadence chunk cut: select by ORIGINAL time window (and global
    frame parity when halving) so retained frames keep source cadence;
    setpts rebases times to the chunk start. I+P, ~2s GOP, NVENC."""
    sel = f"between(t,{t0},{t1 - 1e-4})"
    if halve:
        sel += "*not(mod(n\\,2))"
    w, h = dims
    vf = f"select='{sel}',setpts=PTS-{t0}/TB,scale={w}:{h}"
    subprocess.check_call(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", vf,
         "-fps_mode", "vfr", "-an", "-c:v", "h264_nvenc", "-preset", "p4",
         "-cq", "23", "-g", "24", "-bf", "0", dst])


def chunk_annotation(rows, t0, t1, src_w, src_h, out_fps, video_path):
    """Rows within [t0, t1) -> trackset json dict, chunk-local times."""
    by_time = {}
    for (t, tid, x, y, w, h, name) in rows:
        if not (t0 <= t < t1):
            continue
        cls = CLASS_MAP.get(name, "other")
        rec = {"box": [round(x / src_w, 5), round(y / src_h, 5),
                       round((x + w) / src_w, 5), round((y + h) / src_h, 5)],
               "class": GT_CLASSES.index(cls), "conf": 1.0}
        by_time.setdefault(round(t - t0, 3), {})[str(tid)] = rec
    frames = [{"frame_id": i + 1, "frame_time": t, "objects": objs}
              for i, (t, objs) in enumerate(sorted(by_time.items()))]
    return {
        "metadata": {
            "frame_rate": out_fps,
            "width": src_w, "height": src_h,   # normalized boxes: any consistent pair
            "classes": GT_CLASSES,
            "original_video": video_path,
            # sparse human GT at 1Hz — densify_sparse_gt targets this flag
            "sparse_gt": {"annotation_cadence_s": 1.0,
                          "source": "antare/labelled_videos"},
        },
        "frames": frames,
    }


def import_video(folder, out_root):
    name = os.path.basename(folder.rstrip("/")).replace(".mp4", "")
    files = [f for f in os.listdir(folder) if f.endswith(".mp4")]
    assert len(files) == 1, f"{folder}: expected exactly one video, got {files}"
    src = os.path.join(folder, files[0])
    w, h, fps, dur = probe(src)
    halve = fps >= FPS_HALVE_THRESHOLD
    dims = scale_dims(w, h)
    out_fps = fps / 2 if halve else fps
    rows = load_gt(os.path.join(folder, "MOT"))
    os.makedirs(os.path.join(out_root, "video"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "annotation"), exist_ok=True)
    made = []
    for ci, (t0, t1) in enumerate(chunk_spans(dur)):
        stem = f"{name}_c{ci:03d}"
        vpath = os.path.join(out_root, "video", stem + ".mp4")
        jpath = os.path.join(out_root, "annotation", stem + ".json")
        if os.path.isfile(jpath):
            j = json.load(open(jpath))
            if j.get("metadata", {}).get("sparse_gt") or \
               j.get("metadata", {}).get("autolabel_augmented"):
                made.append(stem)
                continue   # idempotent
        transcode_chunk(src, vpath, t0, t1, dims, halve)
        doc = chunk_annotation(rows, t0, t1, w, h, out_fps, vpath)
        ntracks = len({tid for f in doc["frames"] for tid in f["objects"]})
        with open(jpath + ".tmp", "w") as f:
            json.dump(doc, f)
        os.replace(jpath + ".tmp", jpath)
        made.append(stem)
        print(f"  {stem}: [{t0:.0f},{t1:.0f})s {dims[0]}x{dims[1]}@{out_fps:.2f} "
              f"({'halved' if halve else 'native'}) gt_frames={len(doc['frames'])} "
              f"tracks={ntracks}", flush=True)
    tail = dur - chunk_spans(dur)[-1][1] if chunk_spans(dur) else dur
    if tail > 1e-3:
        print(f"  {name}: dropped {tail:.1f}s tail (< {MIN_TAIL_SECONDS}s)", flush=True)
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mldata/downloaded_datasets/antare/labelled_videos")
    ap.add_argument("--out", default="/mldata/tracking/antare_bwc")
    a = ap.parse_args()
    for d in sorted(os.listdir(a.src)):
        folder = os.path.join(a.src, d)
        if os.path.isdir(folder):
            print(f"== {d}", flush=True)
            import_video(folder, a.out)


if __name__ == "__main__":
    sys.exit(main())
