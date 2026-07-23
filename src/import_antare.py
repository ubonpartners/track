# Antare body-worn labelled videos -> eval dataset (MB spec 2026-07-23).
#
# Source: /mldata/downloaded_datasets/antare/labelled_videos/<name>/
#   <name>            the video (folder name includes .mp4)
#   MOT/gt.txt        MOT16-style rows: frame,id,x,y,w,h,conf,class,vis
#                     FRAME INDEXES THE "1FPS" EXTRACTED IMAGES (Raw
#                     images_1fps/): image k IS source frame k*stride in
#                     PTS order (stride = nframes/nimages, e.g. 24/30) —
#                     pixel-verified against the given jpegs (mean|diff|
#                     ~0.2-2 at the frame hypothesis vs 14-46 at the old
#                     (N-1)*1.0s guess). NOT at t=(N-1)s: the GoPro
#                     escooter clip is VFR (40/50ms intervals mixed) and
#                     justin's stream STARTS at pts 6.009s. Annotation
#                     times are expressed on the timeline the tracker
#                     actually uses — frame_index/out_fps over RETAINED
#                     chunk frames (upyc run_on_video_file feeds a raw
#                     elementary stream and stamps t=n/fps uniformly;
#                     container PTS never reaches it) — so the mapping
#                     that matters is image -> retained-frame INDEX, not
#                     image -> pts. Coordinates are SOURCE pixels.
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


def frame_pts(video):
    """All video frame times in PTS order (packet-level: no decode)."""
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=pts_time", "-of", "json", video])
    return sorted(float(p["pts_time"])
                  for p in json.loads(out)["packets"] if "pts_time" in p)


def image_stride(video, image_dir):
    """(pts_list, stride): image k = source frame k*stride in PTS order.
    Stride recovered from the given jpegs (nframes/nimages); the identity
    was pixel-verified against them — see header."""
    n_images = len([f for f in os.listdir(image_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    pts = frame_pts(video)
    stride = round(len(pts) / n_images)
    assert stride >= 1 and abs(len(pts) / stride - n_images) <= 1, \
        f"{video}: {len(pts)} frames / {n_images} images -> stride {stride} inconsistent"
    assert stride % 2 == 0 or len(pts) / (pts[-1] - pts[0]) < FPS_HALVE_THRESHOLD, \
        f"{video}: odd stride {stride} would put annotations on dropped frames when halving"
    return pts, stride


def load_gt(mot_dir):
    """gt.txt + labels.txt -> [(image_index_0based, track_id, x, y, w, h,
    cls_name)]. gt.txt frame N is 1-based over the extracted images."""
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
        rows.append((fr - 1, tid, x, y, w, h, name))
    return rows


def retained_positions(pts, t0, t1, halve):
    """{global_frame_index: position among this chunk's RETAINED frames}.
    Mirrors transcode_chunk's select exactly: pts window + global parity.
    Position/out_fps is the tracker's uniform timeline for the chunk."""
    pos, i = {}, 0
    for n, t in enumerate(pts):
        if t0 <= t < t1 - 1e-4 and (not halve or n % 2 == 0):
            pos[n] = i
            i += 1
    return pos


def transcode_chunk(src, dst, t0, t1, dims, halve, out_fps):
    """Chunk cut: select by ORIGINAL time window (and global frame parity
    when halving), then restamp to TRUE CFR (setpts=N/out_fps). The
    source may be VFR (escooter GoPro: pts drifts up to 2.7s per chunk
    from a uniform grid) and every downstream consumer — GT times, the
    tracker's raw-h264 n/fps stamping, viewers decoding the mp4 pts —
    must agree on ONE timeline: retained-frame-index / out_fps. Frame
    CONTENT keeps source cadence; only timestamps are normalised.
    I+P, ~2s GOP, NVENC."""
    sel = f"between(t,{t0},{t1 - 1e-4})"
    if halve:
        sel += "*not(mod(n\\,2))"
    w, h = dims
    vf = f"select='{sel}',setpts=N/({out_fps}*TB),scale={w}:{h}"
    subprocess.check_call(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", vf,
         "-fps_mode", "vfr", "-an", "-c:v", "h264_nvenc", "-preset", "p4",
         "-cq", "23", "-g", "24", "-bf", "0", dst])


def chunk_annotation(rows, pos, stride, out_fps, src_w, src_h, video_path,
                     cadence=1.0):
    """GT rows whose image-frame is retained in this chunk -> trackset
    json dict; times on the tracker's uniform retained-index/fps
    timeline."""
    by_time = {}
    for (k, tid, x, y, w, h, name) in rows:
        g = k * stride
        if g not in pos:
            continue
        t_local = pos[g] / out_fps
        cls = CLASS_MAP.get(name, "other")
        rec = {"box": [round(x / src_w, 5), round(y / src_h, 5),
                       round((x + w) / src_w, 5), round((y + h) / src_h, 5)],
               "class": GT_CLASSES.index(cls), "conf": 1.0}
        by_time.setdefault(round(t_local, 4), {})[str(tid)] = rec
    frames = [{"frame_id": i + 1, "frame_time": t, "objects": objs}
              for i, (t, objs) in enumerate(sorted(by_time.items()))]
    return {
        "metadata": {
            "frame_rate": out_fps,
            "width": src_w, "height": src_h,   # normalized boxes: any consistent pair
            "classes": GT_CLASSES,
            "original_video": video_path,
            # sparse human GT (~1Hz; measured mean image spacing) —
            # densify_sparse_gt targets this flag
            "sparse_gt": {"annotation_cadence_s": round(cadence, 4),
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
    pts, stride = image_stride(src, os.path.join(folder, "Raw images_1fps"))
    cadence = stride / out_fps / (2 if halve else 1)  # image spacing on tracker timeline
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
        if not os.path.isfile(vpath):   # video content is annotation-independent
            transcode_chunk(src, vpath, t0, t1, dims, halve, out_fps)
        pos = retained_positions(pts, t0, t1, halve)
        doc = chunk_annotation(rows, pos, stride, out_fps, w, h, vpath, cadence)
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
