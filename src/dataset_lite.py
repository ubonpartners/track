# Lite dataset import (MB spec 2026-07-23): re-encode a labelled-track
# dataset's videos for cheap eval decode, keeping frame-time truth intact.
#
#   - longest side capped at 1280 (never upscaled; boxes are normalized so
#     annotations are resolution-free — only metadata w/h changes);
#   - framerate reduced by an INTEGER divisor N (keep every Nth frame):
#     static cameras -> lowest resulting fps >= 5, moving -> >= 12. The
#     retained frames keep their EXACT original PTS (ffmpeg select + vfr
#     passthrough), so nothing downstream re-times anything;
#   - annotations are subset to exactly the retained frames
#     (round(frame_time*src_fps) % N == 0) — annotated frames always
#     correspond to retained frames. BDD's sparse 5fps GT lands on the
#     retained cadence for any N dividing 6;
#   - no B-frames, I+P only, GOP ~2s of output frames;
#   - optional hard duration cap (MEVA: 120s);
#
# AUTOLABEL ORDERING (the invariant that keeps GT quality): auto-annotation
# ALWAYS runs on the NATIVE-framerate source — its own tracking would
# degrade badly at 5fps — and the fps drop is applied to the RESULTING
# annotation by cadence subsetting here. Time-trimming is safe, rate-drop
# input is not. For time-capped sets this tool therefore also emits
# <root>/video_autolabel/<name>.mp4: a stream-copied (native fps/res)
# duration-trimmed clip that is the ONLY correct autolabel input; the lite
# clip is for EVAL DECODE ONLY.
#   - optionally DROP clips whose source timestamps are broken (backward
#     PTS jitter, the OTW doorbell disease): video+annotation are moved to
#     <root>/dropped_jitter/, never deleted.
#
# Output: <root>/video_lite/<name>.mp4; the annotation json is updated in
# place (metadata.original_video repointed, frame_rate/width/height updated,
# a `lite:` provenance block added — the tool skips clips that already have
# one) with a one-time <name>.json.orig backup beside it.
#
# Usage:
#   python -m src.dataset_lite --root /mldata/tracking/meva  --min-fps 5 --max-seconds 120
#   python -m src.dataset_lite --root /mldata/tracking/otw   --min-fps 5 --drop-jitter
#   python -m src.dataset_lite --root /mldata/tracking/bwc-videotext --min-fps 12
#
# ffmpeg IS allowed here: this is offline dataset preparation tooling, not
# the ai-node runtime.

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from fractions import Fraction


def choose_divisor(src_fps, min_fps):
    """Largest integer N with src_fps/N >= min_fps (N>=1) — the LOWEST
    resulting framerate that still meets the floor."""
    if src_fps <= min_fps:
        return 1
    return max(1, int(math.floor(src_fps / float(min_fps) + 1e-9)))


DEFAULT_TRACKER_CONFIG = "/mldata/config/track/trackers/uc_v11.yaml"


def min_delta_from_config(hint, config_path=DEFAULT_TRACKER_CONFIG):
    """min_time_delta_process for a camera class, read from the production
    tracker config. hint='bodycam' (any moving camera) reads the
    `(hint:bodycam)` variant when present; 'static' (or a hint with no
    variant) reads the base key. Imports derive their decimation from
    THIS so a config change only requires re-running the import."""
    import yaml
    cfg = yaml.safe_load(open(config_path))
    key = "min_time_delta_process"
    if hint and hint != "static":
        v = cfg.get(f"{key}(hint:{hint})")
        if v is not None:
            return float(v)
    return float(cfg[key])


def divisor_from_config(src_fps, hint, config_path=DEFAULT_TRACKER_CONFIG):
    """Smallest integer N with N/src_fps >= min_time_delta_process: the
    decimated grid is then exactly the frame set the tracker's
    min-interval gate selects from the native stream (assuming clean CFR
    stamps), so the decimation is analytics-equivalent by construction —
    the gate passes every retained frame and no un-retained frame would
    ever have been processed."""
    delta = min_delta_from_config(hint, config_path)
    return max(1, int(math.ceil(delta * src_fps - 1e-9)))


def scale_dims(w, h, max_edge=1280):
    """Longest side capped at max_edge, aspect kept, both sides even.
    Never upscales."""
    long_side = max(w, h)
    if long_side <= max_edge:
        return w, h
    s = max_edge / float(long_side)
    return (int(round(w * s / 2)) * 2, int(round(h * s / 2)) * 2)


def probe(video):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate",
         "-of", "json", video])
    st = json.loads(out)["streams"][0]
    fps = float(Fraction(st["avg_frame_rate"]))
    return int(st["width"]), int(st["height"]), fps


def has_backward_pts(video):
    """True when any DISPLAY-order packet timestamp steps backward — the
    broken-container jitter that corrupts time-based analytics."""
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts_time", "-of", "csv=p=0", video])
    last = None
    for line in out.decode().split("\n"):
        line = line.strip().rstrip(",")
        if not line:
            continue
        t = float(line)
        if last is not None and t < last:
            return True
        last = t
    return False


def rewrite_annotation(d, src_fps, divisor, dims, max_seconds, lite_video,
                       hint=None, min_delta=None):
    """Subset an annotation dict to the retained frames + refresh metadata.
    Pure function (unit-tested); returns (new_dict, kept, dropped)."""
    md = dict(d.get("metadata") or {})
    kept, dropped = [], 0
    for fr in d.get("frames") or []:
        t = fr.get("frame_time", 0.0)
        if max_seconds is not None and t > max_seconds:
            dropped += 1
            continue
        idx = int(round(t * src_fps))
        if idx % divisor != 0:
            dropped += 1
            continue
        fr = dict(fr)
        # snap to the same clean grid the transcode restamps the video to
        # (source pts jitter must not leak into eval timing)
        fr["frame_time"] = round(idx / src_fps, 6)
        kept.append(fr)
    md["frame_rate"] = src_fps / float(divisor)
    md["width"], md["height"] = dims
    md["original_video"] = lite_video
    if hint:
        # camera class: evals bind `(hint:x)` config variants through this
        md["hint"] = hint
    md["lite"] = {"source_fps": src_fps, "divisor": divisor,
                  "max_seconds": max_seconds,
                  "hint": hint, "min_time_delta": min_delta}
    out = dict(d)
    out["metadata"] = md
    out["frames"] = kept
    return out, len(kept), dropped


def transcode(src, dst, divisor, dims, out_fps, max_seconds):
    vf = []
    if divisor > 1:
        vf.append(f"select='not(mod(n\\,{divisor}))'")
    # clean-CFR restamp onto the analytics grid: jittery/VFR source pts
    # can land 2-3% off-grid, inside the tracker gate's ~2.5-4% margin
    # (uc_v11 min_time_delta comment) — synthetic stamps make the grid
    # exact by construction; rewrite_annotation snaps GT times to match
    vf.append(f"setpts=N/({out_fps}*TB)")
    w, h = dims
    vf.append(f"scale={w}:{h}")
    gop = max(1, int(round(2 * out_fps)))
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if max_seconds is not None:
        cmd += ["-t", str(max_seconds + 0.5)]  # pad half a frame period; the select+json caps exactly
    cmd += ["-i", src, "-vf", ",".join(vf), "-fps_mode", "vfr", "-an"]
    tail = ["-g", str(gop), "-bf", "0", dst]
    try:
        subprocess.check_call(
            cmd + ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"] + tail)
    except subprocess.CalledProcessError:
        # NVENC session churn fails intermittently (exit 187) on long
        # batch runs; x264 keeps the same I+P contract (-bf 0)
        subprocess.check_call(
            cmd + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
            + tail)


def process_dataset(root, min_fps=None, max_seconds=None, drop_jitter=False,
                    max_edge=1280, hint=None,
                    config_path=DEFAULT_TRACKER_CONFIG):
    """hint mode (preferred): divisor per clip from the tracker config's
    min_time_delta_process for that camera class (analytics-equivalent
    decimation; re-run after a config change). min_fps mode (legacy):
    fixed framerate floor."""
    if (min_fps is None) == (hint is None):
        raise ValueError("pass exactly one of min_fps / hint")
    anno_dir = os.path.join(root, "annotation")
    lite_dir = os.path.join(root, "video_lite")
    os.makedirs(lite_dir, exist_ok=True)
    done = skipped = jittered = 0
    dropped_clips = []
    for name in sorted(os.listdir(anno_dir)):
        if not name.endswith(".json"):
            continue
        jpath = os.path.join(anno_dir, name)
        d = json.load(open(jpath))
        md = d.get("metadata") or {}
        if md.get("lite"):
            skipped += 1
            continue
        src = md.get("original_video")
        if not src or not os.path.isfile(src):
            print(f"  {name}: source video missing ({src}) — SKIPPED", flush=True)
            continue
        if drop_jitter and has_backward_pts(src):
            # broken stamps: quarantine video+annotation, never delete
            qdir = os.path.join(root, "dropped_jitter")
            os.makedirs(qdir, exist_ok=True)
            shutil.move(jpath, os.path.join(qdir, name))
            shutil.copy2(src, os.path.join(qdir, os.path.basename(src)))
            dropped_clips.append(name[:-5])
            jittered += 1
            print(f"  {name[:-5]}: BACKWARD PTS — dropped to {qdir}", flush=True)
            continue
        w, h, fps = probe(src)
        min_delta = None
        if hint is not None:
            min_delta = min_delta_from_config(hint, config_path)
            divisor = divisor_from_config(fps, hint, config_path)
        else:
            divisor = choose_divisor(fps, min_fps)
        dims = scale_dims(w, h, max_edge)
        out_fps = fps / divisor
        dst = os.path.join(lite_dir, os.path.splitext(os.path.basename(src))[0] + ".mp4")
        transcode(src, dst, divisor, dims, out_fps, max_seconds)
        if max_seconds is not None:
            # native-rate autolabel input for the trimmed window (see header)
            adir = os.path.join(root, "video_autolabel")
            os.makedirs(adir, exist_ok=True)
            adst = os.path.join(adir, os.path.basename(dst))
            if not os.path.exists(adst):
                subprocess.check_call(
                    ["ffmpeg", "-y", "-loglevel", "error", "-t", str(max_seconds),
                     "-i", src, "-c", "copy", "-an", adst])
        new, kept, droppedf = rewrite_annotation(d, fps, divisor, dims,
                                                 max_seconds, dst,
                                                 hint=hint,
                                                 min_delta=min_delta)
        orig = jpath + ".orig"
        if not os.path.exists(orig):
            shutil.copy2(jpath, orig)
        tmp = jpath + ".tmp"
        with open(tmp, "w") as f:
            json.dump(new, f)
        os.replace(tmp, jpath)
        done += 1
        print(f"  {name[:-5]}: {w}x{h}@{fps:.2f} /{divisor} -> {dims[0]}x{dims[1]}@{out_fps:.2f} "
              f"frames {kept} kept / {droppedf} dropped", flush=True)
    print(f"{root}: {done} transcoded, {skipped} already lite, "
          f"{jittered} dropped for jitter {dropped_clips or ''}", flush=True)
    return dropped_clips


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--hint", choices=["static", "bodycam"],
                   help="camera class: divisor derived from the tracker "
                        "config's min_time_delta_process (re-run after a "
                        "config change)")
    g.add_argument("--min-fps", type=float,
                   help="legacy fixed framerate floor")
    ap.add_argument("--config", default=DEFAULT_TRACKER_CONFIG,
                    help="tracker config for --hint mode")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--drop-jitter", action="store_true")
    ap.add_argument("--max-edge", type=int, default=1280)
    a = ap.parse_args()
    process_dataset(a.root, min_fps=a.min_fps, max_seconds=a.max_seconds,
                    drop_jitter=a.drop_jitter, max_edge=a.max_edge,
                    hint=a.hint, config_path=a.config)


if __name__ == "__main__":
    sys.exit(main())
