"""The one place that runs ffprobe/ffmpeg (repo_cleanup.md stage 5).

Before this module there were four ffprobe wrappers and six ffmpeg
recipes spread over dataset_lite, trackset_import, import_antare and a
closure in convert_roundabouthd. They are now:

  probe_video(path)        one ffprobe call -> VideoInfo (size, fps, codec,
                           audio codec, b-frames, optional packet count)
  probe_audio(path)        audio codec name or None
  audio_args(codec)        the output flags that PRESERVE source audio
  scale_dims(w, h)         1280-cap, even dims, never upscale
  frame_pts(path)          display-order frame timestamps
  frame_pts_monotonic()    / has_backward_pts()   the two pts-sanity checks
  transcode(src, dst, ...) one ffmpeg runner: encoder fallback list, audio
                           policy, temp-file write, optional output check

Recipes (who calls transcode with what) — kept identical to the flags
each caller used before the unification; tests/unit/test_media.py pins
every one of them to the old command line:

  derive.transcode        select/setpts/scale, -fps_mode vfr, audio kept,
                          nvenc p4 cq23 -> x264 veryfast crf23, -g 2s -bf 0,
                          written straight to dst (caller owns temp names)
  media.transcode_h264    x264 medium crf18, audio kept, faststart, temp+pid
  media.remux_to_mp4      -c copy (+pts check) -> setpts/-r x264 medium crf18,
                          audio kept, faststart, temp ".part.mp4"
  importers roundabouthd  nvenc p5 vbr cq22 -> x264 medium crf22, -g 30,
                          yuv420p, faststart, audio kept, temp ".part.mp4"
  importers uvg_vcm       raw yuv444p16le input, x264 medium crf18, yuv420p,
                          faststart, temp ".part.mp4" (no audio in raw yuv)
  importers bdd100k       x264 fast crf18, yuv420p, faststart, ffmpeg's
                          default audio mapping (unchanged), temp+pid

Encoder fallback: every encoder list is tried in order; a non-zero exit
(NVENC session churn, exit 187, on long batch runs) falls through to the
next. All encoders failing raises RuntimeError with the last stderr tail.
"""
import json
import os
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

# audio codecs that can be stream-copied into mp4 and that the tracker's
# mp4 demuxer hands on (ubon_cstuff mp4_demux.h: audio comes out as raw
# AAC). Anything else present is re-encoded to AAC; no audio -> -an.
MP4_COPY_AUDIO = {"aac"}

# encoder argument lists (the fallback order is the caller's choice)
NVENC_P4_CQ23 = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"]
NVENC_P5_VBR_CQ22 = ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "22", "-b:v", "0"]
X264_VERYFAST_CRF23 = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
X264_MEDIUM_CRF18 = ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]
X264_MEDIUM_CRF22 = ["-c:v", "libx264", "-preset", "medium", "-crf", "22"]
X264_FAST_CRF18 = ["-c:v", "libx264", "-preset", "fast", "-crf", "18"]
STREAM_COPY = ["-c", "copy"]


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float            # avg_frame_rate (0.0 when ffprobe cannot express it)
    r_fps: float          # r_frame_rate; != fps means VFR
    duration: float       # container duration, seconds (0.0 if absent)
    codec: str            # video codec name
    audio_codec: Optional[str]
    has_b_frames: int
    n_frames: int         # packet count, only when probe_video(count=True), else 0


def _frac(x):
    try:
        return float(Fraction(x))
    except (ValueError, ZeroDivisionError, TypeError):
        return 0.0


def probe_video(video, count=False):
    """One ffprobe call. count=True also counts video packets (reads the
    whole file at demux speed) for callers that need an exact frame count."""
    cmd = ["ffprobe", "-v", "error"]
    if count:
        cmd.append("-count_packets")
    cmd += ["-show_entries",
            "stream=codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,"
            "has_b_frames,nb_read_packets:format=duration", "-of", "json", video]
    j = json.loads(subprocess.check_output(cmd))
    streams = j.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        raise ValueError(f"{video}: no video stream")
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    return VideoInfo(
        width=int(v.get("width", 0)), height=int(v.get("height", 0)),
        fps=_frac(v.get("avg_frame_rate")), r_fps=_frac(v.get("r_frame_rate")),
        duration=float((j.get("format") or {}).get("duration", 0.0) or 0.0),
        codec=v.get("codec_name", ""), audio_codec=(a or {}).get("codec_name") or None,
        has_b_frames=int(v.get("has_b_frames", 0) or 0),
        n_frames=int(v.get("nb_read_packets", 0) or 0) if count else 0)


def probe_audio(video):
    """codec name of the first audio stream, or None."""
    return probe_video(video).audio_codec


def audio_args(codec):
    """ffmpeg output args that PRESERVE the source audio (MB 2026-09-06:
    eval media must keep audio when the source has it — the tracker
    runs audio analytics). codec = probe_audio(src)."""
    if codec is None:
        return ["-an"]
    if codec in MP4_COPY_AUDIO:
        return ["-map", "0:v:0", "-map", "0:a:0", "-c:a", "copy"]
    return ["-map", "0:v:0", "-map", "0:a:0", "-c:a", "aac", "-b:a", "160k"]


def scale_dims(w, h, max_edge=1280):
    """Longest side capped at max_edge, aspect kept, both sides even.
    Never upscales."""
    long_side = max(w, h)
    if long_side <= max_edge:
        return w, h
    s = max_edge / float(long_side)
    return (int(round(w * s / 2)) * 2, int(round(h * s / 2)) * 2)


def frame_pts(video):
    """Display-order frame timestamps (seconds) of the first video stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts_time", "-of", "csv=p=0", video],
        capture_output=True, text=True).stdout
    return [float(x.rstrip(",")) for x in out.split() if x.rstrip(",")]


def frame_pts_monotonic(path, fps):
    """True if decoded frame pts sit on a ~uniform 1/fps grid. Catches
    sources whose container timestamps are broken (MEVA r13 AVIs stamp
    pts=dts with B-frames): a -c copy remux of one plays with visible
    glitches even though the bitstream decodes cleanly."""
    ts = frame_pts(path)
    lo, hi = 0.5 / fps, 1.5 / fps
    return all(lo <= b - a <= hi for a, b in zip(ts, ts[1:]))


def has_backward_pts(video):
    """True when any DISPLAY-order packet timestamp steps backward — the
    broken-container jitter that corrupts time-based analytics."""
    last = None
    for t in frame_pts(video):
        if last is not None and t < last:
            return True
        last = t
    return False


def native_fps(path):
    """avg_frame_rate as a float, OpenCV's reading when ffprobe cannot express it."""
    fps = probe_video(path).fps
    if fps > 0:
        return fps
    import cv2
    return float(cv2.VideoCapture(path).get(cv2.CAP_PROP_FPS))


def video_codec(path):
    """Video codec name, "" when the file has no video stream."""
    try:
        return probe_video(path).codec
    except (ValueError, subprocess.CalledProcessError):
        return ""


def build_cmd(src, dst, encoder, *, vf=(), pre_input=(), post_input=(), gop=None,
              bf=None, pix_fmt=None, faststart=False, fps_mode=None, audio=None):
    """The ffmpeg argv for one attempt. `audio`: a list of output flags
    (use audio_args(...)), or None for ffmpeg's default stream mapping."""
    cmd = ["ffmpeg", "-y", "-v", "error", *pre_input, "-i", src, *post_input]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    if fps_mode:
        cmd += ["-fps_mode", fps_mode]
    cmd += list(encoder)
    if gop is not None:
        cmd += ["-g", str(gop)]
    if bf is not None:
        cmd += ["-bf", str(bf)]
    if pix_fmt:
        cmd += ["-pix_fmt", pix_fmt]
    if faststart:
        cmd += ["-movflags", "+faststart"]
    if audio is not None:
        cmd += list(audio)
    cmd.append(dst)
    return cmd


def transcode(src, dst, encoders, *, tmp=".part{pid}.mp4", check=None,
              keep_audio=True, dry_run=False, **opts):
    """Run ffmpeg src -> dst, trying each encoder argument list in turn.

    keep_audio=True   output flags from audio_args(probe_audio(src));
    keep_audio=None   no audio flags at all (ffmpeg's default mapping);
    keep_audio=False  -an.
    tmp               suffix template for the temp file (None = write dst
                      directly; the caller owns temp naming);
    check             callable(path) -> bool run on a finished output; False
                      rejects it and moves to the next encoder;
    dry_run           return the list of argvs instead of running.
    Other keyword arguments go to build_cmd. Returns the index of the
    encoder that succeeded; raises RuntimeError when none did."""
    if keep_audio is True:
        audio = audio_args(probe_audio(src)) if not dry_run else audio_args("<probe>")
    elif keep_audio is False:
        audio = ["-an"]
    else:
        audio = None
    out = dst if tmp is None else dst + tmp.format(pid=os.getpid())
    cmds = [build_cmd(src, out, enc, audio=audio, **opts) for enc in encoders]
    if dry_run:
        return cmds
    err = ""
    for i, cmd in enumerate(cmds):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            err = r.stderr[-300:]
            continue
        if check is not None and not check(out):
            err = "output rejected by check"
            continue
        if out != dst:
            os.replace(out, dst)
        return i
    if out != dst and os.path.exists(out):
        os.remove(out)
    raise RuntimeError(f"ffmpeg could not convert {src} -> {dst}: {err}")


# ---- the two importer-side recipes that used to be private helpers ----

def transcode_h264(src, dst):
    """Exact 1:1 near-lossless x264 transcode (same fps, same frame count,
    crf 18) for source codecs autolabel's rfdetr worker env cannot decode
    (AV1). Temp-name write so interrupted runs can't leave a partial mp4."""
    transcode(src, dst, [X264_MEDIUM_CRF18], faststart=True)


def remux_to_mp4(src, dst):
    """Repackage a video into an mp4 container without transcoding
    (ffmpeg -c copy): bit-identical encoded frames, identical timing.
    Falls back to an exact 1:1 x264 transcode (same fps, same frame
    count, near-lossless crf 18) if the source codec can't be stored in
    mp4 OR the copied stream's decoded frame pts are non-monotonic
    (broken container stamps; the decoder restores true display order
    and setpts stamps a clean CFR grid). Writes via a temp name so
    interrupted runs can't leave a plausible-looking partial mp4."""
    fps = native_fps(src)
    try:
        transcode(src, dst, [STREAM_COPY], tmp=".part.mp4", faststart=True,
                  keep_audio=None, check=lambda p: frame_pts_monotonic(p, fps))
    except RuntimeError:
        transcode(src, dst, [X264_MEDIUM_CRF18], tmp=".part.mp4", faststart=True,
                  vf=[f"setpts=N/{fps}/TB"], post_input=["-r", f"{fps}"])
