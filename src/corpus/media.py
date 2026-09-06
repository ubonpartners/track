"""ffprobe/ffmpeg helpers used by the tier-0 -> tier-1 importers: pts
monotonicity probe, lossless remux, codec/fps probes, exact 1:1 h264
transcode. Stage 5 collapses these and dataset_lite's into one API.

Moved verbatim from src/trackset_import.py (repo_cleanup.md stage 4b).
"""
import os


def _frame_pts_monotonic(path, fps):
    """True if decoded frame pts sit on a ~uniform 1/fps grid. Catches
    sources whose container timestamps are broken (MEVA r13 AVIs stamp
    pts=dts with B-frames): a -c copy remux of one plays with visible
    glitches even though the bitstream decodes cleanly."""
    import subprocess
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts_time", "-of", "csv=p=0", path],
        capture_output=True, text=True).stdout.split()
    ts = [float(x.rstrip(",")) for x in out if x.rstrip(",")]
    lo, hi = 0.5 / fps, 1.5 / fps
    return all(lo <= b - a <= hi for a, b in zip(ts, ts[1:]))


def _remux_to_mp4(src, dst):
    """Repackage a video into an mp4 container without transcoding
    (ffmpeg -c copy): bit-identical encoded frames, identical timing.
    Falls back to an exact 1:1 x264 transcode (same fps, same frame
    count, near-lossless crf 18) if the source codec can't be stored in
    mp4 OR the copied stream's decoded frame pts are non-monotonic
    (broken container stamps; the decoder restores true display order
    and setpts stamps a clean CFR grid). Writes via a temp name so
    interrupted runs can't leave a plausible-looking partial mp4."""
    import subprocess
    tmp = dst + ".part.mp4"
    fps = _native_fps(src)
    copy_args = ["-c", "copy"]
    from src.dataset_lite import audio_args, probe_audio
    transcode_args = ["-vf", f"setpts=N/{fps}/TB", "-r", f"{fps}",
                      "-c:v", "libx264", "-preset", "medium", "-crf", "18"
                      ] + audio_args(probe_audio(src))
    r = None
    for args in (copy_args, transcode_args):
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", src] + args + ["-movflags", "+faststart", tmp]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            if args is copy_args and not _frame_pts_monotonic(tmp, fps):
                continue
            os.replace(tmp, dst)
            return
    if os.path.exists(tmp):
        os.remove(tmp)
    raise RuntimeError(f"ffmpeg could not convert {src} to mp4: {r.stderr[-300:]}")


def _video_codec(path):
    import subprocess
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    return r.stdout.strip()


def _native_fps(path):
    import subprocess
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    try:
        num, den = r.stdout.strip().split("/")
        return float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        import cv2
        return float(cv2.VideoCapture(path).get(cv2.CAP_PROP_FPS))


def _transcode_h264(src, dst):
    """Exact 1:1 near-lossless x264 transcode (same fps, same frame count,
    crf 18) for source codecs autolabel's rfdetr worker env cannot decode
    (AV1). Temp-name write so interrupted runs can't leave a partial mp4."""
    import subprocess
    from src.dataset_lite import audio_args, probe_audio
    tmp = f"{dst}.part{os.getpid()}.mp4"
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", src,
         "-c:v", "libx264", "-preset", "medium", "-crf", "18"]
        + audio_args(probe_audio(src)) + ["-movflags", "+faststart", tmp],
        capture_output=True, text=True)
    if r.returncode != 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"transcode failed for {src}: {r.stderr[-300:]}")
    os.replace(tmp, dst)
