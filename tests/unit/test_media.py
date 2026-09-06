"""src/corpus/media.py: every ffmpeg recipe pinned to the command line its
caller used before the unification (repo_cleanup.md stage 5), plus a real
probe/transcode round trip on a synthetic clip."""
import os

import cv2
import numpy as np
import pytest

import src.corpus.media as media
import src.corpus.derive as derive

W, H, FPS, N = 64, 48, 10.0, 6


def _opts(cmd):
    """(input part, sorted multiset of output options) — ffmpeg output
    options are order-independent, so recipes compare as sets of
    (flag, value) pairs plus the final destination."""
    i = cmd.index("-i")
    head, tail = cmd[:i + 2], cmd[i + 2:-1]
    pairs, k = [], 0
    while k < len(tail):
        if tail[k].startswith("-") and k + 1 < len(tail) and not tail[k + 1].startswith("-"):
            pairs.append((tail[k], tail[k + 1])); k += 2
        else:
            pairs.append((tail[k], None)); k += 1
    return head, sorted(pairs), cmd[-1]


AUDIO = media.audio_args("<probe>")     # placeholder the dry run substitutes for probe_audio(src)


def test_derive_transcode_recipe():
    # old dataset_lite.transcode: nvenc p4 cq23 then x264 veryfast crf23,
    # select/setpts/scale, -fps_mode vfr, -g 2*fps, -bf 0, -t max+0.5, no temp name
    cmds = derive.transcode("in.mp4", "out.mp4", 2, (1280, 720), 15.0, 120.0, dry_run=True)
    old_nvenc = ["ffmpeg", "-y", "-loglevel", "error", "-t", "120.5", "-i", "in.mp4",
                 "-vf", "select='not(mod(n\\,2))',setpts=N/(15.0*TB),scale=1280:720",
                 "-fps_mode", "vfr", *AUDIO,
                 "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23", "-g", "30", "-bf", "0", "out.mp4"]
    old_x264 = old_nvenc[:old_nvenc.index("-c:v")] + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                                                       "-g", "30", "-bf", "0", "out.mp4"]
    assert len(cmds) == 2
    for new, old in zip(cmds, (old_nvenc, old_x264)):
        assert _opts(new)[1:] == _opts(old)[1:]
        assert new[:2] == ["ffmpeg", "-y"] and new[4:8] == old[4:8]      # -v error == -loglevel error
    # divisor 1, no cap: no select filter, no -t
    (c,) = derive.transcode("in.mp4", "out.mp4", 1, (640, 480), 10.0, None, dry_run=True)[:1]
    assert "-t" not in c and "select" not in " ".join(c) and "setpts=N/(10.0*TB),scale=640:480" in c


def test_transcode_h264_recipe():
    cmds = media.transcode("in.mkv", "out.mp4", [media.X264_MEDIUM_CRF18], faststart=True, dry_run=True)
    old = ["ffmpeg", "-y", "-v", "error", "-i", "in.mkv",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", *AUDIO,
           "-movflags", "+faststart", f"out.mp4.part{os.getpid()}.mp4"]
    assert len(cmds) == 1 and _opts(cmds[0]) == _opts(old)


def test_remux_recipes():
    copy = media.transcode("in.avi", "out.mp4", [media.STREAM_COPY], tmp=".part.mp4",
                           faststart=True, keep_audio=None, dry_run=True)
    assert copy == [["ffmpeg", "-y", "-v", "error", "-i", "in.avi", "-c", "copy",
                     "-movflags", "+faststart", "out.mp4.part.mp4"]]
    fb = media.transcode("in.avi", "out.mp4", [media.X264_MEDIUM_CRF18], tmp=".part.mp4", faststart=True,
                         vf=["setpts=N/25.0/TB"], post_input=["-r", "25.0"], dry_run=True)
    old = ["ffmpeg", "-y", "-v", "error", "-i", "in.avi", "-vf", "setpts=N/25.0/TB", "-r", "25.0",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", *AUDIO, "-movflags", "+faststart", "out.mp4.part.mp4"]
    assert _opts(fb[0]) == _opts(old)


def test_roundabouthd_recipe():
    cmds = media.transcode("in.mp4", "out.mp4", [media.NVENC_P5_VBR_CQ22, media.X264_MEDIUM_CRF22],
                           tmp=".part.mp4", gop=30, pix_fmt="yuv420p", faststart=True, dry_run=True)
    old_nv = ["ffmpeg", "-y", "-v", "error", "-i", "in.mp4", "-c:v", "h264_nvenc", "-preset", "p5",
              "-rc", "vbr", "-cq", "22", "-b:v", "0", "-g", "30", "-pix_fmt", "yuv420p",
              "-movflags", "+faststart", *AUDIO, "out.mp4.part.mp4"]
    old_x = ["ffmpeg", "-y", "-v", "error", "-i", "in.mp4", "-c:v", "libx264", "-preset", "medium",
             "-crf", "22", "-g", "30", "-pix_fmt", "yuv420p", "-movflags", "+faststart", *AUDIO, "out.mp4.part.mp4"]
    assert [_opts(c) for c in cmds] == [_opts(old_nv), _opts(old_x)]


def test_uvg_and_bdd_recipes():
    uvg = media.transcode("seq.yuv", "out.mp4", [media.X264_MEDIUM_CRF18], tmp=".part.mp4",
                          pre_input=["-f", "rawvideo", "-pix_fmt", "yuv444p16le", "-s", "3840x2160", "-r", "60"],
                          pix_fmt="yuv420p", faststart=True, keep_audio=None, dry_run=True)
    old = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv444p16le", "-s", "3840x2160", "-r", "60",
           "-i", "seq.yuv", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "out.mp4.part.mp4"]
    assert _opts(uvg[0]) == _opts(old) and uvg[0][:12] == old[:12]
    bdd = media.transcode("clip.mov", "out.mp4", [media.X264_FAST_CRF18], pix_fmt="yuv420p",
                          faststart=True, keep_audio=None, dry_run=True)
    old = ["ffmpeg", "-y", "-v", "error", "-i", "clip.mov", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", f"out.mp4.part{os.getpid()}.mp4"]
    assert _opts(bdd[0]) == _opts(old)


def test_audio_policy():
    # exact lists: the recipe tests substitute audio_args on both sides, so
    # this is the one place that pins what "audio kept" means
    assert media.audio_args(None) == ["-an"]
    assert media.audio_args("aac") == ["-map", "0:v:0", "-map", "0:a:0", "-c:a", "copy"]
    assert media.audio_args("ac3") == ["-map", "0:v:0", "-map", "0:a:0", "-c:a", "aac", "-b:a", "160k"]
    assert media.audio_args("opus") == media.audio_args("ac3")
    (c,) = media.transcode("a", "b", [media.STREAM_COPY], keep_audio=False, dry_run=True)
    assert c[-2:] == ["-an", f"b.part{os.getpid()}.mp4"]


def test_scale_dims():
    assert media.scale_dims(1920, 1080) == (1280, 720)
    assert media.scale_dims(3840, 2160) == (1280, 720)
    assert media.scale_dims(640, 640) == (640, 640)


@pytest.fixture
def tiny_mp4(tmp_path):
    p = str(tmp_path / "tiny.mp4")
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(N):
        vw.write(np.full((H, W, 3), i * 30, np.uint8))
    vw.release()
    return p


def test_probe_and_real_transcode(tiny_mp4, tmp_path):
    info = media.probe_video(tiny_mp4, count=True)
    assert (info.width, info.height, info.fps, info.n_frames) == (W, H, FPS, N)
    assert info.codec == "mpeg4" and info.audio_codec is None and info.r_fps == FPS
    assert media.probe_audio(tiny_mp4) is None and media.video_codec(tiny_mp4) == "mpeg4"
    assert media.native_fps(tiny_mp4) == FPS
    import subprocess
    with pytest.raises(subprocess.CalledProcessError):
        media.frame_pts("/nonexistent/file.mp4")
    assert not media.has_backward_pts(tiny_mp4) and media.frame_pts_monotonic(tiny_mp4, FPS)
    assert derive.probe(tiny_mp4) == (W, H, FPS)
    out = str(tmp_path / "out.mp4")
    media.transcode_h264(tiny_mp4, out)                 # x264 only: runs without a GPU
    o = media.probe_video(out, count=True)
    assert (o.codec, o.n_frames, o.fps) == ("h264", N, FPS) and o.audio_codec is None
    assert not os.path.exists(out + f".part{os.getpid()}.mp4")
    with pytest.raises(RuntimeError):
        media.transcode(tiny_mp4, str(tmp_path / "x.mp4"), [["-c:v", "no_such_encoder"]])


def test_probe_rejects_zero_fps(monkeypatch):
    fake = media.VideoInfo(64, 48, 0.0, 0.0, 1.0, "h264", None, 0, 5)
    monkeypatch.setattr(media, "probe_video", lambda v, count=False: fake)
    with pytest.raises(ValueError):
        derive.probe("x.mp4")
    import src.import_antare as ia
    with pytest.raises(AssertionError):
        ia.probe("x.mp4")


def test_video_codec_missing_file():
    assert media.video_codec("/nonexistent/file.mp4") == ""
