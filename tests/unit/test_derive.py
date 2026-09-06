# Pure-logic tests for the lite dataset import: divisor choice, scaling,
# and annotation subsetting (incl. BDD's sparse 5fps GT case). No ffmpeg.

import src.corpus.derive as dl


def test_choose_divisor():
    assert dl.choose_divisor(30.0, 5) == 6      # meva static -> exactly 5fps
    assert dl.choose_divisor(30.0, 12) == 2     # bwc/bdd moving -> 15fps
    assert dl.choose_divisor(29.97, 12) == 2    # NTSC -> 14.985 >= 12
    assert dl.choose_divisor(26.6, 5) == 5      # otw oddball -> 5.32
    assert dl.choose_divisor(4.0, 5) == 1       # below floor: keep everything
    assert dl.choose_divisor(30.0, 30) == 1


def test_scale_dims():
    assert dl.scale_dims(1920, 1080) == (1280, 720)
    assert dl.scale_dims(1080, 1920) == (720, 1280)
    assert dl.scale_dims(1280, 720) == (1280, 720)   # no-op at the cap
    assert dl.scale_dims(640, 640) == (640, 640)     # never upscale
    w, h = dl.scale_dims(1919, 1079)
    assert w <= 1280 and w % 2 == 0 and h % 2 == 0


def _frames(times):
    return [{"frame_time": t, "objects": {}} for t in times]


def test_rewrite_annotation_dense_30fps():
    d = {"metadata": {"frame_rate": 30.0, "width": 1920, "height": 1080,
                      "classes": ["person"]},
         "frames": _frames([i / 30.0 for i in range(300)])}
    out, kept, dropped = dl.rewrite_annotation(d, 30.0, 6, (1280, 720), None,
                                               "/x/video_lite/a.mp4")
    assert kept == 50 and dropped == 250
    # retained frames are EXACTLY the every-6th cadence, times unchanged
    assert out["frames"][1]["frame_time"] == 6 / 30.0
    md = out["metadata"]
    assert md["frame_rate"] == 5.0
    assert (md["width"], md["height"]) == (1280, 720)
    assert md["original_video"] == "/x/video_lite/a.mp4"
    assert md["lite"]["divisor"] == 6


def test_rewrite_annotation_bdd_sparse_gt():
    # BDD: GT only every 6th source frame (5fps on 30fps video). With the
    # moving divisor N=2 every GT frame lands on the retained cadence —
    # nothing may be dropped.
    d = {"metadata": {"frame_rate": 30.0, "width": 1280, "height": 720},
         "frames": _frames([i * 6 / 30.0 for i in range(200)])}
    out, kept, dropped = dl.rewrite_annotation(d, 30.0, 2, (1280, 720), None, "v")
    assert kept == 200 and dropped == 0


def test_rewrite_annotation_time_cap():
    d = {"metadata": {"frame_rate": 30.0, "width": 1920, "height": 1080},
         "frames": _frames([i / 30.0 for i in range(9000)])}   # 300s meva
    out, kept, dropped = dl.rewrite_annotation(d, 30.0, 6, (1280, 720), 120.0, "v")
    assert kept == 601                       # 0..120s inclusive at 5fps
    assert out["frames"][-1]["frame_time"] <= 120.0
    assert out["metadata"]["lite"]["max_seconds"] == 120.0


def test_audio_args():
    import src.corpus.derive as dl
    assert dl.audio_args(None) == ["-an"]
    assert dl.audio_args("aac") == ["-map", "0:v:0", "-map", "0:a:0", "-c:a", "copy"]
    assert dl.audio_args("ac3") == ["-map", "0:v:0", "-map", "0:a:0", "-c:a", "aac", "-b:a", "160k"]
