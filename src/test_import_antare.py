# Pure-logic tests for the antare importer: chunking, MOT parsing,
# class mapping, retained-frame-index time mapping. No ffmpeg.

import src.import_antare as ia


def test_chunk_spans():
    assert ia.chunk_spans(300.3) == [(0.0, 120.0), (120.0, 240.0), (240.0, 300.3)]
    # 46s tail kept (>=30), 20s tail dropped
    s = ia.chunk_spans(1126.46)
    assert len(s) == 10 and s[-1] == (1080.0, 1126.46)
    assert ia.chunk_spans(260.0) == [(0.0, 120.0), (120.0, 240.0)]
    assert ia.chunk_spans(90.0) == [(0.0, 90.0)]


def test_scale_dims():
    assert ia.scale_dims(1920, 1440) == (1280, 960)
    assert ia.scale_dims(1280, 720) == (1280, 720)


def test_retained_positions():
    # 10 frames at 0.5s spacing starting at pts 6.0 (justin-style offset),
    # halved: retained = even global indices inside the window.
    pts = [6.0 + 0.5 * n for n in range(10)]
    pos = ia.retained_positions(pts, 6.0, 9.0, halve=True)
    assert pos == {0: 0, 2: 1, 4: 2}          # pts 6.0, 7.0, 8.0 (8.5 odd, 9.0 out)
    pos_all = ia.retained_positions(pts, 6.0, 9.0, halve=False)
    assert pos_all == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    # VFR: an extra-long gap shifts pts but positions stay index-based
    pts_vfr = [0.0, 0.04, 0.08, 0.50, 0.54, 0.58]
    assert ia.retained_positions(pts_vfr, 0.0, 1.0, halve=True) == {0: 0, 2: 1, 4: 2}


def test_chunk_annotation_mapping():
    # source: 8 frames, stride 2 (image k = frame 2k), halved -> retained
    # even frames; out_fps = 2.0 -> tracker timeline 0.0, 0.5, 1.0, ...
    pts = [10.0 + 0.25 * n for n in range(8)]        # stream starts at 10.0
    pos = ia.retained_positions(pts, 10.0, 11.5, halve=True)
    assert pos == {0: 0, 2: 1, 4: 2}                  # frames at pts 10.0 10.5 11.0
    rows = [
        (0, 1, 192.0, 144.0, 192.0, 288.0, "person"),   # image 0 -> frame 0 -> t 0.0
        (0, 2, 0.0, 0.0, 960.0, 720.0, "car"),
        (1, 1, 200.0, 150.0, 192.0, 288.0, "person"),   # image 1 -> frame 2 -> t 0.5
        (1, 3, 10.0, 10.0, 50.0, 50.0, "bicycle"),
        (2, 5, 5.0, 5.0, 20.0, 20.0, "unknown_thing"),  # image 2 -> frame 4 -> t 1.0
        (3, 4, 0.0, 0.0, 10.0, 10.0, "truck"),          # image 3 -> frame 6: outside window
    ]
    doc = ia.chunk_annotation(rows, pos, 2, 2.0, 1920, 1440, "/v.mp4", cadence=0.5)
    assert doc["metadata"]["classes"] == ["person", "vehicle", "other"]
    assert doc["metadata"]["sparse_gt"]["annotation_cadence_s"] == 0.5
    f0 = doc["frames"][0]
    assert f0["frame_time"] == 0.0
    assert f0["objects"]["1"]["class"] == 0                  # person
    assert f0["objects"]["2"]["class"] == 1                  # car -> vehicle
    assert f0["objects"]["1"]["box"] == [0.1, 0.1, 0.2, 0.3]
    f1 = doc["frames"][1]
    assert f1["frame_time"] == 0.5                            # retained index 1 / 2.0fps
    assert f1["objects"]["3"]["class"] == 1                  # bicycle -> vehicle
    f2 = doc["frames"][2]
    assert f2["frame_time"] == 1.0
    assert f2["objects"]["5"]["class"] == 2                  # unknown -> other
    assert all("4" not in fr["objects"] for fr in doc["frames"])  # out-of-window excluded
