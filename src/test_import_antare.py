# Pure-logic tests for the antare importer: chunking, MOT parsing,
# class mapping, chunk-local annotation building. No ffmpeg.

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


def test_chunk_annotation_mapping(tmp_path):
    rows = [
        (0.0, 1, 192.0, 144.0, 192.0, 288.0, "person"),
        (0.0, 2, 0.0, 0.0, 960.0, 720.0, "car"),
        (1.0, 1, 200.0, 150.0, 192.0, 288.0, "person"),
        (1.0, 3, 10.0, 10.0, 50.0, 50.0, "bicycle"),
        (0.5 + 120.0, 4, 0.0, 0.0, 10.0, 10.0, "truck"),   # next chunk
        (2.0, 5, 5.0, 5.0, 20.0, 20.0, "unknown_thing"),
    ]
    doc = ia.chunk_annotation(rows, 0.0, 120.0, 1920, 1440, 12.19, "/v.mp4")
    assert doc["metadata"]["classes"] == ["person", "vehicle", "other"]
    assert doc["metadata"]["sparse_gt"]["annotation_cadence_s"] == 1.0
    f0 = doc["frames"][0]
    assert f0["frame_time"] == 0.0
    assert f0["objects"]["1"]["class"] == 0                  # person
    assert f0["objects"]["2"]["class"] == 1                  # car -> vehicle
    assert f0["objects"]["1"]["box"] == [0.1, 0.1, 0.2, 0.3]
    f1 = doc["frames"][1]
    assert f1["objects"]["3"]["class"] == 1                  # bicycle -> vehicle
    f2 = doc["frames"][2]
    assert f2["objects"]["5"]["class"] == 2                  # unknown -> other
    assert all("4" not in fr["objects"] for fr in doc["frames"])  # next chunk excluded
    # chunk 2 gets chunk-LOCAL time
    doc2 = ia.chunk_annotation(rows, 120.0, 240.0, 1920, 1440, 12.19, "/v.mp4")
    assert doc2["frames"][0]["frame_time"] == 0.5
