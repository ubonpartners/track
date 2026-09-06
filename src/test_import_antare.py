# Pure-logic tests for the antare dense-GT importer: MOT parsing, class
# mapping, frame -> time mapping, clipping, empty-frame emission. No ffmpeg.

import os

import src.import_antare as ia


def test_load_gt(tmp_path):
    (tmp_path / "labels.txt").write_text("person\nbicycle\ncar\n")
    (tmp_path / "gt.txt").write_text(
        "1,0,10,20,30,40,1,1,1.0\n"
        "2,1,0.5,0.5,1,1,1,2,1.0\n"
        "2,2,0,0,5,5,1,9,1.0\n"     # class int beyond labels -> other
        "junk\n")
    rows = ia.load_gt(str(tmp_path))
    assert rows == [(1, 0, 10.0, 20.0, 30.0, 40.0, "person"),
                    (2, 1, 0.5, 0.5, 1.0, 1.0, "bicycle"),
                    (2, 2, 0.0, 0.0, 5.0, 5.0, "other")]


def test_build_annotation_mapping():
    rows = [
        (1, 7, 192.0, 144.0, 192.0, 288.0, "person"),
        (1, 8, 0.0, 0.0, 960.0, 720.0, "car"),
        (3, 7, 200.0, 1400.0, 192.0, 100.0, "person"),   # overruns bottom edge
        (3, 9, 5.0, 5.0, 20.0, 20.0, "unknown_thing"),
        (5, 1, 0.0, 0.0, 10.0, 10.0, "truck"),            # beyond n_frames
    ]
    doc, dropped = ia.build_annotation(rows, 4, 10.0, 1920, 1440, "/v.mp4")
    assert dropped == 1
    md = doc["metadata"]
    assert md["classes"] == ["person", "vehicle", "other"]
    assert md["frame_rate"] == 10.0 and (md["width"], md["height"]) == (1920, 1440)
    assert md["original_video"] == "/v.mp4"
    assert [f["frame_id"] for f in doc["frames"]] == [1, 2, 3, 4]
    assert [f["frame_time"] for f in doc["frames"]] == [0.0, 0.1, 0.2, 0.3]
    f1, f2, f3, f4 = doc["frames"]
    assert f1["objects"]["7"]["class"] == 0
    assert f1["objects"]["7"]["box"] == [0.1, 0.1, 0.2, 0.3]
    assert f1["objects"]["8"]["class"] == 1                 # car -> vehicle
    assert f2["objects"] == {}                              # dense: empty frame kept
    assert f3["objects"]["7"]["box"][3] == 1.0              # clipped
    assert f3["objects"]["9"]["class"] == 2                 # unknown -> other
    assert f4["objects"] == {}
    assert all("1" not in f["objects"] for f in doc["frames"])


def test_class_map_covers_labels():
    labels = ["person", "bicycle", "car", "motorbike", "bus", "truck", "other"]
    assert all(l in ia.CLASS_MAP for l in labels)
    assert {ia.CLASS_MAP[l] for l in labels} <= set(ia.GT_CLASSES)


def test_camera_hint():
    assert ia.camera_hint("antare-bwc-04") == "bodycam"
    for cam in ("nc-cam-06", "sm-cam-09", "wh-cam-01"):
        assert ia.camera_hint(cam) == "static"


def _touch(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").close()


def test_discover_flat_and_nested(tmp_path):
    root = str(tmp_path)
    _touch(f"{root}/pub-garden.mp4"); _touch(f"{root}/pub-garden/gt/gt.txt")
    _touch(f"{root}/no-gt.mp4")                                   # no gt -> ignored
    _touch(f"{root}/640-knife-drawn/antare-bwc-04.mp4")
    _touch(f"{root}/640-knife-drawn/antare-bwc-04/gt/gt.txt")
    _touch(f"{root}/640-knife-drawn/nc-cam-06.mp4")
    _touch(f"{root}/640-knife-drawn/nc-cam-06/gt/gt.txt")
    found = ia.discover(root)
    assert [(f[0], f[3], f[4]) for f in found] == [
        ("pub-garden", "bodycam", None),          # flat drop: hint from the drop
        ("knife-drawn-bwc-04", "bodycam", "640-knife-drawn"),
        ("knife-drawn-fixed-06", "static", "640-knife-drawn"),
    ]
    assert found[1][1].endswith("640-knife-drawn/antare-bwc-04.mp4")
    assert found[1][2].endswith("640-knife-drawn/antare-bwc-04/gt")


def test_clip_stem():
    assert ia.clip_stem("643-repeat-offender-refused-entry-at-the-doo", "nc-cam-03") == "refused-entry-fixed-03"
    assert ia.clip_stem("653-heavy-equipment-box-dropped-near-colleag", "antare-bwc-16") == "box-dropped-bwc-16"
    import pytest
    with pytest.raises(AssertionError):
        ia.clip_stem("999-unknown-scene", "nc-cam-01")


def test_build_annotation_hint_and_scene():
    doc, _ = ia.build_annotation([], 2, 10.0, 100, 100, "/v.mp4",
                                 hint="static", scene="640-x")
    assert doc["metadata"]["hint"] == "static"
    assert doc["metadata"]["gt_source"]["scene"] == "640-x"


def test_default_source_layouts():
    # only checked when the drops are present on this box
    for src in ia.SRC_DEFAULTS:
        if os.path.isdir(src):
            found = ia.discover(src)
            assert found
            for _stem, video, gt_dir, _hint, _scene in found:
                assert os.path.isfile(video)
                assert os.path.isfile(os.path.join(gt_dir, "labels.txt"))
