"""One tiny fixture per native format through src/formats/<fmt>.read
(repo_cleanup.md stage 3). Media-backed parsers get a 5-frame synthetic
mp4 written with OpenCV; no /mldata, no GPU."""
import configparser
import json
import os

import cv2
import numpy as np
import pytest

from src import formats
from src.formats import (chirla, jaad, meva, mot, otw, personpath22,
                         roundabouthd, uvg_vcm)

W, H, FPS, N = 64, 48, 10.0, 5


@pytest.fixture
def tiny_mp4(tmp_path):
    p = str(tmp_path / "tiny.mp4")
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    assert vw.isOpened()
    for i in range(N):
        vw.write(np.full((H, W, 3), i * 20, np.uint8))
    vw.release()
    cap = cv2.VideoCapture(p)
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == W
    cap.release()
    return p


def _boxes(ts):
    return [(f["frame_time"], sorted(f["objects"])) for f in ts.frames]


# NB track-id KEY TYPES differ per parser and are asserted as-is below:
# mot/jaad/otw/personpath22 key objects by int, chirla/roundabouthd/uvg_vcm
# by str. Pre-existing; json round-trips make them all str downstream.


def test_mot(tmp_path):
    seq = tmp_path / "MOT-01"
    (seq / "gt").mkdir(parents=True)
    (seq / "img1").mkdir()
    cp = configparser.ConfigParser()
    cp["Sequence"] = {"name": "MOT-01", "imDir": "img1", "frameRate": "10",
                      "seqLength": "4", "imWidth": "64", "imHeight": "48", "imExt": ".jpg"}
    with open(seq / "seqinfo.ini", "w") as f:
        cp.write(f)
    (seq / "gt" / "gt.txt").write_text(
        "1,1,0,0,32,24,1,1,1\n"      # pedestrian -> person
        "1,2,32,24,32,24,1,3,1\n"    # car -> vehicle
        "2,1,8,0,32,24,1,7,1\n"      # static person -> person
        "2,3,0,0,8,8,1,9,1\n")       # occluder -> other
    ts = mot.read(str(seq / "seqinfo.ini"))
    assert ts.metadata["classes"] == ["person", "vehicle", "other"]
    assert ts.metadata["box_convention"] == "fullbody"
    assert len(ts.frames) == 3                       # range(1, seqLength): pre-existing off-by-one preserved
    assert ts.frame_times == [0.0, 0.1, 0.2]
    f1 = ts.frames[0]["objects"]
    assert f1[1]["class"] == 0 and f1[1]["box"] == [0.0, 0.0, 0.5, 0.5]
    assert f1[2]["class"] == 1
    assert ts.frames[1]["objects"][3]["class"] == 2
    assert ts.frames[0]["image_path"].endswith("img1/000001.jpg")
    # the extension dispatch that used to live on TrackSet(path)
    assert _boxes(formats.load(str(seq / "seqinfo.ini"))) == _boxes(ts)


def test_trackset_no_longer_parses_native_formats(tmp_path):
    import src.trackset as tsmod
    p = tmp_path / "seqinfo.ini"
    p.write_text("[Sequence]\n")
    with pytest.raises(ValueError, match="src.formats.load"):
        tsmod.TrackSet(str(p))
    g = tmp_path / "clip.geom.yml"                 # also ends in .yml: must not fall into import_yaml
    g.write_text("- {geom: {id1: 1, ts0: 0, g0: '0 0 1 1'}}\n")
    with pytest.raises(ValueError, match="src.formats.load"):
        tsmod.TrackSet(str(g))


def test_personpath22():
    sample = {"metadata": {"fps": 10.0, "resolution": {"width": 64, "height": 48},
                           "number_of_frames": 5},
              "entities": [
                  {"bb": [0, 0, 32, 24], "labels": {"person": 1}, "id": 7, "blob": {"frame_idx": 0}},
                  {"bb": [0, 0, 64, 48], "labels": {"crowd": 1}, "id": 8, "blob": {"frame_idx": 0}},
                  {"bb": [0, 0, 8, 8], "labels": {"reflection": 1}, "id": 9, "blob": {"frame_idx": 0}},
                  {"bb": [8, 0, 32, 24], "labels": {"person": 1}, "id": 7, "time": 200.0},
              ]}
    ts = personpath22.read("uid_vid_00001.mp4", sample, "/v.mp4")
    assert ts.metadata["width"] == 64 and ts.metadata["original_video"] == "/v.mp4"
    assert ts.frame_times == [0.0, 0.2]                  # keyframes only
    assert ts.frames[0]["objects"][7]["class"] == 0
    assert ts.frames[0]["objects"][8]["class"] == 2      # crowd -> other
    assert 9 not in ts.frames[0]["objects"]              # reflection dropped
    assert ts.frames[1]["frame_id"] == 3                 # 200 ms at 10 fps -> frame 3


def test_meva(tmp_path):
    geom = tmp_path / "clip.geom.yml"
    types = tmp_path / "clip.types.yml"
    geom.write_text("- {geom: {id1: 1, ts0: 0, g0: '0 0 32 24'}}\n"
                    "- {geom: {id1: 2, ts0: 1, g0: '32 24 64 48'}}\n"
                    "- {geom: {id1: 1, ts0: 1, g0: '8 0 40 24'}}\n")
    types.write_text("- {types: {id1: 1, cset3: {person: 1.0}}}\n"
                     "- {types: {id1: 2, cset3: {vehicle: 0.9, person: 0.1}}}\n")
    ts = meva.read(str(geom), width=64, height=48, frame_rate=10.0)
    assert _boxes(formats.load(str(geom), width=64, height=48, frame_rate=10.0)) == _boxes(ts)
    assert ts.metadata["box_convention"] == "fullbody"
    assert ts.frame_times == [0.0, 0.1]
    assert ts.frames[0]["objects"][1]["box"] == [0.0, 0.0, 0.5, 0.5]
    assert ts.frames[1]["objects"][2]["class"] == 1
    assert ts.frames[1]["objects"][1]["class"] == 0
    with pytest.raises(ValueError):
        meva.read(str(geom))                              # no video, no size


def test_otw(tiny_mp4):
    rows = [
        ["v1", "a1", "00039", "person", "0", "0", "0", "32", "24", "True"],
        ["v1", "a1", "00039", "bicycle", "0", "32", "24", "64", "48", "False"],
        ["v1", "a2", "00039", "person", "0", "8", "0", "40", "24", "False"],   # interpolated dup loses
        ["v1", "a1", "00040", "Opening Door", "0", "0", "0", "64", "48", "True"],  # activity region skipped
        ["v1", "a1", "None", "person", "1", "0", "0", "8", "8", "True"],       # no actor
    ]
    ts = otw.read("v1", rows, tiny_mp4)
    assert ts.metadata["width"] == W and ts.metadata["frame_rate"] == FPS
    assert ts.frame_times == [0.0]
    objs = ts.frames[0]["objects"]
    assert len(objs) == 2 and objs[1]["box"] == [0.0, 0.0, 0.5, 0.5] and objs[2]["class"] == 1


def test_chirla(tiny_mp4, tmp_path):
    anno = {str(i + 1): [] for i in range(N)}
    anno["1"] = [{"id": 5, "BboxP": [0, 0, 32, 24]}, {"id": 6, "BboxP": [10, 10, 5, 5]}]
    p = tmp_path / "c.json"; p.write_text(json.dumps(anno))
    ts = chirla.read(str(p), tiny_mp4)
    assert len(ts.frames) == N and ts.frames[0]["frame_id"] == 0
    assert ts.frames[0]["objects"] == {"5": {"box": [0.0, 0.0, 0.5, 0.5], "class": 0, "conf": 1.0}}
    assert ts.frames[1]["objects"] == {}


def test_roundabouthd(tiny_mp4, tmp_path):
    p = tmp_path / "SCT_GT.txt"
    p.write_text("1 11 0 0 32 24 car\n2 11 8 0 40 24 car\n2 12 0 0 0 0 car\n")
    ts = roundabouthd.read(str(p), tiny_mp4)
    assert len(ts.frames) == N
    assert ts.frames[0]["objects"]["11"]["class"] == 1
    assert list(ts.frames[1]["objects"]) == ["11"]       # degenerate box dropped
    assert ts.frames[4]["objects"] == {}


def test_uvg_vcm(tmp_path):
    p = tmp_path / "u.json"
    p.write_text(json.dumps({"version": "1.0",
                             "1": [{"class_id": 1, "track_id": 3, "x_min": 0, "y_min": 0, "x_max": 0.5, "y_max": 0.5},
                                   {"class_id": 8, "track_id": 4, "x_min": 0.5, "y_min": 0.5, "x_max": 1, "y_max": 1},
                                   {"class_id": 17, "track_id": 5, "x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1}],
                             "3": []}))
    ts = uvg_vcm.read(str(p), "/v.mp4", 64, 48, 10.0)
    assert len(ts.frames) == 3
    assert ts.frames[0]["objects"]["3"]["class"] == 0 and ts.frames[0]["objects"]["4"]["class"] == 1
    assert "5" not in ts.frames[0]["objects"]             # non-person/vehicle COCO class dropped


def test_jaad(tiny_mp4, tmp_path):
    xml = f"""<annotations>
  <meta><task><size>{N}</size><original_size><width>{W}</width><height>{H}</height></original_size></task></meta>
  <track label="pedestrian">
    <box frame="0" xtl="0" ytl="0" xbr="32" ybr="24" outside="0"><attribute name="id">p1</attribute></box>
    <box frame="1" xtl="8" ytl="0" xbr="40" ybr="24" outside="0"/>
    <box frame="2" xtl="8" ytl="0" xbr="40" ybr="24" outside="1"/>
  </track>
  <track label="people"><box frame="0" xtl="32" ytl="24" xbr="64" ybr="48" outside="0"/></track>
  <track label="car"><box frame="0" xtl="0" ytl="0" xbr="64" ybr="48" outside="0"/></track>
</annotations>"""
    p = tmp_path / "video_0001.xml"; p.write_text(xml)
    ts = jaad.read(str(p), tiny_mp4)
    assert ts.metadata["box_convention"] == "fullbody"
    assert len(ts.frames) == N and ts.frame_times[1] == 0.1
    assert ts.frames[0]["objects"][1]["class"] == 0
    assert ts.frames[0]["objects"][2]["class"] == 2       # people -> other
    assert len(ts.frames[0]["objects"]) == 2               # car label ignored
    assert 1 in ts.frames[1]["objects"] and 1 not in ts.frames[2]["objects"]   # outside=1 skipped


def test_load_dispatch_rejects_unknown(tmp_path):
    with pytest.raises(ValueError):
        formats.load(str(tmp_path / "x.bin"))
