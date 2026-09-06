import json

from src import autolabel_bridge


def test_manifest_augmentation_uses_worker_pool(tmp_path, monkeypatch):
    folder = tmp_path / "dataset"
    anno = folder / "annotation"
    anno.mkdir(parents=True)
    for stem in ("a", "b"):
        (anno / f"{stem}.json").write_text("{}")
    manifest = folder / "reduced.json"
    manifest.write_text(json.dumps(["a", "b"]))

    monkeypatch.setattr(
        autolabel_bridge, "augment_trackset_file",
        lambda path, **kw: 1)
    result = autolabel_bridge.augment_dataset(
        str(folder), manifest=str(manifest), workers=2)

    assert result == {"done": 2, "skipped": 0, "failed": 0}


def test_autolabel_video_writes_atomically(tmp_path, monkeypatch):
    class Pipeline:
        @staticmethod
        def run(video_path, out_path, convention):
            with open(out_path, "w") as fh:
                json.dump({"frames": [], "metadata": {}}, fh)

    monkeypatch.setattr(
        autolabel_bridge, "load_autolabel",
        lambda: (Pipeline, object()))
    out = tmp_path / "result.json"

    assert autolabel_bridge.autolabel_video(
        "video.mp4", str(out)) == str(out)
    assert json.loads(out.read_text())["frames"] == []
    assert not list(tmp_path.glob("*.tmp*"))


def test_autolabel_video_cuts_enables_scene_cut_config(tmp_path,
                                                       monkeypatch):
    seen = {}

    class Pipeline:
        @staticmethod
        def run(video_path, out_path, convention, config):
            seen["cuts"] = config.cuts
            with open(out_path, "w") as fh:
                json.dump({"frames": []}, fh)

    class Cfg:
        cuts = False

    class Config:
        @staticmethod
        def load_config():
            return Cfg()

    monkeypatch.setattr(
        autolabel_bridge, "load_autolabel", lambda: (Pipeline, Config))
    out = tmp_path / "result.json"

    autolabel_bridge.autolabel_video("video.mp4", str(out), cuts=True)
    assert seen["cuts"] is True


def test_tighten_replaces_loose_gt_boxes_only(tmp_path):
    # GT: obj 1 loose around a true box; obj 2 with only a low-conf
    # candidate; obj 3 pre-added by augmentation (must stay untouched)
    anno = {
        "metadata": {"frame_rate": 30.0, "width": 100, "height": 100,
                     "classes": ["person", "vehicle", "other"]},
        "frames": [{
            "frame_id": 0, "frame_time": 0.0,
            "objects": {
                "1": {"box": [0.35, 0.35, 0.65, 0.65], "class": 0,
                      "conf": 1.0},
                "2": {"box": [0.10, 0.10, 0.20, 0.30], "class": 0,
                      "conf": 1.0},
                "3": {"box": [0.70, 0.70, 0.80, 0.90], "class": 0,
                      "conf": 0.9, "source": "autolabel"},
            }}]}
    al = {"metadata": {}, "frames": [{
        "frame_id": 0, "frame_time": 0.0,
        "objects": {
            "9": {"box": [0.40, 0.40, 0.60, 0.60], "class": 0,
                  "conf": 0.9},
            "8": {"box": [0.10, 0.10, 0.20, 0.30], "class": 0,
                  "conf": 0.2},     # below min_conf: no tightening
        }}]}
    ap = tmp_path / "clip.json"
    ap.write_text(json.dumps(anno))
    (tmp_path / "clip.autolabel.json").write_text(json.dumps(al))

    n = autolabel_bridge.tighten_trackset_file(
        str(ap), work_dir=str(tmp_path), verbose=False)
    assert n == 1
    out = json.loads(ap.read_text())
    objs = out["frames"][0]["objects"]
    assert objs["1"]["box"] == [0.40, 0.40, 0.60, 0.60]
    assert objs["1"]["source"] == "autolabel_tight"
    assert objs["2"]["box"] == [0.10, 0.10, 0.20, 0.30]   # low conf: kept
    assert "source" not in objs["2"]
    assert objs["3"]["source"] == "autolabel"             # untouched
    assert out["metadata"]["autolabel_tightened"]["boxes_tightened"] == 1
    # idempotent
    assert autolabel_bridge.tighten_trackset_file(
        str(ap), work_dir=str(tmp_path), verbose=False) is None


def test_tightening_is_temporally_consistent(tmp_path):
    # 9-frame track; tight anchors only at frames 0 and 8. Frames 3-5
    # are >time_tol from both anchors -> must get interpolated tight
    # geometry (not the loose GT box), so a track never mixes loose and
    # tight boxes frame to frame ("flashing").
    frames = []
    for i in range(9):
        frames.append({
            "frame_id": i, "frame_time": i / 30.0,
            "objects": {"1": {"box": [0.35 + 0.01 * i, 0.35,
                                      0.65 + 0.01 * i, 0.65],
                              "class": 0, "conf": 1.0}}})
    anno = {"metadata": {"frame_rate": 30.0, "width": 100, "height": 100,
                         "classes": ["person", "vehicle", "other"]},
            "frames": frames}
    al_frames = []
    for i in (0, 8):
        al_frames.append({
            "frame_id": i, "frame_time": i / 30.0,
            "objects": {"9": {"box": [0.40 + 0.01 * i, 0.40,
                                      0.60 + 0.01 * i, 0.60],
                              "class": 0, "conf": 0.9}}})
    ap = tmp_path / "clip.json"
    ap.write_text(json.dumps(anno))
    (tmp_path / "clip.autolabel.json").write_text(
        json.dumps({"metadata": {}, "frames": al_frames}))

    autolabel_bridge.tighten_trackset_file(
        str(ap), work_dir=str(tmp_path), verbose=False)
    out = json.loads(ap.read_text())
    srcs = [f["objects"]["1"].get("source") for f in out["frames"]]
    assert srcs[0] == srcs[8] == "autolabel_tight"
    assert all(s in ("autolabel_tight", "autolabel_tight_interp")
               for s in srcs), srcs           # no loose frame survives
    assert "autolabel_tight_interp" in srcs[3:6]
    # geometry moves monotonically: no loose/tight step flashing
    xs = [f["objects"]["1"]["box"][0] for f in out["frames"]]
    assert all(xs[i] <= xs[i + 1] + 1e-9 for i in range(8)), xs
    assert out["metadata"]["autolabel_tightened"]["consistent"] is True
    # standalone repair on an already-consistent file is a no-op
    assert autolabel_bridge.make_tight_consistent(
        str(ap), verbose=False) is None


def test_augmented_tracks_are_dense_on_existing_frames(tmp_path):
    # dense 30fps GT grid (7 frames); autolabel candidate observed at
    # ~5fps keyframes -> after augmentation the added track must appear
    # on EVERY grid frame it spans (a listed frame claims full
    # annotation), while a >max_gap occlusion gap stays absent
    frames = [{"frame_id": i, "frame_time": i / 30.0,
               "objects": {"1": {"box": [0.1, 0.1, 0.2, 0.3],
                                 "class": 0, "conf": 1.0}}}
              for i in range(7)]
    anno = {"metadata": {"frame_rate": 30.0, "width": 100, "height": 100,
                         "classes": ["person", "vehicle", "other"]},
            "frames": frames}
    al_frames = [{"frame_id": i, "frame_time": i / 30.0,
                  "objects": {"9": {"box": [0.60 + 0.01 * i, 0.60,
                                            0.80 + 0.01 * i, 0.90],
                                    "class": 0, "conf": 0.9}}}
                 for i in range(0, 61, 6)]      # 2s track, ~5fps obs
    ap = tmp_path / "clip.json"
    ap.write_text(json.dumps(anno))
    (tmp_path / "clip.autolabel.json").write_text(
        json.dumps({"metadata": {}, "frames": al_frames}))

    added = autolabel_bridge.augment_trackset_file(
        str(ap), video_path=__file__,  # existence-checked only
        work_dir=str(tmp_path), verbose=False)
    assert added == 1
    out = json.loads(ap.read_text())
    grid = [f for f in out["frames"] if f["frame_time"] <= 6 / 30.0 + 1e-9]
    new_id = str(max(int(t) for f in out["frames"] for t in f["objects"]))
    present = [new_id in f["objects"] for f in grid]
    assert all(present), present        # dense on every spanned frame
    assert out["metadata"]["autolabel_augmented"]["dense"] is True


def test_autolabel_export_contract_reads():
    """Reader half of the autolabel<->track schema contract (writer
    half: autolabel tests/test_core.py::test_export_schema_contract).
    Golden fixture is a real autolabel export committed in the
    autolabel repo; if this fails, the exporter drifted from what this
    repo's readers assume (dense 1-based grid, normalized xyxy boxes,
    class indices into metadata classes)."""
    import os
    import src.paths as paths
    golden = os.path.join(paths.autolabel_repo(), "tests", "golden", "contract_trackset.json")
    if not os.path.isfile(golden):
        import pytest
        pytest.skip("autolabel repo not present")
    import src.trackset as trackset
    ts = trackset.TrackSet(golden)
    assert ts.metadata["classes"] == ["person", "vehicle", "other"]
    # TrackSet normalizes away frame_id; the contract surface here is
    # the TIME grid: dense, starting at 0, spaced 1/frame_rate
    fps = ts.metadata["frame_rate"]
    for i, f in enumerate(ts.frames):
        assert abs(f["frame_time"] - i / fps) < 1e-9
        for o in f["objects"].values():
            assert len(o["box"]) == 4
            assert 0 <= o["class"] < len(ts.metadata["classes"])
    assert len(ts.frames) == 8
