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
