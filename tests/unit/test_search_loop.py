# The search LOOP against a fake objective (search_review.md §3): batched
# probes, memoisation across splits, journal + resume, val checkpoint,
# per-group validate deltas, html report. No tracker, no GPU — track_test
# is monkeypatched with a quadratic bowl.

import json
import os

import yaml

import src.track_search as tsr


OPT = {"utrack.kf_weight": 0.6, "utrack.sim_weight": 0.4}


def _fake_track_test_factory(counter):
    def fake_track_test(config, split=None, desc=""):
        rows = []
        for tk, t in config["tests"].items():
            counter["evals"] += 1
            cfg = t["config"]
            x = cfg["utrack"]["kf_weight"]
            y = cfg["utrack"]["sim_weight"]
            score = -((x - 0.6) ** 2) - ((y - 0.4) ** 2)
            base = {"fitness": score, "mota": score}
            rows.append({"params": {"test_key": tk, "ds_key": "_overall"},
                         "result": dict(base)})
            for g, off in (("cctv", 0.01), ("bwc", -0.01)):
                rows.append({"params": {"test_key": tk, "ds_key": f"__ovr{g}"},
                             "result": {"fitness": score + off, "mota": score + off}})
        return rows
    return fake_track_test


def _write_search_yaml(tmp_path):
    tracker = {"utrack": {"kf_weight": 0.2, "sim_weight": 1.0}}
    tracker_path = tmp_path / "tracker.yaml"
    tracker_path.write_text(yaml.dump(tracker))
    logdir = tmp_path / "logs"
    cfg = {
        "result_log_file_path": str(logdir),
        "result_test_opt_key": "search_config",
        "result_dataset_opt_key": "_overall",
        "result_dataset_opt_param": "fitness",
        "tests": {"search_config": {"config": str(tracker_path)}},
        "search_params": {
            "utrack.kf_weight": {"min": 0.0, "max": 2.0, "step": 0.05},
            "utrack.sim_weight": {"min": 0.0, "max": 2.0, "step": 0.05},
        },
        "initial_mult": 4,
        "final_mult": 0.5,
        "do_train_split": True,
        "datasets": {}, "columns": [], "sort_key": "fitness",
        "num_workers": 1,
    }
    path = tmp_path / "search.yaml"
    path.write_text(yaml.dump(cfg))
    return path, logdir


def _latest(logdir, pattern):
    files = sorted(f for f in os.listdir(logdir) if pattern in f)
    assert files, f"no {pattern} in {os.listdir(logdir)}"
    return os.path.join(logdir, files[-1])


def test_search_loop_converges_and_journals(tmp_path, monkeypatch):
    counter = {"evals": 0}
    monkeypatch.setattr(tsr.track_test, "track_test",
                        _fake_track_test_factory(counter))
    monkeypatch.setattr(tsr.track_test, "summary_string",
                        lambda r: f"MOTA:{r['mota']:0.4f}")
    ypath, logdir = _write_search_yaml(tmp_path)
    tsr.search_track(str(ypath))

    log = open(_latest(logdir, "search_log")).read()
    assert "All done!" in log
    assert "best by train" in log and "best by val" in log
    assert "group deltas:" in log or "group levels:" in log

    # journal: parseable, and its best train eval is near the bowl optimum
    entries = [json.loads(l) for l in open(_latest(logdir, "search_journal"))]
    best = max((e for e in entries if e["split"] == "train"),
               key=lambda e: e["score"])
    assert abs(best["vec"]["utrack.kf_weight"] - 0.6) <= 0.101
    assert abs(best["vec"]["utrack.sim_weight"] - 0.4) <= 0.101
    assert best["groups"]["cctv"] > best["groups"]["bwc"]

    # html report exists and embeds the data
    html = open(_latest(logdir, "search_report")).read()
    assert "track search" in html and "cctv" in html

    assert counter["evals"] > 10   # it really searched


def test_search_resume_skips_cached_evals(tmp_path, monkeypatch):
    first = {"evals": 0}
    monkeypatch.setattr(tsr.track_test, "track_test",
                        _fake_track_test_factory(first))
    monkeypatch.setattr(tsr.track_test, "summary_string",
                        lambda r: f"MOTA:{r['mota']:0.4f}")
    ypath, logdir = _write_search_yaml(tmp_path)
    tsr.search_track(str(ypath))
    journal = _latest(logdir, "search_journal")

    # identical search resumed from the journal: every eval is a cache hit
    cfg = yaml.safe_load(open(ypath))
    cfg["resume_from"] = journal
    ypath2 = tmp_path / "search2.yaml"
    ypath2.write_text(yaml.dump(cfg))
    second = {"evals": 0}
    monkeypatch.setattr(tsr.track_test, "track_test",
                        _fake_track_test_factory(second))
    tsr.search_track(str(ypath2))
    assert first["evals"] > 10
    assert second["evals"] == 0, "resume re-ran cached evals"
