# Multi-class scoring (multi_class_and_hints.md §2-§4): per-class metric
# namespacing, the zero-GT skip, fitness_multi, and the suffix-aware
# aggregation in display_results. Uses tiny synthetic tracksets — no
# tracker, no GPU.

import json

import src.trackset as ts
import src.track_test as tt


def _write_trackset(path, frames, classes=("person", "vehicle", "other"),
                    fps=10.0, w=1280, h=720):
    """frames: [{time: t, objects: {tid: (box, cl)}}]"""
    out = {
        "metadata": {"frame_rate": fps, "width": w, "height": h,
                     "classes": list(classes)},
        "frames": [],
    }
    for i, fr in enumerate(frames):
        objs = {str(tid): {"box": list(box), "class": cl, "conf": 1.0}
                for tid, (box, cl) in fr["objects"].items()}
        out["frames"].append({"frame_id": i + 1,
                              "frame_time": fr["time"],
                              "objects": objs})
    path.write_text(json.dumps(out))
    return str(path)


def _box(x, y, s=0.1):
    return (x, y, x + s, y + s)


def _two_class_pair(tmp_path, vehicle_offset=0.0):
    """GT and a 'test' output: one person track and one vehicle track,
    30 frames at 10 fps. vehicle_offset shifts the test vehicle to
    degrade vehicle-only quality."""
    n = 30
    gt_frames = []
    test_frames = []
    for i in range(n):
        t = i / 10.0
        gt_frames.append({"time": t, "objects": {
            1: (_box(0.1 + 0.005 * i, 0.2), 0),        # person
            2: (_box(0.5, 0.6 + 0.005 * i), 1),        # vehicle
        }})
        test_frames.append({"time": t, "objects": {
            11: (_box(0.1 + 0.005 * i, 0.2), 0),
            12: (_box(0.5 + vehicle_offset, 0.6 + 0.005 * i), 1),
        }})
    gt = ts.TrackSet(_write_trackset(tmp_path / "gt.json", gt_frames))
    test = ts.TrackSet(_write_trackset(tmp_path / "test.json", test_frames))
    return gt, test


def test_gt_class_box_counts(tmp_path):
    gt, _ = _two_class_pair(tmp_path)
    counts = tt.gt_class_box_counts(gt)
    assert counts["person"] == 30
    assert counts["vehicle"] == 30
    assert counts["other"] == 0


def test_per_class_metrics_namespaced(tmp_path):
    gt, test = _two_class_pair(tmp_path)
    # person pass: unsuffixed, exactly as today
    r = tt.compute_metrics(gt, test, classes_to_test=["person"],
                           classes_for_det_map=None)
    assert r["mota"] > 0.99
    # vehicle pass: same machinery, its own class
    rv = tt.compute_metrics(gt, test, classes_to_test=["vehicle"],
                            classes_for_det_map=None)
    assert rv["mota"] > 0.99
    assert "fitness" in rv


def test_vehicle_quality_independent_of_person(tmp_path):
    # A displaced test vehicle tanks vehicle MOTA while person stays perfect —
    # proof the classes are scored independently.
    gt, test = _two_class_pair(tmp_path, vehicle_offset=0.3)
    rp = tt.compute_metrics(gt, test, classes_to_test=["person"],
                            classes_for_det_map=None)
    rv = tt.compute_metrics(gt, test, classes_to_test=["vehicle"],
                            classes_for_det_map=None)
    assert rp["mota"] > 0.99
    assert rv["mota"] < 0.1


def test_fitness_multi():
    r = {"fitness": 0.8, "fitness_vehicle": 0.5}
    got = tt.fitness_multi_score(r, {"person": 1.0, "vehicle": 0.3})
    assert abs(got - (0.8 + 0.15)) < 1e-9
    # zero-GT guard: a row without the vehicle keys contributes person only
    got = tt.fitness_multi_score({"fitness": 0.8},
                                 {"person": 1.0, "vehicle": 0.3})
    assert abs(got - 0.8) < 1e-9
    # default weights = person-only
    assert abs(tt.fitness_multi_score({"fitness": 0.7}, None) - 0.7) < 1e-9
    assert tt.fitness_multi_score({}, {"person": 1.0}) is None


def test_aggregation_recomputes_vehicle_suffix(tmp_path):
    # Two clips, one with vehicle metrics, one without: the __ovr rollup
    # must recompute mota_vehicle from vehicle counts alone and produce
    # fitness_multi; base keys must be untouched by the suffix machinery.
    def clip(ds, mota_v=None):
        result = {
            "num_frames": 100, "num_objects": 200, "num_false_positives": 4,
            "num_misses": 10, "num_switches": 1, "num_unique_objects": 20,
            "num_fragmentations": 2, "idtp": 180, "idfp": 5, "idfn": 15,
            "motp": 0.2, "fp_tracks": 1, "fp_tracks_honest_v2": 1,
            "duration": 10.0, "mota": 0.9, "idf1": 0.9,
            "fp_per_frame": 0.04, "fn_per_obj": 0.05,
            "switch_per_obj": 0.05, "frag_per_obj": 0.1, "fitness": 0.85,
        }
        if mota_v is not None:
            result.update({
                "num_frames_vehicle": 100, "num_objects_vehicle": 50,
                "num_false_positives_vehicle": 2, "num_misses_vehicle": 5,
                "num_switches_vehicle": 0, "num_unique_objects_vehicle": 5,
                "num_fragmentations_vehicle": 0, "idtp_vehicle": 45,
                "idfp_vehicle": 2, "idfn_vehicle": 5, "motp_vehicle": 0.3,
                "fp_tracks_vehicle": 0, "fp_tracks_honest_v2_vehicle": 0,
                "duration_vehicle": 10.0, "mota_vehicle": mota_v,
                "idf1_vehicle": 0.9, "fitness_vehicle": mota_v,
                "fp_per_frame_vehicle": 0.02,
            })
        return {"params": {"ds_key": ds, "test_key": "t"}, "result": result}

    results = [clip("a", mota_v=0.86), clip("b")]
    config = {"fitness_weights": {"person": 1.0, "vehicle": 0.3}}
    rollups = tt.display_results(config, results, [], "fitness")
    overall = None
    for r in rollups:
        if r["params"]["ds_key"] == "_overall":
            overall = r["result"]
    assert overall is not None
    # base mota from summed base counts: 1 - (8+20+2)/400
    assert abs(overall["mota"] - (1 - 30 / 400)) < 1e-3
    # vehicle mota from vehicle counts of the ONE clip that has them
    assert abs(overall["mota_vehicle"] - (1 - 7 / 50)) < 1e-3
    assert "fitness_vehicle" in overall
    assert "fitness_multi" in overall
    expected = overall["fitness"] + 0.3 * overall["fitness_vehicle"]
    assert abs(overall["fitness_multi"] - expected) < 1e-9
