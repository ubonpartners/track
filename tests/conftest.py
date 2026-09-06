"""Shared fixtures for the unit tests (repo_cleanup.md stage 0).

Everything here is synthetic and tiny: no GPU, no /mldata. Tests that
need either carry the `gpu` / `data` marker and are skipped by default
in CI (`pytest -m "not gpu and not data"`).

Layout rules: no `__init__.py` anywhere under tests/ (PYTHONPATH carries
another repo whose top-level `tests` package would shadow ours; pytest
loads this conftest by path, which is why it works), and test basenames
must be unique across tests/unit, tests/data and tests/gpu (pytest's
prepend import mode requires it).
"""
import json
import os

import pytest


def write_trackset_json(path, frames, classes=("person", "vehicle", "other"),
                        fps=10.0, w=1280, h=720, extra_metadata=None):
    """Write an annotation json in track's format.

    frames: [{"time": t, "objects": {tid: (box, cl)}}]  (box normalised x1,y1,x2,y2)
    Returns the path as str.
    """
    md = {"frame_rate": fps, "width": w, "height": h, "classes": list(classes)}
    md.update(extra_metadata or {})
    out = {"metadata": md, "frames": []}
    for i, fr in enumerate(frames):
        objs = {str(tid): {"box": list(box), "class": cl, "conf": 1.0}
                for tid, (box, cl) in fr["objects"].items()}
        out["frames"].append({"frame_id": i + 1, "frame_time": fr["time"],
                              "objects": objs})
    with open(path, "w") as f:
        json.dump(out, f)
    return str(path)


@pytest.fixture
def tiny_trackset_json(tmp_path):
    """Two frames, one person and one vehicle, 10 fps. Returns the path."""
    frames = [
        {"time": 0.0, "objects": {1: ((0.1, 0.1, 0.2, 0.3), 0),
                                  2: ((0.5, 0.5, 0.7, 0.6), 1)}},
        {"time": 0.1, "objects": {1: ((0.11, 0.1, 0.21, 0.3), 0),
                                  2: ((0.51, 0.5, 0.71, 0.6), 1)}},
    ]
    return write_trackset_json(tmp_path / "tiny.json", frames)


@pytest.fixture
def tiny_trackset(tiny_trackset_json):
    """The same clip loaded as a TrackSet."""
    import src.trackset as ts
    return ts.TrackSet(tiny_trackset_json)
