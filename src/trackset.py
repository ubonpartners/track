"""Compatibility shim (repo_cleanup.md stage 4c; delete in stage 7).

TrackSet lives in src/core/trackset.py, the viewer in src/core/display.py,
and the tracker-driving import_create is now src.tracker.run.import_create(ts, ...).
"""
from src.core.trackset import TrackSet  # noqa: F401
from src.core.display import display_trackset, onoff  # noqa: F401
