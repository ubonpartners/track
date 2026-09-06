"""Compatibility shim (repo_cleanup.md stage 4c; delete in stage 7).

TrackSet lives in src/core/trackset.py, the viewer in src/core/display.py,
and the tracker-driving import_create is now src.tracker.run.import_create(ts, ...).

Known out-of-repo importers (autolabel repo: eval/iterate.py, eval/score_sweep.py,
eval/error_report.py, eval/score_eval.py import src.trackset for TrackSet):
switch them to src.core.trackset before deleting this file.
"""
from src.core.trackset import TrackSet  # noqa: F401
from src.core.display import display_trackset, onoff  # noqa: F401
