"""Compatibility shim (repo_cleanup.md stage 4c; delete in stage 7).

The upyc wrapper lives in src/tracker/upyc.py.
"""
from src.tracker.upyc import (  # noqa: F401
    trim_aux_outputs,
    h264_for_video,
    RESULT_TYPE_NAMES,
    upyc_tracker,
    upyc_results_view,
)
