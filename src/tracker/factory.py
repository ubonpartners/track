"""Tracker factory (upyc only).

Moved verbatim from the former src/trackers.py (repo_cleanup.md stage 4c).
"""
import src.tracker.upyc as upyct


def create_tracker(
    param_dict,
    track_min_interval,
    debug_enable=False,
    start_time=0,
    end_time=1000000.0,
    classes=None,
):
    tracker_type = param_dict.get("tracker_type", "")
    if not tracker_type.startswith("upyc"):
        raise ValueError(
            f"track only supports upyc trackers now; got tracker_type={tracker_type!r}"
        )
    return upyct.upyc_tracker(
        param_dict,
        track_min_interval=track_min_interval,
        debug_enable=debug_enable,
        start_time=start_time,
        end_time=end_time,
        classes=classes,
    )
