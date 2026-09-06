"""Compatibility shim (repo_cleanup.md stage 4a; delete in stage 7).

The eval engine now lives in src/eval/{matching,metrics,runner,report}.
Everything is re-exported here so old imports keep working, EXCEPT the
mutable module global PM_OVERRIDE: set src.eval.runner.PM_OVERRIDE
instead (a copy here would be silently ignored by the runner).
"""
from src.eval.matching import (  # noqa: F401
    mot_obj,
    permissive_iou_matrix,
    permissive_iou,
    _box_in_ignore,
)
from src.eval.metrics import (  # noqa: F401
    _FP_TRACK_COEF,
    fitness_score,
    gt_class_box_counts,
    fitness_multi_score,
    annotation_floors,
    compute_detection_metrics,
    _MATCHED_TYPES,
    _events_by_hid_from_df,
    _honest_fp_runs_core,
    compute_metrics,
    score_tracksets,
)
from src.eval.runner import (  # noqa: F401
    track_test_work_fn,
    _clip_meta,
    _parse_packed_results,
    _single_metrics_worker_packed,
    _single_metrics_worker,
    _resolve_pm,
    run_single_shared,
    on_result_callback,
    track_test,
)
from src.eval.report import (  # noqa: F401
    summary_string,
    result_string,
    get_avg_scores,
    display_results,
    _summary_metric_keys,
    _result_subset,
    _write_eval_summary_json,
    _write_eval_summary_html,
)
