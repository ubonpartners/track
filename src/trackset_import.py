"""Compatibility shim (repo_cleanup.md stage 4b; delete in stage 7).

The converters now live in src/corpus/importers.py, the ffmpeg helpers in
src/corpus/media.py and the one-off migrations in src/corpus/migrations.py.
"""
from src.corpus.media import (  # noqa: F401
    frame_pts_monotonic as _frame_pts_monotonic,
    remux_to_mp4 as _remux_to_mp4,
    video_codec as _video_codec,
    native_fps as _native_fps,
    transcode_h264 as _transcode_h264,
)
from src.corpus.importers import (  # noqa: F401
    convert_mot,
    convert_personpath22,
    convert_jaad,
    _convert_meva_clip,
    convert_chirla,
    convert_roundabouthd,
    convert_uvg_vcm,
    convert_meva,
    convert_otw,
    convert_cevo,
    convert_bdd100k_kaggle,
    convert_autolabel_folder,
    convert_raw_movies,
    convert_bwc_videotext,
    reduce_dataset,
    apply_reduction,
    fix_cevo25_vfr_times,
)
from src.corpus.migrations import (  # noqa: F401
    estimate_bdd_time_offsets,
    dofix,
)
