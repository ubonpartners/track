"""Compatibility shim (repo_cleanup.md stage 4d; delete in stage 7).

The lite/derive helpers live in src/corpus/derive.py.
"""
from src.corpus.derive import (  # noqa: F401
    choose_divisor,
    min_delta_from_config,
    divisor_from_config,
    scale_dims,
    probe,
    MP4_COPY_AUDIO,
    probe_audio,
    audio_args,
    has_backward_pts,
    rewrite_annotation,
    transcode,
    process_dataset,
    main,
)


if __name__ == "__main__":
    import sys
    sys.exit(main())
