"""Compatibility shim, KEPT for the autolabel repo (ledger 2026-09-06
Repo cleanup): delete once autolabel imports src.corpus.manifest directly.

The registry lives in src/corpus/manifest.py, derivation and the tier-2
check in src/corpus/derive.py. The autolabel repo imports set_audit from
here (eval/gt_audit.py): switch it before deleting this file.
"""
from src.corpus.manifest import (  # noqa: F401
    T1,
    CAPABILITIES_SEED,
    corpus_info,
    _sha256,
    _files,
    build,
    load_capabilities,
    allows,
    set_audit,
    set_file_source,
    verify,
    main,
)
from src.corpus.derive import (  # noqa: F401
    derive_tracking,
    LEGACY_DIRS,
    check_tracking,
)


if __name__ == "__main__":
    main()
