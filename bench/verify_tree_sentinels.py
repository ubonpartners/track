#!/usr/bin/env python3
"""Verify that schema-critical sentinels remain in the tree.

Why this exists: between 2026-05-08 and 2026-05-11 the v3 face-conf /
warped-subbox-DIoU schema additions were silently lost twice by careless
`git checkout <ref> -- <path>` operations that reverted in-flight work,
followed by commits made on the unverified base. Each loss cost ~half a
day of debugging because the downstream `--with-face` build path
appeared broken without anyone noticing the schema field was gone.

This script asserts the sentinels are present. Run it:
  - manually before any state-corpus / state-head / pair-trace commit
  - via the pre-commit hook installed at .git/hooks/pre-commit
  - as part of bench smoke tests

Exit code 0 = OK. Non-zero = at least one sentinel missing; the message
names which file/sentinel so you can recover the lost change before
committing on top of a reverted base.

Add a new sentinel here when a schema-level invariant is added that
must not silently regress (in_dim bumps, new pair-trace fields, new
required CLI args, new corpus _meta fields, etc.).
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UBON_CSTUFF = ROOT.parent / "ubon_cstuff"


def _file_contains(path: Path, needles: list[str | re.Pattern]) -> list[str]:
    """Return list of needle strings NOT found in `path`. Empty = all present."""
    if not path.exists():
        return [f"FILE MISSING: {path}"]
    text = path.read_text(errors="replace")
    missing = []
    for n in needles:
        if isinstance(n, re.Pattern):
            if not n.search(text):
                missing.append(f"regex {n.pattern!r}")
        else:
            if n not in text:
                missing.append(n)
    return missing


# (relative_path, [sentinel strings/regexes], purpose)
PY_SENTINELS = [
    (
        "src/analysis/pair_log_schema.py",
        ["PAIR_LOG_VERSION = 3", '"subbox_diou_warped"', "PAIR_LOG_DTYPE_V2"],
        "pair-trace v3 schema with backward-compat v2 decode",
    ),
    (
        "bench/build_state_corpus.py",
        ['"det_subbox_conf"', '"track_subbox_conf"', '"det_fiqa_score"',
         "--comment", "save_npz_with_meta",
         # Phase 29 scene-aggregate replay — silent removal of this in
         # commit e288eea (2026-05-10) caused the 0.29-fitness V2/V3
         # catastrophe diagnosed 2026-05-11. Never again without
         # disabling --with-scene at the same commit.
         "class SceneStats", "apply_scene_stats_to_examples",
         '"scene_promote_rate"', '"scene_mean_det_conf_TRACKED"'],
        "V3 corpus face-conf fields + Phase 29 SceneStats replay + "
        "mandatory --comment + provenance API",
    ),
    (
        "bench/train_state_head_decoupled.py",
        ["INPUT_DIM_FACE = 28", "--with-face", "require_npz_meta",
         # The fail-loud guard added in commit 877a2da. If this line
         # disappears, --with-scene will silently train on constants
         # again. The corpus-side SceneStats sentinel above is the
         # mirror; together they prevent both halves of the bug.
         "--with-scene requires scene-aggregate columns"],
        "V3 trainer face flag + corpus-meta enforcement + "
        "--with-scene fail-loud guard",
    ),
    (
        "bench/train_phase3.py",
        ["--comment", "require_npz_meta"],
        "Match-cost trainer corpus-meta enforcement + mandatory --comment",
    ),
    (
        "bench/build_pair_dataset.py",
        ["--comment", "save_npz_with_meta"],
        "Pair dataset builder mandatory --comment + provenance API",
    ),
    (
        "bench/_artefact_meta.py",
        ["def save_npz_with_meta", "def require_npz_meta",
         "def attach_npz_meta"],
        "NPZ provenance API (save + require + retrofit)",
    ),
]

C_SENTINELS = [
    (
        "src/track/utrack/nn_state.h",
        ["UTRACK_NN_STATE_GRU_IN_DIM_V3", "UTRACK_NN_STATE_GRU_IN_DIM_V2",
         "UTRACK_NN_STATE_GRU_IN_DIM_MAX"],
        "V1/V2/V3 in_dim sentinels for state-head GRU input",
    ),
    (
        "src/track/utrack/utrack_state.c",
        ["UTRACK_NN_STATE_GRU_IN_DIM_V3", "UTRACK_NN_STATE_GRU_IN_DIM_V2"],
        "V3 face-conf slots 25..27 in state feature builder",
    ),
    (
        "src/track/utrack/utrack_internal.h",
        ["param_subdiou_warped", "last_prev_subbox_conf"],
        "Warped-subbox-DIoU param + carry-over face conf field",
    ),
    (
        "src/track/utrack/utrack_match.c",
        ["compute_subbox_scores", "subbox_diou_warped"],
        "Subbox-score helper (plain IoU + warped DIoU) used in pair build",
    ),
]


def main() -> int:
    failures: list[str] = []
    for rel, needles, purpose in PY_SENTINELS:
        path = ROOT / rel
        missing = _file_contains(path, needles)
        if missing:
            failures.append(f"  [PY] {rel}  ({purpose})")
            for m in missing:
                failures.append(f"      missing: {m}")
    for rel, needles, purpose in C_SENTINELS:
        path = UBON_CSTUFF / rel
        missing = _file_contains(path, needles)
        if missing:
            failures.append(f"  [C ] {rel}  ({purpose})")
            for m in missing:
                failures.append(f"      missing: {m}")
    if failures:
        print(
            "verify_tree_sentinels.py: SCHEMA REGRESSION DETECTED\n\n"
            "The following schema-critical markers are missing. This\n"
            "usually means a recent `git checkout`, `stash`, or `reset`\n"
            "silently reverted in-flight work. Recover them BEFORE\n"
            "committing — see commit 02a0f87 + 49582a5 for the canonical\n"
            "state of the v3 face/diou schema, and feedback memory\n"
            "feedback_methodical_git_state.md for the procedure.\n",
            file=sys.stderr,
        )
        for line in failures:
            print(line, file=sys.stderr)
        return 1
    print("verify_tree_sentinels.py: OK — all v3 schema sentinels present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
