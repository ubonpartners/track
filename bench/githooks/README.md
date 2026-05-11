# Tracked git hooks

Activate with:

    git config --local core.hooksPath bench/githooks

This is set per-clone (it lives in `.git/config`, not in tracked files),
so anyone cloning the repo needs to run the command once.

## Hooks

### `pre-commit`

Runs `python bench/verify_tree_sentinels.py` when the staged set touches
any of the schema-critical files (pair-trace schema, state-corpus
builder, state-head trainer, match-cost trainer, NPZ provenance API, or
the V1/V2/V3 in_dim sentinels in `ubon_cstuff`).

The script asserts that the v3 face-conf + warped-subbox-DIoU schema
fields are still present in the tree. Catches the failure mode where a
careless `git checkout <ref> -- <path>` silently reverts in-flight v3
work and the next commit is made on the regressed base. This pattern
cost ~2 lost days between 2026-05-08 and 2026-05-11; the hook ensures
it can't recur silently.

To bypass for a deliberate schema removal:

    git commit --no-verify

(...and update `bench/verify_tree_sentinels.py` to reflect the new
invariants in the same commit.)
