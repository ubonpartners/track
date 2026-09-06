"""No filesystem roots outside src/paths.py (repo_cleanup.md stage 2).

xfail(strict) until stage 2 lands; the scan is live today and must fail
because the literals exist.
"""
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
PATTERN = re.compile(r"[\"'](/mldata|/home/|~/)")
ALLOWED = {os.path.join(SRC, "paths.py")}      # the one module that may name a root


def _files():
    yield os.path.join(ROOT, "track.py")          # the CLI is part of the package
    for dp, _d, fns in os.walk(SRC):
        for fn in fns:
            p = os.path.join(dp, fn)
            if fn.endswith(".py") and p not in ALLOWED:
                yield p


def literal_paths():
    hits = []
    for p in _files():
        for i, line in enumerate(open(p), 1):
            s = line.split("#", 1)[0]          # comments may mention paths
            if PATTERN.search(s):
                hits.append(f"{os.path.relpath(p, ROOT)}:{i}: {line.strip()[:80]}")
    return hits


def test_no_literal_paths():
    hits = literal_paths()
    assert not hits, "\n".join(hits)


def test_literal_scan_is_live():
    # the scanner must be looking at real code: at least a dozen modules
    assert sum(1 for _ in _files()) >= 12
