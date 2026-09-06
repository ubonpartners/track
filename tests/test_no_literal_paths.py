"""No filesystem roots outside src/paths.py (repo_cleanup.md stage 2).

xfail(strict) until stage 2 lands; the scan is live today and must fail
because the literals exist.
"""
import ast
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
PATTERN = re.compile(r"(/mldata|/home/|~/)")      # anywhere in code, not only at string start
ALLOWED = {os.path.join(SRC, "paths.py"),                     # the one module that may name a root
           os.path.join(ROOT, "tests", "unit", "test_paths.py"),    # asserts the dev-box defaults
           os.path.abspath(__file__)}                                # names the pattern it hunts
SCAN_DIRS = (SRC, os.path.join(ROOT, "tools"), os.path.join(ROOT, "tests"))


def _files():
    yield os.path.join(ROOT, "track.py")          # the CLI is part of the package
    for base in SCAN_DIRS:
        for dp, _d, fns in os.walk(base):
            for fn in fns:
                p = os.path.join(dp, fn)
                if fn.endswith(".py") and p not in ALLOWED:
                    yield p


def _docstring_lines(src):
    """Line numbers covered by module/class/function docstrings."""
    out = set()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                out.update(range(body[0].lineno, body[0].end_lineno + 1))
    return out


def literal_paths():
    hits = []
    for p in _files():
        src = open(p).read()
        skip = _docstring_lines(src)
        for i, line in enumerate(src.split("\n"), 1):
            if i in skip:
                continue                           # prose may name a root
            code = line.split("#", 1)[0]           # so may a comment
            if PATTERN.search(code) or "expanduser(" in code:
                hits.append(f"{os.path.relpath(p, ROOT)}:{i}: {line.strip()[:80]}")
    return hits


def test_no_literal_paths():
    hits = literal_paths()
    assert not hits, "\n".join(hits)


def test_literal_scan_is_live():
    # the scanner must be looking at real code: at least a dozen modules
    assert sum(1 for _ in _files()) >= 12
