"""The allowed intra-package import graph (repo_cleanup.md section 2).

Each module is assigned a layer; an import is allowed only along the
listed edges. xfail(strict) until the split in stage 4/5 lands: the test
runs the real check today and must FAIL for the right reason (the god
modules violate the layering), then flip to a plain test.
"""
import ast
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")

# module (dotted, under src) -> layer. Modules not listed are "unknown"
# and fail the test so a new file has to be placed deliberately.
LAYER = {
    "paths": "paths",
    "trackset": "core", "track_util": "core",
    "upyc_tracker.upyc_tracker": "tracker", "trackers": "tracker",
    "trackset_import": "corpus", "dataset_lite": "corpus",
    "corpus_manifest": "corpus", "import_antare": "corpus",
    "autolabel_bridge": "corpus",
    "track_test": "eval", "eval_compare": "eval",
    "track_search": "search",
}
ALLOWED = {
    "paths": set(),
    "core": {"paths"},
    "tracker": {"paths", "core"},
    "corpus": {"paths", "core"},
    "eval": {"paths", "core", "tracker"},
    # search consults the corpus registry (data_tiers spec section 4), so
    # corpus is an allowed dependency of search by design.
    "search": {"paths", "core", "eval", "corpus"},
    "cli": {"paths", "core", "tracker", "corpus", "eval", "search"},
}


def _modules():
    for dp, _d, fns in os.walk(SRC):
        for fn in fns:
            if fn.endswith(".py") and not fn.startswith("test_") and fn != "__init__.py":
                rel = os.path.relpath(os.path.join(dp, fn), SRC)[:-3].replace(os.sep, ".")
                yield rel, os.path.join(dp, fn)


def _src_imports(path):
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("src."):
                    yield a.name[4:]
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src."):
            mod = node.module[4:]
            # `from src import x` -> module x
            yield mod
        elif isinstance(node, ast.ImportFrom) and node.module == "src":
            for a in node.names:
                yield a.name


def violations():
    bad = []
    for mod, path in _modules():
        layer = LAYER.get(mod)
        if layer is None:
            bad.append(f"{mod}: not assigned to a layer in LAYER")
            continue
        for dep in _src_imports(path):
            dl = LAYER.get(dep)
            if dl is None:
                bad.append(f"{mod} -> {dep}: target not in LAYER")
            elif dl != layer and dl not in ALLOWED[layer]:
                bad.append(f"{mod} ({layer}) -> {dep} ({dl}) not allowed")
    return bad


@pytest.mark.xfail(strict=True, reason="stage 4/5 of repo_cleanup.md: god modules still cross layers")
def test_import_graph():
    bad = violations()
    assert not bad, "\n".join(bad)


def test_import_graph_check_is_live():
    # guard against a vacuous pass: the parser must find a healthy number of
    # intra-package edges (name-independent so the split does not break it)
    edges = [(m, d) for m, p in _modules() for d in _src_imports(p)]
    assert len(edges) >= 8, edges
    assert all(d in LAYER or d.split(".")[0] in LAYER for _m, d in edges), edges
