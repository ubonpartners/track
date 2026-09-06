"""Code comments state the rule; the story lives in docs/ledger.md
(repo_cleanup.md stage 8). A date or a person's name may appear in a
comment or docstring only as a ledger reference of the form
`ledger YYYY-MM-DD <title>`, and every such reference must resolve to a
ledger heading `## YYYY-MM-DD <title>...`."""
import ast
import io
import os
import re
import tokenize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
LEDGER = os.path.join(ROOT, "docs", "ledger.md")
DATE = re.compile(r"\b20\d\d-\d\d(?:-\d\d)?\b")
NAMES = re.compile(r"\bMB\b|\bMark\b")
REF = re.compile(r"ledger (20\d\d-\d\d-\d\d) ([^):;\n]+)")


def _comment_and_docstring_lines(path):
    src = open(path).read()
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            out.append((tok.start[0], tok.string))
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                for i, line in enumerate(body[0].value.value.split("\n")):
                    out.append((body[0].lineno + i, line))
    return out


def _files():
    for dp, _d, fns in os.walk(SRC):
        for fn in fns:
            if fn.endswith(".py"):
                yield os.path.join(dp, fn)


def _headings():
    heads = []
    for line in open(LEDGER):
        m = re.match(r"## (20\d\d-\d\d-\d\d) (.+)$", line.strip())
        if m:
            heads.append((m.group(1), m.group(2)))
    return heads


def _strip_refs(text):
    return REF.sub("", text)


def test_no_dates_or_names_outside_ledger_refs():
    bad = []
    for p in _files():
        for ln, text in _comment_and_docstring_lines(p):
            rest = _strip_refs(text)
            if DATE.search(rest) or NAMES.search(rest):
                bad.append(f"{os.path.relpath(p, ROOT)}:{ln}: {text.strip()[:90]}")
    assert not bad, "\n".join(bad)


def test_every_ledger_reference_resolves():
    heads = _headings()
    assert heads, "docs/ledger.md has no dated headings"
    bad = []
    for p in _files():
        for ln, text in _comment_and_docstring_lines(p):
            for date, title in REF.findall(text):
                title = title.strip().rstrip(".,")
                if not any(d == date and h.lower().startswith(title.lower()) for d, h in heads):
                    bad.append(f"{os.path.relpath(p, ROOT)}:{ln}: ledger {date} {title}")
    assert not bad, "\n".join(bad)
