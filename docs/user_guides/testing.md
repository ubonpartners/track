# Testing

Three tiers, from cheap to expensive. Run the first before every
commit, the second before anything that touches the eval path, the
third by hand when the tracker runtime or the corpora change.

## 1. Unit suite (seconds, no GPU, no /mldata)

```
python -m pytest -q
```

Runs everything under `tests/` that is not marked `gpu` or `data`.
Today that is the whole suite: parsers on tiny fixtures, the media
recipes pinned to their ffmpeg command lines, the search parameter
logic, the CLI translation, and four structure tests that keep the
cleanup honest:

- `tests/test_import_graph.py` — the allowed import edges between the
  package layers (core, formats, corpus, tracker, eval, search, cli).
- `tests/test_no_literal_paths.py` — no `/mldata`, `/home` or `~/`
  literal outside `src/paths.py`.
- `tests/test_comment_hygiene.py` — dates and names appear in code
  comments only as `ledger YYYY-MM-DD <title>` references that resolve
  to a heading in `docs/ledger.md`.
- `tests/unit/test_paths.py` — the dev-box defaults and the `TRACK_*`
  overrides.

Needs the dev environment: the package imports `stuff` and the eval
runner imports `ubon_pycstuff`, both private. A hosted CI runner can
only run the four structure tests (see `.github/workflows/tests.yml`).

## 2. Smoke eval (about 15 seconds, GPU, /mldata)

```
python tests/smoke_eval.py --out /mldata/results/cleanup/<name>
python tests/smoke_eval.py --compare /mldata/results/cleanup/<previous> /mldata/results/cleanup/<name>
```

Three antare clips through the objective config and the shared-stream
runner. The compare must report every clip and rollup cell identical;
float differences under 1e-9 relative are printed as ulp-level noise
and ignored. The results dir also holds `provenance.json` (sha256 of the
objective and tracker yamls, git revision, dirty files), so a config
edit under the repo's feet shows up in the comparison.

## 3. Corpus and full-eval checks (minutes to an hour, by hand)

```
python -m src.cli corpus verify <corpus>      # tier-1 hashes
python -m src.cli corpus check <corpus>       # tier-2 conformance
python -m src.cli eval --split val --results-location /mldata/results/eval/<name>
python -m src.eval_compare /mldata/results/eval/<before> /mldata/results/eval/<name>
```

Mark tests that need the tracker runtime with `@pytest.mark.gpu` and
tests that read `/mldata` with `@pytest.mark.data`; they are skipped by
`-m "not gpu and not data"` and run with `python -m pytest -m gpu` /
`-m data` on the dev box.

## Layout rules

- No `__init__.py` under `tests/` (another repo on `PYTHONPATH` owns a
  top-level `tests` package) and unique test basenames across
  subdirectories.
- Tests import `src.core...`, `src.eval...` etc. directly, never the
  three shims kept for the autolabel repo.
