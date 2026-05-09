# Track-trace regression library

Each trace captures a single (sequence, track_id) from a known corpus that
demonstrates a specific failure mode. Future head versions are regression-
tested against these by running the head over each trace's rows and
asserting that the head's outputs (and the cost-rule's induced state-
machine actions) satisfy a list of expected conditions.

Why this exists: every retrain has uncovered a fresh bug by walking
specific track traces on losing clips. Without a regression library, each
fix's success was measured only by the aggregate fitness number, which
hides per-issue regressions. With it, a head is "ready to ship" only if
every trace's assertions pass.

## Layout

```
bench/trace_library/
  __init__.py
  README.md            ← this file
  runner.py            ← `python -m bench.trace_library.runner --head HEAD.pt`
  capture.py           ← `python -m bench.trace_library.capture --corpus C.npz --seq S --tid T --out trace.npz`
  traces/
    {trace_id}.npz     ← one file per trace, includes inputs + meta + asserts
```

## Trace file format (.npz)

Each trace is a single `.npz` containing:

- `inputs` (T, in_dim) float32 — the head's per-row input matrix slice.
- `matched` (T,) bool — was this row matched this frame.
- `gt_id_now` (T,) int — GT id this row aligned with (-1 if none).
- `frame_idx` (T,) int — original frame index from the corpus.
- `meta_json` (str scalar) — JSON-encoded dict with:
  - `trace_id`, `issue`, `description`, `discovered`, `source_corpus`,
    `source_sequence`, `source_track_id`
  - `asserts`: list of dicts, each:
    - `rows`: row index (int) or "lo-hi" range or "all"
    - `field`: one of {`p_TP`, `mu_TP`, `mu_FP`, `cost_rule_state`}
    - `op`: one of {`>=`, `<=`, `==`, `>`, `<`}
    - `value`: float (for p_TP / mu_*) or string state name
              (UNCONFIRMED | TRACKED | LOST) for cost_rule_state
    - `tag`: short label for failure reporting

## Adding a trace

```bash
# 1. Pick a (corpus, sequence, track_id) that demonstrates the issue.
python -m bench.trace_library.capture \
    --corpus bench/data/state_corpus_v15ld_val.npz \
    --sequence INof_FD_OutFD_Light_FFcam_001 --track-id 42 \
    --trace-id mu_tp_collapse_occlusion \
    --issue mu_TP_collapse_during_occlusion \
    --description "Real TP track with rows 23-26 occluded; head should keep μ_TP > 1.0" \
    --out bench/trace_library/traces/mu_tp_collapse_occlusion.npz

# 2. Edit the meta JSON to add `asserts` (capture.py prints a stub set).

# 3. Run the runner against the head you want to test.
python -m bench.trace_library.runner --head bench/data/state_head_dc_v9.pt
```

## Reporting

`runner.py` exits 0 if every assertion passes; nonzero otherwise. CI / make
target should call it after a head is exported and before any full-178
bench is started — fast (a few hundred ms total).
