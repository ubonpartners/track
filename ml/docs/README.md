# ml/ — utrack training pipeline

This directory contains the **Python training pipeline** for the two
learned components of the shipping pedestrian tracker:

  - **Match-cost NN** — a small two-tower MLP that emits a residual
    on top of the heuristic match score. Used inside the Hungarian
    matcher to disambiguate hard pairs.
  - **State head GRU** — a 19-input, 3-output GRU that decides
    per-track promote / re-promote actions and predicts μ_TP / μ_FP
    for the Bayesian cost rule.

The tracker itself (the **C runtime**) lives in
`ubon_cstuff/src/track/utrack/`. The two `.bin` artefacts produced
here are loaded at runtime via the path keys `utrack.nn_path` and
`utrack.nn_state_path` in the production yaml at
`/mldata/config/track/trackers/uc_v11.yaml`.

If you just want to read about results, jump straight to
[EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md). If you want to run a
retrain, read on.

---

## Mental model

```
                ┌────────────────────────────────────────┐
                │ ubon_cstuff/  (C runtime)              │
                │   utrack.c        loads nn_match.bin   │
                │   utrack_match.c    + nn_state.bin     │
                │   utrack_state.c                       │
                │   utrack_pair_trace.h ↔  schema mirror │
                └────────────────┬───────────────────────┘
                                 │  produces per-frame
                                 │  match-cost trace +
                                 │  UBTRK2 trackset
                                 ▼
       ┌─────────────────────────────────────────────────────────┐
       │ track/  (this repo)                                     │
       │                                                         │
       │   track.py --track / --eval / --search                  │  ← run tracker, measure
       │   track_analysis.py                                     │  ← emit labelled pair-logs
       │   src/pair_log.py + pair_log_schema.py                  │  ← schema + labeller
       │                                                         │
       │   ml/ (this dir)                                        │
       │     data_prep/    pair-log .npz   → labelled corpus     │  ← train data builders
       │     train/        state head + match NN trainers        │  ← .pt checkpoints
       │     export/       .pt → C-readable .bin                 │  ← runtime artefacts
       │     eval/         closed-loop fitness measurement       │  ← did it actually help?
       │     analysis/     permute-importance, dead-feature      │  ← interpret models
       │     cmc/          camera-motion-comp diagnostics        │  ← orthogonal CMC bug-hunt
       │     orchestration/ end-to-end recipe shell scripts      │  ← reproduce the boot
       │     util/         metadata sidecar + pre-commit guard   │  ← provenance + safety
       │     trace_library/ regression-test fixtures             │  ← debug a single track
       │     configs/      pair-log + eval yamls                 │  ← experiment recipes
       │     docs/         this directory                        │
       │     data/         corpora + .pt + .bin  (gitignored)    │
       └─────────────────────────────────────────────────────────┘
```

The two repos share two binary contracts:
1. `src/pair_log_schema.py` ↔ `ubon_cstuff/src/track/utrack/utrack_pair_trace.h` —
   the per-pair trace record. Diff'd at commit time by
   `ml/util/verify_tree_sentinels.py` (installed as pre-commit hook).
2. `.bin` file format — produced by `ml/export/`, loaded by the C runtime.
   Stable magic + version headers (`UP3P` for match NN, `USHT` for
   state head).

---

## Currently shipped artefacts

Both binaries live at `/mldata/config/track/trackers/`:

| Path                              | Magic     | Trainer                                 | Exporter                                | Built from        |
|-----------------------------------|-----------|-----------------------------------------|-----------------------------------------|-------------------|
| `nn_match_v13.bin`                | UP3P v1   | `ml.train.train_phase3`                 | `ml.export.export_phase3`               | `iter1_d05v2`     |
| `nn_state_v23_pw05.bin`           | USHT v3   | `ml.train.train_state_head_decoupled`   | `ml.export.export_decoupled_head`       | `iter1_d05v2`     |

The shipping production yaml is
`/mldata/config/track/trackers/uc_v11.yaml`. As of 2026-05-15 (F5d
ship), the relevant utrack knobs are:

```yaml
utrack:
  match_cheap_filter_delta: 0.7   # delta-from-top NN-skip gate
  delete_dup_iou:           0.70  # post-match dedup
  nn_path:        /mldata/config/track/trackers/nn_match_v13.bin
  nn_state_path:  /mldata/config/track/trackers/nn_state_v23_pw05.bin
  nn_lambda:      0.05            # weight of NN residual in match cost
  bayes_c_FP_track: 0.0007        # cost-rule constants
  bayes_c_FP_frame: 0.002
  bayes_c_MOTA:     0.001
```

Every `.bin` ends with a `META` trailer (magic + JSON, ignored at
runtime, readable via `python -m ml.util._artefact_meta --read <file>`)
holding full provenance: argv, git rev, hyperparams, dataset, host,
UTC timestamp. The same provenance is also written to a sibling
`.meta.json` for grep-friendliness.

---

## Quick reference: where things live

### Code

| Concern                                | Module path                                  |
|----------------------------------------|----------------------------------------------|
| Build labelled pair-log corpus         | `ml.data_prep.build_pair_dataset`            |
| Build state-corpus from pair-log + GT  | `ml.data_prep.build_state_corpus`            |
| Train match-cost two-tower NN          | `ml.train.train_phase3`                      |
| Train state-head decoupled GRU         | `ml.train.train_state_head_decoupled`        |
| Export match NN to `.bin`              | `ml.export.export_phase3`                    |
| Export state head to `.bin`            | `ml.export.export_decoupled_head`            |
| Closed-loop fitness eval               | `ml.eval.eval_head_fitness`                  |
| Offline state-head eval (no tracker)   | `ml.eval.eval_decoupled_offline`             |
| Permutation feature importance         | `ml.analysis.permute_match_features`         |
| Cheap-filter sweep tool                | `ml.analysis.cheap_filter_analysis`          |
| CMC reference comparison               | `ml.cmc.cmc_compare`                         |
| Provenance helpers                     | `ml.util._artefact_meta`                     |
| Yaml-key validator                     | `ml.util._pipeline_checks`                   |
| Schema sentinel guard (pre-commit)     | `ml.util.verify_tree_sentinels`              |
| Pair-log emission (runs the tracker)   | `src.pair_log`  + CLI `track_analysis.py`    |
| Binary record schema                   | `src.pair_log_schema`                        |

### Configs

| Type                              | Path                                                       |
|-----------------------------------|------------------------------------------------------------|
| Pair-log emission yamls           | `ml/configs/pair_log_config_*.yaml`                        |
| Eval yamls (default subsets)      | `ml/configs/eval_base_*.yaml`                              |
| Production tracker yaml           | `/mldata/config/track/trackers/uc_v11.yaml`                |
| No-NN baseline                    | `/mldata/config/track/eval/uc_v11_no_nn.yaml`              |
| Canonical "ship status" eval     | `/mldata/config/track/eval/eval_ship_baseline.yaml`        |

### Data (gitignored, regenerable)

| Type                              | Path                                          |
|-----------------------------------|-----------------------------------------------|
| Pair-log NPZs (raw per-clip)      | `/mldata/track_analysis_runs/<run>/pair_log/` |
| Cached UBTRK2 tracker outputs     | `/mldata/track_analysis_runs/<run>/ubtrk2/`   |
| Aggregated pair datasets          | `ml/data/<corpus>/pairs_{train,val,test}.npz` |
| State corpora                     | `ml/data/<corpus>/state_corpus_{train,val,test}.npz` |
| Trained `.pt` checkpoints         | `ml/data/<corpus>/phase3.pt`, `state_head_*.pt` |
| Local `.bin` exports               | `ml/data/<corpus>/nn_match.bin`, `state_head_*.bin` |

---

## End-to-end retrain workflow

These six steps reproduce the currently shipping
`nn_match_v13.bin` + `nn_state_v23_pw05.bin` pair from the
`iter1_d05v2` corpus. Use them as a template for new retrains —
swap the corpus name and pair-log dir, keep the structure.

### 1. Emit pair-logs (run the C tracker with pair_logger enabled)

```bash
python track_analysis.py \
  --config ml/configs/pair_log_config_iter0_noNN_jaad.yaml \
  --split train
# repeats for --split val, then --split test
```

This runs the C tracker on every clip listed in the analysis yaml,
caches the UBTRK2 trackset under `output_root/ubtrk2/<seq>.ubtrk2`,
and writes a labelled NPZ to `output_root/pair_log/<seq>.npz`. Cached
UBTRK2 files are reused across runs unless `--force-regen` is given.

### 2. Aggregate pair-logs into a training corpus

```bash
python -m ml.data_prep.build_pair_dataset \
  --pair-log-dir /mldata/track_analysis_runs/pair_log_iter0_noNN_jaad/pair_log \
  --analysis-yaml ml/configs/pair_log_config_iter0_noNN_jaad.yaml \
  --split train \
  --delta-filter 0.5 \
  --out ml/data/iter1_d05v2/pairs_train.npz \
  --comment "iter1_d05v2 — δ=0.5 + thr-aware gate"
# repeat for --split val, --split test
```

The `--delta-filter` argument is the **cheap-filter training mirror**:
drop training rows whose pre-NN score is ≥ δ below the per-event
top score, so the model trains on the same input distribution the
runtime cheap-filter passes through (see
[EXPERIMENT_HISTORY.md § matching-cheap-filter](EXPERIMENT_HISTORY.md)).

`--comment` is **required** — without it the provenance trailer
can't be reconstructed.

### 3. Train the match-cost two-tower

```bash
python -m ml.train.train_phase3 \
  --epochs 25 \
  --data_dir ml/data/iter1_d05v2 \
  --save ml/data/iter1_d05v2/phase3.pt \
  --seed 0 \
  --comment "iter1_d05v2 match-cost"
```

`--save` is required so produced models can't be silently lost.
Face/subbox feature inputs auto-detect from the corpus dtype —
`obs_in=16, pair_in=19` if the v3 face/subbox columns are present,
else 13/15. **All A1 conclusions are single-seed** — multi-seed
verification is needed before shipping (see
`project_a1_pw_sweep_was_seed_luck.md` user memory).

### 4. Build the state corpus on top of the new match NN

```bash
python -m ml.data_prep.build_state_corpus \
  --pair-log-dir /mldata/track_analysis_runs/pair_log_iter0_noNN_jaad \
  --gt-config    ml/configs/pair_log_config_iter0_noNN_jaad.yaml \
  --label-driven \
  --phase3-model ml/data/iter1_d05v2/phase3.pt \
  --out          ml/data/iter1_d05v2/state_corpus \
  --comment      "iter1_d05v2 state corpus — phase3 trained on δ=0.5 filtered pair-log"
```

Label-driven replay uses the GT history as oracle for state
transitions; the C runtime's hard floors (`missed≥2`, `K_min`,
buffer) are the safety ceilings. The `--phase3-model` argument is
mandatory — `f_obs` is replayed through the new match NN to fill
the obs-embedding column, so the state corpus is consistent with
the match NN you just trained.

### 5. Train the decoupled state head

```bash
python -m ml.train.train_state_head_decoupled \
  --train ml/data/iter1_d05v2/state_corpus_train.npz \
  --val   ml/data/iter1_d05v2/state_corpus_val.npz \
  --save  ml/data/iter1_d05v2/state_head_pw0.5.pt \
  --epochs 16 --seed 0 --hidden 64 --pos-weight 0.5 \
  --comment "iter1_d05v2 state head, h=64 pw=0.5"
```

The decoupled head is **19-dim input → 3 outputs**: LLR, log μ_TP,
log μ_FP. The C runtime converts to (p_TP, μ_TP, μ_FP) and feeds
them into the Bayesian cost rule. Demote is not modelled — TRACKED
tracks die via the unified deletion pass (gate constants are pinned
in the production yaml).

**Multi-seed for variance bound:** train at least seeds 0, 1, 2 at
the chosen pw before declaring a "win" — σ across seeds can be as
high as 0.013 fitness (see F3 result in
[EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md)).

### 6. Export to the runtime binary format

```bash
python -m ml.export.export_phase3 \
  --in  ml/data/iter1_d05v2/phase3.pt \
  --out ml/data/iter1_d05v2/nn_match.bin

python -m ml.export.export_decoupled_head \
  --in  ml/data/iter1_d05v2/state_head_pw0.5.pt \
  --out ml/data/iter1_d05v2/state_head_pw0.5.bin
```

The exporters reject `in_dim != 19` for the state head, since the C
runtime hardcodes the input width. They also write the `META`
trailer and the sibling `.meta.json`. To inspect:

```bash
python -m ml.util._artefact_meta --read ml/data/iter1_d05v2/nn_match.bin
```

---

## Evaluating a candidate

### Closed-loop fitness (the real metric)

This is the only metric that's allowed to drive ship decisions.
See `feedback_track_eval_metric.md` user memory: AUC, MOTA, and IDF1
in isolation are all misleading.

```bash
# Drop the new bin into a fresh tracker yaml
cp /mldata/config/track/trackers/uc_v11.yaml /tmp/uc_v11_candidate.yaml
# Edit /tmp/uc_v11_candidate.yaml → utrack.nn_state_path: /path/to/new/state_head.bin

# Compare against ship + no-NN baselines on full-176 + JAAD val
python track.py --eval /mldata/config/track/eval/eval_ship_baseline.yaml
```

The orchestrator config sweeps 2 variants × 205 clips = 410 clip-runs
(~7 min wall on a 4-worker queue). Results land in
`/mldata/track_runs/eval_ship_baseline/results/results-<timestamp>.txt`
with `__ovrfull176` and `__ovrjaadval` aggregate rows.

**Fitness** (the primary metric) = `MOTA − 5e-4 × fp_tracks_total`.
The fp_track penalty is what punishes the "more recall at any cost"
direction. See user memory `feedback_track_eval_metric.md`.

### Offline state-head eval

For state-head retrains, `ml/eval/eval_decoupled_offline.py` runs the
decoupled head against the state corpus and reports LLR-AUC, lifetime
MAE, mota_proxy / fitness_proxy. **These numbers do not predict
closed-loop fitness** (see the V2/V3 catastrophe in
[EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md) for the canonical
example). Useful only as a sanity check for early stopping during
training.

### Trace library

`ml/trace_library/` contains 6 hand-picked debugging traces (single
tracks captured from real clips, packaged with expected-behaviour
assertions). They probe failure modes like "μ_TP collapses during
occlusion" or "consistent FP not suppressed". Run:

```bash
python -m ml.trace_library.runner --head ml/data/<corpus>/state_head_*.pt
```

Trace-library pass-rate and fitness are not the same axis — v14
won by aggregate fitness but failed 9/14 traces. Treat traces as
"does this head fail in the specific ways we've already named?"
rather than "should we ship?".

---

## Common pitfalls (learned the hard way)

The user memory files at
`~/.claude/projects/-home-mark-stuff-ubonpartners/memory/` accumulate
the actual lessons. Quick summary of the ones that bite hardest:

- **A1-style single-seed sweeps are unreliable** for state-head
  hyperparameter selection. State-head training is bimodal at
  several pos_weight settings. σ across seeds can hit 0.013 fitness —
  bigger than most apparent "wins". Always K-seed (K≥3) before
  declaring a fitness gain over a previous ship. See
  `project_a1_pw_sweep_was_seed_luck.md`.
- **Default eval to runs=1**. runs=3 only when two candidates differ
  by < 0.003. runs=3 wastes 3× wall time on screening sweeps.
  `feedback_eval_runs_default.md`.
- **Estimate wall-clock BEFORE launching.** Always do a 30-second
  dry-run on a 1-clip subset before kicking off anything that takes
  > 5 min. `feedback_preflight_long_runs.md`,
  `feedback_time_is_primary.md`.
- **Trace config knobs before proposing them.** YAML keys can be
  silently ignored if the C side doesn't read them. The
  `ml.util._pipeline_checks` validator catches this for utrack keys.
  `feedback_verify_config_knobs.md`.
- **Corpus regen drifts retrains** worse than shipped — every retrain
  on a freshly built corpus has underperformed shipped by
  0.04–0.17 fitness. Always also report within-corpus delta vs
  same-corpus baseline. `feedback_track_corpus_drift.md`.
- **More JAAD data does NOT close the cross-domain gap.** Both D1
  (promote JAAD test → train, +117 clips) and D2 (oversample JAAD
  pairs 4×) regressed full-176 and did not move JAAD val. See
  `project_d1_jaad_test_promotion.md`.
- **Don't pitch cheap-filter work as a speed optimisation.** G1
  wall-profile showed δ ∈ {0.0, 0.5, 0.7, 1.0} all within ±1% wall.
  Detector dominates; NN evals weren't constraining throughput.
  `project_cheap_filter_speed_neutral.md`.
- **`nn_lambda` is the IDF1 dial.** Monotonic IDF1 gain with
  increasing λ, paid for in fp_tracks. 0.05 is fitness-optimal; 0.10+
  for IDF1-leaning deployments. `project_nn_lambda_idf1_dial.md`.

---

## Shipping a new model

1. Train + multi-seed verify the candidate (see "End-to-end retrain"
   above). Pick the seed with median fitness, not the best.
2. Copy the `.bin` files to `/mldata/config/track/trackers/` with a
   bumped version suffix (`nn_match_v14.bin`,
   `nn_state_v24_pw05.bin`, etc).
3. Edit `/mldata/config/track/trackers/uc_v11.yaml`:
   `nn_path` / `nn_state_path` → new files.
4. Run the comparison eval:
   `python track.py --eval /mldata/config/track/eval/eval_ship_baseline.yaml`.
   Confirm fitness on full-176 ≥ ship − 0.003 (eval noise band), and
   no JAAD regression of more than 0.003.
5. Update the recipe comment block in `uc_v11.yaml` to point at the
   new bin lineage + git hashes. Append a row to
   [EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md).
6. Commit `/mldata/config` with a message describing the fitness/MOTA
   delta and how multi-seed variance was bounded.

Snapshot the prior ship yaml before each new ship if you need an A/B
baseline — copy `uc_v11.yaml` to a versioned filename in
`/mldata/config/track/eval/` and add it to `eval_ship_baseline.yaml`'s
`tests:` map. The current pattern is to keep only the live ship and a
no-NN reference under `track/eval/`.

---

## Reproducibility chain — what's enforced where

- **Provenance trailer** in every `.pt` and `.bin` carries the full
  command-line + git rev + hyperparams + dataset path. `ml.util._artefact_meta`
  reads it back.
- **`--comment` is mandatory** on every trainer and dataset builder so
  the provenance trailer can record human-readable intent.
- **Schema sentinels** (`ml/util/verify_tree_sentinels.py`) check that
  schema-critical files contain the expected v3 field markers before
  every commit (via the pre-commit hook at `ml/githooks/pre-commit`).
  Activate the hook per-clone with
  `git config --local core.hooksPath ml/githooks`. Catches the
  "silent revert" failure mode that cost ~2 days in May 2026.
- **Yaml-key validator** (`ml/util/_pipeline_checks.py`) rejects
  unknown / dead utrack keys at config-load time. The known-keys list
  mirrors the `yaml_base["..."]` reads in `utrack.c`.

---

## Where to look next

- **What's been tried, what shipped, what failed**: see
  [EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md).
- **Detailed end-to-end retrain checklist**: see
  [PIPELINE.md](PIPELINE.md).
- **Per-feature importance for the match NN**: see
  [FEATURE_AUDIT.md](FEATURE_AUDIT.md).
- **C-runtime contract** (the schema files the Python side mirrors):
  `ubon_cstuff/src/track/utrack/utrack_pair_trace.h`,
  `ubon_cstuff/src/track/utrack/nn_state.h`.
- **Tracker debug format** (UBTRK2): `../../../TRACKER_DEBUG.md`.
- **User memory** (lessons + preferences carried across sessions):
  `~/.claude/projects/-home-mark-stuff-ubonpartners/memory/MEMORY.md`.
