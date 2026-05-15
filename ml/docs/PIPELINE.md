# PIPELINE — end-to-end retrain recipe

This is the short, current, executable recipe for producing the two
shipped runtime binaries from scratch. For the *why* behind each step
and the chronological experiment history, see
[EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md). For the conceptual
overview, see [README.md](README.md).

The currently shipped artefacts (as of 2026-05-15, F5d) are:

| Runtime path                                                  | Source `.pt`                                  |
|---------------------------------------------------------------|-----------------------------------------------|
| `/mldata/config/track/trackers/nn_match_v13.bin`              | `ml/data/iter1_d05v2/phase3.pt`               |
| `/mldata/config/track/trackers/nn_state_v23_pw05.bin`         | `ml/data/iter1_d05v2/state_head_pw0.5.pt`     |

Both bins carry a `META` trailer with full provenance — inspect with:

    python -m ml.util._artefact_meta --read /mldata/config/track/trackers/nn_match_v13.bin

The reproducible launch wrapper for the full pipeline is
[`../orchestration/bootstrap_recipe.sh`](../orchestration/bootstrap_recipe.sh).
The steps below are what it runs.

---

## 0. Inputs

You need:

- A **pair-log directory** built from a tracker run over a labelled
  corpus, with the analysis app's pair-logger enabled. Currently shipped
  bins were trained from
  `/mldata/track_analysis_runs/pair_log_iter0_noNN_jaad/pair_log` (full
  176-clip corpus + 117 JAAD test clips, no-NN tracker pass, permissive
  thresholds so the trace covers candidate pairs the shipped tracker
  would have dropped).
- The matching **analysis yaml** that was used to produce the pair-log,
  e.g. `ml/configs/pair_log_config_iter0_noNN_jaad.yaml`. The
  `--analysis-yaml` flag is required so the downstream stage knows
  exactly which thresholds were in force.

To regenerate the pair-log from scratch, run `track_analysis.py` at the
repo root over the labelled corpus. The config contract is the same
analysis yaml used for `--analysis-yaml` in step 1 below; see
`ml/configs/` for canonical examples.

---

## 1. Build match-NN training corpus

    for split in train val test; do
      python -m ml.data_prep.build_pair_dataset \
          --pair-log-dir /mldata/track_analysis_runs/pair_log_iter0_noNN_jaad/pair_log \
          --analysis-yaml ml/configs/pair_log_config_iter0_noNN_jaad.yaml \
          --split $split \
          --delta-filter 0.5 \
          --out ml/data/iter1_d05v2/pairs_${split}.npz \
          --comment "iter1 bootstrap, delta=0.5 cheap-filter, $split"
    done

`--delta-filter 0.5` keeps only the pairs the runtime cheap-filter would
have evaluated, so train- and inference-time distributions match. The
train-split run also writes `feature_norm.json` next to the npzs.

`--comment` is mandatory — it lands in the npz `_meta` trailer alongside
argv / git / hostname / UTC.

## 2. Train the match-cost NN

    python -m ml.train.train_phase3 \
        --epochs 25 \
        --data_dir ml/data/iter1_d05v2 \
        --save ml/data/iter1_d05v2/phase3.pt \
        --seed 0

Defaults (two-tower, 16+5+19 input features, BCE+contrastive loss) match
the v13 recipe. All paths are required so the trainer cannot be invoked
into a silent-discard.

## 3. Export match-NN to runtime binary

    python -m ml.export.export_phase3 \
        --in  ml/data/iter1_d05v2/phase3.pt \
        --out ml/data/iter1_d05v2/nn_match.bin

The exporter reads the source checkpoint's `_meta`, propagates it into a
`META` trailer on the bin, and also writes a sibling `.meta.json` for
grep-friendliness.

## 4. Build state-head corpus

    python -m ml.data_prep.build_state_corpus \
        --pair-log-dir /mldata/track_analysis_runs/pair_log_iter0_noNN_jaad \
        --gt-config ml/configs/pair_log_config_iter0_noNN_jaad.yaml \
        --label-driven \
        --phase3-model ml/data/iter1_d05v2/phase3.pt \
        --out ml/data/iter1_d05v2/state_corpus \
        --comment "iter1, label-driven, phase3 from same iter"

Notes:

- `--label-driven` is required — it walks GT tracks and records the
  19-dim state-head input that the shipped runtime would have seen at
  each step, using the labels (not the tracker's own decisions) as the
  ground-truth track identity.
- `--phase3-model` is the freshly-trained match NN from step 2. The
  state head's NN-cost input feature has to match what the runtime will
  actually compute at inference.
- The corpus has been observed to be sensitive to pair-log thresholds:
  the v18 state corpus was built from `pair_log_v15_permissive`
  (`new_track_thr=0.05`, 3.65× more rows than the older v9 face corpus).
  See feedback memory `feedback_track_v18_recipe.md` for the resolution
  of that ambiguity.

The output is three npzs (`state_corpus_{train,val,test}.npz`) each
carrying their own `_meta` trailer.

## 5. Train the state head GRU

    python -m ml.train.train_state_head_decoupled \
        --train ml/data/iter1_d05v2/state_corpus_train.npz \
        --val   ml/data/iter1_d05v2/state_corpus_val.npz \
        --save  ml/data/iter1_d05v2/state_head_pw0.5.pt \
        --epochs 16 --seed 0 --hidden 64 --pos-weight 0.5

`--pos-weight 0.5` was the K-seed-fit-pick winner — see feedback memory
`feedback_track_phase20c_failed.md`. `--hidden 64` is the shipped
width. The trainer's three outputs are (LLR, log μ_TP, log μ_FP); μ_TP
target uses dist-to-next-match to avoid the truncation collapse
described in `feedback_track_mu_tp_truncation.md`.

## 6. Export state head to runtime binary

    python -m ml.export.export_decoupled_head \
        --in  ml/data/iter1_d05v2/state_head_pw0.5.pt \
        --out ml/data/iter1_d05v2/state_head_pw0.5.bin

The exporter rejects any `in_dim != 19` (the only width the C runtime
accepts — see `UTRACK_NN_STATE_GRU_IN_DIM_V3` in `nn_state.h`).

---

## Promoting to ship

After verifying the new bins (see Verification below), copy them into
`/mldata/config/track/trackers/` under a versioned name and flip the
yaml pointer in `/mldata/config/track/trackers/uc_v11.yaml`. The current
ship pinned the yaml at v13 + v23_pw05 with cheap-filter δ=0.7 and
dedup-IoU 0.70 — those last two are inference-only yaml changes (no NN
retrain required). See yaml comments at lines 46–53 and 126–137 in
`uc_v11.yaml` for the F5d ship reasoning.

---

## Verification

After producing a new `.bin`, before flipping the yaml pointer:

1. **Round-trip check** — re-export from the active `.pt` and confirm
   bit-identical bytes (excluding the `META` trailer):

       python -m ml.export.export_decoupled_head \
           --in ml/data/iter1_d05v2/state_head_pw0.5.pt --out /tmp/re.bin
       cmp <(head -c $(stat -c%s /mldata/config/track/trackers/nn_state_v23_pw05.bin) \
             /tmp/re.bin) /mldata/config/track/trackers/nn_state_v23_pw05.bin

2. **Closed-loop fitness** — run `ml.eval.eval_head_fitness` against the
   full-176 corpus with `ml/configs/eval_ship_baseline.yaml`. Ship only
   when fitness is within single-eval noise (±0.003) of the prior ship,
   or better. The currently audited number is:

       v13 + v23_pw05 + cFPt=7e-4 + δ=0.5:  0.5736  MOTA=0.6099  fp_tracks=68
       F5d (δ=0.7 + dd=0.70):               0.579   MOTA=0.611   fp_tracks=63

3. **JAAD val** — separate 29-clip dashcam stress test
   (`reference_jaad_dataset.md`). Cross-domain check that the head
   hasn't overfit to the static-camera corpus families.

4. **Live bench** — once `ubon_cstuff` builds (`cmake --build build -j`,
   then `./build/unit_tests` should be 200/200 PASS), run
   `track_benchmark` end-to-end on a labelled clip. This is the only
   check that exercises the full match → state pipeline together.

5. **Pre-commit sentinels** — `ml/util/verify_tree_sentinels.py`
   asserts the schema-critical markers are present in both repos. The
   pre-commit hook at `.git/hooks/pre-commit` runs it automatically;
   run it manually before any state-corpus / state-head / pair-trace
   commit.

---

## Common pitfalls

- **Train/infer distribution mismatch** — if you change the runtime
  cheap-filter δ, you must rebuild the pair-log (or at least re-run
  `build_pair_dataset` with the matching `--delta-filter`). Otherwise
  the NN sees a different feature distribution at inference than it was
  trained on.
- **Silent corpus drift** — every regen of the state corpus drifts the
  retrained head 0.04–0.17 fitness vs the shipped baseline, even with
  identical args. Always report **within-corpus delta** alongside
  absolute fitness when comparing to ship. See
  `feedback_track_corpus_drift.md`.
- **Seed luck** — the A1 finding "pw=0.6 wins" was falsified by F3
  multi-seed (σ≈0.013 across seeds 0/1/2). State-head changes require
  K-seed-fit-pick, not a single-seed sweep. See
  `feedback_track_phase20c_failed.md` and
  `project_a1_pw_sweep_was_seed_luck.md`.
- **Stale eval yaml** — only three eval yamls are kept in
  `/mldata/config/track/eval/`: `eval_ship_baseline.yaml`,
  `uc_v11_pre_F5d.yaml`, `uc_v11_no_nn.yaml`. Anything else under that
  directory was deleted on 2026-05-15.

---

## Build dependency

The `ubon_cstuff` C runtime requires the conda env **with
`nvidia/cu13/include` and `nvidia/cuda_runtime/include` filtered out**
of the include path. That's done in `ubon_cstuff/CMakeLists.txt` — the
conda env's newer cu13 `cuda_runtime.h` ships template overloads that
reference symbols declared only in its own newer
`cuda_runtime_api.h`. The system
`/usr/local/cuda/include/cuda_runtime_api.h` is included first by
`nvjpeg.h`, sets the header guard, and the conda env's
`cuda_runtime_api.h` is silently skipped — leaving the templates
referencing undeclared symbols. Filtering the directory avoids the
collision while keeping cublas/curand/cudnn etc.
