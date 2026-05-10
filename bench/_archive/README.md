# bench/_archive

Stale experiment artefacts moved out of `bench/` and `bench/data/` so the
working directory only carries the actively-shipping pipeline. Nothing here is
referenced by the current trainers, exporters, or runtime configs.

Layout:

- `configs/` — 157 ablation YAMLs from prior parameter sweeps
  (`baseline_test_ablation_*.yaml`). The active baselines stay at
  `bench/baseline_test{,_v2,_v3}.yaml`.
- `data/` — older training artefacts:
  - `state_head_v*.{pt,bin,kseed.json}` — pre-decoupled state-head checkpoints
    (177 files). Replaced by the decoupled-GRU `state_head_dc_*` family.
  - `state_head_dc_v[1-13,15]*.{pt,bin}` — decoupled-GRU iteration history.
    Only `state_head_dc_v14_oldtrainer.{pt,bin}` is shipped.
  - `state_corpus_v[1-16]_*.npz` — older state-head training corpora.
    Active are `state_corpus_v17_*.npz` and `state_corpus_v18_*.npz`.
  - `phase3_*.{pt,bin}` (~30 files) — match-cost iteration history.
    Active are `phase3_v9_face.pt` (which produced the shipped
    `nn_match_v9_face.bin`) and `phase3_v9_noface.pt` / `phase3_e100.pt`
    as references.
  - `pairs_*_v[1-9face,15].npz` — older match-cost pair corpora.
    Active is `bench/data/pairs_{train,val,test}.npz` (latest schema).
  - `bayes_head_*`, `creation_head_*`, `residual_*` — abandoned experiments
    (Bayesian head prototype, separate creation gate, residual MLP).
  - `alpha_sweep/`, `cr_search/`, `multi_alpha/`, `sweep/`, `sweep_etrack/` —
    hyperparameter-search output directories.

Free-disk strategy: this whole directory is safe to delete with
`rm -rf bench/_archive/`. Doing so frees ~5 GB without affecting the
documented retrain pipeline (see `bench/PIPELINE.md`).
