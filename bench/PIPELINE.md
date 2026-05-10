# bench/ — utrack training pipeline

The shipping tracker is `upyc-utrack` (C runtime in `ubon_cstuff/src/track/utrack/`).
It loads two neural artefacts from `/mldata/config/track/trackers/`:

| Path                                  | Role                  | Trainer                                  | Exporter                              |
|---------------------------------------|-----------------------|------------------------------------------|---------------------------------------|
| `nn_match_v9_face.bin`                | Match-cost residual   | `bench.train_phase3`                     | `bench.export_phase3`                 |
| `nn_state_v14_dc.bin`                 | State-machine head    | `bench.train_state_head_decoupled`       | `bench.export_decoupled_head`         |

Both binaries are loaded by `utrack.c` via the path keys `utrack.nn_path`
and `utrack.nn_state_path` in the YAML config. The C runtime accepts:

- `nn_path`: UP3P magic (`0x55503350`), version 1 — produced by
  `export_phase3.py`.
- `nn_state_path`: USHT magic (`0x55534854`), version 3 — produced by
  `export_decoupled_head.py`. `in_dim` must be 19.

After the existing format, every `.bin` produced by these exporters has a
trailing metadata block (`META` magic + JSON) holding the full provenance
(argv, git rev, hyperparams, dataset, hostname, UTC timestamp). The C
loader stops at `fclose` without checking EOF, so the trailer is
silently ignored at runtime. Inspect it with:

    python -m bench._artefact_meta --read /mldata/config/track/trackers/nn_state_v14_dc.bin

A sibling `.meta.json` is also written next to each `.bin` for
grep-friendliness.

## End-to-end retrain — match-cost NN

1. Generate per-pair training records by running the C tracker over a
   labelled corpus with the pair-logger analysis module enabled. Output
   goes to `runs/track_analysis/<run>/pair_log/<seq>.npz`.

2. Aggregate into split-level training datasets:

       python -m bench.build_pair_dataset \
           --pair-log-dir runs/track_analysis/pair_log_v9_face/pair_log \
           --analysis-yaml bench/pair_log_config.yaml \
           --split train \
           --out bench/data/pairs_train.npz
       # repeat for --split val, then --split test

   The train-split run also writes `bench/data/feature_norm.json`.

3. Train the two-tower model:

       python -m bench.train_phase3 \
           --save bench/data/match_phase3.pt
   
   Defaults match the recipe that produced the shipped
   `nn_match_v9_face.bin`. `--save` is required; full provenance lands
   under `ckpt['_meta']`.

4. Export to runtime binary:

       python -m bench.export_phase3 \
           --in  bench/data/match_phase3.pt \
           --out /mldata/config/track/trackers/nn_match_<version>.bin
       # then update utrack.nn_path in the YAML configs

## End-to-end retrain — state head

1. Build the state corpus by running the C runtime in label-driven mode
   over a labelled tracking corpus and capturing the per-row state
   features:

       python -m bench.build_state_corpus \
           --out bench/data/state_corpus_v18

   That writes `state_corpus_v18_{train,val,test}.npz`.

2. Train the decoupled GRU head:

       python -m bench.train_state_head_decoupled \
           --train bench/data/state_corpus_v18_train.npz \
           --val   bench/data/state_corpus_v18_val.npz \
           --save  bench/data/state_head_dc_v15.pt

   `--seed`, `--hidden=32`, `--epochs=8` (defaults) match the v14
   recipe. All three of `--train`, `--val`, `--save` are required so
   the trainer cannot be invoked into a silent-discard.

3. Export to runtime binary:

       python -m bench.export_decoupled_head \
           --in  bench/data/state_head_dc_v15.pt \
           --out /mldata/config/track/trackers/nn_state_v15_dc.bin
       # then update utrack.nn_state_path in the YAML configs

   The exporter rejects any `in_dim != 19` because that's the only
   width the C runtime accepts (see `UTRACK_NN_STATE_GRU_IN_DIM` in
   `nn_state.h`).

## Verification

After producing a new `.bin`, before flipping the YAML pointer:

1. **Round-trip check** — re-run the exporter on the active `.pt` to
   confirm bit-identical bytes (excluding the metadata trailer):

       python -m bench.export_decoupled_head \
           --in bench/data/state_head_dc_v14_oldtrainer.pt --out /tmp/v14_re.bin
       cmp <(head -c $(($(stat -c%s /mldata/config/track/trackers/nn_state_v14_dc.bin))) \
             /tmp/v14_re.bin) /mldata/config/track/trackers/nn_state_v14_dc.bin

2. **Offline fitness** — run `bench.eval_head_fitness` against the
   full-178 corpus. Ship only when fitness is no worse than the current
   shipping head (`v14: 0.4797 ± 0.0011`).

3. **Trace library** — run `bench.trace_library.runner --head <new.pt>`.
   v14 fails 5/14 traces; that's the accepted tradeoff for higher
   aggregate fitness.

4. **Live bench** — once `ubon_cstuff` builds (`cmake --build build -j`,
   then `./build/unit_tests` should be 200/200 PASS), run
   `track_benchmark` against an end-to-end labelled clip. This is the
   only check that exercises the full match→state pipeline together.

## Files in `bench/data/`

The directory carries **only** files referenced by the active trainers
or exporters:

- `pairs_{train,val,test}.npz` — match-cost training corpus
- `feature_norm.json` — per-feature mean/std for the match-cost trainer
- `phase3_v9_face.pt` — checkpoint that produced the shipped match-cost binary
- `phase3_v9_noface.pt`, `phase3_e100.pt` — reference checkpoints
- `state_corpus_v17_*.npz`, `state_corpus_v18_*.npz` — state-head corpora
- `state_head_dc_v14_oldtrainer.{pt,bin}` — shipped state-head

Everything else has been moved to `bench/_archive/` (see its README).

## Build dependency

The C runtime requires the conda env **with `nvidia/cu13/include` and
`nvidia/cuda_runtime/include` filtered out** of the include path.
That's done in `ubon_cstuff/CMakeLists.txt` — the conda env's newer
cu13 cuda_runtime.h ships template overloads that reference symbols
declared only in its own newer cuda_runtime_api.h. The system
`/usr/local/cuda/include/cuda_runtime_api.h` is included first by
`nvjpeg.h`, sets the header guard, and the conda env's
`cuda_runtime_api.h` is silently skipped — leaving the templates
referencing undeclared symbols. Filtering the directory avoids the
collision while keeping cublas/curand/cudnn etc.
