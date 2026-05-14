# matching-cheap-filter — overnight experiments

State at end of session 2026-05-14: `iter1_d05v2` ships
- match NN (`bench/data/iter1_d05v2/nn_match.bin`) trained on iter0_noNN_jaad
  pair-log with the per-event delta filter + threshold-lower-bound gate;
- state head (`bench/data/iter1_d05v2/state_head_pw0.5.bin`) trained on the
  state corpus built from that match;
- C runtime cheap-filter ON via `utrack.match_cheap_filter_delta=0.5`
  (utrack_match.c: delta gate + near-thr rescue + below-thr discard).

Eval (full-176, runs=1):
- baseline (current ship v12/v22, no filter): **0.5746**
- new full pipeline + filter ON δ=0.5: **0.5736**   (−0.001 fitness for ~50% NN evals saved)

Goal of overnight work: (i) close the small fitness gap via hyperparameter
re-tuning that wasn't explored, (ii) trim complexity (smaller NNs, dead-code
removal) without losing parity. Each experiment is one-shot; queue end-to-end.

---

## A. Hyperparameter retunes (cheapest, do first)

All on `bench/data/iter1_d05v2/`. Phase 3 + state corpus are already built.

**A1 — state head pos_weight sweep, seed=0**
Train + export the state head at pos_weight ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 1.0}.
Eval full-176 with filter ON δ=0.5. Pick best. ~10 min train + ~5 min eval each ≈ 90 min.

```
for pw in 0.3 0.4 0.5 0.6 0.7 1.0; do
  python -m bench.train_state_head_decoupled \
    --train bench/data/iter1_d05v2/state_corpus_train.npz \
    --val   bench/data/iter1_d05v2/state_corpus_val.npz \
    --save  bench/data/iter1_d05v2/state_head_pw${pw}.pt \
    --epochs 16 --seed 0 --hidden 64 --pos-weight $pw \
    --comment "iter1_d05v2 pw=${pw}"
  python -m bench.export_decoupled_head \
    --in  bench/data/iter1_d05v2/state_head_pw${pw}.pt \
    --out bench/data/iter1_d05v2/state_head_pw${pw}.bin
  # eval — see template in /tmp/eval_iter1_d05v2_full/eval_full.yaml, swap state_head bin
done
```

**A2 — multi-seed at best pos_weight from A1**
Seed ∈ {0, 1, 2} to bound variance. If +σ > 0.003, single-seed is noisy and
we should use multi-seed for shipping.

**A3 — filter δ sweep at inference (no retrain)**
Reuse the iter1_d05v2 + best-pw state head from A2. Eval at
δ ∈ {0.2, 0.3, 0.4, 0.5, 0.7, 1.0}. The training-time δ is fixed at 0.5;
this only tests how aggressive we can be at inference.

**A4 — joint train+infer δ sweep (full retrain at each δ)**
δ ∈ {0.3, 0.5, 0.7}. Per δ: rebuild pair dataset, retrain phase3 +
state corpus + state head + eval. ~60 min per δ. Only do if A3 shows
clear δ-sensitivity.

---

## B. Ablation — drop NN input features (target: simpler/faster NN, parity)

Hypothesis: not all 24 match-NN / 19 state-head inputs contribute; pruning
gives smaller .bin + faster `nn_pair_score()` at runtime.

**B1 — static importance estimate (no training)**
Load `nn_match_iter1_d05.bin` weights. For each input feature (24 total
across obs[16] + det[5] + pair[19]), compute the L2 norm of its incoming
weights in the first dense layer. Rank features by total magnitude across
all layers it appears in. Drop candidates: bottom-quartile features.

Output: `/tmp/match_nn_feature_importance.txt`. ~5 min, no GPU.

**B2 — drop-one-feature ablation, match NN**
`train_phase3.py` already exposes `--drop-features`. For each candidate from
B1 (5–8 features), retrain phase3 with that feature zeroed, eval. Keep
features whose removal costs ≤ 0.001 fitness. ~3 min train + 5 min eval each.

```
# template (adapt to args.drop_features semantics in train_phase3)
for feat in <candidate list from B1>; do
  python -m bench.train_phase3 \
    --epochs 25 --data_dir bench/data/iter1_d05v2 \
    --save bench/data/iter1_d05v2_drop_${feat}/phase3.pt --seed 0 \
    --drop-features $feat \
    --comment "ablation drop $feat"
  # export + eval ...
done
```

**B3 — drop-multiple-features**
If B2 yields several no-cost drops, train a single NN with all those features
dropped. Confirm fitness parity. Measure .bin size + nn_pair_score throughput.

**B4 — state head ablation (same recipe via train_state_head_decoupled)**
The decoupled state head input is 19-dim (no_state matrix). Use the
`--no-phase3-feature` knob as a working precedent for input-zeroing. Add
a `--drop-state-cols` analogue if it doesn't exist, then sweep.

---

## C. Code cleanup (target: less code, easier maintenance, no fitness change)

**C1 — pending task #88: delete dead C-side knobs guarded by `!nn_state`**
The C runtime is now strictly pure-NN-state. Several yaml keys that were
only consulted in the legacy heuristic-state path are still parsed but
unused. List them via `git grep "param_.*"` against the `!nn_state` guard
sites; remove parse + storage + getter.

**C2 — match-NN input feature dead code**
For features that B3 confirms can be dropped, remove their computation
from `utrack_match.c` and `nn_build_obs/det/pair`. Each removed feature
is ~5–20 lines.

**C3 — drop the heuristic-only pre-pass duplication of work**
`utrack.c` now runs `utrack_match_cost` twice per (det, track) pair when
the filter is on: once in heuristic_only mode for the pre-pass, once
again in the main pass for the same heuristic+NN. The heuristic cost
(motion + reid sim) recomputes both times. Cache it in a `(num_det,
num_tracked)` table during the pre-pass, look up in the main pass.
~5% wall save on NN-heavy clips. Only worth doing if real-time profiling
shows it.

**C4 — `bench/_archive/` is gone; sweep for stale path references**
`grep -rn "_archive" bench/ src/` and prune any leftover lookups.

---

## Bookkeeping

- All experiments operate on the `matching-cheap-filter` branches of `track`
  and `ubon_cstuff`. Commit each experiment's outputs under its own
  `bench/data/<tag>/` so the .bin .meta.json chain stays clean.
- Eval recipe template lives at `/tmp/eval_iter1_d05v2_full/eval_full.yaml`
  (drop in new nn_path / nn_state_path / match_cheap_filter_delta).
- Variance band on a single full-176 eval is roughly ±0.003; treat anything
  within that as noise.
