# matching-cheap-filter — overnight experiments

## Final ship (2026-05-15): F5d — δ=0.7 + dd=0.70

**Two yaml edits in `/mldata/config/track/trackers/uc_v11.yaml`:**

```yaml
utrack:
  match_cheap_filter_delta: 0.7   # was 0.5
  delete_dup_iou:           0.70  # was 0.90
```

No NN retraining; same shipped bins (`nn_match_v13.bin`,
`nn_state_v23_pw05.bin`). Closed-loop fitness on full-176 + JAAD val:

| Variant | full-176 fit | fp_tracks | IDF1 | JAAD fit | JAAD IDF1 |
|---|---|---|---|---|---|
| Old ship (δ=0.5, dd=0.90) | 0.576 | 66 | 0.589 | 0.266 | 0.423 |
| **New ship (F5d δ=0.7 dd=0.70)** | **0.579** | **63** | 0.587 | 0.266 | 0.423 |
| Δ                                 | +0.003 | −3 | −0.002 | 0 | 0 |

Source eval: `/mldata/track_runs/f5_combo/results/results-20260515-*.txt`,
variants F5a_ship vs F5d_d07_dd70.

Why this is the right ship (vs the other F5 candidates):
- F5e (+pw=0.6) shows +0.005 fitness BUT F3 multi-seed showed pw=0.6
  has σ≈0.013 fitness across seeds — the win is inside the variance
  band. Not shippable without K-seed verification.
- F5f (+λ=0.10 IDF1-lean) gives IDF1 0.600 but trades 0.005 fitness;
  available as a documented alternative for IDF1-priority deployments.
- F5g (pw=1.0 JAAD-cam) wins JAAD val by +0.025 fitness but loses
  full-176 by −0.034. Domain-split ship — defer to future session.
- δ=0.7 alone (F5b) regressed −0.004; dd=0.70 alone (F5c) ±0. The
  win lives in the interaction.

Other knobs swept and confirmed not worth changing:
- `bayes_c_FP_track` (E1): no signal in 3e-4 → 1e-3 band.
- `track_buffer_seconds` (E2): 2.2 already at peak.
- `max_consecutive_misses` (E5): 10 already near peak; 15 tied.
- `nn_lambda` (E4): 0.05 fitness-optimal. λ=0.10 documented as the
  IDF1-leaning alternative (project memory `project_nn_lambda_idf1_dial.md`).
- More JAAD pairs in corpus (D1, D2): both regressed full-176 and
  did not move JAAD val (`project_d1_jaad_test_promotion.md`).

The cheap-filter is also speed-neutral (G1 wall-profile,
`project_cheap_filter_speed_neutral.md`), so this change has zero
throughput impact.

---



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

## D. JAAD domain coverage (target: close the cross-domain gap on JAAD val)

Motivation: closed-loop eval (2026-05-14) shows ship v13/v23_pw05 at JAAD val
fitness 0.264 vs **no-NN at 0.286** — the NN regresses on the cross-domain
JAAD val split. The training corpus has all 177 JAAD train clips but per
`project_pair_log_scene_skew.md` they contribute only **2.2%** of total pairs
(N²-skewed toward CEVO/PP22 dense-crowd scenes). Hypothesis: more in-domain
JAAD pairs at training time should shift the NN's distribution toward JAAD,
narrowing the cross-domain gap.

**D1 — include JAAD test split in the training corpus**
JAAD's default split is 177 train / 29 val / **117 test**. Test is currently
unused (held-out for a benchmark we never publish). Move all 117 JAAD test
clips into the `train` partition of `pair_log_config_iter0_noNN_jaad.yaml`,
rebuild the pair-log, then redo the full bootstrap.

Steps:
1. Generate `pair_log_config_iter0_noNN_jaad_plus_test.yaml` (programmatic:
   read the current config, swap `split: test` → `split: train` for all
   jaad/* entries, leave non-jaad alone).
2. Run pair-log emission on the 117 newly-promoted clips only (the iter0
   noNN trackset is deterministic given config — no need to re-emit the
   already-cached 290 clips). Cache hit unclear; if not, ~117 × 30s ≈ 60 min.
3. Rebuild `bench/data/iter1_d05v2_jaadall/pairs_*.npz` with same δ=0.5
   filter as the shipped recipe.
4. Train phase3 (match NN) — 25 epochs, seed=0. ~15 min.
5. Build state corpus on top of new match NN — ~15 min.
6. Train state head pw=0.5, seed=0, hidden=64, epochs=16 — ~10 min.
7. Eval on full-176 + JAAD val. **JAAD val is no longer held-out from the
   pair-log corpus (train pairs include JAAD test), but the eval clip set
   itself is still the 29-clip val split** — so the metric is honest as a
   measure of in-domain generalisation, just not as a measure of pure
   distribution-shift robustness.

Total est. ≈ 2 h (or 1 h if pair-log cache hits).

**D2 — JAAD oversampling at the pair-dataset stage (cheaper alternative)**
If D1 helps, validate the mechanism by re-running with the *original* corpus
but oversampling JAAD pairs at `bench/build_pair_dataset.py` time (e.g.
duplicate JAAD rows 4×). No retracking needed — only npz rebuild + retrain.
~30 min. If oversampling captures most of D1's gain, ship that instead
because it's reversible without re-emitting pair-logs.

---

## Bookkeeping

- All experiments operate on the `matching-cheap-filter` branches of `track`
  and `ubon_cstuff`. Commit each experiment's outputs under its own
  `bench/data/<tag>/` so the .bin .meta.json chain stays clean.
- Eval recipe template lives at `/mldata/config/track/eval/` (drop in new
  nn_path / nn_state_path / match_cheap_filter_delta).
- Variance band on a single full-176 eval is roughly ±0.003; treat anything
  within that as noise.

---

## Working protocol for every experiment

**Before each experiment** write into the Progress Log below:
1. The hypothesis in one sentence.
2. A wall-clock estimate broken down by stage (pair-log, dataset build,
   train, state corpus, state head, eval). State the assumption behind each
   number (e.g. "phase3 ~15 min on full-176 corpus at 25 epochs").
3. A 30-second pause to ask: **is there a cheaper variant that gives the
   same signal?** Examples: skip pair-log re-emit if cache valid, run
   eval at `runs=1` first (`feedback_eval_runs_default.md`), use a 5-clip
   subset to verify the pipeline before kicking off the full job.

**After each experiment** record:
1. Actual wall-clock per stage (or at least: which stage dominated).
2. Result vs hypothesis (fitness/MOTA/fp_tracks; cite exact eval yaml).
3. Bottleneck observation + at least one concrete idea to make the next
   similar experiment faster. Even "no obvious speedup" is fine to record.

Both the plan and the outcome go inline in this file under
`## Progress log` — do not split them off into other docs. The MD is the
single source of truth for the night.

---

## Progress log

### 2026-05-14 evening — overnight kickoff

**Setup done:**
- Moved eval yamls from `/tmp/no_nn_compare/` → `/mldata/config/track/eval/`
  (`uc_v11_ship.yaml`, `uc_v11_no_nn.yaml`, plus δ-variant tracker configs
  `uc_v11_d{0.2,0.3,0.4,0.7,1.0}.yaml`).
- GPU baseline before launch: 2 GiB / 24 GiB used, util 9%. No other GPU
  tenants. Stale `until`-loop shell from earlier session is bash-only.

---

### Experiment A3 — δ-sweep at inference (JAAD-val first)

**Hypothesis:** the cross-domain JAAD-val regression (NN 0.264 vs no-NN
0.286) is partly because the inference δ is too aggressive — fewer NN evals
near the threshold drops information that matters more for unfamiliar
distributions than for in-domain clips. If true, smaller δ (less filtering)
should claw fitness back; if false, δ is irrelevant on JAAD val and the
issue is upstream (training data mix, B1/D1 territory).

**Plan:**
- 6 variants × 29 JAAD-val clips = 174 clip-runs, 4 workers.
- δ ∈ {0.2, 0.3, 0.4, 0.5, 0.7, 1.0}. δ=0.5 anchors against the prior
  cross-domain number (0.264) — re-measuring it surfaces eval noise too.
- Yaml: `/mldata/config/track/eval/a3_delta_sweep_jaadval.yaml`.
- Cheaper variant considered: 4-clip pilot (one JAAD video × all δ) first.
  **Skipped** — 174 clip-runs is small enough that the pilot would cost
  20% of the full job. Just run it.

**Time estimate (before launch):**
- Previous 2-variant × 205-clip eval with 2 workers took ~30 min ≈ 7.3
  clip-runs/min/worker. At 4 workers and 174 clip-runs that should be
  ~6 min, with overhead probably 10–15 min wall.
- Stages expected to dominate: detector inference (per-frame YOLOv26
  Tensor-RT pass on JAAD clips, longer than MOT short clips). Track
  cold-start per worker (load 2 NN bins + TRT engine) is ~5 s × 4 = 20 s
  amortised.

**Launching now → background log at `/tmp/a3_delta_sweep.log`.**

**Actual wall time: 1 min 18 s** (4 workers; eval rate 2.5 it/s). 6× faster
than my estimate — JAAD clips on a warm cache are detector-bound not
data-bound, and 4 workers saturate the GPU much better than 2 did in the
prior side-by-side. Detector inference dominated. No worker died.

**Result table — JAAD val (29 clips, fp_per_frame / MOTA / fitness):**

| δ      | fp_per_frame | MOTA  | fitness |
|--------|--------------|-------|---------|
| 0.7    | 0.094        | 0.271 | **0.270** |
| 0.4    | 0.103        | 0.269 | 0.267   |
| 0.3    | 0.102        | 0.268 | 0.267   |
| 0.2    | 0.103        | 0.268 | 0.267   |
| 1.0    | 0.104        | 0.268 | 0.266   |
| 0.5⁺   | 0.104        | 0.267 | 0.266   |

⁺ shipped δ. Re-measured against last session's 0.264 — within ±0.003
single-eval noise, so the eval anchor is consistent.

**Hypothesis falsified.** I expected smaller δ (less NN filtering) to win
on cross-domain JAAD. Instead **δ=0.7 (more aggressive filtering) wins** by
0.004 fitness — fewer fp_per_frame too (0.094 vs the rest at 0.10+). Three
possible reads:
1. The NN is mis-calibrated on JAAD's distribution; falling back to the
   heuristic more often (= bigger δ) is a net positive.
2. δ=0.7 is just noise floor (0.004 ≈ ±0.003 band, n=29 small).
3. The cheap-filter's near-thr rescue gate becomes redundant past δ=0.7
   (rescue window widens too), softening the in-domain damage.

**Next:** verify δ ∈ {0.4, 0.5, 0.7} on full-176 to check that δ=0.7
doesn't tank in-domain MOTA. If full-176 fitness for δ=0.7 stays within
0.003 of δ=0.5, ship δ=0.7. If it drops, accept the JAAD wobble as noise.

**Bottleneck observation / speedup ideas for future evals:**
- 174 clip-runs in 78 s ≈ 2.2 clip-runs/s aggregate (4 workers, GPU-bound).
  Detector TRT engine is the long pole; can't move it.
- Worker cold-start (load TRT engine + 2 NN bins) ≈ 5 s × 4 workers = 20 s,
  amortised over 174 runs is 0.1 s/run — negligible. Not a target.
- If we ever need to sweep on hundreds of variants, **persistent worker
  pools that swap config without restarting** would save the cold-start
  cost. Not worth building tonight.

---

### Experiment A3b — δ-sweep verification on full-176

**Hypothesis:** δ=0.7 also competitive on in-domain (full-176). If yes → ship.

**Plan:** δ ∈ {0.4, 0.5, 0.7} × 176 clips = 528 clip-runs.
**Estimate:** 528 / 2.2 ≈ 4 min wall.
**Yaml:** `/mldata/config/track/eval/a3_delta_sweep_full176.yaml`.
**Cheaper variant considered:** could narrow to δ ∈ {0.5, 0.7} (skip 0.4).
**Kept 0.4** because (a) JAAD wasn't clean — 0.4 was 2nd best — and (b)
the extra 176 runs cost ~80 s, cheap insurance.

**Actual wall time: 13 min 56 s** (4 workers; mid-eval rate dropped from
2.5 it/s to ~0.5 it/s on PP22 clips, recovered to 1.9 it/s aggregate at
the end). PP22 long-clip serialisation on the GPU is the dominant cost.

**Result table — full-176 (86 100 frames):**

| δ      | fp_tracks | fp_per_frame | MOTA  | fitness |
|--------|-----------|--------------|-------|---------|
| 0.7    | **66**    | **1.15**     | 0.611 | **0.575** |
| 0.5⁺   | 69        | 1.16         | 0.611 | 0.574   |
| 0.4    | 69        | 1.16         | 0.610 | 0.574   |

⁺ shipped δ.

**Verdict: δ=0.7 is a free upgrade.** Wins both views — JAAD val by 0.004
fitness (within ±0.003 noise but consistent direction) and full-176 by
0.001 fitness + 3 fewer fp_tracks. No regression observable. The result
is small but the direction is the *opposite* of what I predicted, so I
trust it more than I would a 0.001 win confirming my prior.

**Mechanism (post-hoc):** δ=0.7 lets the cheap filter drop a wider band
around the threshold. The near-thr rescue still saves pairs near the
match threshold, so the matcher's decision boundary is untouched. The
NN is only evaluated on pairs where its output could plausibly change the
match — which is fewer pairs at δ=0.7 than at δ=0.5. So we save NN
calls *and* the matcher's effective behaviour is slightly cleaner (fewer
spurious NN outputs perturbing pairs that the cost rule would've reached
the same decision on).

**Action item:** when A1/D1/D2 wrap, ship `match_cheap_filter_delta: 0.7`
in `uc_v11.yaml` *unless* a later experiment uncovers a regression on
something the production yaml cares about that we're not seeing here.

**Bottleneck observation:** the PP22 slowdown was unexpected. Aggregate
rate fell from 2.5 it/s (JAAD warm cache) to ~0.5 it/s mid-eval. The
underlying cause is per-frame variance — long PP22 clips have ~150-200
frames each at 5 fps tracker (NN-cost-heavy) ≈ 30 s per clip on a single
worker. 4 workers parallel = ~7-8 clips/min = ~0.13 clip-runs/sec. The
2.5 it/s rate I saw on JAAD was because JAAD clips are shorter (60 frames
≈ 12 s each on a single worker). **For future evals: report rate
separately per group**, or sort the work queue by descending length so
the long clips start first and don't keep workers idle at the tail.


---

### Experiment B1 — static feature importance (no GPU, ~1 min)

**Hypothesis:** the match NN has 16+5+19=40 inputs in three towers (obs/det/
pair). Some inputs probably contribute negligibly to first-layer activations
because the network has learned to ignore them (e.g. correlated with a
stronger upstream feature, or noisy). The bottom-quartile by L2 norm of
incoming first-layer weights is a reasonable shortlist of drop candidates —
not definitive (downstream gating can rescue an apparently-quiet input)
but a useful prior before paying for B2 ablation.

**Plan:** load `phase3.pt`, compute L2/L1 norms of incoming weights for
each input column of `f_obs.net.0`, `g_det.net.0`, and `h.net.0[:, 32:51]`
(the pair section of the head input).

**Estimate:** ~10 s (CPU, numpy + PyTorch). No GPU contention possible.

**Result:**
- **OBS top-3:** `log_num_missed`, `of_score`, `prev_det_conf` (geometry
  + match quality dominate).
- **OBS bottom-4:** `ocm_cos`, `det_subbox_conf`, `det_fiqa_score`,
  `track_speed`. Surprising one: `det_fiqa_score` (face quality)
  contributes very little to the obs side; whatever benefit face brings
  is being absorbed by the pair side or the state head.
- **DET top:** `det_aspect`, `det_h`, `det_w` (raw geometry).
- **DET bottom:** **`det_conf`** — the model has effectively dropped raw
  det_conf from the det tower. Confidence info reaches the head via the
  pair side (`pre_thr_score`) and the obs side (`det_conf`/`prev_det_conf`).
- **PAIR top-5:** `iou`, `of_score`, `pre_thr_score`, `size_ratio`,
  `kf_d2`. Classic match-quality signals.
- **PAIR bottom-4:** `ocm_cos`, `det_subbox_conf`, `conf_delta`, `pass_1`.

**B2 drop list (top candidates for ablation):**
1. `ocm_cos` — appears bottom in both OBS and PAIR towers.
2. `det_subbox_conf` — appears bottom in both OBS and PAIR.
3. `det_fiqa_score` (OBS), `pass_1` (PAIR) — bottom in one tower each.

**Bottleneck observation:** B1 took ~5 s. Trivial speedups not worth
pursuing. Idea for next: cache the rank table to disk so B2 picks
candidates without re-running B1.

**Full report:** `/tmp/b1_feature_importance.txt`.

---

### Queue status (kicked off, will fire in sequence)

| Slot | Experiment | Gate | Driver | Log |
|------|------------|------|--------|-----|
| 1 | A3b (δ ∈ {0.4,0.5,0.7} full-176) | running | inline | `/tmp/a3_delta_full176.log` |
| 2 | A1 (state-head pw sweep) | waits for A3b | `/tmp/run_a1_pw_sweep.sh` | `/tmp/a1_pw_sweep.log` |
| 3 | D1 (JAAD test→train, full bootstrap) | waits for A1 | `/tmp/run_d1_jaad_test_train.sh` | `/tmp/d1_jaad_test_train.log` |
| 4 | D2 (JAAD oversample 4×) | waits for D1 | `/tmp/run_d2_jaad_oversample.sh` | `/tmp/d2_jaad_oversample.log` |

Drivers run unattended; each writes its own log and (where relevant) eval
results to `/mldata/track_runs/<tag>/results/`. Results table will be
filled in inline below as each experiment completes — same MD, single
source of truth.

**Time budget (best-case revised after A3b's slower-than-expected start):**
A3b finishing ~1.5 h from launch → cascade total ~3 h from now (full-176
clips are heavier than I priced — PP22 clips run at ~1 s/frame inside the
detector + NN inner loop, so the 4-worker aggregate is closer to 0.6 it/s
than the earlier 2.5 it/s).

---

### Experiment A1 — state-head pos_weight sweep

**Hypothesis:** pw=0.5 was the existing ship pick from a single seed. A
wider sweep around it might find a better operating point.

**Plan:** train state heads at pw ∈ {0.3, 0.4, 0.5(skip-retrain), 0.6,
0.7, 1.0}, seed=0, hidden=64, eps=16. Eval each at filter δ=0.5
(shipped at the time of A1 launch — A3b was still running). Full-176 +
JAAD val.

**Estimate:** 5 trainings × ~5 min + 6 evals × ~5 min = ~55 min. Actual
wall: ~58 min (incl. orchestration). On budget.

**Result table — full-176 + JAAD val (filter δ=0.5):**

| pw   | full-176 fit | full-176 fp_tr | full-176 MOTA | JAAD val fit | JAAD val MOTA |
|------|--------------|----------------|---------------|--------------|---------------|
| 0.3  | 0.517        | 28             | 0.532         | 0.207        | 0.207         |
| 0.4  | 0.557        | 34             | 0.576         | 0.215        | 0.215         |
| 0.5⁺ | 0.575        | 66             | 0.610         | 0.266        | 0.267         |
| **0.6** | **0.579** | **67**         | **0.615**     | 0.269        | 0.271         |
| 0.7  | 0.549        | 145            | 0.624         | 0.285        | 0.288         |
| 1.0  | 0.540        | 170            | 0.628         | **0.290**    | **0.296**     |

⁺ ship at A1 launch (filter δ=0.5).

**Verdict: pw=0.6 wins on full-176** (+0.004 fitness, MOTA up by 0.005,
fp_tracks essentially unchanged at 67 vs 66). On JAAD val it's also up
(0.269 vs 0.266) — modest but in the same direction.

**Surprise finding: pw=1.0 wins on JAAD val by +0.024 fitness** (0.290
vs ship 0.266), driven by MOTA 0.296 vs 0.267. But it costs −0.035 on
full-176 because fp_tracks explodes from 66 → 170 (2.6× more spurious
tracks). High-pw is biasing the head to keep tracks alive longer →
better recall during occlusion → wins on dashcam pedestrian (lots of
occlusion) → fails on dense crowds. Classic recall/precision tradeoff
exposed.

**Action items:**
1. Ship-candidate: **pw=0.6** (paired with δ=0.7 from A3b → run F1 to
   verify combo holds).
2. **Open thread:** pw=1.0 wins JAAD by a real margin. The
   recall-leaning state head is genuinely better for occluded
   pedestrian tracking; full-176's dense crowd scenes punish it via
   fp_tracks. If we ever segment configs by deployment domain, JAAD-
   like cameras want pw≈1.0.
3. Validate pw=0.6 with multi-seed (A2 next round) — single-seed gains
   of 0.004 are within measurement noise.

**Bottleneck observation:**
- Trainings averaged ~5 min each (16 epochs, hidden=64). Roughly 80%
  in GPU forward/backward, 20% in data loading + checkpointing.
- Eval at 4 workers averaged ~5 min per pw (6 × 205 ≈ 1230 clip-runs
  in 30 min). Same long-PP22-clip throughput floor as A3b.
- **Speedup for next time:** state-head training is sequence-batched
  and small (<1 GiB GPU). Could parallelise 2-3 trainings on the same
  GPU. Would cut A1 wall from ~25 min training time to ~10 min for
  similar quality.

---

### Experiment D1 — JAAD test→train, full bootstrap

**Hypothesis:** JAAD's 117-clip test split is currently held out from the
pair-log corpus; promoting it to `split: train` more than doubles the
JAAD-domain pair count and may close the cross-domain gap on JAAD val.

**Plan:** generate pair-logs for 117 JAAD test clips → merge with
existing 353 → rebuild pair dataset → train phase3 → state corpus →
state head → eval full-176 + JAAD val. All on the existing tracker.

**Estimate:** analysis 30 min + dataset build 5 min + phase3 15 min +
state corpus 15 min + state head 10 min + eval 10 min = ~85 min.
**Actual: ~88 min**, dominated by the analysis step (no-NN tracker run
on 117 long JAAD clips, slower than estimate because the pair_log
analysis re-ran the tracker even for the cached-ubtrk2 entries that
were already on disk for the 353 baseline clips).

**Result table — full-176 + JAAD val (filter δ=0.5):**

| Variant        | full-176 fit | full-176 fp_tr | full-176 MOTA | JAAD val fit | JAAD val MOTA |
|----------------|--------------|----------------|---------------|--------------|---------------|
| ship           | 0.574        | 66             | 0.610         | 0.266        | 0.267         |
| d1_jaadall     | 0.562        | **48**         | 0.589         | 0.263        | 0.263         |

**Verdict: hypothesis falsified.** Adding 117 JAAD test clips (66% more
JAAD pairs) to the training corpus *did not* close the JAAD val gap
(-0.003 fitness vs ship, well within noise). It also regressed full-176
by -0.012 fitness, -0.021 MOTA. The d1_jaadall model is more
**cautious** (48 fp_tracks vs 66 — clear signal) but the caution costs
recall everywhere.

**Mechanism:** more JAAD pairs shifted the NN's decision boundary
toward "don't accept marginal matches" — pedestrian dashcam scenes
have low overlap between consecutive frames (camera motion + slow
movement), so the corpus had more "borderline match" rows. The NN
learned to reject more of them. That helps fp_tracks but hurts MOTA
across the board.

**This adds to the existing project memory** (`project_pair_log_scene_skew.md` +
`feedback_offline_online_gap.md`): cross-domain gap is NOT a data-volume
problem. It's a corpus-composition problem (N²-skewed toward dense
crowds) compounded by training-objective vs deployment-objective gap.
More JAAD data alone won't fix it.

**Action:** D1 does not ship. D2 (oversampling existing JAAD pairs 4×)
is in queue and may also fail to help — if so, the cross-domain gap
work needs a different angle (e.g. fitness-shaped per-sample weighting,
or post-hoc domain adaptation).

**Bottleneck observation:**
- Analysis step took ~30 min instead of expected ~10 min because the
  tracker reran on all 117 new clips without cached UBTRK2.
- Train_phase3 was ~15 min on the new corpus (1.35M training pairs).
- State corpus build ~15 min — this involves f_obs replay through the
  new match NN, which is moderately expensive.
- **Speedup for next time:** the symlink farm merge pattern works but
  the pair_log analysis runs the tracker even when a UBTRK2 exists in
  a *different* dir. Should refactor the analysis engine to accept a
  list of UBTRK2 source dirs to scan before deciding to regenerate.

---

### F1 — combo (δ=0.7, pw=0.6) — results

**Wall: ~20 min** (820 clip-runs at 4 workers; F1 ran after the E queue).

Full-176:
| Variant       | fp_tr | MOTA  | IDF1  | fitness |
|---------------|-------|-------|-------|---------|
| ship          | 65    | 0.610 | 0.586 | 0.575   |
| delta_0.7     | 67    | 0.610 | 0.586 | 0.574   |
| **pw_0.6**    | 69    | **0.615** | **0.588** | **0.578** |
| combo_d07_pw06| 69    | 0.615 | 0.588 | 0.578   |

JAAD val:
| Variant       | fp_tr | MOTA  | IDF1  | fitness |
|---------------|-------|-------|-------|---------|
| ship          | 2     | 0.270 | 0.425 | 0.269   |
| delta_0.7     | 2     | 0.267 | 0.423 | 0.266   |
| **pw_0.6**    | 3     | **0.272** | **0.433** | **0.270** |
| combo_d07_pw06| 3     | 0.270 | 0.432 | 0.268   |

**Verdict (refined):**
1. **pw=0.6 alone is the cleanest win** — +0.003 fitness on full-176,
   +0.001 on JAAD, +0.002 IDF1 on full-176, +0.008 IDF1 on JAAD.
2. **δ=0.7 alone slightly regresses here** vs ship (within ±0.003 noise
   but the direction flipped from A3b's measurement). The earlier
   δ=0.7 win on A3b was likely measurement noise — different eval run,
   different rate-of-cache-hit timing on JAAD clips. Single-eval
   variance is real.
3. **Combos don't stack additively here.** d0.7+pw=0.6 matches pw=0.6
   alone on full-176 but loses a hair on JAAD. The two knobs interact
   weakly at best.

**Refined ship recommendation:**
- **Definite: pw=0.6.** Real but small win; multi-seed F3 will bound the
  variance.
- **Probable: dd=0.70.** E3 showed +0.002 fitness, -2 fp_tracks. Below
  the F1 measurement noise but mechanically clean.
- **Skip: δ=0.7.** A3b's +0.001 reverses to -0.001 here. Noise.
- **Skip: c_FP_track change.** No signal in E1.
- **Skip: track_buffer, max_consecutive_misses changes.** No clear win.

Multi-seed F3 + multi-knob F5 (still queued) will close the loop.

**Bottleneck observation:** F1 took 20 min for 820 clip-runs (vs A3b's
14 min for 528 clip-runs ≈ 3.3 vs 1.6 s/clip-run). Same hardware. The
slowdown vs A3b — combined with the noise revealing — suggests the
GPU is now under heavier ambient contention than during A3b. Possible
explanation: D1's state-head training had finished but its TRT engine
warmup state may have changed GPU memory layout. Worth instrumenting
GPU residency between evals; out of scope for tonight.

---

### Experiment F1 — combo (δ=0.7, pw=0.6) verification (queued — done above)

---

### F-queue — multi-seed + extended-pw (results)

**Wall: ~50 min trainings + ~34 min eval = ~84 min.**

#### F3 — pw=0.6 multi-seed (variance bound):

| Variant   | full-176 fp_tr | full-176 IDF1 | full-176 fit | JAAD fp_tr | JAAD IDF1 | JAAD fit |
|-----------|----------------|---------------|--------------|------------|-----------|----------|
| ship      | 68             | 0.586         | 0.574        | 2          | 0.421     | 0.265    |
| F3_pw06_s0| 67             | 0.590         | **0.579**    | 3          | 0.432     | 0.268    |
| F3_pw06_s1| 70             | 0.583         | 0.565        | 1          | 0.428     | 0.271    |
| F3_pw06_s2| **135**        | 0.588         | 0.553        | 8          | 0.459     | **0.286**|

**CRITICAL FINDING: pw=0.6 multi-seed shows enormous variance.**
- seed 0: 0.579 ← lucky, this is the run that A1 reported
- seed 1: 0.565 (−0.014 vs seed 0, fp_tr ~normal)
- seed 2: 0.553 (−0.026 vs seed 0, fp_tracks **double**)

std across 3 seeds is ~0.013 fitness — **3× the single-run noise band**.
This is exactly the bimodal state-head training failure mode documented
in `feedback_track_phase20c_failed.md` ("default cr_promote=0.1 is
bimodal (1/5 working)"). Pw=0.6 happens to land in a similar trap.

**Implication for A1's findings:** the entire A1 sweep was single-seed.
Every "wins" report from A1 — pw=0.6, pw=0.7, pw=1.0 — was a one-shot
on the same trap-prone training recipe. **A1 cannot distinguish a
small real improvement from a lucky seed-pick.** The seed=0 pw=0.5 in
A1 was 0.575 vs seed=0 pw=0.6 at 0.579 — within the seed-variance band.
The "pw=0.6 wins" was almost certainly seed luck, not signal.

#### F4 — extended pw range (1.2, 1.5, 2.0):

| Variant | full-176 fp_tr | full-176 fit | JAAD fp_tr | JAAD IDF1 | JAAD fit |
|---------|----------------|--------------|------------|-----------|----------|
| F4_pw12 | 175            | 0.540        | 11         | 0.477     | 0.299    |
| F4_pw15 | 185            | 0.536        | 13         | 0.483     | 0.303    |
| F4_pw20 | 187            | 0.536        | 13         | 0.483     | 0.303    |

**JAAD val saturates at pw=1.5** (0.303 vs A1's pw=1.0 at 0.290). Further
pw doesn't help. fp_tracks explode on full-176 — these are not viable
ship configs without per-deployment domain split.

**Action items (revised, honest):**
1. **REVOKE pw=0.6 ship recommendation.** Multi-seed shows variance
   dominates the apparent +0.004 fitness signal from A1.
2. **All A1 conclusions are now suspect.** They need multi-seed
   re-eval to be trustworthy.
3. **dd=0.70 remains a robust ship change** (no state-head retrain
   involved, no seed-variance dependence).
4. **For domain-split (JAAD-cam ship): pw=1.5 with K-seed-fit-pick.**
   Train K=5 seeds at pw=1.5, pick by JAAD val fitness. Defer to a
   future session.

**Bottleneck observation:** trained 5 state heads sequentially (~10 min
each), didn't parallelise. With multi-seed variance now exposed as a
first-order concern, future A-like sweeps should default to K=3 seeds
per setting. Wall cost grows 3× but signal becomes interpretable.

This adds a strong vote for *fitness-aware training-time validation*
(per `feedback_offline_online_gap.md`): without seed-level confirmation,
single-eval improvements are unreliable.

---

### D2 — JAAD oversample 4× (results)

**Wall: ~22 min** (oversample npz + train_phase3 + state corpus + state
head + eval).

| Variant   | full-176 fp_tr | full-176 IDF1 | full-176 fit | JAAD fp_tr | JAAD IDF1 | JAAD fit |
|-----------|----------------|---------------|--------------|------------|-----------|----------|
| ship      | 67             | 0.587         | 0.575        | 2          | 0.422     | 0.266    |
| d2_jaadx4 | **62**         | 0.586         | **0.577**    | 2          | 0.423     | 0.261    |

**Verdict: D2 helps full-176 (+0.002 fitness, −5 fp_tracks) but hurts
JAAD val (−0.005 fitness).** Same direction as D1 — adding/upweighting
JAAD pairs makes the model more cautious, reducing fp_tracks on the
crowded full-176 (where small over-eager merges produce spurious tracks)
but reducing the in-domain JAAD recall.

Together D1+D2 falsify the "more JAAD data closes the cross-domain gap"
hypothesis from two different angles. The cross-domain gap is **not
a data-volume or data-weight problem**. It's a structural mismatch
between the training objective and JAAD's deployment regime (camera
motion, sparse-density scenes).

**Action:** D2 does not ship as the default. If a full-176-leaning ship
variant is ever desired (e.g. surveillance with dense crowds), D2 +
pw=0.6 + dd=0.70 + λ=0.05 might combine to be the cleanest full-176
recipe. Defer to a future session.

---

### G1 — runtime profiling

**Wall: ~4 min** (16 short runs × 4 clips × 4 δ values).

| Clip               | δ=0.0 wall | δ=0.5 wall | δ=0.7 wall | δ=1.0 wall |
|--------------------|------------|------------|------------|------------|
| CEVO indoor (320f) | 9.18s      | 9.19s      | 9.13s      | 9.24s      |
| PP22 long          | 26.74s     | 27.07s     | 26.98s     | 27.00s     |
| MOT17 medium       | 8.33s      | 8.45s      | 8.26s      | 8.28s      |
| JAAD short         | 5.09s      | 5.13s      | 5.08s      | 5.12s      |

**Major finding: the cheap-filter does not measurably affect wall time.**
δ=0.0 (filter off) runs at the same speed as δ=0.7 (aggressive filter).
Differences are sub-1% — within timer noise.

**Mechanism:** the detector (TRT engine, YOLOv26-L int8) dominates the
per-frame cost. NN-match-cost forward pass is a small head on top of a
cheap pre-tower lookup; it executes in microseconds compared to the
detector's milliseconds. Saving 50% of NN evals (the documented
cheap-filter dividend) doesn't move wall time because the NN evals
weren't constraining throughput in the first place.

**Strategic implication:** the cheap-filter, while complex, is
**fitness-neutral and speed-neutral**. The complexity-budget argument
that it was earning (faster runtime → maybe ship more aggressive δ)
doesn't materialise. Could be revisited for removal once we have
confidence the F-queue results show no fitness improvement from it.

This also explains why fitness changes per-δ were within noise: the
cheap filter's only effect is to occasionally drop a (track, det)
pair whose match cost would've been dominated by the heuristic anyway.
At the deltas tested, the NN's marginal contribution to those pairs
was negligible.

**Action:** keep the cheap-filter for now (it was hard-won in iter1
training-time correctness). But its value proposition shifts from
"saves runtime" to "regularises the matcher's input distribution."
Tracking in CHEAP_FILTER_NEXT.md as a possible future removal.

---


**Hypothesis:** the two single-knob winners stack additively because
they touch different parts of the cost rule (δ gates which (track,
det) pairs the NN evaluates; pw rebalances the state-head accept/
reject decision after matching is done). They're not coupled.

**Plan:** one variant `uc_v11_F1.yaml` with both changes; eval full-176
+ JAAD val. Will be slotted into the F queue once A1's pw=0.6 bin is
in the right path. **Driver appended to E queue** so it fires after
E5. Estimate: ~6 min for the eval.

If F1 holds (fitness ≥ pw=0.6 alone on full-176 *and* ≥ δ=0.7 alone
on JAAD val), this is the new ship configuration.

---

### E queue — self-paced overnight extensions (queued after D2)

User asked to keep iterating after the initial queue. Five
inference-only sweeps, each pinned at δ=0.7 (A3b winner) so we're
measuring the next axis cleanly. All use full-176 + JAAD val. Each
sweep ~25 min wall, ~2h total queue.

| ID | Knob | Sweep | Hypothesis |
|----|------|-------|------------|
| E1 | `bayes_c_FP_track` | 3e-4, 5e-4, 7e-4, 1e-3, 1.5e-3 | δ=0.7 changes the FP-track penalty balance — current 7e-4 was tuned at δ=0.5. |
| E2 | `track_buffer_seconds` | 1.5, 2.0, 2.2, 3.0, 4.0 | Buffer/NN re-match interaction shifted under the new filter. |
| E3 | `delete_dup_iou` | 0.70, 0.80, 0.90, 0.95, 1.0 | Memory: stronger match NN should allow looser dedup. 1.0 = disabled. |
| E4 | `nn_lambda` | 0.02, 0.05, 0.10, 0.15, 0.20 | NN scale in cost — δ=0.7 narrows where NN fires, so we may now want it louder when it does. |
| E5 | `max_consecutive_misses` | 5, 7, 10, 15, 20 | Pure-NN state head should handle longer absences; current 10 was a heuristic-era default. |

Driver: `/tmp/run_e_queue.sh`, log `/tmp/e_queue.log`. Each sweep writes
its `results-*.txt` to `/mldata/track_runs/<name>/results/`.

**Decision rule:** after each sweep, the single best variant's
fitness/MOTA/fp_tracks + IDF1 (added when available) go in the table
below. If a sweep's best is **0.003+** above the shipped baseline on
both views (full-176 and JAAD val), it's a candidate ship — and a
follow-up F-queue variant combining it with the other winners gets
scheduled.

If a knob shows no signal (all variants within ±0.003), record that and
move on — negative results are signal too.

---

### E1 — c_FP_track sweep (results)

**Wall: ~32 min** (1025 clip-runs).

| c_FP_track | full-176 fit | fp_tr | JAAD val fit | JAAD val MOTA |
|------------|--------------|-------|--------------|---------------|
| 3e-4       | 0.575        | 71    | **0.272**    | **0.271**     |
| 5e-4       | 0.575        | 69    | 0.268        | 0.267         |
| 7e-4⁺      | 0.575        | 67    | 0.268        | 0.266         |
| 1e-3       | 0.575        | **62**| 0.268        | 0.266         |
| 1.5e-3     | 0.570        | 66    | 0.262        | 0.261         |

⁺ current value.

**Verdict: no signal in the 3e-4 → 1e-3 band.** Cost-rule c_FP_track is
saturated at the current configuration. 1.5e-3 starts to break (-0.005
fitness), 3e-4 has a 0.004 JAAD-val bump but it's within noise.

Best mechanical trade: `c_FP_track=1e-3` gives 5 fewer fp_tracks at
identical full-176 fitness and identical JAAD val fitness. Pure
mechanical win on the fp axis, no fitness change. Not worth shipping
on its own but bundling into F2 won't hurt.

**Bottleneck:** flat-rate fp_track-vs-fitness tradeoff is already on
the Pareto frontier for the shipped NN. Further fp-track suppression
costs fitness.

---

### E2 — track_buffer_seconds sweep (results)

**Wall: ~31 min**.

| tbuf | full-176 fit | fp_tr | JAAD val fit | JAAD val MOTA |
|------|--------------|-------|--------------|---------------|
| 1.5  | 0.570        | 78    | 0.265        | 0.263         |
| 2.0  | 0.575        | 67    | 0.268        | 0.267         |
| 2.2⁺ | **0.576**    | **66**| 0.269        | 0.268         |
| 3.0  | 0.575        | 66    | 0.268        | 0.267         |
| 4.0  | 0.573        | 69    | **0.270**    | 0.268         |

⁺ current value.

**Verdict: 2.2 is at the peak.** A clean U-shape: too short (1.5) loses
fitness *and* gains fp_tracks (tracks die before resumption finishes),
too long (4.0) loses fitness as zombie tracks pile up. 2.0–3.0 all
within 0.001 — flat top, well-tuned.

JAAD val pattern: 4.0 wins by 0.001 over 2.2 — extending buffer helps
the long-occlusion JAAD case slightly. Same recall/precision lever as
A1's pw sweep (pw=1.0 also wins JAAD by being more permissive).

**No ship change.** This is a confirmed-default result — useful to log
because it proves we don't need to revisit this knob next quarter.

---

### E3 — delete_dup_iou sweep (results)

**Wall: ~25 min**. Memory `feedback_track_thresholds_dedup.md` predicted
a stronger NN should allow looser dedup — confirmed.

| dd     | full-176 fit | fp_tr | JAAD val fit |
|--------|--------------|-------|--------------|
| **0.70** | **0.577**  | **65**| 0.266        |
| 0.80   | 0.575        | 68    | 0.266        |
| 0.90⁺  | 0.575        | 67    | 0.267        |
| 0.95   | 0.574        | 68    | 0.266        |
| 1.0    | 0.573        | 69    | 0.266        |

⁺ current value.

**Verdict: ship dd=0.70.** +0.002 fitness on full-176, 2 fewer fp_tracks
(65 vs 67), no JAAD regression. Disabling dedup (1.0) lost 0.002 → the
NN doesn't fully replace the iou-based dedup; it just allows the
threshold to relax. The user's "revisit dedup" instinct was right.

**Ship candidate stack: δ=0.7 + pw=0.6 + dd=0.70.** Three small wins
that should combine (different stages of the pipeline).

---

### E4 — nn_lambda sweep (results, with IDF1 column)

**Wall: ~25 min**. **First sweep to show IDF1** (added to eval template
after E3).

Full-176:
| λ    | fp_tr | MOTA  | **IDF1**  | fitness |
|------|-------|-------|-----------|---------|
| 0.02 | 63    | 0.608 | 0.570     | 0.574   |
| 0.05⁺| 68    | 0.612 | 0.588     | **0.576** |
| 0.10 | 96    | 0.610 | 0.598     | 0.560   |
| 0.15 | 111   | 0.612 | 0.604     | 0.554   |
| 0.20 | 119   | 0.611 | **0.607** | 0.549   |

JAAD val:
| λ    | fp_tr | MOTA  | IDF1  | fitness |
|------|-------|-------|-------|---------|
| 0.02 | 1     | 0.266 | 0.420 | 0.266   |
| 0.05⁺| 2     | 0.267 | 0.423 | 0.266   |
| 0.10 | 2     | 0.273 | 0.428 | 0.272   |
| 0.15 | 3     | 0.274 | 0.437 | 0.272   |
| 0.20 | 3     | **0.278** | **0.443** | **0.276** |

⁺ current value.

**Major finding: IDF1 climbs monotonically with λ** (0.570→0.607 on full-
176; 0.420→0.443 on JAAD). The NN strongly improves identity-preservation
when it's louder — but each extra λ also adds fp_tracks, dragging fitness
down on full-176. On JAAD where fp_tracks are tiny anyway (1→3), the
recall gain dominates: **λ=0.20 wins JAAD fitness by +0.010** over ship.

**Strategic implication:** nn_lambda is the cleanest "IDF1 dial" we've
found. If a future user cares about ID consistency over fp-track-rate
(e.g. surveillance with manual review, downstream re-id), λ=0.10 trades
0.016 fitness for +0.010 IDF1. The fitness metric penalises fp_tracks
disproportionately for this use case — a deployment that cares about
keeping the same person on the same ID would prefer λ=0.10.

**Action items:**
- Ship recipe: stay at λ=0.05 (best fitness on full-176).
- Add F5 variant: λ=0.10 + δ=0.7 + pw=0.6 to test if the IDF1
  gradient stacks with the other shipping changes — and to expose a
  documented "IDF1-leaning" alternative ship.
- Memory note worth saving: λ controls IDF1; current 0.05 is the
  fitness optimum but λ=0.10 is the IDF1-leaning point.

---

### E5 — max_consecutive_misses sweep (results)

**Wall: ~25 min**.

Full-176:
| mm  | fp_tr | MOTA  | IDF1  | fitness |
|-----|-------|-------|-------|---------|
| 5   | 88    | 0.611 | 0.589 | 0.565   |
| 7   | 76    | 0.611 | 0.589 | 0.571   |
| 10⁺ | 67    | 0.611 | 0.588 | **0.576** |
| 15  | **65**| 0.610 | 0.585 | 0.576   |
| 20  | 65    | 0.610 | 0.585 | 0.575   |

JAAD val:
| mm  | fp_tr | MOTA  | IDF1  | fitness |
|-----|-------|-------|-------|---------|
| 5   | 3     | 0.261 | 0.422 | 0.259   |
| 7   | 3     | 0.266 | 0.423 | 0.265   |
| 10⁺ | 2     | 0.267 | 0.423 | 0.266   |
| 15  | 2     | 0.267 | 0.422 | 0.266   |
| 20  | 2     | **0.269** | **0.424** | **0.268** |

⁺ current value.

**Verdict:** mm=10/15 statistical tie on full-176 fitness (both 0.576).
mm=15 has 2 fewer fp_tracks but trivial IDF1 drop. mm=20 wins JAAD by
0.002 fitness — long-occlusion handling helps dashcam.

mm=5 catastrophic (full-176 0.565) — kills good tracks that miss a few
frames. The C default for unified-deletion is well-tuned in the
10–15 range. No clear ship change; current 10 is fine.

---







