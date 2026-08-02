# Revalidation against the real objective (2026-08-02)

Every optical-flow / CMC conclusion from 2026-07-29..08-02 was measured on
`eval_ship_baseline.yaml` — 205 clips, 2 UNWEIGHTED groups, so 29 dashcam clips
carried half the signal. That config was a "where are we now?" snapshot, not the
objective, and it has since been deleted. This file re-runs every conclusion
against the real one and logs the outcome.

## The objective (locked — do not restate it from memory)

```
/mldata/config/track/search/track_search_v11_mc.yaml
  result_dataset_opt_key:   _groupmean
  result_dataset_opt_param: fitness_multi     <-- THE number
  group_weights: movie 0.05, cad_u04/b04/u06 0.1
  clip_weight_cap_pctl: 90
  fitness_weights: person 1.0, vehicle 0.3
```

Command (no yaml path — it runs the objective config directly):

```
python3 track.py --eval --eval-split val --results-location DIR
python -m src.eval_compare DIR [DIR ...]     # quote the _groupmean row
```

Scale: **215 val clips, 60.7 s/run.** Baseline reads **0.4099**, which is the
~0.41 seen in the search logs — the sanity check that we are finally on the
right measurement (the old config read ~0.29 for the same tracker).

## Rules for this campaign

- Only within-batch comparisons. Arms of one comparison run back-to-back;
  cross-session drift on the old config was ~0.003, larger than its own sem.
- No screening-then-promoting. Picking the best of N noisy arms is winner's
  curse — it manufactured a fake +0.0089 on the old config that vanished on
  confirmation. Every candidate gets the same replicate count and is judged
  against its own batch control.
- Quote `_groupmean` / `fitness_multi`. `_overall` is information only.

## Work list

| # | conclusion to revalidate | old claim (WRONG metric) | code needed | status |
|---|---|---|---|---|
| 1 | hardware NVOF vs SW v1 vs SW v2 | SW v2 −0.0033 groupmean / −0.0006 overall | none | **DONE — no difference** |
| 2 | cost weighting into CMC (cost_drop 80 vs 0) | +0.0045 groupmean / +0.0016 overall | env knob `UBON_NVOF_NO_COST` | **DONE — CLAIM REVERSED, worth nothing** |
| 3 | SIMD pyramid downsample | −17.8% time, bit-exact | committed `48847a3` | **no eval needed** — bit-exact by construction (identical field checksums), quality cannot move |
| 4 | penalty shape: truncated / SGM two-tier / log | no gain | done | **DONE — null (unchanged verdict)** |
| 5 | zero-motion bias | −0.0053 | done | **DONE — null, not −0.0053** |
| 6 | probe all 8 candidates | −0.0029 | done | **DONE — null (+0.0014, 1.6σ)** |
| 7 | multi-profile TRT engine quality | "identical" | none | pending (low priority) |
| **8** | **NEW: inter-layer median filter** (Sun/Roth/Black 2010) | — | done | **DONE — null (−0.0003)** |
| 9 | NEW: uniqueness ratio as cost byte | — | new | not started |
| 10 | NEW: census/rank transform instead of SAD | — | new | not started |

Items 4–6 were reverted uncommitted, so they need re-implementing before they
can be re-measured.

## Progress log

- **08:41** Objective confirmed and smoke-tested: 0.4099, 215 clips, 60.7 s.
- **09:05 — item 1 DONE. Software optical flow is indistinguishable from hardware.**
  n=6 per arm, interleaved.

  | arm | objective | vs hw |
  |---|---|---|
  | hardware NVOF | 0.4086 ±0.0007 | — |
  | SW v1 | 0.4091 ±0.0004 | +0.0005 (noise) |
  | SW v2 | 0.4081 ±0.0007 | −0.0004 (noise) |

  Groups are flat too (static 0.5709/0.5721/0.5711, moving 0.2818/0.2827/0.2803).
  The old config's −0.0033 was entirely its dashcam-at-50% weighting. Note the
  objective is also far QUIETER: sem 0.0004–0.0007 against 0.0013–0.0016 on the
  snapshot config, so it resolves smaller effects with the same replicate count.

  **Verdict: switching to software OF everywhere costs nothing measurable.**
  v2 remains preferred on cost (4.4x faster than v1: 0.22 vs 1.11 ms/frame).

- **09:35 — item 2 DONE. The +0.0045 cost-weighting result does not survive.**
  SW v2, costs on vs off (`UBON_NVOF_NO_COST`, which drops the cost plane at
  source — exactly what a backend that cannot report costs gives you). n=6,
  interleaved. Knob proven live first: `cmc_probe` shows the cost plane absent.

  | arm | objective | vs costs-on |
  |---|---|---|
  | costs ON (current) | 0.4083 ±0.0003 | — |
  | costs OFF | 0.4093 ±0.0006 | **+0.0010 ±0.0007 (noise)** |

  Cost weighting is worth nothing on the objective, and the point estimate is
  very slightly AGAINST it. The earlier +0.0045 was the dashcam-at-50%
  weighting: cost weighting mattered on dashcam and dashcam owned that metric.

  **This retires the "plumb costs through on Jetson" idea** — the thing I began
  changing `nvof_jetson.c` for. There is no measured gain to collect. Nothing
  was committed there; the file is untouched.

- **09:50 — items 4, 5, 6 re-implemented and item 8 (median filter) written.**
  All four behind env knobs, all default OFF, and the estimator is bit-exact
  when they are: `of_bench` field checksums match the committed build exactly
  (`e86a303429a1e217` at 512x288, `60cd3b982ecaa9a2` at 960x544).

  Each knob proven live on real dashcam footage before measuring (a stale
  `cmc_probe` binary initially showed all four as no-ops — rebuilt):

  | knob | frac2 | mean_max_abs |
  |---|---|---|
  | base | 0.1702 | 582.8 |
  | `SW2_MEDIAN=1` | 0.1706 | **562.7** |
  | `SW2_PROBE_ALL=1` | 0.1695 | 591.9 |
  | `SW2_ZERO_BIAS=256` | 0.1938 | 583.3 |
  | `SW2_PEN_SHAPE=1` | 0.1709 | 658.3 |

  The median filter measurably SHRINKS the largest vectors (582.8 -> 562.7),
  which is the runaway suppression it is supposed to provide; the truncated
  penalty grows them (658.3), which is the runaway exposure pure truncation
  is known for. Sweep of 6 arms x 6 reps running.

- **10:35 — items 4, 5, 6 and 8 DONE. All null on the objective.**
  6 arms x 6 reps, interleaved, single shared control.

  | arm | objective | vs control |
  |---|---|---|
  | control | 0.4090 ±0.0008 | — |
  | median filter (item 8) | 0.4087 ±0.0005 | −0.0003 |
  | probe all 8 (item 6) | 0.4104 ±0.0004 | +0.0014 (1.6σ) |
  | zero-bias 256 (item 5) | 0.4091 ±0.0008 | +0.0001 |
  | penalty trunc T=16 (item 4) | 0.4090 ±0.0005 | +0.0000 |
  | penalty SGM two-tier (item 4) | 0.4088 ±0.0008 | −0.0002 |

  `probe_all` is NOT promoted on 1.6σ — that is the winner's-curse move that
  produced a fake +0.0089 on the old config. It read −0.0029 on the old metric
  and +0.0014 here: indistinguishable from zero under both.

  The median filter does what it claims mechanically (largest vectors 582.8 ->
  562.7) but that does not convert into tracking quality. Plausible reason: the
  estimator feeds `cmc_fit`, which already runs Huber IRLS over the field, so
  field-level robustness is redundant by the time it matters.

## Summary of the revalidation

Of everything measured on the old config, **not one conclusion survived as a
positive result**, and two REVERSED:

| conclusion | old (wrong metric) | objective | verdict |
|---|---|---|---|
| SW optical flow costs quality | −0.0033 | −0.0004 ±0.0010 | **reversed — costs nothing** |
| cost weighting into CMC helps | +0.0045 | +0.0010 for OFF | **reversed — worth nothing** |
| penalty shape / zero-bias / probe-all hurt | −0.001..−0.005 | all within ±0.0014 | null either way |
| SIMD downsample −17.8% | (timed, not eval) | unaffected | **stands** |

The one durable win of the whole campaign is the SIMD downsample (`48847a3`),
and it is the one result that never went through the eval at all — it was timed
directly and verified bit-exact.

### Still open
- item 7 (multi-profile TRT engine quality) — needs a way to vary the TRACKER
  config without copying the eval yaml; a `--tracker-config` CLI override is the
  clean analogue of `--results-location`.
- items 9, 10 (uniqueness ratio, census transform) — not started.
