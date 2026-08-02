# A software dense optical flow to stand in for NVOF

> **!! ALL SCORES BELOW WERE MEASURED ON THE WRONG CONFIG (corrected 2026-08-02).**
> Every number in this document came from `eval_ship_baseline.yaml` — 205 clips,
> 2 unweighted groups, so 29 dashcam clips carried HALF the signal. That is a
> "where are we now?" snapshot, not the objective. The objective is
> `track_search_v11_mc.yaml`: 546 clips, 6 groups, `group_weights`,
> `clip_weight_cap_pctl: 90`. The snapshot config has since been DELETED and
> `track.py --eval` with no path now runs the objective directly.
>
> The distortion is not cosmetic. Hardware-vs-software optical flow reads
> **-0.0033** on the snapshot weighting and **-0.0006 ±0.0011** — i.e. nothing —
> on `_overall` from the same runs. The dashcam-weighted conclusions in
> "Chasing the dashcam gap" are therefore built on a metric that gave dashcam
> ~50% weight. The mechanism findings (which hypotheses were rejected, and why)
> still stand; the magnitudes do not. Re-measure against the objective before
> quoting any figure here.


**Question.** Can a CPU optical-flow estimator produce a field good enough that
the tracker does not care whether it came from silicon? The only software
backend was `nvof_apple.c`, and it was not close.

**Answer.** Yes, with the motion-estimation design video encoders use — pyramid,
predictors, a smoothness penalty, and a sub-pixel step. On synthetic ground
truth it recovers whole-pixel motion **exactly** across the pyramid's range, and
lands ~100% of cells within half a pixel on fractional motion.

Code: `ubon_cstuff/src/nvof/nvof_sw.c`, `include/nvof_sw.h`.
Tests: `src/tests/test_nvof_sw.cpp` (accuracy), plus the existing
`NvofConventionTest` suite, which this backend passes unmodified.

---

## What was wrong with the old one

`nvof_apple.c` ran a brute-force full search: every integer offset in a ±8 px
window, 4×4 blocks, scalar. That is 289 candidates × 16 pixels per cell, and it
buys:

- **a ±8 px ceiling** — nothing faster than 8 px/frame is representable at all;
- **no regularisation** — 4×4 SAD is close to noise on real texture, and the
  winner is whichever offset wins by a hair;
- **whole pixels only** — fractional camera motion quantises to steps;
- **no SIMD**.

## The design

Coarse to fine over a 2× pyramid. At each level, each cell is matched from a
small **candidate set** rather than a scanned window:

- the co-located vector from the coarser level, doubled;
- the already-decided left / top / top-right neighbours;
- **a temporal hint** — this cell's vector from the previous frame. Camera
  motion is continuous, so last frame's answer is usually within a pixel of this
  one. NVOF does the same thing (`disableTemporalHints=NV_OF_FALSE` in
  `nvof_cuda.c`, described in NVIDIA's header as "flow vectors from previous
  NvOFExecute call", recommended for successive frames of continuous video);
- zero.

The best candidate is then refined by a diamond search with a halving step.

The cost minimised is **not** raw SAD:

```
SAD(window) + lambda * (|mv.x - pred.x| + |mv.y - pred.y|)
```

where `pred` is the median of the decided neighbours. This penalty is what makes
the field regular enough to fit a global transform to, and it is the single
biggest difference from the old backend.

Two further choices:

- **matching window 8×8, output grid 4×4.** The contract (`nvof.h`) is one
  vector per 4×4 cell, which both hardware backends also emit. Matching on an
  8×8 window centred on each cell means neighbouring cells overlap and share
  evidence — smoother than a 4×4 tiling, at no memory cost.
- **edge clamping, not an edge penalty.** A constant out-of-frame penalty biases
  the search toward zero motion.

## Measurements

Synthetic ground truth: band-limited random texture, warped by a known
transform, scored as mean endpoint error over cells. 960×544, 6-frame
constant-motion sequences so the temporal hint is exercised the way a pan
exercises it. Harness: `of_bench.c` (scratch).

### Sub-pixel: needed, but not the obvious way

The question was raised as "not obvious if we NEED subpixel?". Measured three
ways:

| sub-pixel mode | mean EPE | ms/frame | 1.6 px pan: cells <0.5 px |
|---|---|---|---|
| none (whole pixel) | 0.859 | 5.40 | 47.6% |
| half-pel, bilinear search | 0.800 | 16.56 | 80.4% |
| **parabola through the SAD minimum** | **0.667** | **5.91** | **100.0%** |

Sub-pixel matters, and it matters exactly where CMC lives — small fractional
camera motion. But a half-pel search is the wrong way to buy it: it triples the
runtime resampling the reference, and is still *less* accurate than fitting a
parabola through three SAD values the search already has. The parabola is
continuous rather than half-pel quantised, and costs four extra SADs per cell
(+9% runtime).

**Two guards on the fit**, both of which turned out to matter:

- If the match is exact (`s0 == 0`) do not refine. Otherwise the vertex still
  lands off-centre whenever the two neighbours differ, so a *static camera*
  emits a small non-zero vector in every cell and the CMC fit reads steady
  motion that is not there. Caught by `NvofSw.StaticSceneProducesAZeroField`.
- Only fit when the centre is genuinely the raw-SAD minimum. The search
  minimises SAD *plus* the penalty, so its winner need not be, and fitting
  through a non-minimum extrapolates.

With the guards, whole-pixel pans come back exactly (EPE 0.000, from 0.003).

### Accuracy, final configuration

| case | mean EPE | cells <1 px |
|---|---|---|
| pan 2 px/frame | 0.000 | 100% |
| pan 6 px/frame | 0.000 | 100% |
| pan 18 px/frame | 0.001 | 100% |
| pan 40 px/frame | 3.93 | 93.2% |
| sub-pixel 0.4 px | 0.153 | 100% |
| sub-pixel 1.6 px | 0.209 | 100% |
| zoom 1.02 + 1° | 0.165 | 100% |
| zoom 1.05 + 3° | 1.36 | 95.9% |

The 40 px/frame residual is content genuinely leaving the frame: 40 px right and
12 px down past a 16 px margin accounts for ~4.7% of cells, which is the whole
of the 6.8% miss. No estimator can match what is no longer there.

### SIMD

SAD kernels for SSE2 and NEON, scalar fallback. **An AVX2 kernel was written and
rejected**: four 8-byte rows per 256-bit register, identical results, no faster
(6.66 vs 6.41 ms/frame). An 8×8 block's rows do not fill a 256-bit lane without
shuffle work that costs more than the wider SAD saves, and SSE2 is baseline on
every x86-64. x86 uses SSE2 unconditionally.

---

## A trap worth recording

The first accuracy run said large motion was broken: 40 px/frame left 28% of
cells stuck at zero, and the failure rate was suspiciously *stable* across
lambda values (71.7% → 72.2% over a 4× sweep). Lambda was not the cause. The
**harness** was: it warped each frame from the original image, so by frame 6 a
40 px/frame pan had shifted 240 px and a quarter of the frame was edge-replicated
with no texture to match. Rendering every frame from an oversized source instead
turned 92.2% into 100.0% with no change to the estimator.

A test that fabricates texture-free regions and then measures matching failure
in them is measuring itself. `test_nvof_sw.cpp` crops from a padded texture for
this reason, and says so.

## Track quality vs hardware NVOF

**Result: -0.0043 groupmean. Indistinguishable from hardware on 176 of 205
clips; the entire loss is on 29 dashcam clips.**

| arm | groupmean | full176 | jaad_val (dashcam) |
|---|---|---|---|
| hardware NVOF | **0.3013** | 0.4257 | 0.1769 |
| software, final | **0.2970** (-0.0043) | 0.4255 (-0.0002) | 0.1684 (-0.0085) |

The eval turned out to be **deterministic** on this config — three hardware
replicates gave 0.3013 / 0.3013 / 0.3013, with 5 of 204 clips moving by =<0.004
and no movement at the group level. (The 0.0055 run-to-run sd measured in the
detection-free-frames campaign is config-dependent and does not apply here.)
Three software replicates were likewise bit-identical, so the -0.0043 is a real
difference and not a sampling artefact.

### Chasing the dashcam gap

Three hypotheses, all tested on the 29 jaad clips, all rejected or nearly so:

| hypothesis | test | result |
|---|---|---|
| The CMC fit accepts a fiction it used to reject | gate forced to reject (`max_resid_frac: 1.0`), both arms | **rejected** — gap persists and widens (0.0096 -> 0.0121) |
| Smoothness penalty smears object motion into the background | lambda swept 2 / 6 / 12 / **24** / 48 / 96 / 192 | **rejected** — 24 is an interior optimum; both directions are worse |
| Cost byte is miscalibrated against NVOF's scale | scale/offset swept | **mostly rejected** — +0.0011 from matching the noise floor, scaling hurts |

The cost probe is worth keeping: on real dashcam frames the hardware backend
puts 83.1% of cells under `cost_ok`, the software one 94.8% — the software field
reports itself as *more* reliable than it is. Only part of that matters, because
neither backend ever reaches `cost_drop` (0.0% of cells), so nothing is
discarded; only the weighting differs.

What did land: `motion_track.c` hard-codes NVOF's properties, subtracting a
noise floor of 5 and calling a cell decisive above 12. A block matcher that
finds an exact match reports **0**, so `scene_cost` collapsed toward zero and
`scene_cover` read systematically low. Adding a floor of 5 to the cost byte
(`SW_COST_OFF`) recovers +0.0005 overall. A multiplicative rescale to match the
distribution's *shape* was also swept and made things worse (1.5x: -0.0029 on
dashcam), so only the floor is matched.

**What is left is the vectors themselves on dashcam**, not how they are weighted
or consumed. The residual shows up as tracking instability rather than a global
mis-fit: switches/object 0.158 vs 0.124, fragmentations/object 0.312 vs 0.253,
false tracks 16 vs 10. The untested hypothesis is the matching window: an 8x8
window on a 4x4 grid means a small pedestrian's cell draws most of its evidence
from a fast-moving background, and the penalty then pulls it the rest of the
way. Testing that needs a 4x4 SAD kernel and is the obvious next step if the
gap ever needs closing.

Method (SUPERSEDED — see the warning at the top): `track.py --eval` on
`/mldata/config/track/eval/eval_ship_baseline.yaml`, val split, 205 tests, 3
replicates per arm. That config no longer exists; the objective is
`track_search_v11_mc.yaml`, reached by `track.py --eval` with no path.

The backend is selected at runtime with `UBON_NVOF_SW=1`, which
`nvof_cuda.c` honours — so both arms are the same binary, same frames, same
tracker, differing only in where the flow comes from. Liveness was proven
before measuring, not assumed: the binding sets `log_set_level(LOG_WARN)` at
import, so the backend's INFO line is invisible in a normal run; raising the
level shows `nvof: using SOFTWARE backend (sse2 kernels)` under `=1` and nothing
under `=0`.

---

# Second campaign: is SW good enough to replace NVOF *everywhere*? (x86)

**Question.** Retire the hardware backend and run the software one on every
platform, for the simplicity of a single implementation. What does it cost?

**Answer.** ~0.003 groupmean and ~2% throughput on x86, all of the quality loss
still on dashcam. Cheap enough to be a reasonable trade — but a second attempt
at closing the dashcam gap **failed**, and four hypotheses are now ruled out.

Measured on the canonical eval, **8 replicates per arm** (the eval now runs on
the single-shared-state path, which is 3-5x faster but no longer
bit-reproducible, so replicates are mandatory — see `eval_ship_baseline.yaml`).

| | hardware | SW v1 | **SW v2** |
|---|---|---|---|
| groupmean | 0.2982 +-0.0015 | 0.2931 +-0.0031 | 0.2949 +-0.0016 |
| vs hardware | — | -0.0051 | **-0.0033 +-0.0022** |
| full176 | 0.4270 +-0.0003 | 0.4267 | **0.4269 +-0.0002** |
| jaad_val (dashcam) | 0.1694 +-0.0030 | 0.1596 | **0.1630 +-0.0033** |
| switch/obj | 0.726 | 0.756 | 0.738 |

The 176 non-dashcam clips are **identical** — 0.4270 vs 0.4269 against a
+-0.0003 error bar. This reproduces the first campaign's -0.0043 on a different
eval path and a different engine, so it is a stable property of the estimator,
not a sampling artefact. **v2 is strictly the one to use**: cheaper than v1 AND
better on every axis.

### Cost on x86

Normalised against video processed (7,002 s over 205 clips), which needs no
assumption about how often the estimator runs:

| | extra CPU | per realtime stream | wall clock |
|---|---|---|---|
| SW v1 | 21.4 CPU-s | 0.31% of a core | +2.6% |
| **SW v2** | 13.4 CPU-s | **0.19% of a core** | **+2.1%** |

The v1/v2 ratio is 1.6x, matching v2's known speedup — good evidence the deltas
really are the optical flow. CPU is a non-issue (64 realtime streams cost ~0.12
of a core on a 16-core box). The 2% wall clock is **not** CPU: it is the
synchronous `cudaMemcpy2D` D->H of the luma plane in `nvof_cuda.c`, which
serialises against the CUDA stream. That cost is x86-only — Jetson's memory is
unified.

## Chasing the dashcam gap, second attempt: four dead ends

Iterated on a jaad-only eval (29 clips, **8.5 s per run**, 6-12 replicates per
config). Note it scores ~0.207 where the same clips score ~0.169 inside the full
eval — lighter load changes the dynamics, so it is a proxy, and a win there
would still need confirming on the full eval. None was.

| hypothesis | test | result |
|---|---|---|
| Cost is residual, not reliability: flat road/sky matches read cheap but are meaningless, and `cmc_fit` weights by cost | added an ambiguity penalty from the SAD-minimum **curvature** (`min` over the two axes, to catch the aperture problem); free on cells that reach the 4x4 tier, since the sub-pixel parabola already probes those four SADs. Swept K and REF | **rejected** — no K improved anything, and see the correction below |
| The 4x4 descent under-fires, so a small pedestrian's vector is dominated by the fast background it shares a quadrant with (the first campaign's "obvious next step") | forced the descent on for every cell, and off entirely, to bracket it | **rejected** — always-split is *identical* to the trigger (mean cost 13.6 vs 13.6): on dashcam the trigger already fires almost everywhere. Never-split is worse (-0.0098 vs -0.0057), so the descent is already earning its keep |
| Sub-pixel noise stops `cmc_fit`'s static gate (95% of cells under 4 ticks) from firing, so a stopped car still gets a bogus transform | dead-zone snapping near-zero vectors to exactly 0, swept 2..12 ticks | **rejected**, and the premise was an artefact — see below |
| The smoothness penalty over-regularises a divergent (forward-motion) field | lambda 8 / 12 / **24** / 48 on jaad | **rejected** — 12 looked like +0.004 at n=6 but collapsed to +0.0020 +-0.0023 at n=12 with `switch/obj` identical (0.1935 vs 0.1939). Noise. 24 stands |

### Two corrections to the record

**The first campaign's cost-calibration finding no longer applies to v2.** It
recorded hardware putting 83.1% of dashcam cells under `cost_ok` against the
software backend's 94.8% — the software field calling itself more reliable than
it was. Re-measured on v2 (`cmc_probe` now reports the cost distribution):

| | mean cost | under `cost_ok` | dropped |
|---|---|---|---|
| hardware | 12.4 | 74.5% | 2.3% |
| **SW v2** | 6.9 | **80.5%** | 0.4% |

6 points apart, not 11.7. v2's cost is already close to honest, which is why
there was nothing for the ambiguity penalty to win — and why the first
calibration attempt should not be repeated on v2 without re-measuring first.
The penalty was also wildly mis-scaled on the first try (K=20 drove under_ok to
29%, K=80 discarded 41% of all cells) because per-pixel curvature truncates to
an integer and bucket "0" spans everything from a dead-flat wall to a usable
cell. Raw window units are needed for that knob to be tunable at all.

**`cmc_probe`'s `static_fired` column cannot be trusted where it matters.** It
infers a gate fire by comparing the gated and ungated fits, so when the field is
exactly zero the two agree and it reports *no* fire. The software backend
produces exact zeros MORE often (its two sub-pixel guards), so it reads 0.0%
where hardware reads 2.9% — which looks like a smoking gun and is nothing of the
kind. On the one frame examined directly, the software field had frac4 = 1.000
against hardware's 0.999 while reporting "not fired".

### What is left

Not the cost byte, not the descent granularity, not the penalty weight, and not
sub-pixel quantisation. The CMC fit itself is not obviously worse either —
frame-to-frame jitter in the fitted transform is slightly *lower* for software
(0.0024 vs 0.0036 mean |d tx| on video_0156).

What remains is the vectors themselves on fast forward ego-motion, and the next
honest step is a direct field comparison against hardware on dashcam frames —
where do they disagree, and is the disagreement on the background (which would
corrupt the fit) or on the pedestrians (which would corrupt association)? That
needs a harness that runs both backends over the same decoded frames and dumps
both fields; every diagnostic used above was indirect, and indirect is how three
of these four hypotheses looked plausible in the first place.
