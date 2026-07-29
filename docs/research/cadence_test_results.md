# Cadence test — phase 1 results (current code baseline)

*2026-07-28. Instrument: docs/plans/cadence_test.md. Set: 30 clips
(cadence_manifest.json), 167 built variants, epsilon min-interval
(0.001 s), preflight-verified frame-exact delivery (sole artifact: final
frame lost to decoder EOF flush, uniform across variants). Eval:
cadence_eval.yaml, binding @ ubon_cstuff 8c05deb, uc_v11 config of this
date. Raw log: /mldata/results/cadence_test/ + /home/mark/cadence_eval_phase1.log.*

## Headline: gaps, not irregularity, destroy tracking

Paired per-clip fitness deltas vs the same clip's uniform (`U`) variant —
all variants deliver the SAME 5 fps average:

| variant | cadence character | median Δ | mean Δ* | worse than U | group mean |
|---|---|---|---|---|---|
| U    | uniform 200 ms (control)          | —      | —      | —     | 0.334 |
| J50  | ±50 % jitter (100/300 ms)         | −0.072 | −0.076 | 23/29 | 0.267 |
| G2   | sparse/dense alternating seconds  | −0.220 | −0.261 | 28/29 | 0.193 |
| B50  | 5 frames @100 ms + 600 ms gap     | −0.229 | −0.343 | 27/29 | 0.135 |
| B17  | 5 frames @33 ms + 867 ms gap      | −0.390 | −0.508 | 27/29 | 0.051 |
| Pfpt | first-past-threshold: clean 208 ms but 4.8 fps (non-integer-ratio clips only) | +0.000 | −0.28* | 5/16 | — |

*means include heavy tails; one pair excluded: `U`/homes_00352 scored a
sentinel-garbage fitness of +501 (fitness-function robustness bug, same
family as the FPTf=13550 sentinels — track separately).

## Conclusions

1. **Dose-response in burstiness** at constant average rate: fitness falls
   monotonically J50 → G2 → B50 → B17; the max-burst regime destroys ~85 %
   of the uniform group score. Gap length is the damage variable.
2. **Jitter alone is cheap** (−0.07 median): the tracker tolerates uneven
   spacing when gaps stay short — consistent with cadence.md's
   "the KF is already dt-parameterised" analysis.
3. **The decimator policy question (cadence.md P0-5) is a quality wash**:
   `Pfpt` (today's first-past-threshold — rate error, clean cadence) is
   statistically indistinguishable from the nearest-to-grid gear cadence
   on 24→5 fps clips (median 0.000). P0-5 still matters for rate
   accounting, not for tracking quality at this operating point.
4. **Phase-2 priority ordering follows directly**: the gap-regime changes
   (P0-2 deletion OR→AND; P1-6 dt-adaptive association + OF/CMC gap
   gates) target where the loss lives; estimator-driven constants
   (P0-3/P0-4) are secondary (G2's moderate −0.22).

## Phase 2 (to be appended)

Config-gated implementations on ubon_cstuff branch `cadence`; same eval,
same pairing, one gate at a time then best-combo. Success bar (from the
plan): recover ≥ half of the B50/B17 deficit with no U regression.

## Phase 2 — P0-mechanics gates: measured null result (2026-07-28)

Implementation: ubon_cstuff branch `cadence` (config-gated, defaults-off
byte-identical — golden digest unchanged). Same 167 clips, one gate per
run + combo, absolute per-variant group scores:

| run | U | J50 | G2 | B50 | B17 |
|---|---|---|---|---|---|
| baseline | 0.334 | 0.267 | 0.193 | 0.135 | 0.051 |
| `cadence_deletion_and` | 0.333 | 0.267 | 0.196 | 0.136 | 0.049 |
| `cadence_vbox_dt` | 0.333 | 0.264 | 0.192 | 0.133 | 0.053 |
| `cadence_of_max_gap_s=0.5` | 0.334 | 0.267 | 0.193 | 0.135 | 0.051 |
| `kf_fps_scale_auto` | 0.333 | 0.267 | 0.193 | 0.133 | 0.048 |
| all four | 0.332 | 0.265 | 0.192 | 0.133 | 0.049 |

**Every difference is within noise (±0.003).** Mechanistic reading — the
result is informative, not merely disappointing:

- **Deletion is not the loss mechanism at sub-buffer gaps.** During a
  0.87 s gap no frames arrive, so no misses accrue (misses are
  per-processed-frame); bursts contribute ≤5 < the count gate of 10, and
  0.87 s < the 2.2 s time gate. Tracks already survive B17's gaps under
  OR. The AND-gate's correctness value (cadence.md §2) is real but only
  observable at gaps crossing `track_buffer_seconds` — needs the `B17x3`
  variant (≈2.6 s gaps) proposed in the plan's open questions.
- **vbox dt-scaling is mildly negative** (B50 median −0.229→−0.264
  paired): the vbox already unions the KF prediction, which widens with
  dt; blind symmetric expansion mostly admits false candidates.
- **OF gap-gating is neutral**: silently-failed flow at large dt already
  behaves like the unmoved-box fallback it formalizes.
- **Live fps-scale is neutral** at these patterns.

Conclusion: the burst-regime damage lives in **association failing
across the gap** (world moves ~0.9 s between looks; last-box IoU breaks;
identities fragment) — the one P1-6 lever NOT implemented this round:
the dt-conditional cost blend (d²/Mahalanobis-dominant + appearance
weighted up at dt >> nominal, discounting last-box IoU). That, plus the
long-gap `B17x3` variant to give the deletion AND-gate its real test,
is round 2. The four gates remain merged on the branch as
infrastructure: proven-safe defaults, the cadence estimator, and the
config surface round 2 needs.

## Round 2 — cost blend, low-res NVOF, long-gap AND retest: all null (2026-07-28)

New variant `B17x3` (15-frame burst / 3 s cycle / ~2.5 s gaps — crosses
`track_buffer_seconds` 2.2 s): baseline group score **−0.218** — fitness
goes NEGATIVE once tracks can die mid-gap; the cadence dose-response now
spans 0.334 → −0.218 at identical 5 fps average.

| run | U | B50 | B17 | B17x3 |
|---|---|---|---|---|
| baseline | 0.334 | 0.135 | 0.051 | −0.218 |
| cost blend (d²-dominant at large dt) | 0.332 | 0.131 | 0.051 | −0.221 |
| NVOF half-res on gap pairs | 0.334 | 0.135 | 0.051 | −0.218 |
| deletion AND (binds here) | 0.333 | 0.136 | 0.049 | −0.216 |
| all three | 0.330 | 0.132 | 0.050 | −0.221 |

Six interventions across two rounds, all within noise.

**RETRACTED (2026-07-28, MB review): an earlier revision of this section
asserted a vbox-candidate-culling mechanism for the nulls. That was
conjecture fitted to the numbers — no match-level evidence had been
examined. Mechanism claims below this point must cite the match-level
diagnostic (next section) or say "unknown".**

Scope correction (MB): production only needs gaps ≤ ~0.5 s to work for
moving scenes — B50 (0.6 s max gap, −0.23 median deficit) is the
operating regime that matters; B17/B17x3 are instrument extremes.
Practical footnotes: the fast iteration profile must keep the PROVEN
worker count (12-worker attempt OOM'd: each eval worker loads ~3 GB of
engines; auto=4 exists for a reason); duration-capped rows (60 s) are
statistically sufficient (20+ gap cycles/clip) and ~3× cheaper.

## Match-level gap anatomy (eval-path verified, 2026-07-28)

Two light B50 clips (uid_vid_00231, video201), C-tracker at epsilon
interval (the eval substrate; an earlier python-path version re-decimated
bursts via its 0.199 gate — coarse ratios matched, fine cases did not):

| | B50 gaps (0.6 s) | U steps (0.2 s) |
|---|---|---|
| identity kept | 199 (67 %) | 1508 (90 %) |
| switched | 17 (6 %) | 29 (2 %) |
| missing at re-entry | 81 (27 %) | 147 (9 %) |
| …recovered ≤3 frames | 54/81 | 70/147 |
| GT displacement (broken) | median 0.68 w, p90 1.15 | 0.31 w |
| GT displacement (kept) | 0.21 w | 0.11 w |

Evidence-backed characterisation: the dominant B50 cost is 1–3 frame
re-attach latency at gap re-entry for targets that moved ~0.7 box-widths;
targets are visibly present (frame inspection) and detected within a few
frames. Quick-probe levers eliminated by measurement: immediate-confirm
(0.6: WORSENS both groups — FP-track cost exceeds blackout savings),
match_thr_initial 0.7 (neutral), plus the six round-1/2 gates.
Open observable: where `kf_predicted_box` actually pointed at re-entry —
requires a one-line debug export on the branch; without it, prediction-
error vs gating-error cannot be distinguished from outputs.

Candidate direction (MB, recorded not run): search at fixed low rate
(e.g. 2 fps max) to test whether a materially different parameter
optimum exists per rate; if so, interval-conditioned parameter
switching/interpolation via the existing (hint:) machinery + the
branch's live cadence estimator.

## Root cause found: prediction, not gating (debug-box export, 2026-07-28)

Instrument: branch gate `cadence_debug_boxes` — every emitted det carries
the kf/of predicted boxes the associator scored; alive-but-unmatched
tracks are additionally emitted as coasted entries. Rerun of the two B50
diagnostic clips under eval conditions:

| gap-crossing outcome | kf_pred IoU vs GT | kf center err | of_pred IoU |
|---|---|---|---|
| kept (n=196) | 0.66 | 0.19 w | 0.73 |
| broken, coasted present (n=27) | 0.19 (p10 0.00) | **0.71 w ≈ full GT displacement** | **0.15** |
| broken, absent from output (n=58) | — (track dead/demoted by t1; age median 1.3 s, NOT young) | | |

Both predictors point at the OLD location for broken cases (kf_pred vs
own t0 box IoU 0.44–0.64 in all groups — near-zero effective velocity
extrapolation across the gap). Association at re-entry then fails on
geometry; a new track claims the det (miss@reentry anatomy: 61/81 no
output at t1, 18/81 stolen/below-thresh, only 2/81 old IDs ever
re-attach) and the old track starves.

**Why OF degenerates (code-level, verified by the NVOF investigation):**
NOT a search-range problem. The flow field is backward vectors indexed
on the current frame (correct), but `motion_track_predict_box_inplace`
samples it AT THE OLD BOX LOCATION (5-point stencil inside the old box,
motion_track.c:918-924). Beyond ~0.5 box-widths of subject motion every
sample lands on vacated background → sampled flow ≈ camera motion ≈ 0 →
predicted box = unmoved box. At walking pace (~1.13 w/s measured) the
budget dies at ~0.45 s — exactly bracketing the measured cliff (U/J50
fine, ≥0.5 s gaps broken), and resolution-independent — which is why the
half-res tier gate measured null. `of_weight` is hard-coded 1.0
(utrack_cost.c:116) so the degenerate of-score fully participates.

**Fixes in flight (branch, config-gated, being implemented):**
1. `cadence_of_forward_map` — inverse-lookup consumer: vote over flow
   cells whose vectors point back INTO the old box; new box = old box +
   robust (median) displacement of voters. Testable directly on the B50
   set.
2. MOTION frame class (MB design): `min_time_delta_motion` — frames
   between analytics ticks run NVOF only and carry each alive track's
   of anchor through chained short hops (each hop within the stencil's
   valid regime), no detections/association/misses. NOT testable on the
   B50 files (gaps contain no container frames — cadence is baked into
   the file); test instrument = full-rate tier-1 sources with
   min_time_delta_process 0.6 vs same + motion 0.2.

## KF velocity lag measured (2026-07-28, debug-box data)

Velocity-lead ratio (projection of kf_pred's lead onto the true
inter-frame motion; 1.0 = perfect constant-velocity extrapolation,
0 = static prediction), STEADY linear movers only (consecutive steps
with direction cosine > 0.7, displacement > 0.08 w):

| dt | n | median lead | p25 |
|---|---|---|---|
| 0.1 s | 339 | 0.58 | 0.08 |
| 0.2 s | 631 | 0.71 | 0.29 |
| 0.6 s | 126 | 0.62 | 0.26 |

The KF systematically extrapolates only ~2/3 of true motion even on
steady movers (small attenuation bias from detection noise inflates the
deficit at 0.1 s; at 0.6 s the noise is relatively small so ~35–40 %
under-lead is genuine). Not a warm-up artifact (older tracks are no
better). At 0.2 s the resulting error (~0.1 w) is invisible to IoU;
at 0.6 s it alone accounts for ~0.25 w of the 0.68 w median miss.
Mechanism candidate: std_weight_vel = 1/160 per internal frame
(kalman_tracker.h:29, ByteTrack heritage) — heavy velocity smoothing.
Third candidate gate queued: `kf_std_vel_scale` (scale velocity process
noise; default 1.0 byte-identical), to be A/B'd alongside the OF fixes.

## Round 3 — targeted fixes measured (2026-07-28 evening)

Branch commits d80c0da (`cadence_of_forward_map`), 0b81226 (MOTION frame
class), bd3c006 (`kf_std_vel_scale`). All default-off, golden-digest
clean. Fast profile (60 s caps, 4 workers); scores below are group mean
per-clip fitness from the eval logs (same parser both sides, sentinels
excluded) — comparable within this section only.

**Forward-map OF consumer, baked-gap B50/B17 files (single-shot
long-baseline flow): no recovery.**
baseline U +0.326 / B50 −0.011 / B17 −0.227 / B17x3 −0.645;
fwdmap U +0.338 / B50 −0.036 / B17 −0.249 / B17x3 −0.772.
The unit tests prove the consumer recovers 1.5-w displacements on clean
fields, and the live probe shows of_pred travelling further at re-entry
— but on real gap pairs the improvement doesn't materialise in scores.
CONFIRMED (debug-box rerun, forward-map on): broken-case of_pred IoU at
re-entry 0.18 vs 0.15 off — the single-shot 0.6 s flow field itself does
not reliably contain the person's motion (small target, appearance
change, no temporal-hint history on the pair), so no consumer can
extract it. The stencil bug is real but not the binding constraint on
long baselines; chained short hops (MOTION frames) are the fix that
works because they keep every flow pair inside NVOF's good regime.

**MOTION frames (the MB proposal): FIRST MEASURED WIN of the campaign.**
Sparse instrument = U-variant files (real 0.2 s container frames),
analytics gated to 0.6 s, so gaps contain genuine frames as in
production; `min_time_delta_motion: 0.2` turns the in-between frames
into OF-only hops that carry each track's of-anchor (no detections, no
association).

| run | group mean | paired vs sparse baseline |
|---|---|---|
| sparse baseline (0.6 s analytics) | +0.219 | — |
| + MOTION 0.2 s | **+0.263** | median +0.025, mean +0.043, 19/27 clips improved |
| + MOTION 0.2 s + forward-map | ≈ +0.263 | ≈ motion alone |

(Corrected 2026-07-28 late: the first parse dropped 7 clips on
carriage-return handling; full set is 27 paired clips + 2
sentinel-excluded of 30 rows.)

Winner clips are exactly the motion-heavy ones (bwc +0.115/+0.123, otw
homes00334 +0.170, movies +0.025..+0.086, PP2299999 +0.141); static MEVA
unchanged; worst regression bwcvideo182 −0.052. Forward-map adds nothing
on top — chained short hops keep every step inside the stencil-valid
regime, as designed. Cost: NVOF-only hops (hardware engine + frame
scale), no detector/tracker work.

Next measurements in flight: `kf_std_vel_scale` 4/8 on the baked-gap
B50 files (the only lever when gaps contain no frames), same on the
sparse instrument, and MOTION at 0.1 s hops.

## Round 3b — kf_std_vel_scale ruled out; MOTION hop-rate dose-response

`kf_std_vel_scale` (velocity process noise ×4 / ×8, the "counter the
measured KF under-lead" lever): **dose-dependent WORSENING** on the
baked-gap files (B50 −0.011 → −0.034 → −0.049; B17/B17x3 similar;
J50 mildly positive) and neutral on the sparse instrument (+0.215 vs
+0.211). Velocity-noise amplification during dense bursts costs more
than the improved lead pays. The measured ~2/3 under-lead stands as a
finding, but raising Q_vel globally is the wrong fix. RULED OUT.

MOTION hop rate on the sparse instrument (0.6 s analytics):

| hops | group mean | paired vs baseline |
|---|---|---|
| none | +0.219 | — |
| 0.2 s | +0.263 | +0.043 mean, 19/27 |
| 0.1 s | +0.263 | **+0.044 mean, 24/27, median +0.038** |

Monotone in hop rate; biggest gains otw homes00334 +0.163, bwc
video201 +0.137, PP2299999 +0.128; only consistent regression is
bwcvideo182 (−0.041..−0.052 at both rates — worth a later look).

**Recommendation to MB**: MOTION frames (0b81226) are the real
candidate — a measured +0.04 fitness on 0.6 s-gap conditions at
NVOF-only cost, monotone with hop rate, mechanism-verified end to end
(measured failure → targeted fix → measured recovery concentrated on
motion-heavy clips). Open items: bwcvideo182 regression; production
operating-point A/B (0.18 s analytics + PM-shed/dropped-frame
scenarios); whether the v2 KF pseudo-measurement is worth trying on
top; merge decision for the `cadence` branch.

## NVOF direction fix on main (2026-07-28 evening): measured, committed

MB identified the OF consumer bug as a frame-order/indexing mismatch and
ruled: fix it on main with a single flipped execute (no dual field, no
second NVOF run). Landed as ubon_cstuff main `77f670b`: upload the new
frame as NVOF reference (execute = input:PREV / reference:CUR → field
prev-indexed, prev→cur), predictors `pos+d`, CMC fit negation bridge
(fit is odd in its flow input — exact), scene stats untouched
(cost-residual only), Apple backend mirrored, Jetson TODO (VPI docs
silent on convention — verify on-device), sign tests updated, 666/666,
golden digest legitimately unchanged (golden path runs mt=null).

Full 504-row search set at production cadence, baseline-tuned params:

| binding | fitness | idf1 | MOTA |
|---|---|---|---|
| pre-fix baseline | 0.2132 | 0.5134 | 0.3281 |
| fixed main | 0.2050 | **0.5169** (235 better / 199 worse) | 0.3211 |

Isolation probes (all measured, 504 rows each):
- CMC exonerated: with `kf_cmc_enabled: false` on both bindings the same
  signature persists (idf1 +0.0015, fitness −0.005) — neither the
  negation bridge nor field-quality-through-CMC drives the dip.
- Family attribution: fitness dip concentrates in uvg (−0.022, n=173)
  and bwc (−0.018, n=48) — in BOTH, idf1 improves and (uvg) switches
  drop; the cost is `fp_tracks` (+0.07/clip uvg, +0.15/clip bwc). An
  earlier read that MEVA drove the losses was wrong at family level
  (MEVA fitness delta +0.0002) — individual movers misled.

Mechanism (evidence-consistent): the fixed predictor follows real motion,
so marginal detections chain into confirmed tracks more readily → more
FP tracks under thresholds (new_track_thr, confirm) that were auto-tuned
against the old always-lags predictor. Identity continuity improves
broadly; FP-track cost is a tuning artifact, per MB's "parameters are
auto-tuned for the current broken setup".

DECISION PENDING (MB): re-tune on the fixed binding — full search vs
reduced sweep over threshold/OF-coupled params — to price the fix's real
ceiling and restore the fitness headroom.

## NVOF convention tests + two more production bugs (2026-07-29)

Per MB: proper tests, running on the target devices. Added
`test_nvof_convention.cpp` (ubon_cstuff 3c54677): three end-to-end tests
with real frames / known motion / the REAL backend — (1) the nvof.h
field convention itself, (2) predictor moves a box WITH the target,
(3) CMC compensation follows a global pan. Verified 669/669 on desktop
CUDA and 3/3 on the Jetson VPI backend (branch `nvof-verify` pushed to
the Jetson clone; its checkout restored to 8c05deb).

Findings the tests produced on day one:
1. **Jetson/desktop convention divergence (test 1, run on-device):** VPI
   natively emits prev-indexed prev→cur vectors — the OPPOSITE of the
   pre-fix CUDA backend. Jetson deployments have therefore been running
   an OF predictor that moved boxes AGAINST motion and a sign-inverted
   CMC the whole time. The direction fix (77f670b) harmonizes both
   platforms; nvof_jetson needs no code change (verified on-device,
   3e0a522). VPI quirk: first executed pair returns a zero field.
2. **CMC dead on moving frames in production (test 3):**
   `CMC_MAX_FLOW_CELLS` was 6400 (the OLD 320px motiontrack default's
   80×80 grid); uc_v11 sets motiontrack 512px → 9216–16384 cells → every
   moving-camera fit path silently returned identity. Only the uncapped
   static fast-out ever fired — CMC worked EXCEPT when the camera moved.
   Fixed: cap raised to 128×128 (846a28f). NOT yet in the installed
   binding (the overnight search is running on the 21:07 build); install
   + A/B after it completes.
3. Test-authoring facts recorded: motion_track's predictors are gated on
   the set_roi driver call (mirrors track_stream); an iid-noise pan is
   learned as sensor noise by the adaptive floor (use structured
   texture); scene detection must be off for synthetic pans.

## CMC cap fix A/B (2026-07-29 morning, current uc_v11 params)

Search stopped per MB (train-only gains). 504-row set, three bindings:

| binding | fitness | idf1 | MOTA |
|---|---|---|---|
| (a) pre-fix baseline | 0.2132 | 0.5134 | 0.3281 |
| (d) + direction flip | 0.2050 | 0.5169 | 0.3211 |
| (e) + CMC cap fix | **0.2133** | **0.5178** | **0.3298** |

The cap fix recovers the flip's entire fitness cost (+0.0084) with MOTA
+0.0087, and the gains sit exactly where the mechanism says: bwc +0.027
fitness / +0.028 MOTA, uvg +0.018, UKof +0.011; static MEVA flat. One
negative family: bdd dashcam −0.023 (n=25) — forward-driving flow is
scale-dominated (p), worth a look at similarity-term behaviour later.
Big single-clip wins: otw_homes_00347 idf1 0.52→0.85 (switches →0),
video_0246 +0.26.

Net of the whole NVOF campaign at FIXED current params (e vs a):
fitness parity (+0.0002), idf1 +0.0044 (228/205 clips), MOTA +0.0017 —
plus the correctness wins that don't show on this set: Jetson no longer
motion-inverted, CMC alive on moving cameras, contract enforced by
on-device tests. Re-tune remains the upside lever.

## The "proper look" at CMC scale (2026-07-29): machinery exact, model gated

MB challenged the translation-only idea as a bodge and suspected another
underlying bug. Verdict from ground truth:

1. **The similarity machinery is EXACT** — new zoom ground-truth test
   (first end-to-end exercise of the p/q path ever): on a real 2% zoom,
   flow slope +0.0200 vs true +0.0196 (corr 0.999), fitted p −0.02002 vs
   −0.01961, tx/ty on the money. The NVOF direction change did NOT break
   scale (and pre-change, the cap meant this path never ran at all).
2. **A first zoom-test failure was a stimulus artifact worth knowing**:
   periodic texture (48px grating) under scale change period-locks the
   block matcher — a 2% zoom read as a 26% CONTRACTION. Translation is
   immune. Real-world analogue: fences/brick under zoom.
3. **The real-clip collapses are model-vs-scene**: on dashcam forward
   motion (jaad video_0210) the fit yields |p|≈0.12/frame with 65–78%
   residual — depth parallax + traffic is not a global similarity, and
   applying the fiction at full strength scaled/dragged every KF box.
4. **Fix: fit-validity gate** (0ff2b91) — minimum inlier fraction 0.5
   (residual < max(1px, 0.3·rms_field)); inlier-fraction (not rms) so
   foreground movers can't veto CMC on pans. Measured discrimination:
   dashcam 0% applied / bodycam pan 94% / close-quarters 40% /
   handheld 13%. 671/671 tests.

504-clip results (current uc_v11 params):

| | fitness | idf1 | MOTA | val-weighted (search obj, person) |
|---|---|---|---|---|
| (a) original baseline | 0.2132 | 0.5134 | 0.3281 | 0.2978 |
| (e) CMC live, ungated | 0.2133 | 0.5178 | 0.3298 | 0.2869 |
| (f) + validity gate | **0.2168** | 0.5136 | **0.3310** | **0.2985** |

Collapsed clips restored (video_0210 −1.03 → +0.17 ≈ baseline). Open
cost: switches/obj +0.03 vs baseline (gate flapping mid-sequence on
mixed clips like bwc201 at 40%, and association thresholds tuned for
the never-fires CMC era) — candidate follow-ups: gate hysteresis/EMA,
damped application instead of hard zero, and the re-tune.
