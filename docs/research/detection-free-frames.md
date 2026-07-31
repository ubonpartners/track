# Detection-free frames (v2)

Status: design investigation, 2026-07-29 (MB request). Scope: the whole skip
chain — decoder cadence, PM controller, `track_stream.c` input path,
`motion_track.c`, `utrack` — and what a frame class that runs *everything
except the detector* would buy us.

Line references are `ubon_cstuff` @ `0cca0a9`. Claims are marked **[measured]**
(a number we have taken), **[code]** (read off the source), or **[conjecture]**
(reasoned, untested). Nothing here is a result.

---

## 0. TL;DR

1. **The three questions in the brief are three operating points on one curve**:
   tracking quality vs *detector invocations per second*. We do not plot that
   curve at all today — the eval reports quality only, and the load counters
   live in a different subsystem. **Step zero is putting cost on the eval axis**
   (§8.1); the numerator already exists (`nonskipped_input_image_count`,
   `force_skip_count`).
2. **Goal (a) splits into two very different cases**, and conflating them is the
   main trap:
   - **motion-skip frames**: by the time the gate fires, decode, two image
     scales, MAD, the mask kernel, a blocking sync and often NVOF are *already
     paid* (`track_stream.c:1092`), and the tracker is never touched. This
     looked like a free win. **It is not — measured neutral (E6).** A
     motion-skip fires precisely when nothing is moving, so there is nothing
     to carry. Implemented anyway as `predict_on_motion_skip` (default off).
   - **load-shed frames**: the frame **never reaches `track_stream` at all**
     (§5.2). There is no image, no result, no `result_type`. Running anything
     on a shed frame means *giving back* most of what shedding saved. This is
     a new rung on the PM ladder, not a free win.
3. **Goal (b) fits the existing architecture exactly**: `preview_framerate.md`
   already establishes "one decode grid, N consumers each decimating it"; a
   motion-only rate is a third consumer. The `cadence` branch prototype
   (`0b81226`) proved the mechanism and **[measured]** +0.044 mean fitness at
   0.6 s analytics with 0.1 s hops (24/27 clips improved). Whether it pays at
   the **production 0.18 s cadence is untested** and is the main open question.
4. **Goal (c) has a signal already computed and thrown away.** With production
   thresholds, an ROI in the band (0.01, 0.05) makes NVOF run
   (`motion_track.c:593`) while the detector is skipped — on exactly the
   marginal frames MB proposes to adjudicate with flow, **we already compute the
   flow field and discard it**. Using it is a gate change, not new compute.
5. **Two probable bugs found on the way**: the motion-skip hysteresis is
   inverted relative to its own comment and to what hysteresis is for (§3.4),
   and the decoder's `force_skip` flag can leak onto the wrong picture when the
   carrying AU is a dropped non-reference frame (§5.4).
6. **The hard constraint on any design**: the tracker's *filter* is
   time-correct but its *lifecycle is frame-counted* (§4.2). Extra frames are
   safe if and only if they never reach `utrack_run`.

---

## 1. Why this is three questions, not one

| Goal | Operating point | What "better" means |
|---|---|---|
| (a) frames dropped anyway | detector budget fixed *below* demand | recover quality lost to shedding |
| (b) 2× frame rate, same budget | detector budget fixed *at* today's | buy quality with cheap frames |
| (c) idle CCTV | quality already at ceiling | spend less, lose nothing |

One mechanism (a frame processed without the detector), one metric (quality vs
detections/second), three policies. The policy differs per goal; the frame
class does not.

---

## 2. The pipeline as it is today — the frame's journey

```
RTP/NAL → storage_add_video (ALWAYS — recording never sheds)
        → [S3] R1 decode shed .................... AU freed, NVDEC never runs
        → [S5] PM force_skip decision (per AU)
        → NVDEC decode                           (always, if not S3)
        → HandlePictureDisplay
             ├ [S5] force_skip ..................  return BEFORE map/convert
             ├ [base cadence] delta < base_interval  same return, same counter
             └ map + NV12 convert + scale + rotate
        → MAIN_PIPELINE job (work queue paused: ≤1 frame in flight)
        → thread_stream_run_input_image_job
             ├ [analytics decimator] .............  SKIP_FRAMERATE (preview tap only)
             ├ scale to analytics format (reads the UNCAPPED 1280 source)
             ├ motion_track_add_frame  (blur, MAD, mask, sync, NVOF?, histogram)
             ├ scene change → resets
             ├ [no-motion gate] ..................  SKIP_NO_MOTION (set_roi(ROI_ZERO))
             ├ utrack_predict_positions  (KF + CMC + OF warp)   ← NOT reached on any skip
             ├ roi_scan HQ tile union
             └ detector submit (PM tier, per-stream cadence)
```

### 2.1 One decode grid, several decimators

The decoder is constrained to a **base cadence** = min(analytics, preview)
(`track_stream.c:183-193`). Frames failing it are dropped in the CUVID display
callback (`simple_decoder_cuda.c:340-362`) — **after** NVDEC decoded them
(reference integrity) but **before** map/convert/scale. So the marginal cost of
delivering one more frame to the pipeline is map + convert + scale, **not**
decode. This is the structural fact behind goal (b): the base grid already
exists as a concept, and a motion rate joins it as a third term (which is
exactly what the `cadence` prototype does: `base = min(analytics, preview,
motion)`).

**Corollary worth knowing before designing anything**: with no preview
configured, `base_interval == min_time_delta_process`, so the *decoder* already
decimates to the analytics rate and `SKIP_FRAMERATE` **essentially never fires
in production**. It exists for preview-only frames. Adding a motion rate
therefore does not "use frames we already have" — it widens the decode-output
grid, and the extra frames cost decoder map + colour-convert + scale.
`preview_framerate.md:185-188` quantifies the same trade for preview:
"3× the decoder map/colour-convert and 3× the scale; inference — which
dominates — is unchanged … paid in memcpy/resize, not in NVDEC or TensorRT."

### 2.2 Four skip classes, four different amounts of preserved state

| | trigger | image exists downstream? | motion tracker | `mt->ref` | utrack | result emitted |
|---|---|---|---|---|---|---|
| **S3** decode shed | age > 5 s or qlen > 40 | **no — AU freed** | ✗ | ✗ | ✗ | **none** |
| **S5** PM frame skip | `skip_frac = pm_continuous − 3` | **no — returns pre-map** | ✗ | ✗ | ✗ | **none** |
| `SKIP_FRAMERATE` | time gate / test mask | yes (discarded) | ✗ | ✗ | ✗ | yes, `track_dets=0` |
| `SKIP_NO_MOTION` | ROI area < thr | yes | **✅ ran** | **held** | **✗** | yes, `track_dets=0` |
| processed | — | yes | ✅ | advanced over the *actual* inference ROI | ✅ | yes |

The fourth row is the free lunch for goal (a): everything expensive has already
run, and then the frame returns at `:1115` without ever calling
`utrack_predict_positions` (`:1141`). Tracks coast blind through motion-skip
runs, which on a quiet scene last up to `min_time_delta_full_roi` = 1 s.

### 2.3 The reference-frame policy — the good idea already in the code

`motion_track_set_roi` (`motion_track.c:649-691`):

| ROI passed | effect on `mt->ref` | when |
|---|---|---|
| area > 0.99 | `ref ← in_img` wholesale | full-frame inference |
| **area == 0** | held frame destroyed, **`ref` frozen** | motion-skip (`track_stream.c:1096`) |
| 0 < area ≤ 0.99 | `image_blend`: old ref, **only the inspected rectangle refreshed** | normal partial-ROI inference |

Consequences worth preserving:
- **Differences accumulate across skips** — MAD compares against the last frame
  *delivered to the detector*, so a subject moving 1 px/frame (invisible
  frame-to-frame) eventually breaks the gate.
- **The reference is a spatial mosaic of differently-aged content** — regions
  the detector has not inspected recently compare against older content and are
  biased toward re-triggering. Free coverage insurance.

The reference advances over `ts->inference_roi` — what the infer thread
*actually cropped* (`track_stream.c:758`), not what we asked for. **"Don't
advance where we didn't look" is the invariant any v2 must keep.**

### 2.4 Four different "previous frame" clocks

1. `mt->ref` — MAD baseline, ROI-mosaic, advances only where the detector looked.
2. **NVOF's internal previous frame** — advances only when `nvof_execute` runs.
   `nvof_set_no_motion` (`nvof_cuda.c:301-306`) zeroes the output buffers and
   returns *without executing or swapping*, so the flow baseline also holds
   across no-motion frames — same property as `mt->ref`, entirely different
   mechanism.
3. `ts->last_run_time` — advanced on analytics **and** motion-skip frames, not
   on framerate-skips (`:975-977`, called out as "the one correctness trap" in
   `preview_framerate.md:153`).
4. Per-track `kf->last_update_time_` — advanced by every `predict()`.

### 2.5 What already exists — do not reinvent

- **HQ scan-tile** (`:1152-1210`): 3×3 age-driven tiles, **ON by default**
  (`rs["cols"]` defaults to 3 and `uc_v11.yaml` has no `roi_scan:` section),
  one tile/frame unioned into the ROI *only if the union stays under
  `hires_max` 0.5* so the detector never downscales. Deliberately load-neutral.
- **PM cadence** (`:1224-1231`): per-stream pattern of detector performance
  modes; a bench API today (`track_stream_set_pm_cadence`), off in production.
  **The natural home for a "detection-free" tier** — already a per-frame,
  per-stream schedule of detector cost.
- `prevent_consecutive_skip`, `force_full_roi`, `cmc_min_motion_area`.
- **`debug_analytics_mask`** (`:971-973`, added this week) — imposes an
  arbitrary duty cycle on unmodified video; the experiment instrument for all
  of this.

---

## 3. The no-motion gate in practice

```c
skip_roi_thr = ts->last_skip ? min_roi_after_skip     /* 0.01 */
                             : min_roi_after_nonskip; /* 0.05 */   // :1059
if (roi_area(&motion_roi) < skip_roi_thr && !force_full_roi && !prevent_consecutive_skip)  // :1092
```

### 3.1 What the ROI is

A **single axis-aligned bounding box** over 8×8-pixel blocks whose MAD exceeded
an adaptive per-block noise floor (`motion_track.c:555-589`; floor updated every
analytics frame, fast-fall α=0.9 / slow-rise β=0.9951). Two movers in opposite
corners ⇒ near-full-frame ROI, and the gate saves nothing. `motion_score` is the
post-erode *fraction* of blocks over floor.

### 3.2 What a smaller ROI actually buys

**[code]** The ROI genuinely crops and the network input is **dynamically shaped
per batch**, not a fixed letterbox (`infer.c:1443-1467`). At 1280×720, PM0:
full ROI → 640×352 (225 kpx); a 10 %-area ROI → 416×224 (93 kpx) — ~2.4× fewer
pixels **and** no 2× downscale, i.e. twice the pixels-on-target for small
subjects. That cliff is what `roi_scan_hires_max = 0.5` encodes. Caveats: the
saving saturates below ~0.5×0.5; **batching takes the max over the batch**
(`infer.c:1445`); PM tiers cap it further at `{0,512,416,320}` (`infer.c:1456`).

### 3.3 The three bypasses

`force_full_roi` (no non-skip run for 1 s), `cadence_lock_on` (lowest-PM slot
always runs), `prevent_consecutive_skip` (never two consecutive skips while
confirmed tracks exist). Today's idle floor is therefore roughly **1
detection/s from the timeout**, plus whatever the noise band forces.

### 3.4 ~~Probable bug: the hysteresis is inverted~~ — **RETRACTED, refuted by experiment 2026-07-29**

**The claim below was wrong.** See §11.E1: swapping the thresholds into
textbook Schmitt order makes load **worse on every clip tested** (mean 4.07 →
4.22 det/s; the dead camera G329 went 0.93 → 2.60 det/s). The current
direction is not a broken Schmitt trigger, it is a deliberate **skip-biased**
policy — "after any run, try to skip again; only stay skipping when the scene
is genuinely dead" — and it dominates the alternative. The original reasoning
is kept below because the *mechanism* description is still accurate; only the
"this is a bug" conclusion is retracted.

The alternation it produces is real (flip rates 0.7–0.9 on quiet cameras) but
is **not** the dominant idle cost — see §11.E2 for what actually is.

#### (original claim, retracted)

Read literally: while **running**, we start skipping when area < 0.05; while
**skipping**, we keep skipping only while area < 0.01. That is backwards for a
Schmitt trigger and backwards from the yaml's own comment (`uc_v11.yaml:47-52`,
which describes `after_nonskip` as *raising* the bar to start skipping — a
higher threshold lowers it).

**[code]** On a scene whose chronic noise sits in the band (say area ≈ 0.02):
frame 1 skips (0.02 < 0.05), frame 2 runs (0.02 ≮ 0.01), frame 3 skips… The
detector runs at **half rate indefinitely instead of settling into skip**. The
effective stable-skip threshold is 0.01; the 0.05 buys exactly one skipped
frame per wake. Fixing this may be the single largest idle saving available,
and it is a two-line change plus a re-tune.

---

## 4. What a detection-free frame may legitimately do

### 4.1 `utrack_predict_positions` is already detection-free

`utrack.c:299-419` reads **no detections** — only each track's last observed box
and its KF: CMC fit once per frame (`:320-322`), then per track CMC-apply
(`:340-341`), KF predict (`:343`), reset `of_predicted_box` from the last
observation (`:354-357`), OF warp (`:365`), pose warp (`:367-380`), ROI union.
It is exactly the "carry the tracks forward without looking" primitive we want.

### 4.2 The hazard: the lifecycle is frame-counted

The filter is time-correct (KF dt, `Q ∝ dt`, `track_buffer_seconds`, eviction
age). The lifecycle is not:

| counter | site | budget | effect of interleaved frames |
|---|---|---|---|
| `num_missed` | `utrack_class.c:574` | `max_consecutive_misses` 10 | at 5 fps ≈ 2.0 s; at 10 fps ≈ 1.0 s — becomes the binding deletion gate |
| LOST demotion | `:618` | `num_missed ≥ 2` | demotes 2× sooner in wall-clock |
| `border_degenerate_run` | `:640` | 3 frames (`:631`) | sticky on unmatched tracks; burns 2× faster |
| NEW-track death | `:590` | **first miss** | a tentative track dies before it can ever confirm |

**Therefore a detection-free frame must not call `utrack_run`** — not a
preference, a correctness requirement. This is also how the codebase already
behaves: both existing skip classes call neither `predict_positions` nor `run`,
and `track_stream.c:1060` states the intent that skipped runs age tracks by
wall-clock only.

Corollary worth stating: `num_missed` means *the detector looked and found
nothing*. A frame with no detector run produces no such evidence, so
incrementing it would fabricate evidence.

### 4.3 KF split-predict: exact in the mean, conservative in the covariance

`F(dt₁)·F(dt₂) = F(dt₁+dt₂)` for the constant-velocity block ⇒ splitting one
predict into N leaves the **mean identical**. The covariance is not:
`F₂Q₁F₂ᵀ + Q₂` vs `Q₁₊₂` adds a small extra position term and pos–vel
correlation — conservative (slightly larger, better-correlated), benign for
gating, **but it moves the golden digest**, so the change must be blessed
deliberately rather than slipped in.

### 4.4 CMC: correct to apply, dangerous to double-apply

Camera motion is measurable with no detections. Applying it in smaller
increments should be *more* accurate. The hazard is structural: `apply_cmc`
mutates the KF mean in place and `get_cmc_transform` re-derives from whatever
`mt->of_results` currently holds — it neither consumes nor clears. If a
detection-free frame's flow interval and the next analytics frame's flow
interval overlap, the same physical motion is applied twice. **"One flow
interval, one CMC application" must be structural, not conventional.**

### 4.5 OF needs a persistent anchor — which v1 already built

On `main`, `of_predicted_box` is **reset from the last observation every predict
call** (`utrack.c:354-357`) and warped once; it cannot chain. The `cadence`
branch added `of_anchor_box`/`of_anchor_valid` (`0b81226`): seed from the
observation, warp per hop, consume as-is at the next analytics frame
(re-warping there would double-count). **[measured]** that prototype is the
campaign's only win: 0.219 → 0.263 group fitness at 0.6 s analytics with 0.2 s
hops; 24/27 clips improved at 0.1 s hops.

Rough edges to fix before promotion: anchor invalidation relies on the implicit
`memset` of a fresh utdet (nothing clears `of_anchor_valid` by name); the
`cadence_of_max_gap_s` guard is bypassed while an anchor is valid;
`motion_track_motion_step` overwrites the shared `mt->of_results`, safe today
only by call ordering.

### 4.6 Consumers are already safe

Every existing skip class emits `track_dets = 0`, so `track_aux_run`
(`track_aux.c:803`), events (`track_events.c:541`), storage
(`storage.cpp:1802`), serialisation (`track_serialise.c:1483`) and the python
bindings all null-check first. `ts->tracked_object_roi` is refreshed **only**
when detections exist (`track_stream.c:793`), so it correctly persists.
Predicted-but-unmatched tracks are **never** emitted (`utrack_class.c:696`) and
should stay that way. Protocol rules: **append** any new `result_type` (it goes
to the wire as a raw uint32 and into the `analytics_debug` table), and call
`end_of_main_pipeline` on every early return or the work queue stalls.

### 4.7 Summary

| state | may a detection-free frame update it? | why / hazard |
|---|---|---|
| KF predict | **yes** | dt-based; mean exact, covariance conservative (§4.3) |
| CMC | **yes** | measurable without detections; must not double-apply (§4.4) |
| OF anchor | **yes, with a persistent anchor** | `main` resets per call (§4.5) |
| pose warp | yes (read-only) | never mutates `pose_points` |
| ROI for next detection | yes | already persists correctly |
| ReID | no | embeddings come from the detector head; already inert |
| miss accrual / deletion | **no** | frame-counted; would kill NEW tracks (§4.2) |
| track birth | impossible | needs detections |
| emitting tracks | no | a predicted box reads downstream as an observation |

---

## 5. Load shedding — where the frames actually go

### 5.1 The controller

`pm_controller.c` is pure math; the loop is `track_shared.c:664-902` at 10 Hz.
The control signal is **time, not queue depth** (`:741-778`): the max of
(RT inference latency / `skip_target_sec` 0.30 s), (p95 upstream latency /
0.6×`max_analytics_latency_s` 5 s), and a reactive backstop while decode
shedding is active. Queue depth is explicitly diagnostic-only.

State is **one continuous scalar** `pm_continuous ∈ [0,4]`: `[0,3]` selects the
detector **resolution tier** (Bresenham-cadenced), `[3,4]` becomes
`skip_frac = pm_continuous − 3` — i.e. **"PM first, skip last"**
(`pm_controller.h:24-32`). The degradation ladder in force:

| rung | action | what it costs the pipeline |
|---|---|---|
| S1 | NRT/batch throttle (delayed requeue) | nothing dropped |
| S6 | detector resolution tier `{0,512,416,320}` | detections preserved, precision reduced |
| **S5** | PM frame skip (`skip_frac`) | frame decoded, then **dropped before map/convert** |
| S3 | R1 decode shed to next keyframe | AU freed, **NVDEC never runs** |
| S4 | queue OOM net (drop-oldest at 48) | emergency |

### 5.2 The finding that matters: a shed frame never reaches `track_stream`

Both PM shed points terminate below the input job. S3 frees the AU
(`track_stream_jobs.c:482`); S5 sets a flag consumed inside the CUDA display
callback, which returns **before any `image_t` is created**
(`simple_decoder_cuda.c:337, 356-363`). So `decoder_process_image` is never
called, no MAIN_PIPELINE job is queued, and **no `track_results_t` exists** —
there is no `TRACK_FRAME_SKIP_LOAD` because there is nothing to emit it from.

**Design consequence for goal (a):** running T1 on a shed frame is not
"reclaiming work already done" — it is *undoing part of the saving*. What a
shed frame costs today is: I/O + NVDEC (S5) or I/O only (S3). What T1 would add
is map + convert + scale + a second analytics scale + MAD + mask + a blocking
sync + optionally NVOF. That is real GPU work, on a box whose *detector* is
already late.

The saving grace is that the PM signal is dominated by **inference latency**,
and T1's work is on NVDEC/OFA/SM-light paths that don't queue behind TensorRT.
So the trade is plausible but **must be measured, not assumed** — and it is
exactly the kind of thing that can make the controller oscillate if the added
work feeds back into the latency it is reacting to.

### 5.3 Counters (the cost axis already exists)

`nonskipped_input_image_count`, `motion_skip_input_image_count`,
`skipped_input_image_count` (analytics skips only — **excludes load-shed**),
`force_skip_count` + gap mean/variance (S5), `h26x_dropped`/`h26x_shedding`
(S3/S4), decoder `stats_frames_output_skipped` (⚠ conflates PM skip with
base-cadence decimation — one code path, one counter).

### 5.4 Probable bug: `force_skip` can land on the wrong picture

`HandlePictureDisplay` returns for a dropped non-reference surface
(`simple_decoder_cuda.c:317-318`) **before** `force_skip` is read and cleared
(`:337-338`), so a skip decision carried by a dropped non-reference AU survives
onto the *next* displayed picture. `force_skip` is also a single bool, so two
AUs before one display callback collapse to one skip. Net effect: achieved skip
rate can be **below** `skip_frac` on B-pyramid content, with phase drift from
what the Bresenham cadence intended. No test covers it. Flagged, not confirmed.

---

## 6. What things cost

### 6.1 The ladder for one delivered frame

| stage | avoidable by | notes |
|---|---|---|
| RTP/NAL + `storage_add_video` | nothing | recording never sheds |
| NVDEC decode | S3 only | S5 pays it and throws the result away |
| map + NV12 convert + decoder scale | **S5 / base cadence** | this is what the base grid saves |
| analytics re-scale (`track_stream.c:1004`) | — | normally a **no-op** (same geometry ⇒ `image_reference`) |
| motion tracker's internal scale (`motion_track.c:418`) | analytics decimator | 1280-cap frame → 512 motion frame; convert-then-scale for device targets, so it reads the full source |
| blur + MAD + mask + erode | analytics decimator | cheap: MAD is 3 launches; the mask kernel is **one CUDA block** |
| **NVOF** | `roi_area > 0.01` | **OFA hardware engine** (not SMs) + a second scale/convert + a blocking sync |
| scene histogram | `scene_change_sensitivity: 0` | DtoH overlapped by design |
| **detector** | the no-motion gate / this proposal | the expensive one; ROI and PM tier scale it |
| tracker associate | — | CPU |

Two facts favour detection-free frames: the expensive optional stage (NVOF) runs
on the **OFA block, which does not contend with the SMs the detector needs**,
and the genuinely cheap stages are launch-latency or nanosecond CPU work.

Two facts temper it: the analytics scale reads the full 1280-cap frame
regardless of the 512 motion cap, and every analytics frame carries **two or
three hard sync points** (mask DtoH, histogram event, NVOF stream sync) — the
real limiter at 16 streams. **The motion tracker is the second-largest per-frame
GPU cost after inference, and it sits above the no-motion gate by design.**

### 6.2 What the detector actually costs — the model is already in the code

`infer_batch_pick.c:16-29`, refit 2026-05-20 from `bench/bench_trt_engine.py`
(CUDA-event timing, residual 2.4 %), on **Orin Nano**:

```
T(B, area) = K_LAUNCH + K_PER_ITEM·B + K_PER_PIXEL·B·area
K_LAUNCH   = 4.007 ms   (per BATCH)
K_PER_ITEM = 0          (fit ≈ 0, clamped)
K_PER_PIXEL= 46.4 ns/px (per image)
```

**This is the number the whole design turns on.** The marginal cost of one more
image in an existing batch is `46.4 ns × area`:

| detector input | area | marginal GPU cost of that image |
|---|---|---|
| 640×352 (PM0, full ROI) | 225 kpx | **10.5 ms** |
| 512×288 (PM1) | 147 kpx | 6.8 ms |
| 416×224 (PM2) | 93 kpx | 4.3 ms |
| 320×192 (PM3) | 61 kpx | **2.9 ms** |

Two consequences:
- **A detection-free frame saves the per-pixel term, not the launch term.** At
  the batch sizes seen under load (≈8), skipping one image saves ~10.5 ms of
  GPU, not ~14.5 ms — the 4 ms launch is still paid by the surviving batch.
  Only if the frame was the *sole* job in its batch do you save the launch too.
- **Detector work is fully serialised**: one CUDA stream inside `infer_t`,
  `infer_batch` runs under `inf->lock` (`infer.c:351,1427`), one global
  detection thread shared by every stream. So the "spare capacity" a shedding
  stream frees is **global, not returned to that stream** — the 10 Hz
  controller converts it into a lower PM tier / lower `skip_frac` for
  everybody.

Against that, T1's own cost: MAD + mask + a blocking sync, plus NVOF
**~1-2 ms on Jetson** (`bench/cmc/STATIC_TUNING.md:66-68`) on the OFA engine,
CMC fit **45 µs** dynamic / 0.45 µs static (AVX2, `e434904`), tracker predict
sub-ms. So T1 vs T3 is roughly **~1-2 ms of OFA + <0.5 ms CPU, versus 2.9-10.5 ms
of contended SM time** — the ratio that makes the idea worth testing, and the
one to confirm on the actual deployment hardware (all of the above except the
scale figures is Orin, not the RTX box).

Secondary heads matter here too: **ReID, pose and face keypoints are columns of
the detector's own output tensor** (`infer.c:610-660`), inseparable from the
call — a detection-free frame gets none of them, so continuity must come from
KF/CMC/OF alone. The heavy aux engines (face embed, CLIP, vehicle, OCR) are
separate passes that self-disable with zero detections, since every one sits
inside the per-detection loop.

### 6.3 Instrumentation: better than expected, with one hole

Already available with **no code changes**:
- `infer_thread_stats_node()` — `thread_time_breakdown` (wait/pick/prep/infer/post)
  and `infer_batch_breakdown` with **CUDA-event-measured `gpu_s`** separated from
  `enqueue_s`/`sync_s`/`post_s`, plus batch-size and latency histograms and
  `performance_mode_count[]`.
- Per-work-queue `total_time / jobs_run` — `H26X_DECODE`'s ratio *is* the
  per-frame decode cost.
- Two zero-overhead binary traces: `UBON_INFER_TRACE_PATH` (per enqueue) and
  `UBON_INFER_BATCH_TIME_TRACE_PATH` (per batch), with `bench/fit_batch_cost.py`
  to refit the `T(B,A)` constants on *our* hardware.

**The hole**: no per-stage timer inside `thread_stream_run_input_image_job` —
motion tracker, NVOF, CMC and association are lumped into `h_input_image_time`.
Also computed but unexposed: `scene_cover` (no accessor anywhere),
`scene_hist_dist`, the per-block MAD bitmap (host-side already), and the
**graded MAD field, destroyed at `motion_track.c:553` without readback**.

**Every T1-side cost figure above is a bound or an Orin number, not a
measurement on our box.** Closing that is task 2 in §9.

---

## 7. Proposal: one frame-tier decision

Today seven mechanisms decide a frame's fate (analytics decimator, ROI gate,
`force_full_roi`, `cadence_lock_on`, `prevent_consecutive_skip`, roi_scan tile
picker, PM controller) and they interact in ways nobody can hold in their head.
The v2 shape is **one policy function assigning each delivered frame a tier**:

| tier | work | detector |
|---|---|---|
| **T0 DROP** | nothing (decoder level) | — |
| **T1 MOTION** | scale + MAD + (NVOF) + `predict_positions` (KF/CMC/OF anchor). **No association, no miss accrual, no births.** | none |
| **T2 DETECT-LITE** | T1 + detector at reduced PM and/or tight ROI | reduced |
| **T3 DETECT-FULL** | T1 + detector at full PM, expanded/scan-tile ROI | full |

Policy inputs, all already computed: motion evidence (ROI area, `motion_score`,
and — new — flow coherence), track state, time since *that region* was last
inspected (roi_scan tile ages), PM pressure, cadence position.

This **subsumes** the existing gates rather than adding to them: `force_full_roi`
becomes "T3 when the oldest tile age exceeds X"; `prevent_consecutive_skip`
becomes "never two consecutive sub-T3 frames while tracks live"; the PM cadence
becomes the T2/T3 mix; and — the important one — **load shedding demotes a tier
instead of dropping a frame**, which is precisely the "PM first, skip last"
philosophy the controller already documents, extended by one rung:

> resolution tier → **detection-free** → frame skip → decode shed

---

## 8. The three goals as tier policies

### 8.1 (a) Frames dropped anyway

Two distinct changes, in order of confidence:

1. **Motion-skip frames → call `predict_positions`.** Free (§0.2), immediate,
   and independent of everything else in this document. **[conjecture]** the
   benefit is largest exactly where motion-skips run longest — quiet scenes
   with a stationary-then-moving subject, where today the KF sits frozen for up
   to 1 s and then jumps.
2. **Load-shed frames → T1 instead of T0**, as a new PM rung. Costs real work
   (§5.2), so it must be gated on the *reason* for shedding (SM/detector
   pressure yes; decode/bandwidth pressure no) and measured for controller
   stability. **[conjecture]** the shape of the benefit is the measured
   MOTION-frame result, since a shed run *is* a sparse-analytics run — but shed
   runs must be ≥0.5 s to land in the regime where **[measured]** carry is
   worth +0.044.

### 8.2 (b) Double the frame rate at constant detector budget

Add a motion rate to the base-grid plan (`base = min(analytics, preview,
motion)`) and run T1 in between. Detector invocations/second are unchanged **by
construction**. A pleasing property falls out of §5: under load, PM skip culls
the *extra* frames first, so the feature degrades gracefully to today's
behaviour.

Mechanisms that could make it pay, in my order of confidence:
1. **Better OF carry** — shorter hops keep flow inside its valid regime (the
   >0.5-box-width stencil degeneracy is measured and documented).
2. **Better CMC** — camera motion estimated at 2× rate with smaller baselines,
   fewer parallax-dominated fits. Directly relevant to the bodycam and movie
   cases in the brief.
3. **Fresher ROI** for the next detection ⇒ tighter crop ⇒ possibly
   quality-positive *and* cost-negative.
4. Faster scene-change response.

Against it, two things: at 0.18 s the tracker is already in the flat part of the
**[measured]** cadence dose-response (damage is small below ~0.3 s gaps); and
the extra frames are **not free at the decoder** (§2.1) — doubling the base
grid doubles decoder map/convert/scale plus the motion tracker's own scale and
two GPU syncs per extra frame. The detector budget is untouched, but "no added
load" is not literally true. Quantifying that per-frame T1 cost on our hardware
is exactly task 2 in §9.

### 8.3 (c) Idle CCTV: evidence-based backoff instead of a blind timeout

1. **Fix the inverted hysteresis** (§3.4) so a quiet scene settles into skipping
   instead of alternating. Probably the biggest single idle saving.
2. **Flow-corroborated gating.** In the marginal band the flow field is *already
   computed and discarded*. Promote it: MAD says "something changed", flow says
   "nothing moved coherently" ⇒ noise/lighting/compression ⇒ skip **and** count
   it as positive evidence of quiescence. Signals available with no new compute:
   per-cell flow magnitude and `costs`, `scene_cover`, the per-block MAD bitmap,
   and the graded MAD field (currently thrown away).
3. **Adaptive backoff of the forced look.** Today `force_full_roi` fires every
   1 s unconditionally — a floor of ~1 detection/s on a dead camera. With
   positive quiescence evidence and no live tracks, back off geometrically to a
   bounded ceiling (1 s → 8 s); snap back instantly on any flow coherence or
   live track. The roi_scan tile ages already provide "when did we last look
   *there*", so coverage insurance survives.
4. **Cheaper idle frames.** Since the motion tracker is the second-largest cost
   and sits above the gate, a confidently-idle scene could also drop the blur,
   skip the histogram (already possible via `scene_change_sensitivity: 0`), or
   run MAD at reduced resolution. This is the only lever that reduces the
   *floor* rather than the detector rate. **[conjecture]**, needs the §6.2
   instrumentation first.

**The risk to state plainly**: a perfectly stationary person produces neither
MAD nor flow. Backoff must be bounded and track-aware, and the tile sweep must
continue. The failure mode of getting this wrong is not "slightly worse
tracking" — it is "misses a person standing still for 8 seconds". The
experiment must therefore measure **detection latency for a stationary
subject**, not just aggregate fitness.

---

## 9. Experiment plan

Instruments in place: `debug_analytics_mask`, the cadence rows in
`track_search_v11_mc.yaml`, `min_time_delta_motion` on the `cadence` branch,
`eval_compare.py`.

1. **Put cost on the eval axis** (prerequisite): carry
   `nonskipped_input_image_count` / duration into the per-clip metrics so every
   run reports **detections per second** beside fitness. Without it we cannot
   distinguish a saving from a regression.
2. **Instrument the stages** — the only real hole (§6.3): timers around
   `motion_track_add_frame`, NVOF, and CMC/predict inside
   `thread_stream_run_input_image_job`. Combine with the *existing*
   `infer_batch_breakdown.gpu_s` and a `bench/fit_batch_cost.py` refit on the
   RTX box to get the true **T1 : T3 cost ratio** on our hardware instead of
   §6.2's Orin-derived bound.
3. **Goal (b) A/B** (cheapest real test): production 0.18 s analytics ± T1
   frames at 0.09 s, full search set. Success = fitness up at equal
   detections/s.
4. **Goal (a) simulation**: mask out 50 % of analytics frames (simulated shed),
   compare (i) nothing, (ii) `predict_positions` carry, (iii) full T1 with OF
   anchor. This is the sparse-instrument experiment at production-like rates.
5. **Goal (c) idle bench**: static-camera subset, measuring detections/s **and
   time-to-first-detection for an entering subject** under: today → hysteresis
   fixed → + flow corroboration → + backoff. The second metric is the safety
   metric and must not regress.
6. **Stationary-subject probe**: a clip where a person stops moving >10 s;
   assert the track survives and re-detection latency stays bounded.

---

## 10. Open questions for MB

1. Is the inverted hysteresis (§3.4) a bug, or deliberate for a reason not in
   the comments?
2. For (c), what is the acceptable worst-case detection latency on an idle
   camera? That number sets the backoff ceiling directly.
3. Should T1 frames emit a result at all? v1 emits `TRACK_FRAME_MOTION` with
   `track_dets = 0`; safe, but it adds wire/storage rows at the higher rate.
4. Does the `cadence` branch MOTION prototype get rebased onto current main
   (post NVOF-direction and CMC fixes) and promoted, or is this a fresh
   implementation inside the tier framework?
5. For (a): is it acceptable for the PM controller to spend *more* GPU work
   (T1) while it is shedding, given the feedback risk (§5.2)?

---

## 11. Experiment log

Running log; newest last. Instrument: `scratchpad/skipprobe.py` — runs a clip
through the real pipeline (`c_track_stream`, production `uc_v11.yaml`,
mp4-direct) and reports the per-frame decision sequence from `result_type`,
`motion_roi` and `motion_score`. Load metric = **detector invocations per
second** (`TRACKED` + `TRACKED_FULL` results ÷ clip duration).

### E1 — is the motion-skip hysteresis inverted? **NO. Claim refuted.**

18 clips (MEVA static, cevo office, MOT20, uvg, bwc), 60 s each, baseline vs
thresholds swapped into Schmitt order (`after_skip: 0.05`, `after_nonskip: 0.01`):

| | baseline | swapped | verdict |
|---|---|---|---|
| mean det/s (18 clips) | **4.07** | 4.22 | swap is worse |
| G329 (dead camera) | **0.93** | 2.60 | swap is **2.8× worse** |
| G506 (quiet) | **2.68** | 3.00 | worse |
| G341 (quiet) | **2.97** | 3.33 | worse |
| G424 / G339 | 3.22 / 3.08 | 3.32 / 3.13 | worse |
| busy clips (MOT20, uvg, bwc) | 3.7–5.6 | unchanged | no effect |

**Conclusion**: the current order is deliberate and correct for load. Entering
skip easily (area < 0.05) and leaving it easily (area < 0.01) means the system
*attempts* to skip after every run; the textbook order makes it hard to enter
skip at all, so it runs more. **§3.4 retracted.**

**What is real**: on quiet static cameras the decision flips almost every frame
(flip rate 0.72–0.91) and the detector still runs ~3 times/second on a nearly
empty scene. So the *symptom* stands; the diagnosis was wrong.

### E2 — what actually forces the runs on a quiet camera?

Sweep on three quiet MEVA cameras (60 s each), det/s:

| config | G329 (dead) | G506 | G424 | mean |
|---|---|---|---|---|
| baseline (0.01/0.05, `full_roi` 1 s) | 0.93 | 2.68 | 3.22 | 2.28 |
| thresholds wide open (0.99/0.99) | 0.85 | 2.48 | 2.48 | 1.94 |
| wide open + `full_roi` 8 s | **0.13** | 2.18 | 2.18 | 1.50 |
| baseline + `full_roi` 8 s | **0.25** | 2.55 | 3.22 | 2.01 |

Two distinct regimes, and neither is the ROI threshold:

1. **Dead camera** — the `force_full_roi` **timeout dominates**. It alone
   accounts for 0.93 → 0.25 det/s (3.7×); with thresholds also open, 7×.
   Motion thresholds are nearly irrelevant here (0.93 → 0.85) because the area
   is already ~0.
2. **Quiet but occupied** — everything bottoms out at **~2.2 det/s ≈ 44 % of
   the 5 fps rate even with skipping wide open**. That is
   `prevent_consecutive_skip` (`track_stream.c:1089`): while confirmed tracks
   exist, never skip twice in a row ⇒ a hard 50 % ceiling on skipping.

**This re-frames goal (c) and directly motivates T1.** The two levers are:
- dead scenes → back off the forced look (large, low-risk: no tracks to lose);
- occupied scenes → the 50 % ceiling exists *because a present subject must not
  go >2 frames without a detection opportunity*. **A detection-free frame is
  exactly the thing that lets you relax that rule safely**: carry the track with
  T1 instead of paying a detector run to keep it alive. This is the strongest
  evidence-backed argument for T1 found so far, and it is an argument about
  *load*, not about tracking quality.

### E3 — does backing off the forced look cost quality? **No.**

Full val split, canonical objective:

| | objective | static | moving | movie |
|---|---|---|---|---|
| baseline (`full_roi` 1 s) | 0.4255 | 0.5720 | 0.2821 | 0.3646 |
| `full_roi` 8 s | **0.4274** | 0.5708 | 0.2882 | 0.3411 |

Per-clip 85 better / 89 worse — a wash. So **8× less forced looking is
quality-neutral on this corpus while cutting dead-camera load 3.7× (E2)**.
Caveat that matters: the eval corpus is mostly *active* footage and contains
almost no idle-camera-with-stationary-subject case, which is exactly what the
timeout is insurance against. The safety probe (E7) is still owed before this
becomes a recommendation.

### E4 — what does a detector invocation actually cost? (both targets)

Method: same clip, analytics forced on every frame, detector suppressed via a
wide-open skip gate; wall-clock differenced against a decode-only run. Min of
3 reps.

| | RTX 5090 desktop | **Jetson Orin** |
|---|---|---|
| per detector invocation | 1.49 ms | **19.46 ms** |
| per delivered frame (decode) | 0.39 ms | **7.36 ms** |
| per T1 frame (marginal wall-clock) | 0.08 ms | 0.10 ms |
| detector : T1 | 18 : 1 | 189 : 1 |

Cross-check on the desktop by regressing wall-clock against detection count
over masks 1/2/3/4 (602-frame clip): **3.25 ms per analytics frame**
(detector + motion tracker together), r² visually linear — consistent with the
1.49 ms figure once the motion tracker's share is separated out.

**Caveat, stated because it changes the interpretation**: single-stream
wall-clock measures the *critical path*, not GPU time. The ~0.1 ms T1 figure
means "T1 adds almost nothing to a pipeline whose critical path is decode",
which is the operationally relevant number for one stream but **not** the
multi-stream capacity number. Work-queue attribution on Jetson confirms decode
dominates (7.3–7.7 ms/job × 1470 jobs ≈ 11 s of an 11 s floor), and that the
detector's ~19 s is **additive, not overlapped**. The Jetson build predates the
per-stage histograms, so the true GPU cost of MAD+NVOF is still unmeasured
(§6.3) — Jetson is being updated to current main to close this.

**The design consequence is large**: on the target that matters, one avoided
detection buys ~19.5 ms, and the frame it is avoided on costs ~0.1 ms of
critical path. That is the whole case for T1 in one line.

### E5 — the quality-vs-detection-rate curve (the Pareto the design must beat)

Val split, `debug_analytics_mask` decimating analytics frames, everything else
production. This is the curve every mechanism in this document is trying to
move:

| detector rate | objective | static | moving | movie |
|---|---|---|---|---|
| every frame (≈5.5/s) | 0.4249 | 0.5711 | 0.2829 | 0.3418 |
| **every 2nd (≈2.8/s)** | **0.4127** (−0.012) | 0.5649 (**−0.006**) | 0.2676 (−0.015) | 0.2734 (−0.068) |
| every 3rd (≈1.8/s) | 0.3754 (−0.050) | 0.5417 (−0.029) | 0.2173 (−0.066) | 0.2095 (−0.132) |

**Halving the detector rate costs 1.2 % of the objective; on static cameras it
costs 0.6 %.** Thirding costs 12 %. Two conclusions:

1. **Goal (c) has enormous headroom on static cameras** — the quality gradient
   with respect to detection rate is nearly flat there, and E2 showed the load
   is dominated by a *timeout*, not by evidence of motion.
2. **This sets T1's bar precisely.** At half rate the deficit to recover is
   0.012 overall / 0.006 static / 0.068 movie. T1 does not need to be
   spectacular to pay — on Jetson it would be buying that back for ~0.1 ms of
   critical path per frame against 19.5 ms per avoided detection.

The movie group is the most rate-sensitive (−0.068 at half rate) and the
moving group next, which matches the cadence campaign's finding that motion
between looks is what costs — i.e. exactly the deficit T1's OF/CMC carry
targets.

### E6 — carry tracks through motion-skip frames? **Neutral. "Free win" retracted.**

Implemented on main as `predict_on_motion_skip` (default off, byte-identical,
golden digest unchanged, 675/675): on a `SKIP_NO_MOTION` frame call
`utrack_predict_positions` (KF predict + CMC + OF warp) — never `utrack_run`.

| corpus | gate off | gate on |
|---|---|---|
| full val, detector every frame | 0.4240 | 0.4270 (+0.0030) |
| full val, detector every 2nd | 0.4130 | 0.4096 (−0.0034) |
| **quiet static subset (66 clips, where skips are frequent)** | **0.5269** | **0.5259** (−0.0010), 28 better / 29 worse |

Run-to-run noise on this harness is ±0.0015 (same config, different run:
0.4240 / 0.4249 / 0.4255), so all three deltas are noise.

**Conclusion: no measurable benefit, including where the gate fires most.**
§0.2's "free win" claim is **retracted**. The reason is obvious in hindsight
and worth stating because it re-points the whole design: **a motion-skip fires
precisely when nothing is moving**, so there is nothing for KF/CMC/OF to carry.
Coasting blind through a period of no motion costs nothing.

Where carry *should* matter is frames skipped **despite** motion — load
shedding and rate decimation — which is exactly the regime where the cadence
campaign **[measured]** +0.044. The gate is kept (default off) as it is free
and may matter on cameras where the motion gate mis-fires.

### E7 — how far can the forced look back off, on quiet cameras?

Quiet static subset (66 clips), varying `min_time_delta_full_roi`:

| | objective | idf1 | fp_tracks | per-clip |
|---|---|---|---|---|
| 1 s (today) | **0.5269** | 0.5068 | 157 | — |
| 8 s | 0.5243 (−0.0026) | 0.5092 | 152 | 23 better / 37 worse |
| 30 s | 0.5248 (−0.0021) | 0.5097 | 146 | — |

**It is a trade, not a free win** — and the cost saturates: 30 s is no worse
than 8 s, so nearly all of the (small) quality cost is paid by the first step
away from 1 s. Against E2's measured 3.7× reduction in dead-camera detections,
the exchange rate is roughly **0.5 % of quality on quiet-but-active cameras for
3.7× less load on dead ones**. On the *full* corpus the same change measured
neutral-to-positive (E3), because most clips never hit the timeout.

Whether that trade is worth taking is MB's call (open question 2) — but the
shape is now known, and the flat 8 s → 30 s segment says: if it is taken at
all, take it well past 8 s.

### E8 — the cost side of the Pareto, on the Jetson (the target that matters)

Same decimation as E5, measuring throughput instead of quality (Orin, clip
`UKof_MD_Indoor_Light_OHcam_001`, 1468 frames, min of 2 reps):

| detector rate | wall | throughput | ms/delivered frame |
|---|---|---|---|
| every frame | 29.20 s | 3.3× real-time | 19.89 |
| **every 2nd** | **16.96 s** | **5.7× real-time (+72 %)** | 11.59 |
| every 3rd | 12.77 s | 7.6× real-time (+129 %) | 8.72 |

**Put E5 and E8 together and the headline trade is measured on both axes:**

> **Halving the detector rate costs 1.2 % of tracking quality (0.6 % on static
> cameras) and buys 72 % more throughput on the Jetson.**

That is the number every proposal in this document has to be judged against —
including doing nothing but turning the analytics rate down, which is the
trivial baseline any cleverer scheme must beat.

Caveat on the absolute Jetson figures: TRT warns the engine plan was built on a
different device ("using an engine plan file across different models of devices
… likely to affect performance"), so absolute ms may be pessimistic. All
configs share the engine, so the *ratios* stand; a native-engine rebuild would
be needed before quoting absolute Orin latency anywhere else.

---

## 12. Revised plan (after E1–E8)

What the evidence changed:

| original belief | status after experiment |
|---|---|
| inverted hysteresis is a bug worth fixing | **refuted** (E1) — current order is better; symptom real, diagnosis wrong |
| carrying tracks through motion-skips is a free win | **refuted** (E6) — neutral; skips fire when nothing moves |
| idle load is dominated by the motion gate | **refuted** (E2) — dominated by the `force_full_roi` timeout (dead scenes) and the `prevent_consecutive_skip` 50 % ceiling (occupied scenes) |
| detection-free frames are attractive because the detector is expensive | **supported** (E4/E8) — 19.5 ms/detection on Jetson vs ~0.1 ms of added critical path |
| there is headroom to spend | **quantified** (E5/E8) — half rate = −1.2 % quality, +72 % throughput |

Priorities now, in evidence order:

1. **Goal (c), dead cameras — ready to decide.** Back off `force_full_roi` well
   past 8 s (the cost saturates there): 3.7× fewer detections on a dead camera
   for −0.5 % on quiet-but-active ones, neutral-to-positive corpus-wide. Needs
   only the safety probe (stationary subject) and MB's latency tolerance.
2. **Goal (b)/(a), T1 where motion exists — the open question.** E6 showed carry
   is worthless when nothing moves; the cadence campaign **[measured]** +0.044
   when things do. The decisive experiment is T1 on *rate-decimated* frames
   (half-rate substrate, deficit to recover = 0.012). MOTION-frame port to main
   in progress.
3. **The 50 % skip ceiling** (`prevent_consecutive_skip`) is the binding
   constraint on occupied-but-quiet cameras and is exactly what T1 could relax —
   but only if (2) shows carry works at production cadence.
4. **Deprioritised**: the tier framework (§7) is premature until (2) lands; the
   flow-corroborated gate (§8.3.2) targets the marginal band, which E2 showed is
   not where the load is.

### E9 — the forced look and the track buffer are a COUPLED pair (goal (c) win)

`force_full_roi` exists because "an all-skipped sequence ages tracks out via
`track_buffer_seconds` (wall-clock) and a returning subject re-enters with a
new track ID" (`track_stream.c:1060-1065`). So backing off the forced look
without extending the buffer guarantees exactly the loss E7 measured. Testing
the pair on the quiet subset (66 clips):

| `full_roi` / `track_buffer_seconds` | objective | vs baseline |
|---|---|---|
| 1 s / 2.0 s (today) | 0.5269 | — |
| 8 s / 2.0 s | 0.5243 | −0.0026 |
| 1 s / 10 s | 0.5234 | −0.0035 |
| 8 s / 10 s | 0.5216 | −0.0053 |
| **30 s / 30 s** | **0.5265** | **−0.0004 (noise)** |

Neither knob alone survives; **moved together to 30 s/30 s the quality cost
disappears**. Load on the same clips:

| config | G329 (dead) | G506 | G424 | mean |
|---|---|---|---|---|
| today (1 s / 2.0 s) | 0.93 | 2.68 | 3.22 | 2.28 |
| 8 s / 2.0 s | 0.25 | 2.55 | 3.22 | 2.01 |
| **30 s / 30 s** | **0.15** | 2.55 | 3.22 | 1.97 |

**Result: a dead camera drops from 0.93 to 0.15 detections/second — 6.2× — at
indistinguishable quality.** Occupied-but-quiet cameras barely move (2.68 →
2.55, 3.22 → 3.22) because there the binding constraint is not the timeout but
the 50 % skip ceiling (E2).

This is the first outright win in the campaign: **config-only, no code, no
quality cost, 6× less load on idle cameras.** The remaining idle lever is the
skip ceiling, which needs a knob (`max_consecutive_motion_skips`, in progress).

Caveats before shipping: (i) the quiet subset is 66 clips of *this* corpus and
contains no idle-camera-with-stationary-subject case — the safety probe is
still owed; (ii) `track_buffer_seconds` 30 s means a genuinely departed subject
keeps a live track for 30 s, which costs memory and may affect
counting/event semantics downstream — that is a product question, not a
tracking-quality one, and the eval cannot answer it.

### E10 — SAFETY: does the backoff delay detecting someone who appears?

The risk of widening the forced-look interval is that a subject appears during
a blind window. Measured directly: for every GT person's **first appearance**
on the three quiet clips, how long until the detector next actually runs?

| config | appearances | median latency | p90 | max | max gap between detector runs |
|---|---|---|---|---|---|
| today (1 s / 2.0 s) | 74 | 0.00 s | 0.20 s | 0.20 s | 1.20 s |
| **30 s / 30 s** | 74 | **0.00 s** | **0.20 s** | **0.20 s** | 30.20 s |
| 60 s / 60 s | 74 | 0.00 s | 0.20 s | 0.20 s | 4.20 s |

**Detection latency is completely unchanged** (max 0.20 s = one analytics
period) even though the maximum blind window grows to 30 s. The reason is the
mechanism working as designed: the blind window only extends while *nothing is
happening*; the instant a person appears they generate motion, the ROI gate
fires, and the detector runs on the very next analytics frame. The forced-look
timeout is insurance against motion the gate *misses*, and across 74
appearances it was never the thing that caught them.

Residual risk not covered by this probe: a subject who enters below the motion
threshold entirely (very distant, very slow, or appearing already-stationary
after a cut). That is a real but different failure mode, and the tile-scan
(`roi_scan`, on by default) is the mechanism that covers it — independent of
this timeout.

**Goal (c) conclusion — ready to ship, pending MB's call on the buffer:**
config-only, no code change:
```yaml
min_time_delta_full_roi: 30        # was 1
utrack: { track_buffer_seconds: 30 }   # was 2.0 — MUST move together (E9)
```
6.2× fewer detections on a dead camera, quality neutral-to-positive on the
full corpus (0.4255 → 0.4267), unchanged detection latency. The open question
is not tracking quality but product semantics: a 30 s track buffer keeps a
departed subject's track alive for 30 s, which affects counting/dwell/event
logic the eval cannot see.

### E11 — MOTION frames on main: the T1 hypothesis CONFIRMED

Ported to main (`4b70ec6`, `min_time_delta_motion`, default off, digest
unchanged, 688/688, flow sign falsification-tested on the real backend, hop
buffer proven isolated from the analytics signals). Val split:

| | objective | moving | static | movie | switch/obj | fp_tracks |
|---|---|---|---|---|---|---|
| full detector rate | 0.4249 | 0.2829 | 0.5711 | 0.3418 | 0.963 | 250 |
| half rate | 0.4143 | 0.2750 | 0.5607 | 0.2724 | 0.907 | 188 |
| **half rate + MOTION** | **0.4190** | **0.2828** | 0.5630 | 0.2629 | **0.809** | 170 |
| full rate + MOTION | **0.4297** | 0.2882 | 0.5752 | 0.3499 | 0.918 | 235 |

- **Half rate + MOTION recovers 44 % of the deficit overall and ~100 % on the
  moving group** (0.2828 vs 0.2829 at full rate) — for half the detector cost.
- It also *reduces* ID switches (0.907 → 0.809) and false tracks (188 → 170).
- At full rate it is still positive (+0.0048).

This is the first confirmation of the core T1 hypothesis, and it lands exactly
where E6 predicted it would after the motion-skip null: carry pays when frames
are skipped **despite** motion, not when they are skipped because of its
absence.

### E12 — is adaptive detector rate by object speed worth building? (MB's idea)

Per-clip GT speed (median box-widths/second) against the measured loss from
halving, 187 val clips:

**corr(GT speed, loss from halving) = −0.165** — weak and backwards.

| GT speed quartile | median speed | mean loss from halving | mean MOTION recovery |
|---|---|---|---|
| slowest 25 % | 0.24 w/s | −0.0134 | +0.0033 |
| 25–50 % | 0.88 w/s | +0.0136 | +0.0095 |
| 50–75 % | 1.95 w/s | +0.0019 | +0.0061 |
| **fastest 25 %** | 3.66 w/s | −0.0211 | **+0.0374** |

Two conclusions, in opposite directions:

1. **Gating the DETECTOR rate on speed has no measurable headroom here.** The
   loss from halving is not speed-predicted — the quartile means alternate sign
   and are within per-clip noise. Whatever makes a clip lose from a lower
   detector rate, it is not how fast its objects move. (Plausible reason, not
   tested: fast clips are already hard and score low either way, so there is
   little left to lose.)
2. **Speed predicts the VALUE OF T1 very strongly — an 11× gradient.** MOTION
   frames buy +0.037 on the fastest quartile and +0.003 on the slowest. So the
   adaptive lever MB is reaching for is real, but it belongs on the *motion-frame
   rate*, not the detector rate: **run hops when things move, skip them when
   they don't.** The signal is already in hand (KF track speed, or NVOF flow
   magnitude — the latter needs no tracks at all).

Whether that adaptivity is worth implementing depends on what a hop costs, which
E4 bounds at ~0.1 ms of critical path plus one NVOF execute — small. The honest
framing: adaptivity here saves a little cost on slow scenes rather than buying
quality, so it is an optimisation of an optimisation. **Always-on MOTION frames
should be measured first** (E11 says they are positive at both rates), and
speed-gating only if the hop cost turns out to matter on Jetson under
multi-stream load.

### E13 — density/crowding as the adaptive signal (MB's hypothesis) — **SUPPORTED**

Hypothesis: denser scenes need more observations to disambiguate, so the loss
from a lower detector rate should grow with density. Tested per-clip against
the measured loss from halving (187 val clips). Two measures: mean persons per
frame, and **crowding** = mean number of other persons within 2 box-widths
(the association-ambiguity proxy).

| crowding quartile | median crowding | mean loss | **median loss** | **% clips hurt** |
|---|---|---|---|---|
| lowest 25 % | 0.00 | −0.0724 | **−0.0001** | 45 % |
| 25–50 % | 0.49 | +0.0035 | +0.0187 | 57 % |
| 50–75 % | 1.01 | +0.0138 | +0.0256 | 61 % |
| highest 25 % | 2.47 | +0.0362 | **+0.0384** | **68 %** |

Both the median loss and the fraction of clips hurt rise **monotonically** with
crowding. Raw persons-per-frame shows the same trend more weakly, so it is
*proximity*, not headcount, that matters — exactly what "needed to
disambiguate" predicts.

**Robustness caveat, stated because it changes the claim**: the sparse
quartile's *mean* of −0.0724 is driven by a single −0.759 outlier; its median
is −0.0001. So the correct reading is "halving is FREE in sparse scenes", not
"halving improves them". Pearson correlation is only +0.134 because the
relationship is a threshold effect, not linear — the quartile medians are the
honest summary.

**Oracle bound**: run full rate on the crowded half of clips and half rate on
the sparse half ⇒ **25 % of all detections saved for ~zero quality change**
(median loss on the halved half: +0.002).

Comparison of the three candidate adaptive signals tested so far:

| signal | predicts loss from halving? | verdict |
|---|---|---|
| object speed (E12) | no (corr −0.165, non-monotone) | **rejected** for detector rate — but predicts T1's *value* with an 11× gradient |
| persons/frame | weakly (corr +0.049, monotone-ish) | subsumed by crowding |
| **crowding** | **yes, monotone in median and in %hurt** | **the signal to build on** |

**Why this is the most promising unbuilt thing in the document**: the runtime
signal needs no ground truth. The tracker already knows how many confirmed
tracks it has and where they are — crowding is computable from
`tracked_object_roi` / the live track list at zero cost, every frame. And it
composes with T1 (E11), which pays most in *fast* scenes, whereas crowding
gates *sparse* ones — the two adaptations are close to orthogonal.

Proposed next experiment (not yet run): a runtime crowding estimate driving the
analytics interval between 1× and 2× the configured value, measured on the same
val split against detections/second. Success = the 25 % oracle saving with a
quality delta inside noise.

### E14 — crowding-adaptive detector rate at RUNTIME: **negative**

E13's oracle promised 25 % of detections for ~free. Implemented for real
(`crowding_adapt_max`, `a1b12dc`): crowding estimated from the tracker's own
output each frame, asymmetric EMA (rise instantly so a re-converging group gets
the full rate at once; relax at 0.05 so a momentary dispersal inside a busy
scene does not stretch the interval), interval stretched up to `max` when
crowding ≤ 0.2 and back to 1× at ≥ 1.0.

| | objective | vs off |
|---|---|---|
| off | 0.4294 | — |
| adaptive 2×, symmetric EMA | 0.4218 | −0.0076 |
| adaptive 2×, asymmetric EMA | 0.4229 | −0.0065 |
| adaptive 1.5×, asymmetric | 0.4229 | −0.0065 |

Load, measured over 9 mixed clips (MEVA / MOT20 / cevo / pp22):
**4.48 → 4.31 detections/second — a 4 % saving.**

| mechanism | detections saved | quality cost | **objective per % saved** |
|---|---|---|---|
| crowding-adaptive 2× | 4 % | 0.0065 | **0.00171** |
| uniform half rate | ~50 % | 0.0106 | **0.00021** |

**The adaptive rule is 8× less efficient per detection saved than simply
turning the rate down.** Verdict: negative, kept default-off as infrastructure.

**Why the oracle did not transfer** — worth recording, because it is a lesson
about the oracle method rather than about crowding:
1. The oracle adapted **per clip, using ground truth for the whole clip**. The
   runtime rule adapts per frame, and to be safe it must react instantly to
   crowding rising — so in any clip that is busy *at any point*, it spends most
   of its time at 1×.
2. **The sparse clips where stretching is safe were already being skipped by
   the motion gate.** The oracle's headroom was measured against an
   analytics-rate baseline, but the detector rate on those clips is set by the
   motion gate, not the analytics interval — so there was little left to take.

This is the same trap as E6 (motion-skip carry): a mechanism aimed at frames
that the existing gates have *already* dealt with. The general lesson for this
document: **before valuing any new gate, check what fraction of frames it would
actually change** — the existing motion gate and the analytics decimator have
taken most of the easy savings already.

What would still be worth trying (not run): applying the crowding signal to the
**motion-skip thresholds** rather than the analytics interval, since that is
the gate actually deciding detector runs on quiet cameras; or per-camera (not
per-frame) rate selection, which is closer to what the oracle actually modelled.

### E15 — rate vs resolution (MB's question) — **rate wins by ~7×**

**First attempt was VOID and MB caught it.** Forcing `performance.force_perf_mode`
produced *zero* change in quality or cost on either platform — impossible given
the published engine matrix. Cause: the python binding creates streams with
`realtime=false` (`ubon_pycstuff.cpp:819`) and `track_stream.c:1237` pins
non-realtime work to PM0 — *"Non-realtime (batch) work runs its detections at
FULL resolution (PM0), never the shared realtime-driven PM tier."* So PM tiers
are unreachable from the eval path, and every PM number taken before this was
measuring nothing. Re-running with `realtime=true` was also invalid: file
ingest at full speed trips the R1 decode shed (frames age past
`max_analytics_latency_s`), leaving 4 detections in the whole clip.

Fixed by adding `detection_max_size` (`track_shared.c`) — a detector input cap
independent of the PM controller, which is also the knob an adaptive-resolution
scheme would need.

**Cost of the cap — FIRST VERSION WAS WRONG (MB challenged it).** I timed
`inf.run()` on a single image and reported that as cost. At batch=1 a fixed
per-call overhead (enqueue + sync + preprocess round trip) dominates and masks
the resolution scaling entirely. Corrected by measuring the *same* shape across
batch sizes:

| batch | Orin 640 | Orin 320 | ratio | desktop 640 | desktop 320 | ratio |
|---|---|---|---|---|---|---|
| 1 | 14.35 ms | 7.86 | 1.83× | 1.76 ms | 1.43 | 1.23× |
| 4 | 11.12 | 4.03 | 2.76× | 0.84 | 0.47 | 1.79× |
| 8 | 10.68 | 3.28 | **3.26×** | 0.71 | 0.25 | **2.84×** |
| 16 | 10.67 | 3.29 | 3.24× | 0.67 | 0.24 | 2.79× |

The true per-image network cost at 640 is **10.7 ms on Orin and 0.67 ms on the
desktop**; the batch=1 figures were ~3.7 ms and ~1.1 ms of pure overhead on
top. **Resolution scales ~3.2× (Orin) and ~2.8× (desktop) at production batch
sizes** — the earlier "resolution is useless on desktop" claim was an artifact
of measuring at batch=1 and is retracted.

Per-image cost at batch 8 (Orin): 640 → 10.73 ms, 512 → 7.24, 416 → 4.99,
320 → 3.28.

**Quality cost of the cap (val split, cap actually applied):**

| cap | objective | vs 640 |
|---|---|---|
| 640 | 0.4280 | — |
| 512 | 0.4186 | −0.0094 |
| 416 | 0.4038 | −0.0242 |
| 320 | 0.3530 | **−0.0750** |

**The comparison MB asked for — corrected, at matched compute saving on Orin
(batch 8):**

| lever | compute saving | quality cost |
|---|---|---|
| resolution 512 px | 1.48× | 0.0094 |
| resolution 416 px | 2.15× | 0.0242 |
| resolution 320 px | 3.27× | 0.0750 |
| **rate 50 %** | **2.00×** | **0.0106** |
| **rate 33 %** | **3.00×** | **0.0500** |

At matched saving:
- **~2×**: rate costs **0.0106**, resolution (~470 px, interpolated) ~0.0217 —
  **rate ~2× better**
- **~3×**: rate costs **0.0500**, resolution (320 px) **0.0750** —
  **rate ~1.5× better**

**Rate still wins, but by ~1.5–2×, not the 7× claimed from the batch=1 cost
numbers.** That earlier factor is retracted. The two levers are much closer
than the first pass suggested, and resolution becomes relatively *better* the
harder you push (its cost curve is flatter between 512 and 416 than the rate
curve is between 50 % and 33 %).

The tie-breaker is that the rate lever composes with MOTION frames (E11),
which recover 44 % of its cost — 0.0106 → ~0.006 at 2× — while nothing
analogous exists for resolution. On that basis rate remains the primary lever,
but a **combined** setting (moderate rate cut + moderate resolution cut) is
now worth measuring and has not been tried.

### E16 — adaptive resolution when there are no small objects (MB's idea) — weak

Oracle test: does per-clip object size predict the loss from a 320 px cap?
185 clips.

**corr(median object height, loss) = +0.050; corr(smallest-decile height, loss)
= +0.010** — essentially zero, and the quartile medians are non-monotone
(+0.032 / +0.059 / +0.085 / +0.005 by smallest-decile height).

The one suggestive datum: the **largest-object quartile has a near-zero median
loss (+0.005)** versus +0.03–0.09 elsewhere — so "resolution is free when
everything is big" is directionally right for the extreme quartile, but the
signal does not order the middle of the distribution at all, and the
correlation is ~0.

Verdict: not worth building as stated. Even if the top quartile were detected
perfectly, it is 25 % of streams × 1.8× ≈ 11 % total saving from a noisy
signal, against a rate lever that is 7× more efficient and already implemented.
If resolution adaptation is revisited, the honest framing is **per-camera
commissioning** (a camera whose subjects are always large can be pinned to a
lower cap once, measured) rather than a per-frame runtime signal.

### E17 — the full rate × resolution Pareto (MB request) — **mixing beats either lever**

12-point grid (3 detector rates × 4 resolution caps), val split, plus MOTION
frames at the frontier. Compute = rate × per-image cost at that cap (Orin,
batch 8: 640→10.73 ms, 512→7.24, 416→4.99, 320→3.28).

| config | speedup | objective | vs full | frontier |
|---|---|---|---|---|
| 640 px @ 100 % | 1.00× | 0.4256 | — | ✓ |
| 512 px @ 100 % | 1.48× | 0.4190 | −0.0066 | ✓ |
| 640 px @ 50 % | 2.00× | 0.4145 | −0.0111 | ✓ |
| 416 px @ 100 % | 2.15× | 0.4054 | −0.0202 | |
| **512 px @ 50 %** | **2.96×** | **0.4075** | **−0.0181** | ✓ |
| 640 px @ 33 % | 3.00× | 0.3788 | −0.0468 | |
| 320 px @ 100 % | 3.27× | 0.3505 | −0.0751 | |
| **416 px @ 50 %** | **4.30×** | **0.3802** | **−0.0454** | ✓ |
| 512 px @ 33 % | 4.45× | 0.3681 | −0.0575 | ✓ |
| 416 px @ 33 % | 6.45× | 0.3499 | −0.0757 | ✓ |
| 320 px @ 50 % | 6.54× | 0.3359 | −0.0897 | ✓ |
| 320 px @ 33 % | 9.81× | 0.3054 | −0.1202 | ✓ |

**Every frontier point above 2× is a MIX.** The clearest comparison, at ~3×:

| ~3× speedup | quality cost |
|---|---|
| **mixed: 512 px @ 50 %** | **−0.0181** |
| pure rate: 640 px @ 33 % | −0.0468 |
| pure resolution: 320 px @ 100 % | −0.0751 |

**Mixing is 2.6× better than the best pure lever and 4× better than the other.**
That is what two levers with diminishing returns and different failure modes
should do — a moderate cut on each stays in the gentle part of both curves,
where either lever alone is already into its knee.

**MOTION frames improve every frontier point at zero detector cost**, and more
so the deeper the cut (there is more gap to bridge):

| config | speedup | plain | + MOTION |
|---|---|---|---|
| 512 px @ 50 % | 2.96× | −0.0181 | **−0.0162** |
| 416 px @ 50 % | 4.30× | −0.0454 | **−0.0340** (25 % recovered) |
| 512 px @ 33 % | 4.45× | −0.0575 | **−0.0433** (25 % recovered) |

### The recommended operating points

| target | config | quality cost |
|---|---|---|
| **3× detector compute** | 512 px cap, half rate, MOTION frames on | **−1.6 %** |
| **4.3×** | 416 px cap, half rate, MOTION frames on | **−3.4 %** |
| conservative 1.5× | 512 px cap, full rate | −0.7 % |

All three are config-only on current main (`detection_max_size`,
`min_time_delta_motion`), and compose with the idle win (E9/E10:
`min_time_delta_full_roi` 30 s + `track_buffer_seconds` 30 s), which is
orthogonal — it removes detections on *dead* cameras rather than reducing the
cost of the ones that run.

Caveats: the compute axis is a **model** (measured per-image cost × rate), not
an end-to-end throughput measurement of each of the 12 configs; and the
resolution figures assume batch ≈ 8, which is production-like under
multi-stream load but not what a single stream sees.

### E18 — END-TO-END on the real pipeline (`rt_benchmark`, Jetson): **the box is decode-bound, and the model-based Pareto does not apply at scale**

MB asked for the actual pipeline rather than the cost model, and pointed at
`rt_benchmark` (realtime throughput vs latency, sweeps stream counts, accepts
`--tracker-yaml`). Three configs from the E17 frontier, 20 s measurement
windows, h265 720p5 pcap inputs:

| streams | base 640 @ full | 512 @ half + MOTION | 320 @ full |
|---|---|---|---|
| 24 | 120.9 fps | 121.4 | 121.3 |
| **32** | **147.0** | **148.7** | **144.5** |
| 40 | 146.3 | 148.0 | 143.2 |
| 48 | 142.4 | 141.6 | 140.9 |
| 56 | 139.2 | 140.5 | 137.9 |

**Every config plateaus at ~140–148 fps.** A 3× cheaper detector configuration
delivers **no** additional end-to-end throughput, and per-stream fps degrades
identically (at 40 streams: 3.25 / 3.35 / 3.20). An earlier 4→20 stream sweep
across all seven configs was likewise indistinguishable — the box was simply
not saturated there (full 5 fps × N delivered every time).

**Diagnosis — decode-bound.** E4 measured decode at **7.36 ms/frame** on this
Orin. At 145 fps aggregate that is 145 × 7.36 ms ≈ **1.07 s of decode per
second of wall clock**, i.e. NVDEC saturated. Corroboration: by 32 streams the
PM controller has already driven itself to the lowest resolution tier and ~48 %
frame skip, and throughput still does not respond to being handed a cheaper
detector config.

**What this means for everything above:**
- E17's Pareto is a **detector-compute** Pareto. It is valid where the detector
  is the binding constraint — a moderate number of streams, or a box with
  decode headroom — and it is **irrelevant** where decode saturates first. On
  this Jetson at ≥32 streams, all detector levers are free *and worthless*.
- The quality numbers in E5/E11/E15/E17 stand (they are platform-independent);
  the *compute-saving* half of every claim needs qualifying by "if the detector
  is the bottleneck".
- **The corollary is more useful than the finding**: on a decode-bound box the
  detector budget is effectively free, so the right move is the opposite of
  economising — spend it. Run the detector at full resolution and full rate
  (and turn MOTION frames on, which cost no detector time at all and measured
  +0.0048 at full rate in E11), because reducing any of it buys no throughput.

**Caveats stated plainly**: (i) the `--csv` header has fewer names than the row
has values, so I read only the unambiguous leading columns (streams, fps_total,
fps_min/max) and the trailing PM-tier fractions — the latency columns are not
safely attributable and are not quoted; (ii) one input resolution/codec
(720p5 h265) and one clip set — a 4K or h264 input mix would move the decode
ceiling and could restore detector-boundedness; (iii) this is the Orin, and the
desktop's decode:detector ratio is entirely different (0.39 ms vs 0.67 ms per
image at batch 8), so the desktop may well remain detector-bound.

**The single most valuable follow-up** is therefore not another detector
experiment: it is to establish where the decode ceiling sits per input format
(720p vs 4K, h264 vs h265) and whether it can be raised, since on this hardware
that ceiling — not the detector — sets stream density.

### E19 — decode-bound CONFIRMED by starving the detector (720p5 h265, the production format)

E18 inferred decode-boundedness from a plateau plus a separately-measured
per-frame decode cost. Direct test instead: hold everything constant and starve
the detector by lowering the analytics rate, on the production-representative
input (MB's steer: 720p5 h265 is the format that matters, so the 4K/h264
ceiling sweep was dropped).

| streams | analytics 5 fps | analytics 1 fps | analytics 0.1 fps |
|---|---|---|---|
| 28 | 138.0 fps | 140.0 | 139.3 |
| 36 | 148.4 | 150.2 | **156.0** |
| 44 | 145.1 | 147.9 | **153.1** |

**A 50× reduction in detector work buys ~5 % throughput** (148 → 156 fps).
Per-stream fps and skip fraction degrade near-identically across all three.
If the detector were the binding constraint, removing 98 % of its work would
multiply throughput; it does not. **Confirmed: the ceiling on this Orin is
ingest/decode at ≈150 fps aggregate ≈ 30 streams at 5 fps.**

This closes the loop on the whole campaign's cost reasoning:

| claim | status |
|---|---|
| detector quality-vs-rate / -resolution curves (E5, E15, E17) | **valid** — platform-independent |
| detector compute savings translate to throughput | **false on this box at scale** (E18, E19) |
| ~150 fps / ~30 streams @ 720p5 h265 is the Orin ceiling | **measured directly** |

**Practical consequence, and it inverts the original brief**: on a
decode-saturated box the detector budget is free, so the correct configuration
is the *most* detector work that fits — full resolution, full rate, and MOTION
frames on (zero detector cost, +0.0048 at full rate, E11). Every
economy measured in E15/E17 would trade quality for throughput that the box
cannot deliver anyway.

Where the E17 Pareto *does* apply: deployments below the decode ceiling
(≲28 streams here), any box with more decode headroom relative to its GPU, and
the desktop class (0.39 ms decode vs 0.67 ms detector per image at batch 8 —
plausibly still detector-bound, untested at scale).

### E20 — **RETRACTION of E18/E19.** The box is not decode-bound; the detector is ~88 % busy

MB rejected the decode-bound conclusion and pointed at `decode_benchmark`. He was
right, and there were **two independent errors**.

**Error 1 — mislabelled measurement.** I called the 7.36 ms/frame figure from E4
"decode". It is the whole *decode job*: `cuvidMapVideoFrame` + NV12→YUV420
convert + scale-to-cap, plus queue plumbing. Actual NVDEC capacity, measured
with `decode_benchmark` on this Orin:

| input | threads | fps |
|---|---|---|
| 720p h265 | 32 | **820** |
| 720p h265 | 8 | 711 |
| 720p h264 | 32 | 456 |

Decode has ~5× headroom over the observed ~150 fps. It is **not** the
constraint. (`scale_perf` likewise: 4K→640 convert+scale is 179 µs.)

**Error 2 — wrong axis.** I compared configs on `fps_total`, which counts frames
**ingested**, not detector work. Detector-boundedness cannot be read off it.

**What the infer thread's own accounting says** (36 streams, `--verbose`):

```
thread_time_breakdown: wait 3.06 s | pick 0.003 | prep 0.19 | infer 24.08 s
total_batches 508 | total_images 2784 | mean_batch 5.48 | mean_roi_area 0.60
```

**infer_time is 24.08 s of a ~27 s window ⇒ the detection thread is ~88 % busy**,
at **8.65 ms per image** (47.4 ms per batch of 5.5). So the detector *is* a
first-order constraint, and E17's economy Pareto is **not** invalidated — I
simply measured it on the wrong axis.

**Why the configs still converged (the real reason).** Two mechanisms, both
visible in the corrected CSV columns (the `--csv` header is stale — the true
order is `streams, fps_total, fps_min, fps_max, skipped_%, force_skip_rate, cv,
mean_batch, lat50, lat90, iq50, iq99, roi, pm0..pm3`):

1. **The PM controller already does the economising.** At ≥32 streams it has
   driven itself to `pm3 = 0.99` (the 320 px tier) unprompted, then sheds frames
   (`force_skip_rate` 0.05 → 0.24 → 0.39 → 0.48 at 32/40/48/56 streams).
   Pinning a cheaper config manually just *replaces* what the controller would
   have done — hence no change.
2. **Batching partly defeats per-stream tiers**: batch effective PM is
   `min(req_pm)` over the batch (`infer_thread.c:298-319`), so one full-resolution
   stream upgrades the whole batch. Measured 8.65 ms/image against the 3.28 ms
   an all-320 px batch of 8 achieves in isolation — a 2.6× gap explained by
   batch 5.5 (less launch amortisation) plus tier upgrading.

**Also true, and why `fps_total` was insensitive**: with analytics starved 50×
the ingest side still only reached 156 fps of the 180 fps demanded, so there is
*additionally* an ingest-path ceiling around ~155 fps in this harness
(pcap replay + RTP depacketisation + decode-job plumbing — **not** NVDEC).
Two near-simultaneous constraints is why no single change moved the headline
number.

**Net effect on the campaign's conclusions:**

| claim | status |
|---|---|
| E18/E19 "decode-bound, detector budget is free" | **RETRACTED — wrong** |
| E17 detector Pareto is irrelevant at scale | **RETRACTED — it is relevant** |
| detector economy is worth pursuing on Orin | **restored**: the thread is 88 % busy |
| quality curves (E5/E11/E15/E17) | unaffected throughout |

**Methodological lesson, recorded because it caused both errors**: measure the
*resource you are reasoning about*, on its own counter. The infer thread has
published `thread_time_breakdown` and CUDA-event `gpu_s` all along; I inferred
from wall-clock and an ingest-rate proxy instead, and got the bottleneck
backwards. Any future capacity claim in this document should cite
`thread_time_breakdown`, `decode_benchmark`, or `scale_perf` — never a plateau.

### E21 — half-rate detection END-TO-END: **GPU time does not fall, it rises** (batch collapse)

MB asked directly whether half-rate detection had been tried end to end. It had
not been tested on the right metric. Doing so properly, 36 streams on Orin,
`rt_benchmark --verbose`, CUDA-event `gpu_s` (not wall clock, not thread
lifetime):

| config | detector images | batches | mean batch | **gpu_s** |
|---|---|---|---|---|
| full rate | 2788 | 504 | 5.53 | **20.01 s** |
| **half rate** (`debug_analytics_mask: "10"`) | 1789 (−36 %) | **597 (+18 %)** | **3.00** | **21.14 s (+5.7 %)** |

**Halving the detection rate cuts invocations by 36 % and makes the GPU work
MORE.** The mechanism is unambiguous: mean batch collapses 5.53 → 3.00, so the
batch count *rises* 504 → 597, and the fixed per-batch cost (the repo's own
model puts `K_LAUNCH` at ~4 ms) is paid 18 % more often. Per-image cost rises
from 7.2 ms to 11.8 ms.

Same effect visible one step further out (`infer_time`, thread lifetime):
full 5.43 batch / half 3.13 / third 1.30, with per-image cost 8.7 → 13.2 →
18.0 ms.

**This invalidates the compute-saving half of E17 for multi-stream deployments.**
The quality numbers stand (single-stream eval, platform-independent), but "half
the rate ⇒ half the detector compute" is **false whenever batching is what makes
the detector efficient**. In the eval each stream runs alone (batch 1), so rate
reduction translated directly; at 36 concurrent streams it does the opposite.

**Corollary — the lever that actually exists here is batch efficiency, not
rate.** Anything that raises mean batch size (more streams per batch, longer
picker linger, aligning stream phases) is worth more than reducing work.

**Resolution end-to-end: NOT MEASURED — my test was void.** `detection_max_size`
did not bind in `rt_benchmark` (verified by the absence of its log line): the
parse I added sits inside the `inference_config` entry loop in
`track_shared.c`, so it only reaches the top-level key via the *set-config*
path the eval uses, never the *create* path `rt_benchmark` uses. The pin640 vs
pin320 rows (20.04 vs 20.18 s) were the same configuration twice and must be
disregarded. **Fixing that plumbing is the prerequisite for any end-to-end
resolution claim.**

**Correction to E20**: I reported the detection thread as "~88 % busy" from
`infer_time_s`. That field is thread *lifetime* minus wait — `wait_time_s +
infer_time_s` equals the window in every run — so it cannot measure utilisation.
The CUDA-event figure is `gpu_s = 20.0 s` in a ~27 s window ⇒ **~74 % GPU
utilisation** at 36 streams. Detector-boundedness is real but less extreme than
E20 said. Also note `performance_mode_count {0:15, 1:51, 2:21}`: the *effective*
batch tiers are mostly 1–2, not the pm3 the CSV column implies, because batch PM
is `min(req_pm)` — one full-resolution stream upgrades the whole batch.

**Running tally of metric mistakes in this section, since the pattern matters
more than any single number**: wall-clock latency mistaken for cost (E15);
`fps_total` (ingest) mistaken for detector work (E18/E19); decode *job* mistaken
for NVDEC (E18); thread lifetime mistaken for GPU utilisation (E20); an unbound
config mistaken for a measurement (E21). Every one was caught by checking the
resource's own counter. **`gpu_s`, `decode_benchmark`, `scale_perf`, and
`total_images`/`batch_count` are the only capacity numbers this document should
quote.**

### E22 — why the batch is small, and why every end-to-end knob measured null

MB rejected the "half rate is slower" result as absurd. It was. Root causes,
established by *checking the running binary* rather than reading code:

**Methodology failure first.** Three of my end-to-end tests measured knobs that
were never live:
- `performance.force_perf_mode` / `rt_linger_ms`: the whole `performance:` block
  was parsed **only in `track_shared_state_set_config`**, never in
  `track_shared_state_create`. `rt_benchmark` (and the python binding, and the
  apps) only call create ⇒ silently defaulted. Fixed by factoring
  `ts_apply_performance_config()` and calling it from both.
- `detection_max_size`: not present in the Jetson build when the pin640/pin320
  test ran (rsync'd later).
- The Jetson's `/mldata/config/track/trackers/uc_v11.yaml` **differs from the
  desktop's** (`min_time_delta_full_roi: 10`, `min_time_delta: 0.1`), so
  desktop-derived configs were a different baseline.

**New standing rule: never measure a knob without positive confirmation it is
live in the binary under test.** Both knobs now emit a log line on bind
(`infer_thread[0]: RT linger 25 ms, batch target 12`,
`detection_max_size: capping detector input at 320px`), and those were checked
before the numbers below.

**With both knobs verified live, 36 streams, 2 reps each — all identical:**

| config | images | mean batch | gpu_s |
|---|---|---|---|
| baseline | 2784 / 2796 | 5.60 / 5.57 | 20.16 / 20.13 |
| + RT linger 25 ms | 2805 / 2742 | 5.69 / 5.72 | 20.03 / 19.64 |
| + 320 px cap | 2752 / 2809 | 5.56 / 5.72 | 19.88 / 20.16 |
| both | 2791 / 2763 | 5.71 / 5.51 | 19.71 / 19.92 |

**The explanation, from the per-batch shape trace**
(`UBON_INFER_BATCH_TIME_TRACE_PATH`):

| config | batches | mean num | mean elapsed | dominant shapes |
|---|---|---|---|---|
| baseline | 412 | 5.40 | 48.5 ms | **320×320: 291 (71 %)**, 416²: 23, 512²: 9 |
| +320 px cap | 411 | 5.41 | 48.5 ms | 320×320: 303 (74 %) |

1. **The PM controller had already driven resolution to 320 px in the baseline.**
   Manually capping resolution is redundant — it is doing my optimisation for me.
   This is the real reason every resolution config measured null, on every run.
2. **`gpu_s` is pinned at ~20 s (≈74 % of the window) in every config** because
   the controller sheds until latency meets its target. It fills whatever
   capacity exists, so **`gpu_s` at fixed stream count cannot discriminate
   configs** — and my "half rate uses more GPU" claim was an artefact of exactly
   that. Retracted.
3. **Batch size is set by arrival rate × service time, not by picker greed.**
   At 48.5 ms/batch and ~140 jobs/s, ~7 jobs accumulate during each inference —
   hence the observed 5.4. It is a queueing equilibrium.
4. **My RT linger cannot fire under load — a design error.** I bounded the wait
   by the *oldest* job's age, but median queue latency is 147 ms, so the head job
   is always older than any sane linger and `remaining < 0` dispatches
   immediately. The linger only engages when the queue is nearly empty, i.e.
   when batching is irrelevant. To grow batches under load the wait must be
   relative to the **dispatch decision**, not job age — which trades latency
   directly and needs a deliberate decision about the latency budget.

**Where this leaves the compute question**: on Orin at 36 streams the pipeline is
already at its self-optimised operating point — 320 px, batch ~5.4, 74 % GPU,
shedding to hold latency. There is **no headroom to reclaim by configuring the
detector cheaper, because the controller has already done it.** Detector economy
only has room *below* saturation, which is precisely where the E17 quality/cost
Pareto applies.

The one remaining untested lever with real upside is **batch efficiency**: 9.0
ms/image at batch 5.4 versus 3.3 ms/image at batch 8 in isolation. Realising it
requires dispatch-relative lingering (a latency trade) or phase-aligning stream
arrivals, not a cheaper detector.

### E23 — the capacity CURVE, measured: the mixed ladder is a new Pareto frontier

MB's reframing: capacity is not a number, it is the curve of tracking quality
versus concurrent stream count, because the PM controller degrades to hold
latency and therefore *chooses* the operating point at every N. The goal is to
move that curve out.

Built two tools so the curve is a measurement rather than an argument:

- `src/quality_table.py` — quality at each (resolution cap × analytics rate)
  operating point, per content type, from `track.py --eval` on the val split,
  reading the eval's own `results-*.json` rollups. Content type = the corpus the
  clip came from; the mapping is explicit and unmapped clips are reported, not
  silently bucketed.
- `src/capacity_curve.py` — joins that table with `rt_benchmark --csv` operating
  points (PM tier distribution + shed rate) to emit quality vs streams.

**Long-window sweep on Orin** (12–60 streams, 60 s measurement + 15 s warmup per
point, 720p5 h265, `degrade_policy` the only variable):

| streams | resolution-first (default) | mixed ladder | Δ quality |
|---|---|---|---|
| 12 | 548 px, rate 1.00 → **0.4209** | 549 px, rate 1.00 → **0.4209** | +0.0001 |
| 24 | 393 px, rate 1.00 → **0.3921** | 511 px, rate 0.62 → **0.4101** | **+0.0180** |
| 36 | 320 px, rate 0.89 → **0.3474** | 466 px, rate 0.50 → **0.3941** | **+0.0467** |
| 48 | 320 px, rate 0.64 → **0.3400** | 419 px, rate 0.44 → **0.3703** | **+0.0304** |
| 60 | 322 px, rate 0.50 → **0.3368** | 416 px, rate 0.36 → **0.3540** | **+0.0172** |

Read the curve horizontally rather than vertically: the mixed ladder holds
quality 0.37 at **48** streams where the default reaches it at **~27**. The two
policies are identical at 12 streams — as they must be, since neither degrades
when nothing is saturated, which is also a useful negative control on the
harness.

The shape of the win is exactly what E17 predicted: the default spends its first
currency on resolution, which is the most expensive quality per unit of compute
saved. The mixed ladder trades rate earlier and keeps the detector sharper —
at 36 streams, 466 px @ half rate beats 320 px @ 89 % rate by 0.047.

### E24 — **RETRACTION: the `min_time_delta_motion` arms of that sweep were no-ops**

Two of the four sweep arms (`L_res_nvof`, `L_mixed_nvof`) set
`min_time_delta_motion: 0.09` intending to measure the NVOF carry under load.
Added a `motion_percent` column to `rt_benchmark` (reading the
`motion_frame_count` the stream stats already emitted) to confirm the path was
live — and it was not: **`Mot=0.000` at 36 streams**, with `FSkip=0.505`.

The reason is structural, and it is the whole argument for MB's requested change.
In the default shed mode the PM controller drops the frame in the decoder display
callback, so it never reaches `track_stream`. At 5 fps input with
`min_time_delta_process = 0.18`, every frame that *does* arrive is analytics-due.
So there is no such thing as a spare frame for the MOTION class to occupy: the
frames that could have carried flow are precisely the ones thrown away. The two
"nvof" arms were measuring nothing, and their near-identical curves were correct
readings of a disabled feature.

This is the third time a config knob has been parsed, plausible and inert. The
`motion_percent` column exists so this class of error is caught by the harness
rather than by a later contradiction.

### E25 — `performance.skip_mode: motion` — confirmed live, and not free

The change MB asked for (rather than approximating it by widening the base grid):
PM shed no longer drops the frame, it delivers it flagged `MD_NO_ANALYTICS`, so
it routes to the MOTION path — no detector, but the tracks keep their
optical-flow anchor. `force_skip_count` is still incremented at the decision
point, so the shed rate remains measurable.

Confirmed on Orin at 36 streams: **`Mot=0.523` against `FSkip=0.527`** — every
shed frame becomes a carry frame. Log line binds, counter agrees.

It costs more than the flow hop, though, and both costs are visible at 36
streams versus the same config in drop mode:

| | drop | motion |
|---|---|---|
| latency p50 / p90 | 0.232 / 0.426 s | **0.701 / 0.968 s** |
| PM tier mix | 512 px 0.40 / 416 px 0.60 | **416 px 0.99** |
| shed rate | 0.505 | 0.527 |

Not dropping at the decoder means decoding, converting and scaling every frame
plus one NVOF hop per shed frame, and the controller pays for that work by
degrading further — effective resolution falls from ~455 px to ~416 px and
latency roughly triples. At 48 streams a stream starves outright (`fps_min` 0.07).

So the question is not "does the carry help" (E9 says it does, at fixed rate)
but whether the carry is worth the resolution and latency it costs at equal
stream count. That is a strict Pareto question and it needs the carry axis in the
quality table, which is why `quality_table.py` now measures
`gridm_r{2,3}_{res}` — the same grid with the carry on — as `table_motion_carry`.

### E26 — **the cheapest degradation lever is content-dependent, and that is the biggest lever found**

With the quality table indexed by content type, the per-content curves show the
global ladder is a compromise between two populations that want opposite things.
Quality at the corners (val split, `fitness_multi`):

| content | 640 @ 1.0 | 320 @ 1.0 | 640 @ 0.33 | 640→320 costs | rate→1/3 costs |
|---|---|---|---|---|---|
| cctv_static    | 0.4874 | 0.3325 | 0.4689 | **−0.155** | −0.019 |
| cctv_dense     | 0.3156 | 0.1856 | 0.2574 | **−0.130** | −0.058 |
| office_indoor  | 0.7214 | 0.6518 | 0.6155 | −0.070 | −0.106 |
| handheld_crowd | 0.2757 | 0.2348 | 0.2393 | −0.041 | −0.036 |
| bodycam        | 0.4443 | 0.4044 | 0.2992 | −0.040 | **−0.145** |
| doorway        | 0.6040 | 0.6284 | 0.5948 | **+0.024** | −0.009 |
| dashcam_jaad   | 0.1959 | 0.2304 | 0.1446 | **+0.035** | −0.051 |
| movie          | 0.3227 | 0.3589 | 0.1960 | **+0.036** | **−0.127** |

320 px @ full rate and 640 px @ 1/3 rate are roughly compute-comparable (area
ratio 0.25, measured per-image ratio 1/3.24, versus rate 0.33). Compared at that
equal compute the population splits cleanly, and along a physically sensible
line:

- **Static camera, small or dense subjects** — cctv_static, cctv_dense — pay
  heavily for resolution and almost nothing for rate. Consecutive frames are
  nearly redundant, but distant subjects need pixels. Rate-first, by a margin of
  0.07–0.14.
- **Ego-motion content** — movie, bodycam, dashcam — pay heavily for rate and
  *gain* from lower resolution. Inter-frame displacement is large, so dropping
  frames breaks association; and a smaller detector input suppresses the small
  spurious detections that ego-motion generates. Resolution-first, by 0.09–0.16.

For three content types, dropping the detector to 320 px **improves** quality
outright — it is a free compute saving, not a degradation, which no global ladder
can exploit without hurting the static-camera streams.

**Headroom estimate.** Taking, at each stream count, the better of the two
measured operating points per content type (unweighted mean over the eight
types):

| streams | resolution-first | mixed | per-content best | gain over the better global |
|---|---|---|---|---|
| 12 | 0.4195 | 0.4195 | 0.4197 | +0.0001 |
| 24 | 0.4018 | 0.3987 | 0.4124 | **+0.0106** |
| 36 | 0.3728 | 0.3839 | 0.3954 | **+0.0115** |
| 48 | 0.3601 | 0.3631 | 0.3768 | **+0.0137** |
| 60 | 0.3536 | 0.3437 | 0.3665 | **+0.0129** |

The gain is over *whichever* global policy is better at that point, so it is
additive to the E23 mixed-ladder win rather than competing with it.

**This is an estimate from measured points, not an end-to-end measurement.** It
assumes a stream can take its own ladder at the same total compute; a mixed fleet's
aggregate compute is not exactly the mean of the two homogeneous fleets'. To
measure it properly, `rt_benchmark` needs per-stream content assignment and
per-stream quality attribution — the current harness runs one content type across
all streams. That harness gap is the blocker, and it should be closed before the
feature is trusted, not after.

**Implementation shape** (not yet built): keep the controller's single
`pm_continuous` pressure scalar — it is the closed loop and must stay global —
but make the *mapping* from pressure to (resolution tier, skip fraction)
per-stream, selected by `stream_hint`. The plumbing already suits this: detector
jobs carry a per-job `req_pm`, and `skip_frac` is consulted per stream in
`track_stream_jobs.c`. Two ladders (rate-first, resolution-first) keyed on hint
would capture most of the table above.

### E27 — batch builder: the linger fix works, and the lever is worth ~2 %

E22 closed by naming batch efficiency "the one remaining untested lever with real
upside". Implemented the fix it prescribed — linger measured from the **dispatch
decision** rather than the oldest job's age, with `rt_max_age_ms` as a hard
ceiling — and measured it at 36 streams on Orin:

| config | mean batch | lat50 / lat90 | shed rate | effective res |
|---|---|---|---|---|
| greedy (default) | 3.94 | 0.167 / 0.295 | 0.497 | 467 px |
| linger 10 ms | 4.58 | 0.173 / 0.286 | 0.501 | 458 px |
| linger 25 ms | 4.63 | 0.181 / 0.294 | 0.501 | 456 px |
| linger 50 ms | 4.73 | 0.181 / 0.303 | 0.500 | 451 px |

The mechanism works — the batch grows 20 % for a few ms of latency, where the
age-relative version could not engage at all. **But the operating point does not
improve**: the shed rate is identical to three decimals and effective resolution
drifts slightly *down*. So the lever is real and negligible.

**Why, arithmetically.** Detector cost is `K_LAUNCH (4.0 ms/batch) +
K_PER_PIXEL · B · area`. At 22.8 batches/s (batch 3.94) launch overhead is 91
ms/s ≈ 9 % of the detector; at batch 4.73 it is 76 ms/s. The saving is ~1.5 % of
GPU time. **Launch overhead is only ~9 % of detector time at these batch sizes,
so batching cannot yield more than that even if perfected.**

This also corrects E22's own framing: I had contrasted "9.0 ms/image at batch 5.4"
against "3.3 ms/image at batch 8" and attributed the gap to batch size. It is
mostly **area** — the isolated batch-8 figure was at 320 px while the loaded
figure is 416–512 px. Batch size and resolution were conflated; the batch part of
that gap is a couple of percent, not 2.7×.

`rt_linger_ms` stays defaulted to 0 (greedy). It is kept, with the corrected
semantics, because it is the right shape if a future engine has a larger
`K_LAUNCH`.

### E28 — per-content ladders, built and measured: **+0.001, not the estimated +0.012**

Built the feature E26 specified and measured it end-to-end rather than trusting
the estimate.

**Implementation.** The pressure signal stays global — it is the closed loop and
must be. Only the *spending* is per stream: `pm_ladder_apply()` in
`pm_controller.h` is now the single ladder implementation, the controller
publishes the raw pressure level as `tss->global_pm_continuous`, and a stream
with its own `performance.degrade_policy` maps that level itself for both its
requested resolution tier (via the per-job `submit_pm`, already plumbed) and its
skip fraction. Per-content selection needs no new mechanism: the config's
hint-variant resolution already applies `degrade_policy(hint:x)` per stream.

Compute-effective because `infer_batch_pick_area` groups by requested shape — at
inflation 0 it batches only identical `req_w/req_h`, and its floor stops small
jobs being upgraded — so per-stream tiers drain in their own batches.

**Harness.** `rt_benchmark --stream-hints cctv,egomo` cycles hints across
streams, so one run carries a mixed fleet, with per-hint rollups. This closes the
harness gap E26 flagged.

**It binds, and each group spends along its own axis** (36 streams, Orin):

| group | ladder | shed rate | fps/stream |
|---|---|---|---|
| hint cctv | mixed | 0.527 | 2.37 |
| hint egomo | resolution_first | 0.125 | 4.50 |

Both groups' shed rates imply the *same* global pressure level (L ≈ 3.11 solved
independently from each ladder's skip curve) — a useful consistency check that
the two are reading one controller.

**But the gain is +0.0012, an order of magnitude below the +0.0115 estimate:**

| content | split fleet | homogeneous mixed | homogeneous res-first |
|---|---|---|---|
| cctv_static    | 0.3891 | **0.4185** | 0.3349 |
| cctv_dense     | 0.2159 | 0.2184 | 0.1877 |
| office_indoor  | 0.6659 | 0.6847 | 0.6453 |
| handheld_crowd | 0.2407 | 0.2501 | 0.2291 |
| dashcam_jaad   | 0.2089 | 0.2256 | 0.2183 |
| bodycam        | 0.3945 | 0.3945 | 0.3957 |
| doorway        | 0.6277 | 0.6090 | 0.6278 |
| movie          | 0.3416 | 0.2738 | 0.3437 |
| **mean** | **0.3855** | 0.3843 | 0.3728 |

**Why — the ladders are not on a common scale.** The resolution-first group landed
exactly where its homogeneous run did (doorway 0.6277 vs 0.6278, movie 0.3416 vs
0.3437). The *entire* loss fell on the mixed group, which got 416 px @ 0.473
instead of the 466 px @ 0.50 it reaches alone. The cause is that one pressure
level means different amounts of compute depending on the ladder — relative
detector cost (rate × area) at equal L:

| L | resolution-first | mixed | ratio |
|---|---|---|---|
| 2.50 | 0.423 | 0.263 | **1.61** |
| 3.00 | 0.250 | 0.211 | 1.18 |
| 3.11 | 0.223 | 0.200 | 1.11 |
| 3.60 | 0.100 | 0.148 | 0.68 |

So a mixed fleet's pressure is dragged by whichever ladder is greedier at that
level, and the other ladder's streams pay. Here resolution-first is ~11 % heavier
around L≈3, which pushed L up just far enough to cost the mixed group a whole
resolution tier — the mixed ladder's tier is flat at 2 across L ∈ [3,4], so the
466→416 px step lands right at that boundary.

**My estimate assumed the two groups were independent. They are not, because they
share one pressure scalar.** The estimate should have been labelled as assuming
compute-neutrality between ladders, which is measurably false.

**The fix, well defined but not built:** parameterise both ladders by *relative
compute fraction* rather than by an arbitrary level scale, so that at any level
every ladder consumes the same detector budget. Then a mixed fleet's equilibrium
pressure is ladder-independent, each group lands where its homogeneous run does,
and the E26 per-content quality advantage should actually materialise. It also
makes the two ladders directly comparable — "which ladder gives more quality per
unit of compute for this content" becomes a well-posed question instead of an
artefact of how the scales happen to line up. This touches the controller's
calibration (`pm_signal_to_continuous`, `pm_static_from_streams`), so it carries
regression risk on the default path and wants a deliberate decision, not a
drive-by change.

**Standing so far on the frontier**: the mixed ladder (E23, +0.017 to +0.047, all
measured end-to-end) is the banked win. Per-content ladders are built,
verified live, and currently worth ~nothing until the ladders are put on a common
compute axis.

### E29 — equalising the ladders on DETECTOR compute: still wrong, and why

Implemented E28's fix — ladders parameterised by relative compute rather than by
an arbitrary level scale, with the budget curve anchored on the resolution-first
ladder so the default path is exactly unchanged (pinned by a unit test over the
whole range), and each ladder inverting the budget exactly (waypoints differ in
one axis at a time, so a rate-only segment solves `rate = budget/area` and a
tier-only segment solves the fractional tier whose cadenced *mean* area hits the
budget — the Bresenham mean is linear in the fraction, so that inversion is exact
rather than approximate).

Result at 36 streams: **+0.0005**, versus +0.0012 before. No better.

The diagnosis is visible in the latencies. Homogeneous resolution-first ran at
**Lat50 0.661 / Lat90 1.053** against mixed's **0.210 / 0.360** at the same stream
count — while using a *smaller* detector budget. Two ladders can be exactly
detector-equal and still be far apart in true cost:

| | rate | tier area | detector | non-detector | total |
|---|---|---|---|---|---|
| resolution-first | 0.825 | 0.25 | 2.21 ms | 6.07 ms | 8.28 ms |
| mixed | 0.488 | 0.4225 | 2.21 ms | 3.59 ms | 5.80 ms |

**Resolution scales only the detector; rate scales the entire pipeline** — decode
map, NV12→YUV420 convert, scale, tracker, plumbing. Identical detector spend,
1.43× different total cost. Equalising the detector alone was the wrong axis.

### E30 — the cost model, calibrated from fleet equilibria — and the structural limit

Added the non-detector term to the ladder cost model, `cost = rate × (w + area)`.

**Fitting w from stage timings was wrong.** 7.36 ms non-detector against 10.7 ms
for a 640 px detector pass gives w = 0.688, and measuring that pushed the entire
imbalance onto the *other* ladder: the ego-motion group's shed rate went from
0.114 alone to 0.348 in a mixed fleet. Latency is not the sum of stage costs —
decode and scale run on per-stream worker threads in parallel while the detector
is a single serialised shared resource.

**The fleet equilibrium is what measures it.** Two homogeneous fleets on the same
hardware at the same stream count and latency target must settle at the same cost
per stream, which pins w exactly. At N=36: resolution-first settles at rate 0.886
/ mean area 0.2522, mixed at rate 0.516 / mean area 0.5628, giving **w = 0.181**.

That calibration works, and is checkable: with w = 0.181 the two homogeneous
fleets now settle at the *same pressure level* — L = 3.126 and L = 3.121, against
3.114 vs 3.312 before. Equal pressure now means equal true cost, which is exactly
what the parameterisation was for.

**But the per-content gain still does not appear.** Three iterations of the same
measurement:

| ladder parameterisation | per-content vs best homogeneous |
|---|---|
| level-indexed tables (E28) | +0.0012 |
| detector-compute equalised (E29) | +0.0005 |
| total-cost equalised, w fitted (E30) | **−0.0023** |

**The residual coupling is queueing latency, not cost, and no cost
reparameterisation can remove it.** At equal cost and equal pressure level,
homogeneous resolution-first still runs at Lat50 0.621 / Lat90 1.009 versus
mixed's 0.247 / 0.456. A high-rate ladder keeps roughly four times as many frames
in flight per stream, so it generates far more queueing latency per unit of
compute. The controller regulates *latency*. So a fleet containing high-rate
streams sheds harder for everybody, and the split fleet settles at L = 3.258 when
both pure fleets settle at ≈ 3.12.

Two ladders cannot be made latency-equivalent by pricing compute, because they
differ in frames-in-flight at equal compute. **Per-content ladders are
structurally limited under a single shared latency-regulated scalar** — which is
the honest verdict after three attempts, each of which fixed a real modelling
error and moved the result by ~0.002.

The design that could work is **per-group control loops**: each ladder group
deriving its own pressure from its own measured latency, rather than one global
scalar. That is several coupled controllers sharing one resource, with real
stability questions (and the existing single loop was itself adopted because a
multi-band predecessor limit-cycled — see the `PMSkipControllerSim` history). It
wants a deliberate decision, not a drive-by change.

**What the calibration did buy**: the global mixed ladder's own operating point
improved, because its mapping is now priced correctly rather than accidentally.
At 36 streams homogeneous mixed moved from 470 px @ 0.495 to 483 px @ 0.505,
worth **+0.0038** on the same eight-content mean. The E23 curve is being
re-measured with the calibrated ladder; E23's numbers were taken with the old
level-indexed table.

**The E23 curve, re-measured with the calibrated ladder** (12–60 streams, 60 s
windows, 15 s warmup):

| streams | resolution-first | mixed | Δ | Δ before calibration |
|---|---|---|---|---|
| 12 | 550 px, 1.00 → 0.4240 | 557 px, 1.00 → 0.4243 | +0.0002 | +0.0001 |
| 24 | 384 px, 1.00 → 0.3855 | 512 px, 0.68 → 0.4116 | **+0.0260** | +0.0180 |
| 36 | 321 px, 0.88 → 0.3473 | 479 px, 0.50 → 0.3971 | **+0.0498** | +0.0467 |
| 48 | 320 px, 0.65 → 0.3391 | 431 px, 0.38 → 0.3647 | +0.0256 | +0.0304 |
| 60 | 321 px, 0.51 → 0.3353 | 397 px, 0.27 → 0.3439 | +0.0086 | +0.0172 |

Calibration helps the mid-load range (24 and 36 streams gain 0.008 and 0.003) and
costs a little at the top end (48 and 60 lose 0.005 and 0.009). The reason is
visible in the operating points: pricing resolution correctly makes it look *less*
effective than the detector-only model implied — 640→320 px is a 2.74× saving on
total cost, not 4× — so at a given budget the mixed ladder is pushed further along
its rate axis, reaching rate 0.27 at 60 streams where before it held 0.36. That
over-sheds rate once the budget gets tight. A rate floor on the mixed ladder's
waypoint path is the obvious remedy and is untested.

The banked win is unchanged in shape and slightly better where it matters: the
mixed ladder is worth **+0.026 to +0.050** across 24–48 streams, which is the
realistic operating range.

### E31 — a TRUE baseline column, and the cost of enabling per-stream ladders

Correcting a methodological gap: the `L_res_first` arm sets
`performance.degrade_policy: resolution_first`, and under the per-stream ladder
change *the presence of that key* is what switches a stream onto its own ladder.
So that arm was never a clean baseline — same ladder arithmetic, but tier
cadencing moves from one globally-shared tier to each stream dithering its own.

Ran a genuine baseline with the key absent entirely, binding-checked both ways
(baseline: 0 `PM ladder` log lines; `L_res_first`: 12, one per stream):

| streams | TRUE baseline (no key) | same ladder, per-stream path | Δ |
|---|---|---|---|
| 12 | 553 px, 1.00 → 0.4241 | 550 px, 1.00 → 0.4240 | −0.0001 |
| 24 | 392 px, 1.00 → 0.3901 | 384 px, 1.00 → 0.3855 | **−0.0045** |
| 36 | 321 px, 0.88 → 0.3473 | 321 px, 0.88 → 0.3473 | +0.0001 |
| 48 | 320 px, 0.65 → 0.3391 | 320 px, 0.65 → 0.3391 | −0.0000 |
| 60 | 320 px, 0.51 → 0.3349 | 321 px, 0.51 → 0.3353 | +0.0004 |

**Enabling per-stream ladders is free at 12, 36, 48 and 60 streams and costs
0.0045 at 24** — reproducible, since the earlier old-binary sweep showed the same
~0.005 gap at 24 and nowhere else. The mechanism fits batch shape fragmentation:
when each stream cadences its own tier the queue holds several request shapes at
once, and `infer_batch_pick_area` groups by shape. At 24 streams the batch is ~4.9
and mixed shapes bite; at 36+ the controller is pinned near a single tier so there
is nothing left to fragment, and at 12 the batch is ~1.1 so batching is irrelevant
either way.

This is a third, independent cost of heterogeneity, on top of E30's
queueing-latency coupling — and it points the same way: the pipeline is built
around streams that agree with each other.

**The headline result against the true baseline** (12–60 streams, 60 s windows,
15 s warmup, no MOTION/NVOF frames anywhere — `motion_percent` is 0.000 at every
point):

| streams | baseline | mixed ladder | Δ |
|---|---|---|---|
| 12 | 553 px, rate 1.00 → **0.4241** | 557 px, rate 1.00 → **0.4243** | +0.0001 |
| 24 | 392 px, rate 1.00 → 0.3901 | 512 px, rate 0.68 → 0.4116 | **+0.0215** |
| 36 | 321 px, rate 0.88 → 0.3473 | 479 px, rate 0.50 → 0.3971 | **+0.0498** |
| 48 | 320 px, rate 0.65 → 0.3391 | 431 px, rate 0.38 → 0.3647 | **+0.0256** |
| 60 | 320 px, rate 0.51 → 0.3349 | 397 px, rate 0.27 → 0.3439 | +0.0090 |

At 12 streams the two are identical and undegraded (rate 1.00, ~550 px) — worth
confirming rather than assuming, since it is the control that says the harness
isn't inventing differences.

### E32 — motion-carry frames on the capacity curve: valuable, but priced wrong

Re-measured `skip_mode: motion` (PM shed delivers the frame flagged
`MD_NO_ANALYTICS` so it becomes a MOTION/NVOF carry frame) on top of the mixed
ladder across the full 12–60 range. Worth redoing rather than citing E25: both the
ladder pricing (E30) and the quality table (NaN fix, corrected content mapping)
have changed since.

Live and binding-checked — 12 `PM skip_mode: motion` lines, and `motion_percent`
tracks the shed rate closely:

| streams | shed | carried | coverage | mean batch | lat90 |
|---|---|---|---|---|---|
| 12 | 0.000 | 0.000 | — | 1.21 | 0.101 |
| 24 | 0.354 | 0.351 | 0.99 | 3.66 | 0.812 |
| 36 | 0.549 | 0.546 | 0.99 | 2.72 | 0.944 |
| 48 | 0.692 | 0.594 | 0.86 | 2.24 | 1.447 |
| 60 | 0.877 | 0.705 | 0.80 | 2.71 | 1.723 |

Coverage falls below 1.0 at the top end — beyond ~0.7 shed the decoder cannot
deliver every shed frame, so some are lost outright rather than carried. The curve
tool blends the two measured quality tables by that measured coverage.

**On the curve it loses, and loses harder the more it is used:**

| streams | mixed | mixed + carry | Δ |
|---|---|---|---|
| 12 | 557 px, 1.00 → 0.4243 | 539 px, 1.00 → 0.4236 | −0.0006 |
| 24 | 512 px, 0.68 → 0.4116 | 512 px, 0.65 → 0.4134 | **+0.0018** |
| 36 | 479 px, 0.50 → 0.3971 | 429 px, 0.45 → 0.3813 | **−0.0158** |
| 48 | 431 px, 0.38 → 0.3647 | 407 px, 0.31 → 0.3530 | −0.0118 |
| 60 | 397 px, 0.27 → 0.3439 | 320 px, 0.12 → 0.3124 | **−0.0315** |

**The mechanism is self-defeating.** Carrying a shed frame means decoding,
converting, scaling it and running an NVOF hop — so the cost scales with the shed
rate, which is exactly the quantity that makes the carry valuable. At 60 streams
the controller sheds to 0.877 and collapses to 320 px @ rate 0.12 to afford it,
against 397 px @ 0.27 without. The one point where it helps is 24 streams, where
the shed is light enough (0.354) that the bill is small.

**But the carry's VALUE is real — the problem is its price.** At a *fixed*
operating point the carry-on table beats the carry-off table substantially, and
most for exactly the content you would expect (crowded scenes, where a coasting
track has neighbours to be confused with):

| content | 416 @ 0.5 | 416 @ 1/3 | 320 @ 0.5 | 320 @ 1/3 |
|---|---|---|---|---|
| cctv_dense | +0.0133 | +0.0232 | +0.0059 | **+0.0466** |
| handheld_crowd | +0.0134 | **+0.0292** | +0.0206 | +0.0141 |
| dashcam_jaad | +0.0205 | +0.0273 | +0.0112 | +0.0202 |
| office_indoor | −0.0004 | +0.0187 | −0.0030 | **+0.0456** |
| bodycam | +0.0033 | +0.0074 | +0.0033 | +0.0324 |
| doorway | +0.0056 | +0.0080 | −0.0064 | +0.0124 |
| cctv_static | −0.0024 | −0.0032 | +0.0042 | +0.0024 |
| movie | −0.0127 | −0.0207 | −0.0031 | +0.0080 |
| **ALL** | +0.0052 | +0.0033 | +0.0020 | +0.0139 |

The benefit also grows as the rate falls (1/3 beats 1/2 almost everywhere) —
there is more gap for the flow to bridge. So this is a genuinely useful mechanism
being defeated by its implementation cost, not a bad idea.

**Where the price actually goes, and the untested fix.** NVDEC is not the
expensive part — `decode_benchmark` gives 820 fps for 720p h265, about 1.2 ms per
frame. The ~7.4 ms per carry frame is the decode *job*: surface map, NV12→YUV420
convert, full-size scale, plumbing. At 36 streams that is 0.549 shed × 36 × 5 fps
× 7.4 ms ≈ 0.73 core-seconds per second spent to obtain flow.

If the carry ran NVOF directly on the decoder's NV12 surface at reduced scale —
skipping the YUV420 conversion and the full-resolution scale, neither of which a
flow hop needs — the per-frame price should fall to roughly 1–2 ms, and 0.73
core-s/s would become ~0.15. That is the difference between the carry paying for
itself and not. **Untested**, and it is the one remaining lever in this campaign
with a plausible path to a real win.

Until then: `skip_mode` stays defaulted to `drop`, and the banked
recommendation remains the mixed ladder alone.

### E33 — **RETRACTION ×2 of E32's cost attribution.** The carry's cost is an OFA lease queue

MB rejected E32's explanation on the grounds that convert and scale should be
trivial compared with everything else. Correct, and I was wrong twice — both times
by attributing a cost instead of measuring it.

**Wrong claim 1**: "~7.4 ms per carry frame is the decode job — surface map,
NV12→YUV420 convert, full-size scale, plumbing." The 7.4 ms came from a decode-job
wall-clock figure that includes waiting, the same error retracted in E18/E19.

**Wrong claim 2**: having instrumented the pipeline and found the flow hop was
17.8 ms, I then asserted the residue was the `image_scale`/`image_convert` *inside*
`nvof_execute` queueing behind detector GPU work. Also inferred, also wrong.

**Measured.** Added per-stage accumulators to the MOTION branch and to
`nvof_execute` (`motion_frames:` and `nvof_execute:` lines in `rt_benchmark`), plus
a new `nvof_benchmark` app for isolated OFA characterisation. At 36 streams:

| stage | pool = 2 (default) | pool = 8 |
|---|---|---|
| scale (to flow size) | 0.288 ms | 0.460 ms |
| convert (to NV12) | 0.324 ms | 0.631 ms |
| **OFA lease wait** | **8.248 ms** | **0.001 ms** |
| flow submit + sync | 8.162 ms | 12.214 ms |
| total per flow call | 17.02 ms | 13.31 ms |

Scale plus convert is **about 1 ms** — exactly as MB said, and nowhere near the
~10 ms I attributed to it.

**The real cost was the OFA engine pool, which defaults to 2**
(`NVOF_POOL_DEFAULT_SIZE`). `nvof_execute` blocks in `pool_acquire()`, and at 36
streams that wait was 8.2 ms per call — half the total. Raising the pool removes it
entirely.

**Isolated OFA capacity is not the constraint** (`nvof_benchmark`, synthetic
720p input):

| flow size | 1 session | 2 | 4 | 8 |
|---|---|---|---|---|
| 640×384 | 161 calls/s (5.97 ms) | 390 | 922 | 1030 (7.48 ms) |
| 512×288 | 234 calls/s (4.27 ms) | 456 | 1040 | 1169 (6.60 ms) |
| 320×192 | 318 calls/s (3.13 ms) | 711 | 1267 | 1342 (5.79 ms) |

OFA sustains >1000 flow calls/s at 640×384. The pipeline asks for ~240/s at 36
streams (carry frames plus the analytics frames' CMC flow), i.e. under a quarter of
capacity. So the hardware was never the limit — the lease queue in front of it was.

**This is not carry-specific.** Analytics frames use OFA for camera-motion
compensation, so the default pool costs the no-carry configurations too:

| config | pool | lease wait | lat50 | shed |
|---|---|---|---|---|
| baseline | 2 | 5.72 ms | 0.631 | 0.117 |
| baseline | 8 | 0.010 ms | 0.549 | 0.108 |
| mixed | 2 | 3.38 ms | 0.248 | 0.502 |
| mixed | 8 | 0.001 ms | 0.240 | 0.497 |

**Effect on the carry's verdict.** With the pool raised, `skip_mode: motion` goes
from clearly negative to a wash:

| streams | carry − mixed, pool 2 | carry − mixed, pool 8 |
|---|---|---|
| 12 | −0.0006 | −0.0007 |
| 24 | +0.0018 | +0.0018 |
| 36 | −0.0158 | −0.0066 |
| 48 | −0.0118 | **+0.0044** |
| 60 | −0.0315 | −0.0054 |

So E32's conclusion ("valuable but priced wrong") survives in outline, but its
*reason* was wrong and its magnitude was inflated by a fixable configuration
default rather than an intrinsic cost.

**What is left, and what is still only a hypothesis.** The dominant remaining term
is `flow_submit_sync` at 10–16 ms across every configuration, against 3–6 ms for
the same call isolated on an idle machine. At 240 calls/s the isolated data
predicts ~4–5 ms, so OFA queueing alone does not account for it. The leading
hypothesis is that the sync absorbs GPU-queue delay — `image_scale`/`image_convert`
are cheap to *submit* (the ~1 ms above) but the OFA cannot start until that CUDA
work actually executes on a GPU that the detector keeps ~74 % busy. **That is a
hypothesis, not a measurement.** The test is to run `nvof_benchmark` alongside a
GPU-loaded `rt_benchmark` and see whether isolated flow latency inflates.

**Concrete levers now, in order of evidence:**
1. **Raise the OFA pool** — measured, free, helps every configuration. Costs
   memory: each engine holds an OFA payload and MV buffer set, so 8 engines is 4×
   the buffers of the current default. Wants a deliberate default change rather
   than my unilateral one.
2. **Lower the flow working resolution.** MB's suggestion, and the isolated
   numbers support it: 640×384 → 320×192 takes a call from 5.97 ms to 3.13 ms and
   raises ceiling throughput 30 %. Flow is used only to carry a box anchor, so the
   quality cost is plausibly ~nil — untested, and cheap to test on the eval grid.
3. Skip the redundant NV12→YUV420→NV12 round trip (the decoder gives NV12,
   `nvof_execute` wants NV12, and we pass YUV420 between them). Worth ~1 ms of
   submit plus whatever GPU execution it forces the sync to wait for — small, but
   it is on the contended path.

### E34 — flow resolution: **correction**, then measured — a real but marginal lever

**Correction first.** E33 listed "lower the flow working resolution" as an
untested lever, and I then claimed the pipeline was already at 320 because
`motion_track.max_width` defaults to 320. Wrong on two counts: the config block is
`motiontrack:` (no underscore), so I was reading an absent key, and MB was right
that it is set to **512**. Added the actual size to the instrumentation output
rather than reasoning about it — the pipeline reports `flow_size=512x288`. Assume
nothing about a config value that a log line can state.

**GPU contention on flow latency — hypothesis from E33, now measured.** Running
`nvof_benchmark` against an idle machine and then alongside 36 detector streams, at
a fixed 320×192:

| | 1 session | 4 sessions |
|---|---|---|
| idle GPU | 3.60 ms, 264 calls/s | 3.24 ms, 1211 calls/s |
| GPU ~74 % busy | **5.22 ms**, 182 calls/s | **7.80 ms**, 505 calls/s |

So detector load roughly doubles flow-call latency and halves flow throughput,
even though OFA is separate silicon: the flow cannot start until the CUDA
scale/convert feeding it actually executes. The pipeline's 12.4 ms at 512×288 is
fully accounted for by size plus this contention — no unexplained residue left.

**The lever, measured.** Added `motiontrack.of_max_width` / `of_max_height`,
defaulting to the motion-image size so unset is exactly the previous behaviour.
This matters for isolation: `max_width` sizes the motion mask and detector ROI as
well as the flow, so changing it wholesale would confound a flow experiment with a
motion-detection change. At 36 streams, pool 8:

| config | flow size | flow_submit_sync | total/call | eff res | shed | lat50 |
|---|---|---|---|---|---|---|
| carry | 512×288 | 12.12 ms | 13.12 ms | 454 px | 0.526 | 0.319 |
| carry | 320×180 | **10.12 ms** | 11.35 ms | 449 px | 0.513 | 0.304 |
| mixed (no carry) | 512×288 | 16.21 ms | 17.81 ms | 481 px | 0.498 | 0.249 |
| mixed (no carry) | 320×180 | **14.61 ms** | 16.55 ms | 479 px | 0.497 | 0.249 |

**A flow call gets 10–16 % cheaper and the operating point does not move.** Effective
resolution and shed rate are flat to within run-to-run noise in both configs.

**Why, and it is the useful part of this result.** Flow is not the binding
constraint. OFA runs at ~25 % of its isolated capacity, and the detector owns the
GPU. What flow costs the pipeline is **worker-thread occupancy** — a thread is held
for the duration of the call — which is why the pool fix mattered so much (it
removed 8.2 ms of pure blocking per call, and the batch went 2.68 → 4.77 with
lat50 halving) and why shaving 2 ms off an unblocked call does almost nothing.

So the ordered verdict on flow cost:
1. **OFA lease wait at the default pool of 2 was the real problem** — 8.2 ms of
   blocking per call, fixed by raising the pool, and it helps every configuration
   including those with no carry frames at all.
2. **Flow resolution is real but marginal.** 512→320 is a 10–16 % cost reduction
   worth ~nothing on the curve. The knob is committed and defaulted off; since it
   buys no measurable capacity there is no reason to accept even a small
   motion-quality risk by enabling it, and no eval was spent on it.
3. Convert/scale is ~1 ms and never mattered — as MB said twice.

Net position on the carry after all of this: a wash (+0.002 to −0.007 across
24–60 streams with the pool raised), and the mixed ladder alone remains the banked
recommendation.

### E35 — more worker threads: **negative**, and my third wrong mechanism retracted

MB's objection: the flow hop is a latency, not GPU occupancy — OFA is separate
silicon sitting at ~25 % utilisation — so a blocking wait should be fixable with
more worker threads rather than accepted as a cost.

The arithmetic supported it. `work_queue_init` binds every stream's queues to the
SHARED `tss->thread_pool`, `auto` resolves to **5 worker threads** on this 6-core
Orin, and at ~240 flow calls/s × 12.4 ms that is ~3 thread-seconds/s of blocking —
apparently 60 % of the pool parked. The no-carry baseline looked like ~22 %.

**Measured at 36 streams, pool 8 — it makes no difference:**

| config | workers | shed | batch | lat50 | eff res |
|---|---|---|---|---|---|
| carry | 5 (default) | 0.527 | 4.68 | 0.332 | 453 px |
| carry | 10 | 0.537 | 4.92 | 0.331 | 447 px |
| carry | 16 | 0.535 | 4.88 | 0.301 | 447 px |
| no carry | 5 | 0.502 | 4.55 | 0.243 | 480 px |
| no carry | 10 | 0.502 | 4.43 | 0.244 | 474 px |
| no carry | 16 | 0.506 | 4.43 | 0.245 | 478 px |

Shed rate flat, effective resolution flat to slightly worse, only latency
marginally better. **So the pool was never starved and "worker-thread occupancy"
was wrong** — the third mechanism I proposed in this thread and the third one a
measurement killed (after the 7.4 ms decode-job attribution and the ~10 ms
internal-conversion inference).

Worth stating why the thread arithmetic misled: a thread blocked in
`vpiStreamSync` is not consuming a *scheduling* slot the pool needs — the pool had
spare capacity, so parking 3 of 5 threads cost nothing measurable. Time-in-a-thread
is not the same as thread starvation, and I conflated them.

**What the evidence now points at, NOT yet verified**: the carry's cost is GPU
occupancy, not latency. Each carry frame adds ~1.1 ms of surface map/copy and
~0.9 ms of scale/convert; at ~91 carry frames/s that is ~0.2–0.3 core-seconds/s of
extra CUDA on a GPU the detector already keeps ~74 % busy. This is consistent with
threads changing nothing — contention for one GPU is not a threading problem — but
it is a hypothesis and the direct test (comparing GPU-side volume between carry and
no-carry runs, both already instrumented) has not been run.

If that attribution holds, MB's original point sharpens rather than dissolves: the
flow *hop* really is free silicon, and what costs is the **scale/convert feeding
it** — the redundant NV12→YUV420→NV12 round trip (lever 3, untested). Unlike the
flow-resolution lever, removing that takes work off the contended resource instead
of shortening a wait, so it is the only remaining candidate that could make the
carry pay under load.

**Standing conclusions after this thread:**
- The carry is a real quality win at a fixed operating point: E11 (half rate +
  MOTION recovers ~100 % of the moving group, fewer switches, fewer false tracks)
  and the quality table's carry axis (+0.005 to +0.047 by content).
- Charged against stream capacity via PM-shed frames, it is a wash.
- The E11 configuration has **never been load-tested**, because rt_benchmark's
  720p5 input against `min_time_delta_process: 0.18` leaves no decimated frames for
  the carry to occupy. Testing goal (b) properly needs higher-framerate input.
- The one unambiguous free win from this thread is raising the OFA engine pool
  from its default of 2, which helps every configuration including those with no
  carry frames at all.

### E36 — OFA pool default raised to 8 (the one unambiguous win), and the NV12 question

**Done.** `NVOF_POOL_DEFAULT_SIZE` 2 → 8. Verified on device with no override flag:
`lease_wait` 0.001 ms, against 3.38 ms for the same config at the old default.
692/692 tests pass.

Sized on utilisation rather than on the smallest value that helped: at 36 streams
the pipeline issues ~240 flow calls/s at ~14.5 ms, i.e. ~3.5 engine-seconds/s —
~87 % of a 4-engine pool but ~44 % of 8. A pool of 4 captured most of the win at 36
streams and would re-saturate at higher stream counts. An engine costs a VPI
stream, a CUDA stream and per-size MV buffers (~37 KB at 512×288), well under 1 MB.

This helps every configuration, including those with no carry frames, because
analytics frames run flow for camera-motion compensation.

**Does a small-NV12 decode path for skipped frames help? The pieces all exist.**
MB's proposal: have the decoder emit a ≤512 NV12 image for frames that will only
feed flow, instead of the current NV12 → YUV420(full size) → scale → NV12 route.
MB doubted a native NV12 scaler existed. It does:

- `nvSurfToImageNV12Device()` — decoder surface → NV12 device image.
- `image_scale()` on `IMAGE_FORMAT_NV12_DEVICE` → `image_scale_nv12_device()` →
  the `ResizeNv12` CUDA kernel (`src/cuda/cuda_kernels_nvidia.cu:96`). Native on
  the device backends; only the Apple/host path falls back to a YUV420
  intermediate.
- `nvof_execute` converts its input to NV12 internally, so feeding NV12 makes that
  conversion a no-op.

So the change is assembly of existing parts, and it removes work from the
*contended* device rather than shortening a wait — structurally the right shape,
unlike the flow-resolution lever.

**But I cannot currently predict it will move the curve, and I should say so.**
The GPU-occupancy hypothesis from E35 is **not supported** by the data available.
Comparing the same 36-stream runs with and without carry:

| | surface copies | copy_y per call | total copy_y |
|---|---|---|---|
| carry | 13112 | 2.68 ms | 35.1 s |
| no carry | 6532 | 8.32 ms | 54.3 s |

The carry run performs **twice as many** surface copies for **less** total copy
time, and per-call cost varies 3× between configurations. Those counters therefore
include contention and waiting, so they cannot measure occupancy — and they
certainly do not show the carry adding GPU work.

That makes four proposed mechanisms for the carry's residual cost, none of which
survived: the decode-job attribution (E33), the internal-conversion inference
(E33), worker-thread occupancy (E35), and now GPU occupancy. **The mechanism is
unidentified.** Building the small-NV12 path is defensible on first principles, but
after this thread I will not claim a predicted gain for it without an instrument
that can actually attribute cost on the contended device — clean GPU-time
attribution (CUDA events per stage, or nvidia-smi/tegrastats sampling correlated
with config) is the missing tool, and it is the prerequisite for the next honest
attempt.

### E37 — mechanism FOUND, with an instrument: the flow hop occupies the ordered per-stream pipeline

Four earlier explanations for the carry's cost were wrong because every counter on
the contended path mixes work with waiting. Built the missing instrument
(`src/gpu_attrib.py` over `nsys cuda_gpu_kern_sum`, kernels bucketed by the actual
names in `src/cuda/*.cu` and the TRT/NPP families) and then isolated the mechanism
by config alone.

**GPU attribution, 36 streams, carry vs no carry (GPU seconds per wall second):**

| bucket | no carry | carry | Δ |
|---|---|---|---|
| detector | 0.774 | 0.663 | −0.111 |
| npp | 0.017 | 0.025 | +0.008 |
| image_convert | 0.004 | 0.009 | +0.005 |
| motion_detect | 0.003 | 0.003 | −0.000 |
| **total** | **0.810** | **0.710** | **−0.100** |

The carry adds **0.013 GPU s/s — 1.3 % of one GPU** — and the run uses *less* GPU
overall because the controller sheds more. So it is not a GPU-occupancy cost. That
is the fifth failed mechanism (E33 ×2, E35, E36) and the first one killed by a
real instrument rather than by argument.

**MB's correction, which located the right time.** I said the controller "regulates
latency, so more frames means more shedding". MB pointed out nothing counts queue
lengths — the signals are times. Correct, and the code says so: `pm_sig` is
`max(RT inference latency / skip_target_sec, tail_upstream_lat / (0.6 ×
max_analytics_latency_s), shed backstop)`. Also worth noting my next guess was
wrong too — MOTION frames do *not* enter `h_pipeline_latency`, because line 948
samples `ts->inference_image`, set only on the analytics path.

The time that actually grows is **`decode_age`**: `tail_upstream_lat` is built from
the decode-dequeue age, and `MAIN_PIPELINE` backpressures `H26X_DECODE` with a
queue length of 3. A carry frame holds the stream's ordered main-pipeline slot for
~14.5 ms, that backpressure reaches the decode queue, decode_age rises, and the
controller sheds. Time, not depth — exactly as MB said.

**Isolated by config, no code change:** `skip_mode: motion` *without*
`min_time_delta_motion` delivers the shed frame and marks it no-analytics, but the
MOTION branch requires the motion clock, so the frame falls through to a plain
SKIP_FRAMERATE — extra frames through the pipeline, no flow hop.

| config | shed frames | flow hop | FSkip | eff res | lat50 |
|---|---|---|---|---|---|
| drop at decoder (baseline) | dropped | — | 0.497 | 478 px | 0.246 |
| **delivered, no flow hop** | delivered | **no** | **0.497** | **474 px** | **0.249** |
| delivered + flow hop | delivered | yes | 0.522 | 455 px | 0.327 |

**Not dropping the frames is free.** Identical shed rate, resolution within noise,
latency unchanged. **The flow hop is the entire cost**, and it is neither GPU nor
OFA capacity (OFA runs at ~25 %) — it is holding an ordered per-stream pipeline
slot.

This also explains E35's null: adding threads to the shared pool cannot remove
per-stream serialisation of `MAIN_PIPELINE`, so the flow hop keeps its slot however
many threads exist.

**The fix this implies** (not built): run the flow hop off the ordered main
pipeline — its own queue, or async submit with the OF anchor consumed on the next
analytics frame — so a carry frame does not hold the slot that backpressures
decode. If the carry then costs what "delivered, no flow hop" costs (nothing), its
measured quality benefit lands: +0.005 to +0.047 by content type, biggest on dense
and crowded scenes, and the E11 half-rate result becomes available under load.
That is the first candidate in this campaign with an identified mechanism, a
measured cost, and a predicted gain that follows from both.

### E38 — relaxing the latency target: **negative**, and E37's mechanism retracted

MB asked whether simply relaxing the latency judgement would recover the carry's
operating point. Two measurements answer it, and the first also retracts part of
E37.

**Which control term actually binds** (`rt_benchmark -L`, `PM_TICK` rows, warmup
third dropped). Note the knob had to be set on the command line: `performance.pm_log`
in the yaml is silently overwritten by rt_benchmark's own default at line 741 — the
same trap as `skip_target_sec`, and caught only because the row count was zero.

| | pm_signal | tail_lat | e2e term (tail/3.0 s) | decode shedding |
|---|---|---|---|---|
| no carry | 0.660 | 0.436 s | 0.145 | 0 % of ticks |
| carry | 0.753 | 0.600 s | 0.200 | 0 % of ticks |

**The RT-inference term binds 100 % of ticks.** `decode_age` does rise with the
carry (0.436 → 0.600 s) but the e2e term sits at 0.15–0.20 against a binding
0.66–0.75, and the decode shed never fires. **So E37's mechanism — carry frames
backpressuring the decode queue and raising decode_age — is retracted.** It was
built from reading the backpressure wiring, not from the controller's own trace.
The binding quantity is `qlat_rt` (detector enqueue→dequeue): 0.198 s without carry,
0.226 s with.

**Relaxing `skip_target_sec` (the knob that gates the binding term) does nothing
useful:**

| config | target | FSkip | eff res | lat50 | tail_lat | pm_signal |
|---|---|---|---|---|---|---|
| mixed | 0.30 | 0.498 | 481 px | 0.245 | 0.428 | 0.689 |
| mixed | 0.45 | 0.492 | 474 px | 0.449 | 1.017 | 0.694 |
| mixed | 0.60 | 0.487 | 477 px | **0.948** | **2.038** | 0.684 |
| carry | 0.30 | 0.524 | 452 px | 0.328 | 0.569 | 0.763 |
| carry | 0.45 | 0.534 | 452 px | 0.620 | 2.118 | 0.751 |
| carry | 0.60 | 0.553 | 455 px | 0.792 | 2.178 | 0.727 |

**`pm_signal` returns to ~0.69 whatever the target is.** Raising the target raises
the tolerated `qlat_rt` proportionally, queues grow to fill it, and the loop settles
at the same signal — so shed rate and resolution are unchanged while lat50 gets
4× worse and tail latency 5×. Relaxing the judgement buys latency and nothing else.

**The general lesson, for the third time this campaign**: the box is
*compute*-limited, and the latency target only decides how much queueing is
tolerated before shedding begins. At equilibrium the shed rate is set by the compute
deficit, not by the target. Anything that does not change the compute/quality
exchange (as the mixed ladder does) or genuinely remove work will be absorbed by
the loop. Same shape as the E22 retraction (`gpu_s` pinned) and E30 (ladder
equilibria).

**Where the carry stands.** Its cost is real — the isolation in E37 holds (frames
delivered without a flow hop are free; with the hop they cost) — but the mechanism
is again unidentified, since it is neither GPU (1.3 %, E37), nor OFA capacity
(~25 % utilised), nor decode_age, nor thread starvation, nor the latency target. The
one candidate left standing is that the flow hop de-phases each stream's detection
submissions, making arrivals at the detector burstier and so raising `qlat_rt` at
unchanged mean rate — consistent with the batch growing (4.48 → 4.72) alongside the
latency. **That is a hypothesis and I am not going to act on it without measuring
it**; the test is the detection-job inter-arrival distribution, which
`UBON_INFER_TRACE_PATH` already records.

### E39 — PRE-REGISTERED: what does the flow hop cost? (written before the measurement)

MB's diagnosis of this campaign's failure mode: I have been running an experiment,
observing the result, and then inventing an explanation that fits. That is
unfalsifiable by construction, and it produced six mechanisms of which five died and
one contradicted my own data. The fix is to state the hypothesis and its predicted
observable BEFORE the run. This entry is committed before the measurement it
describes completes, so the ordering is checkable in the git history.

**Question.** A carry frame costs ~0.02 of objective at 36 streams (E32/E33). It is
NOT: GPU work (measured +0.013 GPU s/s, 1.3 %, E37), OFA capacity (~25 % utilised,
E33), the OFA lease queue (fixed, E36), worker-thread starvation (E35), decode_age
(never binds, E38), or the latency target (relaxing it changes nothing, E38). What
is left?

**H1 — detector-arrival burstiness.** The flow hop delays each stream's analytics
submission, de-phasing arrivals across streams. At unchanged mean arrival rate, a
burstier arrival process raises queue wait, so `qlat_rt` rises and the controller
sheds.
- *Predicts*: the coefficient of variation of detection-job inter-arrival times is
  materially HIGHER with carry, and rises together with `qlat_rt`.
- *Falsified if*: inter-arrival CV is equal or lower with carry while `qlat_rt`
  still rises.
- *Instrument*: `UBON_INFER_TRACE_PATH` already records per-enqueue timestamps.

**H2 — the GPU is actually saturated and my utilisation figure was wrong.** The
0.810/0.710 GPU s/s in E37 came from dividing total kernel time by an arbitrary 30 s
window without establishing the GPU-active span, so it may understate utilisation
badly. If the GPU is in fact at ~100 % in steady state, then the carry's small extra
GPU work does bite, and "compute-limited" — which I asserted in E38 in flat
contradiction of my own profile — would turn out right for the wrong reasons.
- *Predicts*: tegrastats `GR3D_FREQ` during the measurement window is ≥ 95 % in
  both configs.
- *Falsified if*: steady-state GR3D is materially below saturation (< 85 %).
- *Instrument*: tegrastats at 500 ms, windowed to the contiguous active span's
  middle 60 % (stated method, not a guess; the first attempt at this was
  contaminated by including teardown samples and read 44 %).

**H3 — per-stream pipeline serialisation.** The hop consumes the stream's ordered
main-pipeline slot, limiting that stream's throughput.
- *Predicts*: the cost should scale with hop duration. **Already weakened**: E34 cut
  the hop 12.1 → 10.1 ms (−16 %) by dropping flow resolution and the operating point
  did not move.

H2 is the one the in-flight run tests. Recording the predictions before reading it.

### E40 — the pre-registered tests, and the answer is "I don't know"

Results against the predictions committed in E39, before the data was read.

**H2 (GPU actually saturated) — NOT SUPPORTED.** tegrastats, windowed to the middle
60 % of the contiguous active span as pre-stated:

| | GR3D mean | p50 | p10 | ≥95 % | <85 % |
|---|---|---|---|---|---|
| no carry | 85.5 % | 93 % | 58 % | 49 % of samples | 40 % |
| carry | 83.7 % | 90 % | 60 % | 39 % | 40 % |

Predicted ≥ 95 %; measured 84–86 % mean with 40 % of samples below 85 %. The GPU is
*near* saturation much of the time and has real idle gaps — so E38's
"compute-limited" was wrong as stated, and "not compute-limited" would be equally
sloppy. The carry config uses *less* GPU (83.7 vs 85.5), consistent with E37.

**H1 (detector-arrival burstiness) — FALSIFIED.** From `UBON_INFER_TRACE_PATH`,
middle 60 % of enqueues:

| | mean inter-arrival | CV | p50 gap | enqueues/s |
|---|---|---|---|---|
| no carry | 13.26 ms | **1.486** | 5.61 ms | 75.4 |
| carry | 13.93 ms | **1.158** | 9.03 ms | 71.8 |

Predicted materially higher CV with carry. Measured **lower** — 1.158 against 1.486.
Falsified by the pre-registered criterion.

**H3 (per-stream serialisation)** stays weakened by E34 and untested further.

**The effect is real, and it is not noise.** Across seven-plus runs of each config at
36 streams with the pool fixed, the separation is consistent: carry settles at
452–457 px effective resolution with shed 0.520–0.527, no-carry at 478–486 px with
shed 0.495–0.502. So there is something to explain.

**And the shape of it is genuinely counter-intuitive**, stated as observations only:
the carry configuration issues *fewer* detector jobs per second (71.8 vs 75.4), uses
*less* GPU (83.7 % vs 85.5 %), and has a *less* bursty arrival process (CV 1.158 vs
1.486) — yet the controller sheds more and holds a lower resolution. Its binding
signal `qlat_rt` is higher (0.226 s vs 0.198 s, E38) despite all three of those
moving the other way.

**I do not know why.** Three specific candidates are now eliminated by
pre-registered tests rather than replaced by a seventh story, which is the useful
part: the space is narrower and the record is honest. The next hypothesis should be
about detector *service* time or batch formation rather than arrival or resource
volume — but it needs pre-registering with a prediction, and an instrument that can
separate "waiting for a batch to form" from "waiting behind an in-flight inference",
which does not exist yet.

### E41 — PRE-REGISTERED: defer-and-batch the flow chain (MB's proposal)

MB's proposal: on a skipped frame do *only* the async scale→convert to a small
NV12 and queue it; run the actual NVOF work on the next NON-skipped frame,
draining all queued frames then. Written before implementing or measuring.

**Why the measurements support trying it.** `run_flow` issues three hardware
submits and then one blocking sync per call:
`vpiSubmitConvertImageFormat` (VIC, NV12 pitch→block-linear) →
`vpiSubmitOpticalFlowDense` (OFA) → `vpiSubmitConvertImageFormat` (VIC, MV
block→pitch-linear) → `vpiStreamSync`. So every carry frame pays a full
round-trip wait through queued CUDA→VIC→OFA→VIC work.

That wait is measurably not engine capacity: isolated, 8 sessions sustain 1169
calls/s at 6.6 ms each (E33) while the pipeline asks ~240 calls/s and sees
12.4 ms (E38); and detector load alone roughly doubles flow latency at fixed size
(3.60 → 5.22 ms, 1 session, E34). Round-trip wait that K frames could share is
therefore the one cost structure consistent with all the measurements — unlike
the six mechanisms already killed.

**Semantically free, which is the part worth checking first.** A hop's output is
not consumed until the next analytics frame: `utrack_motion_frame` only updates
`of_anchor_box`, and `utrack_predict_positions` reads it at analytics time. Running
the chain in order immediately before that frame yields the same warp sequence and
the same final anchor. Retained frames cost ~221 KB each at 512×288 NV12 (~16 MB
across 36 streams at two deferred frames each).

**H4.** The per-call cost is dominated by the round-trip sync through queued
engines, not by per-frame engine work, so K hops sharing one sync cost far less
than K separate syncs.
- *Predicts (a)*: in isolation, submitting K flows per sync gives per-flow time
  falling materially with K — approaching engine time (~4.3 ms at 512×288) rather
  than the 6.6–12.4 ms round-trip figure.
- *Predicts (b)*: in the pipeline, the operating point moves from the current carry
  point (452–457 px, shed 0.520–0.527) toward the "delivered, no flow hop" point
  (478–486 px, shed 0.495–0.502, E37) — i.e. most of the carry's cost disappears.
- *Falsified if*: per-flow time falls with K but the operating point stays at
  ~454 px / 0.525. That would mean the carry's cost was never the blocking, and
  would also contradict E37's isolation, so it is a genuine test of both.

**Staging, so the cheap test comes first.** Prediction (a) needs only a chained
nvof API (acquire once, submit K, sync once, K MV buffers) plus a `--chain K` option
in `nvof_benchmark`. If (a) fails, stop — no pipeline change. Only if (a) holds is
the `track_stream` work (queue scaled frames on skip, drain on the next analytics
frame) worth doing, and (b) then tests it end to end.

### E42 — H4 stage (a): chaining buys ~1 ms alone and nothing under concurrency. STOP.

Built the chained path (`nvof_execute_chain`: acquire one engine, submit n hops,
sync once) plus `nvof_benchmark --chain K`, and ran E41's pre-registered
prediction (a) at the pipeline's real flow size.

| hops per sync | 1 session | 4 sessions |
|---|---|---|
| K = 1 | 4.84 ms/flow, 199 calls/s | 3.65 ms, 1077 calls/s |
| K = 2 | 4.00 ms, 239 calls/s | 3.67 ms |
| K = 4 | 3.89 ms, 245 calls/s | 3.81 ms |
| K = 8 | 3.81 ms, 250 calls/s | **4.39 ms**, 885 calls/s |

At one session chaining saves ~1 ms (21 %), saturating by K = 2. **Under
concurrency it saves nothing, and at K = 8 it is worse.** The pipeline runs 36
streams — squarely in the regime where the effect disappears. The mechanism is
visible: with several sessions in flight the engines are already kept busy, so
there is no idle round trip left for a chain to fill, and a long chain just adds
head-of-line delay.

**Also, a flaw in the test I designed, worth recording.** Prediction (a) was framed
as "per-flow time should fall toward engine time (~4.3 ms) rather than the
6.6–12.4 ms round-trip figure". But the isolated K = 1 baseline is **already**
4.84 ms — the isolated harness never reproduced the in-pipeline 12.4 ms at all. So
the premise ("there is a large shareable round trip") was never exhibited in the
environment I chose to test it in, and this experiment could not have confirmed or
refuted it. Pre-registering caught that the result did not support the change; it
did not stop me choosing a harness that could not see the effect.

**Decision, per the staging committed in E41: stop. No `track_stream` change.**
The chained API stays as a documented timing instrument (default unused, other
backends fall back to sequential) rather than being reverted, so the measurement
is reproducible.

**What remains unexplained**: the in-pipeline flow hop costs 12.4 ms where the same
call costs 3.6–4.8 ms isolated, and detector load only accounts for part of that
(E34: 3.60 → 5.22 ms at 320×192, 1 session). Nothing in isolation reproduces the
full gap, which is why five mechanisms have died against it. The next attempt needs
a harness that reproduces the in-pipeline cost — otherwise it will keep testing a
premise the harness cannot exhibit.

### E43 — PRE-REGISTERED: is the in-pipeline cost engine size-set churn? (MB's question)

MB asks whether the unexplained in-pipeline flow cost is "something dumb like engine
size changes". Written before instrumenting or measuring.

A size miss is expensive by inspection: `engine_ensure_size` evicts the LRU slot and
then allocates two `vpiImageCreate` buffers plus a `vpiCreateOpticalFlowDense`
payload. `pool_acquire` prefers a free engine already holding (w,h) but will hand
back a wrong-sized one rather than block, and each engine caches only
`NVOF_ENGINE_SIZE_CACHE` size sets. The isolated benchmark uses ONE size and few
sessions, so it would never show this — which is exactly the shape of a cost that
appears in the pipeline and not on the bench.

**H5.** A material fraction of in-pipeline `nvof_execute` calls miss the engine
size cache and rebuild the OFA payload and MV buffers, and that rebuild is a
significant share of the 12.4 ms.
- *Predicts*: rebuilds occur on a non-trivial fraction of calls in the pipeline
  (say > 5 %), and are ~0 % in the isolated benchmark at a single size.
- *Falsified if*: rebuilds are ~0 % of pipeline calls too — in which case the size
  cache is working and the cost is elsewhere.
- *Instrument*: counters on size-cache hit / miss-rebuild, and on `pool_acquire`
  returning an engine that lacks the requested size, reported alongside the existing
  per-stage `nvof_execute:` timing line.

Note the reasoning that says H5 should be FALSE — all streams are 720p, both the CMC
and carry paths derive their flow size from `motiontrack.max_width`, so every request
should be 512×288 — is exactly the kind of reasoning that has been wrong six times in
this campaign. Measuring instead.

### E44 — H5 FALSIFIED (size churn is not it), and a harness that reproduces the magnitude

**H5 falsified.** Instrumented the size cache and ran the carry config at 36 streams:

```
nvof_sizes: cache_hit=17934  build=5 (0.03% of lookups)
            wrong_size_lease=5  build_time=13.8 ms TOTAL (0.001 ms/call)
```

Predicted > 5 % rebuilds; measured **0.03 %**, and those 5 builds are the pool
warming up (5 engines created during the run, hence 5 wrong-size leases too). Size
churn costs 13.8 ms across the entire 75-second run. Cheap to check, worth checking,
and not the answer.

**A harness that finally reproduces the magnitude.** The gap all along has been that
nothing isolated showed the pipeline's ~12.3 ms. Sweeping concurrent sessions does:

| sessions | 4 | 8 | 16 | 32 |
|---|---|---|---|---|
| per-call | 4.60 ms | 6.72 ms | **14.00 ms** | 28.37 ms |
| aggregate | 815 calls/s | 1149 | 1097 | 1067 |

Aggregate throughput plateaus at ~1100 calls/s from 8 sessions on, and past that
point per-call latency grows linearly with concurrency — a saturated server with
closed-loop clients, exactly as Little's law requires. The pipeline's 12.3 ms sits
between the 8- and 16-session rows.

**But it does not close the gap, and saying so is the point.** Little's law applied
to the pipeline's own numbers (17939 calls over 75 s at 13.27 ms) gives **~3.2 flow
calls in flight** — and at that concurrency the bench reads 4.6 ms, not 12.3 ms. So
the pipeline is ~2.7× slower per call than the isolated harness at equivalent
in-flight concurrency. Detector load accounts for roughly a factor of 2 at fixed size
(E34), which is the right order but was measured at a different size and session
count.

What this does give, for the first time, is a bench that exhibits the phenomenon at
all — the prerequisite E42 identified. The next step is to measure in-flight flow
concurrency in the pipeline directly (a counter around the acquire/release pair)
rather than inferring it from Little's law, and to run the bench at matched
concurrency under matched GPU load. No mechanism is proposed here.

### E45 — PRE-REGISTERED: is the pipeline's flow latency just concurrency + GPU load?

E44 left a ~2.7× gap: the pipeline sees 12.3 ms per flow call, while the isolated
bench at the concurrency Little's law implies (~3.2 in flight) reads 4.6 ms. Both
inputs to that comparison were inferred rather than measured — the concurrency came
from Little's law, and the bench ran on an idle GPU while the pipeline does not.
Measuring both directly. Written before the runs.

**H6.** The pipeline's per-call flow latency is fully accounted for by its actual
in-flight concurrency plus concurrent GPU load; there is no additional pipeline-only
cost.
- *Predicts (i)*: measured in-flight concurrency is close to the Little's-law value,
  ~3–5.
- *Predicts (ii)*: the bench at that matched concurrency, run against a loaded GPU,
  reads close to the pipeline's 12.3 ms (within ~25 %), where idle it reads 4.6 ms.
- *Falsified if*: matched-concurrency, matched-load bench stays materially below the
  pipeline (< 8 ms) — leaving a pipeline-only cost still unexplained.
- *Also informative*: if measured concurrency is much HIGHER than 3–5 (say 10–16),
  then concurrency alone explains the latency and my Little's-law arithmetic was
  wrong, because the completed-call rate I used spanned non-steady-state.

Instrument: an in-flight counter incremented on entry to the engine phase of
`nvof_execute` and decremented on exit, reporting the concurrency an arriving call
sees (itself included) and the maximum.

### E46 — H6: concurrency measured (prediction i confirmed), but H6 FALSIFIED

**(i) confirmed.** In-flight flow concurrency, measured with a counter rather than
inferred: **mean_seen 3.17, max 5** at 36 streams with carry. Little's law's 3.2 was
right, so E44's arithmetic stands.

**(ii) falsified, and the confound matters.** The loaded bench runs *alongside* the
36-stream pipeline and shares the same OFA, so its total concurrency is its own
sessions PLUS the pipeline's 3.17:

| bench sessions | total concurrency | bench per-call | |
|---|---|---|---|
| 2 | ~5.2 | 6.93 ms | idle equivalent 5.08 ms |
| 4 | ~7.2 | 9.19 ms | idle 3.76 ms |
| 6 | ~9.2 | 11.18 ms | idle 4.87 ms |
| **pipeline itself** | **3.17** | **12.10 ms** | |

**The bench at higher concurrency, sharing the same loaded GPU, is still ~40 %
faster than the pipeline.** At total concurrency 5.2 it reads 6.93 ms where the
pipeline at 3.17 reads 12.10 ms. So concurrency and GPU load together do not account
for the pipeline's flow latency; something specific to the pipeline's own flow calls
is roughly 2× slower under equal-or-worse conditions.

GPU load is nonetheless confirmed as a real term: at fixed bench concurrency,
loading the GPU costs 5.08 → 6.93 ms (2 sessions) and 3.76 → 9.19 ms (4 sessions),
and bench throughput collapses from 1042 to 429 calls/s at 4 sessions.

**No mechanism proposed.** What is now pinned, all measured: the flow hop costs
12.1–12.7 ms in the pipeline; ~1 ms of that is scale+convert; the OFA lease wait is
~0 with the pool at 8; size-set churn is 0.03 % of lookups; in-flight concurrency is
3.17; and an isolated call at that concurrency, on the same loaded GPU, takes about
7 ms. The residual is ~5 ms per call and it is pipeline-specific.

**Process note.** Part (i) was first run against a stale Jetson binary — I edited and
built on the desktop, then ran on the Jetson without syncing, and the missing
`nvof_inflight` line is the only reason I noticed. The same class of error as the
config knobs that were parsed but never bound (E24, E38's `pm_log`). Sync-then-verify
belongs in the loop, not in a checklist I re-derive each time.

---

## §12 — State of play, and how to pick this up again

Written as a handover. Everything below is measured unless explicitly marked
otherwise. Branch `pm_opt` in **both** repos (`track` and `ubon_cstuff`); nothing
pushed; `main` in both is back at `origin/main`.

### 12.0 Headline (updated after E47–E57)

The flow hop's in-pipeline cost was **our own per-frame VPI wrapper work**, not
anything intrinsic. Writing the flow input through `vpiImageLockData` instead of
wrapping user memory takes a flow call from 13.2 ms to 6.3 ms (p99 43 → 14), and
that flips `skip_mode: motion` from a wash into a consistent win. Current best
configuration — mixed ladder WITH carry — against a true baseline:

| streams | 12 | 24 | 36 | 48 | 60 |
|---|---|---|---|---|---|
| Δ quality | +0.0001 | **+0.0173** | **+0.0515** | **+0.0329** | **+0.0142** |

Correctness of the flow path is now gated on-device by `nvof_benchmark --verify`;
before E56 there was no such check and every optimisation was judged on timing alone.

### 12.0a What actually landed

This log records the whole campaign, including the ideas that failed — deliberately,
because the failures are most of the evidence. Only a subset was ported to the clean
branch. **Landed**: the mixed degradation ladder (cost-parameterised), the MOTION
frame class and OF-anchor carry, `skip_mode: motion`, the OFA pool default, the
`vpiImageLockData` flow input path, the four detector-free crash fixes,
`detection_max_size`, `debug_analytics_mask`, `nvof_benchmark` (with `--verify`),
and the `rt_benchmark` motion column and CSV-header fix.

**Not ported**, each measured null or negative and each documented above:
`predict_on_motion_skip` (E6), `crowding_adapt_max` (E14), the RT batch linger
(E27), per-stream/hint ladders (E28-E30), `motiontrack.of_max_width` (E34), the
high-priority CUDA stream (E51), `nvof_execute_chain` (E42), and all the
investigation instrumentation (stage/CPU/in-flight timers, size-churn counters,
per-caller distributions, detector-overlap sampling).

### 12.1 Banked — worth shipping

**The mixed degradation ladder** (`performance.degrade_policy: mixed`) is the
result of this campaign. Against a TRUE baseline (no `performance` block at all —
see the trap in 12.6), 12–60 streams, 60 s windows, 15 s warmup, 720p5 h265 on Orin:

| streams | baseline | mixed | Δ |
|---|---|---|---|
| 12 | 553 px, rate 1.00 → 0.4241 | 557 px, 1.00 → 0.4243 | +0.0001 |
| 24 | 392 px, rate 1.00 → 0.3901 | 512 px, 0.68 → 0.4116 | **+0.0215** |
| 36 | 321 px, rate 0.88 → 0.3473 | 479 px, 0.50 → 0.3971 | **+0.0498** |
| 48 | 320 px, rate 0.65 → 0.3391 | 431 px, 0.38 → 0.3647 | **+0.0256** |
| 60 | 320 px, rate 0.51 → 0.3349 | 397 px, 0.27 → 0.3439 | +0.0090 |

Default is unchanged (`resolution_first`); the mixed ladder must be enabled
deliberately. At 60 streams it over-sheds rate (0.27) — a rate floor on the mixed
waypoint path is the obvious untested remedy.

**The OFA engine pool default, 2 → 8.** Already changed and verified. A pool of 2
cost 8.2 ms of pure lease wait per flow call at 36 streams; the new default gives
0.001 ms. Helps every configuration, not just carry ones, because analytics frames
run flow for camera-motion compensation.

**MOTION carry frames are a real quality win at a FIXED operating point** — E11
(half rate + MOTION recovers ~100 % of the moving group, with fewer ID switches and
false tracks) and the quality table's carry axis (+0.005 to +0.047 by content type,
largest on dense/crowded scenes). This is what `min_time_delta_motion` buys when
analytics runs slower than the delivered frame rate *by choice*.

### 12.2 Measured negatives — do not re-try without new information

- ~~`skip_mode: motion` — a wash~~ **SUPERSEDED by E57**: with the flow input path
  fixed it is a consistent win (+0.0025 to +0.0053, growing with load).
- Per-content ladders by `stream_hint` — +0.0012, +0.0005, −0.0023 across three
  parameterisations. Streams share one pressure scalar; the loop absorbs it.
- Relaxing `skip_target_sec` — `pm_signal` returns to ~0.69 at every target. Buys
  latency (lat50 4× worse at 0.60) and nothing else.
- More worker threads (5 → 10 → 16) — shed rate and resolution flat.
- Flow resolution 512 → 320 — cuts a flow call 10–16 %, moves the operating point by
  nothing.
- RT batch linger — works (batch 3.94 → 4.73) and is worth ~2 %; `K_LAUNCH` is only
  ~9 % of detector time.
- Crowding-adaptive and speed-adaptive detector rate — E13/E14, negative.

### 12.3 The open question — **RESOLVED, see E48–E53**

**Answered.** A flow hop costs ~4.5 ms alone and ~12 ms beside a running detector.
The excess is not a per-call penalty: the median matches an isolated bench at the
same load, and the mean is dragged up by stalls on ~17 % of calls (p99 43 ms, max
107 ms). Removing the detector removes both — mean 12.37 → 4.60 ms, p99 43.01 →
5.80, stalls 17.2 % → 0.1 % (E52). And it cannot be scheduled around: at 36 streams
99.8 % of flow calls already coincide with a detector inference, so there is no idle
window to defer into (E53). That is why `skip_mode: motion` is a wash and why
chained submission (E42), CUDA stream priority (E51) and flow resolution (E34) all
measured flat.

The original table of what had been ruled out, kept because it is what narrowed the
search:

| | |
|---|---|
| hop total, in-pipeline | 12.1–12.7 ms |
| scale + convert | ~1 ms |
| OFA lease wait (pool 8) | ~0.001 ms |
| size-set churn | 0.03 % of lookups, 13.8 ms per 75 s run |
| in-flight concurrency | 3.17 mean, 5 max (measured, not inferred) |
| isolated call, same concurrency, same loaded GPU | ~7 ms |
| **residual** | **~5 ms/call, pipeline-specific** |

Ruled out by test: GPU work (+1.3 %), OFA capacity (~25 % utilised), the lease
queue, worker-thread starvation, `decode_age` (never binds — the RT-inference term
binds 100 % of ticks), the latency target, engine size churn, and
concurrency+GPU-load together (the bench at HIGHER concurrency on the same loaded
GPU is still ~40 % faster).

If this residual were removed the carry would plausibly become free, and E37's
isolation says that is where its whole cost lives — but that is an inference, not a
measurement.

### 12.4 Never load-tested, and it matters

The E11 configuration — `min_time_delta_motion` with analytics decimated below the
delivered frame rate — has **never been measured on the capacity curve**, because
`rt_benchmark` runs 720p**5** input against `min_time_delta_process: 0.18`, so
analytics is due on essentially every frame and there are no decimated frames for the
carry to occupy (this is why the early "nvof" sweep arms read `Mot=0.000`, E24).
Testing goal (b) properly needs higher-framerate input — a 720p30 or 720p15 pcap.

### 12.4a Bugs found and fixed (committed)

Detector-free operation was documented as supported (`track_shared.c:783`) but
crashed four different ways; all four are fixed and 692/692 tests pass:
`infer_thread_configure` on a NULL thread; `infer_thread_infer_async_callback`
returning silently and stranding the caller's frame so the work queue could never
drain; `utrack_run` on a NULL detection list; and the analytics path submitting into
a void instead of treating frames as non-analytics. The stranding bug also explains
a teardown crash seen with `min_time_delta_process: 10`. `rt_benchmark` is guarded
too, so a detector-free control run is now possible — which is how E52 was measured.

### 12.5 Tools built (all committed)

| tool | what it does |
|---|---|
| `src/quality_table.py` | builds the quality table from `track.py --eval` JSON rollups, per content type, with a MOTION-carry axis |
| `src/capacity_curve.py` | joins that table with `rt_benchmark --csv` operating points → quality vs streams |
| `src/gpu_attrib.py` | buckets `nsys cuda_gpu_kern_sum` by pipeline stage → GPU seconds per wall second |
| `src/make_rt_configs.py` | regenerates every campaign config from the base tracker yaml |
| `nvof_benchmark` (app) | isolated OFA: latency/throughput vs flow size, session count, hops-per-sync |
| `rt_benchmark` additions | `motion_percent`, `nvof_execute:` per-stage line with flow size, `nvof_sizes:` churn, `nvof_inflight:` concurrency, `--stream-hints`, fixed CSV header |

Data locations (outside git, on /mldata, persistent): quality table
`/mldata/config/track/quality_table.yaml`; eval grids
`/mldata/config/track/search/grid_*.yaml` and `gridm_*.yaml`; eval results
`/mldata/tracking/results/qtab/`. Jetson configs regenerate to `~/rtcfg_pm_opt`
via `make_rt_configs.py` (the originals were in `/tmp` and would not survive a
reboot; the regenerated ones are semantically identical, verified key-by-key).

### 12.6 Traps that cost time here — check these first

1. **`rt_benchmark` overrides the yaml** for `performance.pm_log` (use `-L`),
   `skip_target_sec` (use `-S`) and the OFA pool (`--of-pool-size`). Set in yaml,
   they are silently ignored. Caught only because a row count was zero.
2. **The config block is `motiontrack:`, not `motion_track:`.** Reading the wrong
   key returns the default and looks plausible.
3. **Build on the desktop, run on the Jetson** — they are separate trees and the
   Jetson is on a commit the desktop does not have. Sync the specific files and
   confirm the binary rebuilt, or you measure stale code.
4. A knob that is parsed is not a knob that is live: require a log line or a counter
   (E24 burned a whole 4-arm sweep on a feature that never fired).
5. The controller is a closed loop: anything it regulates (`gpu_s`, shed rate,
   latency at the target) is pinned at equilibrium and cannot be A/B'd at fixed
   stream count.

### 12.7 Process rules adopted mid-campaign, after six wrong mechanisms

State the hypothesis and its predicted observable BEFORE the run, and commit it, so
the ordering is checkable (E39 onward). Report only what a measurement shows; if
asked why and there is no measurement, the answer is "I don't know". Do not
volunteer mechanisms — propose experiments. And check the harness can actually
exhibit the phenomenon before designing a test around it (E42 tested a premise the
bench could not show).

Retractions in this document, kept deliberately: E18/E19 (decode-bound), E20 (88 %
utilisation), E21 (half-rate slower), E24 (nvof arms inert), E28 (per-content
estimate 10× high), E33 (×2, cost attribution), E35 (worker-thread occupancy),
E37 (`decode_age` backpressure), E38 ("compute-limited"), E40 (H1/H2), E44 (H5),
E46 (H6).

### E47 — PRE-REGISTERED: is the ~5 ms residual constant, or a tail? (MB's question)

Every pipeline flow measurement so far has been a MEAN. A 12.1 ms mean is equally
consistent with every call taking 12 ms and with most calls taking 5 ms and a few
taking 200 ms — and those imply completely different causes. Not asked until MB
asked it. Two cuts, both pre-registered before instrumenting.

**H7 — the residual is a tail, not constant slowness.**
- *Predicts*: p50 materially below the mean (p50 < 9 ms) with a heavy upper tail
  (p99 > 3 × p50).
- *Falsified if*: p50 ≈ mean and the tail ratio resembles the isolated bench's
  (loaded, 4 sessions: mean 9.19, p50 8.77, p95 14.80, so p95/p50 ≈ 1.7) — which
  would mean uniformly slower calls and point at a fixed per-call cost instead.

**H8 — the two flow callers differ.** Each stream owns two nvof objects: `mt->nvof`
(camera-motion compensation, on analytics frames) and `mt->nvof_motion` (the carry
hop, on MOTION frames). Every number reported so far pools them.
- *Predicts*: the two have materially different distributions, localising the cost
  to one caller.
- *Falsified if*: their p50/p99 match within ~20 %, i.e. the cost is generic to any
  flow call made from inside the pipeline.

Instrument: per-caller sample rings (exact percentiles, not bucket estimates) tagged
at `nvof_create`, reported per tag alongside the existing `nvof_execute:` line.

Other approaches not taken yet, recorded so they are not lost: correlate flow time
with concurrent detector activity; test whether the first call after an idle gap is
the slow one (engine or VIC power state); and run the pipeline with detection
disabled but flow still active, which isolates "pipeline context" from "GPU load"
more cleanly than running the bench alongside a loaded pipeline (E46's confound).

### E48 — H7 CONFIRMED, H8 FALSIFIED: the residual is a TAIL, and the median is fine

| caller | n | mean | p10 | p50 | p90 | p99 | max | p99/p50 |
|---|---|---|---|---|---|---|---|---|
| cmc (analytics) | 4900 | 11.66 | 5.89 | **8.72** | 22.11 | 41.19 | 73.1 | 4.72 |
| motion (carry) | 13059 | 12.36 | 6.07 | **9.07** | 23.83 | 42.87 | 106.9 | 4.73 |

**H7 confirmed.** p50 is 8.7–9.1 ms against a 12.2 ms mean, with p99/p50 ≈ 4.7 —
predicted p50 < 9 ms and p99 > 3 × p50. The distribution is heavily right-skewed.

**The median pipeline call is the same speed as the bench.** Isolated under GPU
load at 4 sessions: mean 9.19, p50 8.77, p95 14.80, i.e. p95/p50 ≈ 1.7. The
pipeline's p50 is 8.72–9.07 — a match. **So there is no constant per-call penalty in
the pipeline at all.** The ~5 ms residual that six mechanisms failed to explain is
produced entirely by occasional stalls: p90 ≈ 22–24 ms, p99 ≈ 41–43 ms, max 73–107 ms.

Every mean-based comparison in E33–E46 was therefore asking the wrong question. The
means were correct; the quantity was uninformative.

**H8 falsified.** The two callers are indistinguishable — p50 8.72 vs 9.07, p99
41.19 vs 42.87, tail ratio 4.72 vs 4.73, all within 4 % and far inside the 20 %
criterion. The cost is generic to any flow call made from inside the pipeline, and
nothing about it is specific to the MOTION carry. Note the carry makes 13059 of the
17959 calls, so it pays most of the aggregate tail simply by making most of the
calls — which is consistent with delivering shed frames being free (E37) while
adding a flow hop is not.

**What this reframes.** The open question is no longer "why is every flow call
slower in the pipeline" — it isn't. It is "what stalls roughly 10 % of flow calls by
10–100 ms". That is a different and more tractable question, and it is the first
time in this campaign the target has been stated in terms a measurement chose rather
than one I assumed.

One structural difference worth noting, NOT yet tested: `nvof_benchmark` is a
separate process with its own CUDA context, so its flow work cannot queue behind the
detector's kernels; in the pipeline both share one process and one context. A
detector batch was measured at ~48 ms (E22), which is the right order for the
observed tail.

### E49 — PRE-REGISTERED: what stalls ~10 % of flow calls?

E48 reframed the question: the median flow call matches the isolated bench, so the
target is the tail (p90 ≈ 22 ms, p99 ≈ 42 ms, max 107 ms on ~10 % of calls).

**H9 — the stalls are head-of-line blocking behind detector work.** Flow and
detector share one process and one CUDA context in the pipeline; `nvof_benchmark` is
a separate process with its own context and shows a far milder tail (p95/p50 ≈ 1.7 vs
4.7). A detector batch was measured at ~48 ms (E22), the right order for the stalls.
- *Predicts*: calls that overlap a detector inference are stalled (> 2× median) at a
  materially higher rate than calls that do not — say ≥ 2× the rate — and their p99
  is materially higher.
- *Falsified if*: stall rate and p99 are similar whether or not a detector inference
  overlapped, i.e. the stalls are independent of detector activity.
- *Instrument*: an atomic in-flight detector counter sampled at flow-call entry and
  exit; each flow sample tagged with whether it overlapped.

**H10 — the tail is load-dependent.** If stalls come from contention, they should
worsen with stream count.
- *Predicts*: p99/p50 rises monotonically from 12 → 36 → 60 streams.
- *Falsified if*: the tail ratio is flat across load, which would point at something
  periodic or structural instead (power state, engine handoff, a fixed-interval
  housekeeping task).

### E50 — H9 and H10 both CONFIRMED: the stalls are detector overlap, and they scale with load

| streams | caller | overlap | n | p50 | p99 | stalled (>2× median) |
|---|---|---|---|---|---|---|
| 12 | cmc | YES | 3012 | 5.81 | 17.35 | 5.7 % |
| 12 | cmc | **NO** | 340 | **4.62** | **6.75** | **0.0 %** |
| 12 | motion | YES | 4039 | 6.03 | 25.12 | 8.6 % |
| 12 | motion | **NO** | 337 | **4.70** | **7.42** | **0.3 %** |
| 36 | cmc | YES | 4789 | 8.80 | 40.59 | 15.7 % |
| 36 | motion | YES | 13034 | 9.01 | 41.84 | 17.5 % |
| 60 | cmc | YES | 471 | 7.95 | 117.86 | 16.3 % |
| 60 | motion | YES | 1224 | 8.85 | 134.69 | 18.4 % |

**H9 confirmed.** Predicted overlapping calls stall at ≥ 2× the rate of
non-overlapping ones. Measured: **0.0 % and 0.3 % without overlap against 5.7 % and
8.6 % with**, and p99 6.8–7.4 ms against 17–25 ms. A flow call that does not overlap
a detector inference runs at **4.6–4.7 ms** — the isolated bench's idle speed — and
essentially never stalls.

**H10 confirmed.** Tail ratio p99/p50 rises monotonically with load: **3.09 → 4.63 →
14.90** (cmc) and **4.28 → 4.65 → 15.42** (motion) at 12 / 36 / 60 streams, with p99
reaching 118–135 ms and a maximum of 173 ms at 60 streams.

**So the ~5 ms mean residual is now accounted for**: it is not a per-call cost, it is
detector work blocking roughly a sixth of flow calls for tens of milliseconds. Flow
and detector share one process and one CUDA context in the pipeline; `nvof_benchmark`
is a separate process with its own context, which is why nothing isolated ever
reproduced it (E42's harness flaw, now explained rather than merely noted).

**The honest caveat.** At 36 and 60 streams almost every call overlaps a detector
inference (4789 of 4807; 13034 of 13049), so the overlap=NO sample there is 15–23
calls and proves little. The 12-stream rows carry the result, with 340 and 337
non-overlapping calls. And "overlap" is correlational: when the detector is idle the
whole box tends to be quiet, so this does not by itself prove head-of-line blocking
rather than general quiescence. The clean causal test is to run the pipeline with
detection disabled but flow live, and check the tail vanishes — not yet done.

**Where this points, as a candidate rather than a conclusion**: the VPI stream wraps
a CUDA stream (`e->cu`) created without priority. The flow chain's CUDA work
therefore queues behind detector kernels in the same context. `cudaStreamCreateWithPriority`
on the engine's stream is a small, contained change that would test whether the
stalls are schedulable away.

### E51 — CUDA stream priority for the flow engine: **no effect**

E50's candidate was that flow CUDA work queues behind detector kernels in the shared
context. Tested directly: the nvof engine's CUDA stream created at the device's
highest priority, env-gated (`UBON_NVOF_STREAM_PRIO=0/1`) so both arms run the same
binary. Knob verified live by log line.

| | p50 | p99 | stalled | FSkip | PM mix |
|---|---|---|---|---|---|
| default stream | 9.06 | 43.01 | 17.2 % | 0.529 | 0.432/0.566 |
| HIGH priority | 9.06 | 42.50 | 17.6 % | 0.531 | 0.437/0.560 |

Flat — differences of 2–5 % are inside run-to-run variation, and the operating point
is unchanged. **Stream priority does not move the stalls.**

This is informative rather than merely negative. The CUDA portion of a flow call is
small (scale 0.45 ms, convert 0.60 ms); the 12.2 ms sits in `flow_submit_sync`,
which covers VIC convert → OFA flow → VIC convert → sync. Those are separate
hardware engines with their own queues, and a CUDA stream priority does not touch
their scheduling. So the stall is most likely NOT CUDA-kernel queueing — which also
means E50's detector-overlap correlation, though strong and reproduced five times,
is not yet explained by a mechanism that survives intervention.

**Correlation replicated five ways** (det_overlap=NO vs YES, p50 and stall rate):

| config | NO: p50 / stalled | YES: p50 / stalled |
|---|---|---|
| 12 streams | 4.62–4.70 ms / 0.0–0.3 % | 5.81–6.03 / 5.7–8.6 % |
| 36 streams | 4.78–5.05 / 0.0–6.7 % | 8.80–9.01 / 15.7–17.5 % |
| 60 streams | 3.08–6.06 / 0.0–4.3 % | 7.95–8.85 / 16.3–18.4 % |
| analytics 1.0 s | 4.60–4.90 / 0.0–4.0 % | 8.48–9.83 / 20.6–22.8 % |
| analytics 0.5 s | 4.76–5.16 / 0.0–6.7 % | 8.66–9.25 / 20.3–21.6 % |

**Two crashes found along the way, both worth fixing separately:**
1. `rt_benchmark` segfaults on a tracker yaml with `inference_config.detection`
   removed — it dereferences the detection infer thread unconditionally, though
   `track_shared.c:783` documents model-less shared state as legitimate. This blocked
   the clean causal test (detector absent, flow live).
2. A config with `min_time_delta_process: 10` (analytics ~0.1 Hz, so nearly every
   frame is a MOTION frame) ran and then segfaulted at teardown. Not investigated;
   possibly MOTION frames arriving before any analytics frame has initialised the
   motion tracker.

**State**: the tail is real, reproducible, correlated with detector activity, scales
with load, and is unaffected by CUDA stream priority. The next intervention worth
trying is on the VIC/OFA side rather than CUDA — but the honest position is that the
mechanism is identified only as far as "flow calls stall when the detector is
working", and the one intervention tried did not shift it.

### E51a — clarification: contention with detection is ESTABLISHED; the resource is not

MB pushed back on the E51 summary, correctly. That write-up let a failed
intervention blur an established result. Separating them:

**Established.** Flow stalls are contention with detector work. Calls that do not
overlap a detector inference stall ~0 % and run at 4.6 ms; calls that overlap stall
16–23 % with p99 up to 135 ms. Replicated across five configurations (12/36/60
streams, and two detector-starvation variants), and the tail scales with load
(p99/p50 = 3.1 → 4.6 → 14.9). That is a strong, repeatedly reproduced result and it
stands.

**Not established.** WHICH shared resource. E51 ruled out exactly one candidate —
CUDA kernel scheduling order in the shared context — because raising the flow
stream's priority changed nothing. Still open: memory bandwidth (a 416 px batch-5
detector pass is bandwidth-heavy, and VIC/OFA also stream through memory); SoC power
or thermal budget shared across GPU/VIC/OFA; or the CUDA convert that must finish
before VIC/OFA can start being delayed by bandwidth rather than by queue order —
which priority would not fix either.

**Discriminating test not yet run**: record, per stalled flow call, how long the
overlapping detector inference had been running and how long it continued. If stall
duration tracks the detector batch's remaining time, the flow call is waiting for the
batch to finish (whatever the resource); if stalls are shorter and scattered within
the batch, it is bandwidth or power sharing rather than queueing behind it. Both
timestamps are already available — `g_infer_inflight` would need to carry the current
batch's start time alongside the count.

### E52 — the causal test, at last: **the detector causes the whole thing**

Four bugs had to be fixed before this control run was possible (all committed; see
"fix: detector-free operation actually works"). `track_shared.c:783` documents
model-less shared state as legitimate, but any tracker without
`inference_config.detection` crashed — `infer_thread_configure` on a NULL thread,
then `infer_thread_infer_async_callback` returning silently and STRANDING the frame
(the work queue could never drain: `Assertion iters<20000 failed`,
"main_pipeline 0 l 3 jr 1"), then `utrack_run` on a NULL detection list, then the
analytics path submitting into a void at all. That last one also explains the
earlier `min_time_delta_process: 10` teardown crash — same stranding, reached a
different way.

**The control: flow at full rate, detector absent, 36 streams.**

| | detector present | detector absent | change |
|---|---|---|---|
| mean | 12.37 ms | **4.60 ms** | −63 % |
| p50 | 9.06 | **4.54** | −50 % |
| p90 | 24.15 | **4.86** | −80 % |
| p99 | 43.01 | **5.80** | −87 % |
| tail p99/p50 | 4.75 | **1.28** | — |
| stalled (>2× median) | 17.2 % | **0.1 %** | — |

**The tail vanishes and the median halves.** So the detector causes both effects,
not just the stalls. A flow hop's intrinsic cost on this hardware is **4.5 ms**;
everything above that — the elevated median AND the 10–100 ms stalls — is
interaction with detector execution.

This closes the question opened in E33. The chain of reasoning that survives:
the flow hop costs 12.1 ms in the pipeline (E33) → not GPU work, OFA capacity, the
lease queue, threads, decode_age, the latency target, size churn, or concurrency
(E35–E46) → the excess is a tail, not a constant, and the median matches an
isolated bench under equivalent GPU load (E48) → stalls correlate with detector
overlap across five configurations and scale with load (E50) → and removing the
detector removes them entirely (E52).

Note the median result also reconciles E48 and E46: the pipeline's p50 matches the
isolated bench *under load* (8.77 ms), and the detector-free pipeline's p50 (4.54 ms)
matches the bench *idle* (3.7–5.1 ms). The pipeline was never anomalous — it was
simply always measured with a detector running.

**What this does and does not give.** It is a complete causal account, but not a
fix: the detector is the product. The lever it implies is to stop flow calls
coinciding with detector execution — scheduling them into gaps, or batching them
behind a detector batch boundary — rather than to make flow itself cheaper. E51
already showed CUDA stream priority is not that lever, and E42 showed chained
submission is not either. Whether any scheduling arrangement can avoid a resource
the detector saturates ~85 % of the time is untested and, on the evidence of this
campaign, should be pre-registered before anyone tries it.

### E53 — and there is no scheduling window, so the cost is irreducible at load

E52's implied lever was "stop flow calls coinciding with detector execution". That
requires the detector to be idle sometimes. It is not. From the E50 overlap tagging:

| config | caller | overlapping | total | % overlap | idle window |
|---|---|---|---|---|---|
| 12 streams | cmc | 3012 | 3352 | 89.9 % | 10.1 % |
| 12 streams | motion | 4039 | 4376 | 92.3 % | 7.7 % |
| 36 streams | cmc | 4867 | 4881 | **99.7 %** | **0.3 %** |
| 36 streams | motion | 13028 | 13043 | **99.9 %** | **0.1 %** |
| 60 streams | cmc | 471 | 478 | 98.5 % | 1.5 % |
| 60 streams | motion | 1224 | 1247 | 98.2 % | 1.8 % |

At 36 streams **99.8 % of flow calls already coincide with a detector inference** —
there is essentially no idle window to schedule into. Deferring, batching or
prioritising flow cannot move work into gaps that do not exist, which is consistent
with chained submission (E42) and CUDA priority (E51) both measuring flat, and means
no scheduling arrangement will recover the difference at load.

**So the account is complete and the conclusion is negative but firm.** A flow hop
costs 4.5 ms alone and ~12 ms beside a detector; the detector runs essentially
continuously above ~24 streams; therefore in-pipeline flow costs ~12 ms and cannot be
scheduled out of it. The remaining levers are only: make flow calls cheaper (E34:
512→320 buys 10–16 %, worth nothing on the curve), or make fewer of them — and the
carry's whole purpose is to make more (13043 against 5100 for CMC alone at 36
streams).

**This is why `skip_mode: motion` is a wash and will stay one** on this hardware at
these stream counts. It is not an implementation defect to be fixed; the flow hop is
intrinsically ~4.5 ms and unavoidably ~12 ms in situ, and the carry adds ~2.5× more
of them. The measured quality gain (+0.005 to +0.047 by content) does not cover
that. The feature remains correct and default-off, and it is the right thing to
enable where analytics runs below the delivered frame rate BY CHOICE and the detector
is not saturated — which is E11's configuration and the untested case in §12.4.

### E54 — PRE-REGISTERED: which stage stretches? (MB: "really no way to run trivial work in parallel?")

MB pushed back on E53's conclusion and is right that it overreached. "99.8 % of flow
calls coincide with a detector inference" establishes there is no idle window; it
does NOT establish that the contended resource is saturated. VIC and OFA are separate
silicon from the GPU SMs and should in principle run alongside detector kernels. I
inferred "irreducible" from "always overlapping", which does not follow.

A comparison already in the data argues against simple hardware saturation:

| | idle | beside loaded pipeline | ratio |
|---|---|---|---|
| flow OUT of process (nvof_benchmark, 2 sessions) | 5.08 ms | 6.93 ms | **1.36×** |
| flow IN process (pipeline) | 4.5 ms | 12.1 ms | **2.7×** |

Same hardware and the same detector load, but an in-process flow call suffers twice
as much as an out-of-process one. If the cause were a globally saturated engine or
memory bandwidth, both should degrade alike. Something process-local is implicated.

`tegrastats` on this build reports only `GR3D_FREQ` — no VIC, OFA, NVDEC or EMC
counters — so engine utilisation cannot be read directly. Measuring instead by
splitting the call.

**H11 — one of the three hardware stages stretches, and which one names the
resource.** `run_flow` submits VIC convert (NV12 pitch→block-linear) → OFA dense
flow → VIC convert (MV block→pitch-linear), then one `vpiStreamSync`. A diagnostic
mode syncs after each submit and times the stages separately.
- *H11a VIC contention* — predicts the VIC stages stretch most under detector load.
  The decoder also uses VIC, so 36 streams of decode plus 2 VIC ops per flow call
  could saturate it.
- *H11b OFA contention* — predicts the OFA submit stretches most.
- *H11c neither: CPU wakeup / scheduling* — predicts all three stages stay near their
  idle cost and the SUM of stage times is materially less than the unsplit
  12.1 ms, i.e. the time is spent returning from the sync rather than in any engine.
- *Falsified/uninformative if*: splitting changes total cost so much that the arms
  are not comparable (adding syncs serialises what was pipelined, so the sum is
  expected to be ≥ the unsplit total; H11c is the case where it is markedly LESS).

If H11c holds, "run trivial work in parallel with the detector" is a thread-scheduling
problem and is fixable. If H11a or H11b holds, it is an engine contention problem and
the lever is doing less of that engine's work — e.g. feeding OFA block-linear NV12
directly from the decoder, which is already block-linear, instead of the current
NV12→YUV420→NV12→block-linear round trip.

### E55 — FOUND IT: per-call VPI wrapper create/destroy is the dominant cost

Splitting `run_flow` further (E54's stage sum only accounted for half the call)
puts the missing time in one place — the per-call VPI wrapper churn:

| ms/call at 36 streams | detector absent | detector present | stretch |
|---|---|---|---|
| wrapper_create | 0.459 | 1.071 | 2.3× |
| **wrapper_destroy** | **0.326** | **4.405** | **13.5×** |
| mv_export | 0.103 | 0.128 | 1.2× |
| cuda_dep | 0.050 | 0.356 | 7.1× |
| ensure_size | 0.002 | 0.001 | — |
| pool_release | 0.001 | 0.001 | — |
| *(whole call)* | *4.77* | *13.22* | *2.8×* |

**`vpiImageDestroy` on the per-frame wrapper alone is 4.4 ms of a 12.2 ms call,
and it stretches 13.5× under detector load.** With create, the churn is 5.48 ms
against 0.79 ms detector-free — most of the difference between a flow call in an
idle process and the same call beside a detector. It is driver-side work, done
every frame, on a resource that evidently serialises against whatever the detector
is doing.

That also explains the in-process versus out-of-process asymmetry (E54): the bench
process runs the same VPI calls but does not have a TensorRT engine hammering the
same driver, so its wrapper churn stays cheap.

**MB's question — do we need to destroy/recreate, or only on a size change?** VPI
provides exactly that: `vpiImageSetWrapper(VPIImage, const VPIImageData *)` rebinds
an existing wrapper to new memory, requiring only that dimensions, format and buffer
type match and the image is not locked. Flow geometry is fixed for a stream, so a
rebind is legal on every call and a create/destroy is needed only when the size
changes.

**H12 (pre-registered).** Replacing per-call create/destroy with a rebind removes
most of the 5.48 ms churn.
- *Predicts*: whole-call cost at 36 streams falls from ~13.2 ms toward ~8 ms, and the
  carry's operating point improves (currently 455 px / shed 0.525 against the
  no-carry 482 px / 0.497).
- *Falsified if*: call cost does not fall materially — which would mean the destroy
  time was an artefact of measurement placement rather than real work.

One hazard to note: the existing code destroys the wrapper on lease release
specifically so "the engine holds no reference to this stream's (about to be freed)
frame memory between leases". Rebinding leaves a stale pointer in the wrapper between
calls. Nothing reads it — submits only follow a rebind, and the stream is synced
before release — but it is a deliberate weakening of an invariant the author wrote
down, so it is called out rather than quietly changed.

### E56 — the static wrapper: measured beautifully, produced no flow. Reverted.

MB: "why do we need rebind at all, why can't we use a static wrapper?" — and "you
shouldn't need a copy, can't you convert to NV12 into a fixed buffer?" Both are the
right shape. E55's rebind still cost 3.58 ms/call under load, and an address-keyed
wrapper cache would only have helped 35.3 % of the time (measured: the image
allocator does not reliably recycle buffers). A wrapper that never changes removes
the cost rather than reducing it.

Implemented exactly that: the stream owns one persistent NV12 buffer wrapped once,
and each frame's scale/convert writes **directly into it** — the same Y copy plus UV
interleave `image_convert` already does, aimed at a fixed destination, so no extra
copy and no per-frame allocation.

**It measured superbly:**

| | before | rebind (E55) | static wrapper |
|---|---|---|---|
| call total | 13.22 ms | 11.01 | **6.13** |
| p50 | 9.06 | 8.23 | **5.49** |
| p99 | 43.01 | 29.96 | **16.18** |
| shed rate | 0.525 | 0.523 | **0.495** |
| eff res | ~455 px | ~452 | **477 px** |

The carry's operating point had risen to meet the no-carry baseline (477 px / 0.495
against 482 px / 0.497). The carry's cost appeared to have vanished.

**It produced no flow at all.** A correctness gate built for this change
(`nvof_benchmark --verify`: a textured patch translated a known distance must show
that displacement in the field) passes on the rebind version and fails on the static
wrapper — 0.00 px measured, 38 non-zero cells against 245. The entire improvement
was the cost of not doing the work.

**Why, and why the original code churns the wrapper:** VPI appears to cache its view
of wrapped memory, so CUDA writes into that buffer behind its back are not observed.
`vpiImageSetWrapper` evidently performs the necessary invalidation — which is
precisely why rebinding works and a static wrapper does not, and probably why the
author recreated the wrapper every frame in the first place.

**Reverted.** The banked, verified result is E55's rebind: **13.22 → 11.01 ms/call
(−17 %), p99 43.0 → 30.0 (−30 %), p90 24.2 → 15.3 (−37 %)**, with the gate passing.

**What this cost and what it bought.** Two hours and a wrong conclusion I would have
shipped: the static-wrapper numbers were the most impressive of the campaign and
every one of them was meaningless. The Jetson build has no gtest suite, so before
this there was no on-device correctness check on the flow path at all — every
optimisation in E42–E55 was measured on timing alone. The gate now exists and is
cheap to run; it should gate any further change here.

### E57 — the wrapper was the wrong route entirely. The carry now PAYS.

MB kept asking why a per-frame wrapper was needed at all. It is not: VPI offers two
interop routes and we were on the expensive one.

- **Wrap user memory** (`vpiImageCreateWrapper`) — VPI must be re-told whenever the
  memory changes. Measured 5.48 ms/call under detector load; rebinding instead of
  recreating cut it to 3.58 ms but no further.
- **Let VPI own the buffer** (`vpiImageCreate` + `vpiImageLockData` /
  `vpiImageUnlock`) — the documented way to write image data. No wrapper at all.

Switched to the second: the stream owns a VPI-allocated NV12 image, and each frame
locks it for CUDA write, runs the same Y copy plus UV interleave `image_convert`
would have done straight into VPI's buffer, syncs its own stream (the documented
caller obligation, since lock/unlock are oblivious to the stream queue) and unlocks.
Both the per-frame `image_convert` allocation and the wrapper disappear.

**Measured at 36 streams, correctness verified by `nvof_benchmark --verify`:**

| | before | after |
|---|---|---|
| flow call, carry | 13.22 ms | **6.33 ms** |
| p50 / p90 / p99 | 9.06 / 24.15 / 43.01 | **5.78 / 8.51 / 13.59** |
| flow call, no carry | 15.44 ms | **5.07 ms** |
| carry: shed / eff res | 0.525 / ~455 px | **0.494 / 474 px** |
| no carry: shed / eff res | 0.501 / ~482 px | **0.491 / 486 px** |

**And the carry finally pays.** `skip_mode: motion` on the capacity curve, against
the same ladder without it:

| streams | mixed | mixed + carry | Δ | Δ before the fix |
|---|---|---|---|---|
| 12 | 0.4243 | 0.4243 | +0.0000 | −0.0006 |
| 24 | 0.4123 | 0.4148 | **+0.0025** | +0.0018 |
| 36 | 0.3988 | 0.4014 | **+0.0026** | −0.0066 |
| 48 | 0.3682 | 0.3736 | **+0.0053** | −0.0118 |
| 60 | 0.3460 | 0.3513 | **+0.0053** | −0.0054 |

Consistently positive and growing with load, which is the right shape: more shedding
means more carry frames and more tracks coasting. The best configuration is now the
mixed ladder WITH the carry, against the true baseline:

| streams | 12 | 24 | 36 | 48 | 60 |
|---|---|---|---|---|---|
| mixed + carry − baseline | +0.0001 | **+0.0173** | **+0.0515** | **+0.0329** | **+0.0142** |

**Route not taken, and why.** A truly static wrapper — never rebound — reads exactly
zero flow even with stream ordering correct, established by controlled A/B. VPI must
be told when wrapped memory changes, consistent with its docs noting an image may
carry more than one backing representation. So wrapping was the wrong *route*, not
the wrong details, and the two failed attempts at making wrapping cheap (E55 rebind,
E56 static) were optimising something that should not have been there.

**What this closes.** E52/E53 concluded the carry could not pay because a flow hop
cost ~12 ms in situ and there was no scheduling gap. Both premises were right and the
conclusion was still wrong: the 12 ms was not intrinsic — 7 ms of it was our own
per-frame wrapper work. The intrinsic hop is ~4.5 ms and is now ~6 ms in situ. E53
should have said "irreducible *given this implementation*", which is exactly the
overreach MB called out.

### E58 — MB's three ideas: (a) un-gating CMC, (b) carrying every decoded frame

**(a) The CMC fit-validity gate is content- and gap-dependent.** `motiontrack.cmc.
min_motion_area` defaults to 0.50 — camera compensation is disabled on any frame
whose motion ROI covers under half the image — and `uc_v11.yaml` does not override
it. The code's own comment says this is backwards for body-worn cameras ("a gentle
pan lights 20-40 % of blocks and is exactly when compensation is most needed").
Setting it to 0 (never disable), val split:

| content | n | full rate | half rate |
|---|---|---|---|
| dashcam_jaad | 86 | −0.0063 | **+0.0304** |
| bodycam | 23 | +0.0007 | **+0.0102** |
| handheld_crowd | 20 | −0.0036 | +0.0074 |
| doorway | 16 | −0.0022 | +0.0041 |
| cctv_static | 15 | −0.0103 | +0.0011 |
| office_indoor | 11 | −0.0065 | −0.0092 |
| dashcam_bdd | 13 | −0.0157 | **−0.0332** |
| movie | 3 | −0.0132 | −0.0049 |
| **aggregate** | | **−0.0052** | **+0.0018** |

Un-gating is a net loss at full rate and a net (small) gain at half rate, with
switches 0.870 → 0.848 and false tracks 195 → 186. The value of camera compensation
GROWS with the analytics gap, and the right threshold is content-dependent — which
is what the existing per-stream knob is for; it simply is not set.

**(b) Carrying every decoded frame beats a coarse motion cadence.** At the eval's
usual 9.9 fps floor every clip is decimated to ~10 fps, where a 0.09 s motion clock
lands on the analytics frames and yields NO carry at all. Re-run at native
framerate with analytics at 1/3:

| | objective | switch/obj | fp_tracks |
|---|---|---|---|
| no carry | 0.3791 | 0.849 | 191 |
| carry @ 0.09 s | 0.3893 (+0.0102) | 0.738 | 174 |
| **carry every frame** | **0.3914 (+0.0123)** | 0.740 | **170** |

Per content (none → every frame, and every-frame vs coarse):

| content | n | Δ vs none | Δ vs coarse |
|---|---|---|---|
| dashcam_jaad | 86 | **+0.0707** | +0.0078 |
| office_indoor | 11 | +0.0360 | +0.0054 |
| handheld_crowd | 20 | +0.0270 | +0.0000 |
| bodycam | 23 | +0.0176 | −0.0003 |
| doorway | 16 | +0.0007 | +0.0030 |
| cctv_static | 15 | −0.0092 | −0.0166 |
| cctv_dense | 3 | −0.0152 | −0.0390 |
| movie | 3 | −0.0305 | +0.0032 |
| dashcam_bdd | 13 | −0.0540 | −0.0430 |

So the carry is worth far more than the earlier 720p5 measurements suggested — up to
+0.07 on JAAD dashcam — once the input rate genuinely exceeds the analytics rate,
which is the production case the benchmark could not reproduce (§12.4). Carrying
EVERY frame adds a further +0.0021 overall, but the split is sharp: it helps
ego-motion and crowded content and hurts static and dense scenes, the same axis the
degradation ladder splits on (E26). Carry cadence looks like another hint-keyed
setting rather than a global one.

**Method note.** An earlier probe reported "CMC is never called" and that reading was
JUNK: it read stats through the installed python binding, which was stale and lacked
the counter key entirely, so `dict.get(...)` returned zeros that looked like real
measurements. The eval results above do not depend on it — `min_motion_area` is a
pre-existing knob — but the silent-absent-key failure is the same shape as the
parsed-but-inert config knobs of E24 and E38. A counter that reads exactly zero
deserves the same suspicion as a knob that changes nothing.

### E59 — bodycam at native rate: the carry cadence is already saturated

MB asked whether `/mldata/tracking/bwc-videotext` improves with carry on *all*
non-analytics frames, believing them 12.5 fps. Two facts settle it.

**The clips are 7.49 fps** (GT metadata `frame_rate`), not 12.5 — the eval divisor
at `eval_min_framerate: 9.9` is therefore 1 and they always run native.

**At 7.5 fps a 0.09 s motion clock fires every 0.675 frames**, i.e. it is due on
every frame. Direct check on `video116.mp4` with analytics forced to 1/3
(`debug_analytics_mask: '100'`), `min_time_delta_motion` 0.09 vs 0.001:

    both: frames=450 dur=59.9s  TRACKED=150  TRACK_FRAME_MOTION=300

Identical. Every non-analytics frame is already carried at the "coarse" setting,
which is why E58 measured bodycam at −0.0003 for every-frame vs coarse — there was
no difference to measure. **The +0.0176 in E58 IS the all-frames number.**

Carry benefit for bodycam therefore scales with the analytics gap, not with cadence:

| content | 1/2 rate none → carry | Δ | 1/3 rate none → carry | Δ |
|---|---|---|---|---|
| dashcam_jaad | 0.1830 → 0.2382 | **+0.0551** | 0.1205 → 0.1912 | **+0.0707** |
| office_indoor | 0.7042 → 0.6880 | −0.0162 | 0.6259 → 0.6620 | +0.0360 |
| handheld_crowd | 0.2479 → 0.2677 | +0.0198 | 0.2351 → 0.2621 | +0.0270 |
| **bodycam** | 0.4072 → 0.4142 | **+0.0070** | 0.2900 → 0.3076 | **+0.0176** |
| doorway | 0.5999 → 0.6081 | +0.0082 | 0.6055 → 0.6062 | +0.0007 |
| cctv_static | 0.4725 → 0.4845 | +0.0120 | 0.4783 → 0.4691 | −0.0092 |
| cctv_dense | 0.2960 → 0.2970 | +0.0010 | 0.3026 → 0.2874 | −0.0152 |
| movie | 0.2556 → 0.2378 | −0.0178 | 0.1971 → 0.1666 | −0.0305 |
| dashcam_bdd | −0.7213 → −0.7752 | −0.0538 | −0.6070 → −0.6610 | −0.0540 |

Ego-motion content (JAAD, bodycam, handheld) gains monotonically with the gap;
static/dense content is flat-to-negative at both. BDD loses at both rates and by
the same amount — that is not a gap effect, it is the carry itself being wrong on
that content, and it is the one bucket worth chasing separately.

**Consequence for the fleet.** There is no cadence knob left to turn on bodycam:
any input under ~11 fps already carries every frame at the default 0.09 s. The
remaining levers on this content are the CMC gate (E58a, +0.0102 at half rate) and
the per-hop CMC composition below.

### E60 — per-hop CMC composition (PRE-REGISTERED before the A/B was run)

**Mechanism.** CMC is fitted from ONE flow field spanning the whole analytics gap.
OFA resolves a bounded displacement, so that fit should degrade as the gap grows.
The MOTION chain already flows every pair of consecutive ingested frames — and,
crucially, `track_stream.c` already runs the chain's FINAL hop on the analytics
frame itself (the `min_time_delta_motion` block immediately before
`motion_track_add_frame`). The gap is therefore ALREADY available as a chain of
small hops, and composing them costs **no extra GPU work** — only one CPU
`cmc_fit_solve_fast` per hop. This was the assumption most likely to sink the
idea (a per-analytics-frame extra NVOF hop would have cost ~6 ms) and it is
simply not needed.

**The composition is exact, not approximate.** Writing the fit's raw forward map
as `x' = a·x − α·b·y + tx`, `y' = (b/α)·x + a·y + ty` with `a = 1+p`, `b = q`,
applying hop 1 then hop 2 gives
`a = a₁a₂ − b₁b₂`, `b = a₁b₂ + a₂b₁` (i.e. `(a,b)` multiply as complex numbers),
`tx = a₂·tx₁ − α·b₂·ty₁ + tx₂`, `ty = (b₂/α)·tx₁ + a₂·ty₁ + ty₂`.
`α` is a grid property, shared. `apply_cmc` reconstructs exactly this raw map, so
the flow-convention bridge is applied ONCE at the end. Three unit tests pin it:
translation hops compose to their sum, rotation+scale+translation matches a
direct fit of the analytically composed field, and a hop with no flow falls back.

**Falsifier, measured FIRST (mean inlier fraction / reject rate of the single
full-gap fit, vs gap size):**

| clip | gap 1 | gap 2 | gap 3 |
|---|---|---|---|
| bodycam video116 | 0.485 / 55 % | 0.441 / 68 % | 0.425 / 71 % |
| JAAD video_0001 | 0.587 / 48 % | 0.537 / 41 % | 0.584 / 26 % |

Bodycam confirms the mechanism; JAAD contradicts it. **Confound not
pre-registered and stated here rather than buried:** the inlier band is
`0.3 × rms` of the field itself, so a larger gap widens its own tolerance.
Bodycam degrading *despite* the widening band is the stronger signal; JAAD's
apparent improvement may be entirely the band. So the mechanism is live on at
most some content, and the quality A/B — not the inlier fraction — decides it.

**Also measured:** carry on/off leaves the CMC counters byte-identical, so the
carry hops feed CMC nothing today. That is the gap being closed.

**Chain completion, measured before the A/B (composed / gaps):** JAAD 84–87 %,
bodycam **23 %** at 1/3. A hop that fails the fit-validity gate breaks the chain
and sends the gap back to the very full-gap fit the composition was meant to
replace. That policy is a choice, not a given, so it is a knob
(`cmc.compose_gate_hops`, default true) and both settings are measured.

**Pre-registered predictions.** All arms: native framerate, analytics 1/3, carry
every frame, `cmc.min_motion_area: 0` in EVERY arm so CMC is actually applied
(at the 0.5 default it is skipped on most bodycam frames and the treatment could
not show at all — one variable, and it is `compose_hops`).

1. `cmp_on` > `cmp_off` on ego-motion content (bodycam, JAAD, handheld).
2. Static content (cctv_static, cctv_dense, doorway) ≈ 0 — camera is not moving,
   so there is no gap transform to get wrong.
3. If (1) is null AND bodycam chain completion stays ~23 %, dilution is the
   explanation and `cmp_ung` should recover it. If `cmp_ung` is ALSO null, the
   mechanism is not worth what it costs and the idea is dead — recorded as such.
4. dashcam_bdd is expected to stay negative: E59 showed its carry loss is
   rate-independent, i.e. not a gap-transform problem.

**RESULT — a measured null, and a measurement-hygiene finding that matters more.**

Every prediction above failed. 4 replicates per arm (val split, 190 scored clips):

| arm | mean | sd | replicates |
|---|---|---|---|
| off | 0.24063 | 0.00500 | 0.23835 0.23543 0.24169 0.24708 |
| on (gated hops) | 0.24287 | 0.00406 | 0.23839 0.24758 0.24465 0.24085 |
| on (ungated hops) | 0.24124 | 0.00737 | 0.23018 0.24501 0.24482 0.24495 |

`on − off = +0.0022 (se 0.0032, 0.7σ)`; `ungated − off = +0.0006 (0.1σ)`. Per
content, with the run-to-run sd of the baseline arm alongside:

| content | off | Δ gated | Δ ungated | noise sd |
|---|---|---|---|---|
| bodycam | 0.3066 | −0.0005 | −0.0034 | 0.0016 |
| dashcam_jaad | 0.1882 | +0.0040 | +0.0024 | 0.0107 |
| handheld_crowd | 0.2689 | −0.0052 | −0.0080 | 0.0042 |
| dashcam_bdd | −0.6543 | +0.0147 | +0.0128 | 0.0106 |
| cctv_static | 0.4779 | +0.0026 | +0.0005 | 0.0043 |
| doorway | 0.6095 | −0.0025 | −0.0054 | 0.0088 |
| office_indoor | 0.6591 | +0.0022 | +0.0063 | 0.0041 |
| cctv_dense | 0.3077 | −0.0025 | +0.0001 | 0.0256 |
| movie | 0.1738 | −0.0029 | −0.0027 | 0.0065 |

Nothing clears its own noise, and the pre-registered direction is absent exactly
where it was predicted (bodycam −0.0005, handheld −0.0052). The dilution escape
hatch is closed too: un-gating takes bodycam chain completion from 23 % to
**100 %** (150/150 gaps composed, 0 fallbacks — verified on the counters) and the
effect gets *smaller*. **Idea (1) is dead.** The code stays behind a default-off
flag with its unit tests, because the composition itself is proven correct and
costs no GPU; it simply buys nothing.

**The finding that outlives it: this eval is NOT deterministic.** Re-running one
config unchanged moves **137/197 clips**, and the run-to-run sd of the overall
weighted mean is **0.0055** — an order of magnitude above the ±0.0002 that a
single pair of runs happened to suggest. Small-n content buckets are far worse
(cctv_dense, n=3: sd 0.026).

**This is NOT a bug, and must not be "fixed" (MB).** It is the batch builder
doing its job. The detector's adaptive picker chooses its inflation percentage
from whatever is in the queue at that instant, and `infer_batch_pick_area` sets
`batch_w/h = max(req dims)` over the batch — so a frame is inferred at whatever
resolution its batchmates need, and which batchmates it gets depends on arrival
timing. Timing-dependent batching IS the production system; the default picker
was chosen partly *for* its lower run-to-run variance (infer_thread.c:519).
So two runs of one config are two samples of a real distribution, not a broken
measurement. Pinning the picker to remove the variance would measure a
configuration we do not ship.

The consequence is therefore purely methodological — replicate and report a
spread — never a hunt for a defect.

This was caught only because a stale python binding made `cmp_ung` an accidental
config-identical REPLICATE of `cmp_on` — and the two "different" arms disagreed
on 136/197 clips. The trap that has now cost three separate results this campaign
(§12.6, E58, here) paid for itself once.

**Consequence for everything already banked.** Single-run deltas below ~0.005 on
the overall mean are at or under 1σ and cannot be distinguished from noise:
- SAFE (≥2σ, or per-content deltas far above their bucket noise): mixed ladder
  (+0.022 to +0.050), carry at native rate (+0.0102/+0.0123 overall, JAAD
  +0.0707/+0.0551), the E59 gap-scaling result.
- NOT ESTABLISHED, needs replicates before it is claimed again: `skip_mode:
  motion` at +0.0025 to +0.0053, CMC un-gating at +0.0018 (E58a), and any other
  single-run delta of that size.

**Process rule added.** An A/B on this eval needs ≥3 replicates per arm and a
reported sd, or a delta ≥0.01. Replicates are cheap — a val-split arm is ~35 s —
so there is no excuse for a single-run claim. Prefer per-content deltas compared
against that bucket's own measured noise, not against zero.
