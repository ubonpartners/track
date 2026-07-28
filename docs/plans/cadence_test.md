# Cadence test set — measuring tracking quality vs analytics cadence

Status: PLANNED 2026-07-28 (MB + agent). Companion to
`ubon_cstuff/docs/research/cadence.md` (the code review of what must change
for non-uniform cadence; its §4.5 calls for exactly this eval). This plan
builds the measurement instrument: a controlled test set where **average
framerate is held constant (5 fps) while cadence regularity varies**, so
score deltas are attributable to cadence alone.

## 1. Question and method

How much tracking quality do we lose when the ~5 fps analytics stream
arrives unevenly instead of on a clean 200 ms grid — and how much of that
loss do the proposed ubon_cstuff changes (deletion OR→AND, live
`kf_fps_scale`, EMA time constants, dt-conditional association) recover?

Method: bake the cadence INTO the test videos themselves (select source
frames by pattern, keep their true PTS), then run the standard eval with an
**epsilon analytics min-interval** so the tracker processes every delivered
frame. The file's frame spacing becomes the analytics cadence, and the
whole current pipeline — decode, motion tracker, utrack, scoring — sees it
exactly as a production stream with that cadence would.

## 2. Clip selection (~30 clips)

Criteria, applied mechanically over the union of the search config's
train+val rows:

1. **Source fps ≥ 24** (probe the tier-1 `tracking_original` video — the
   *source*, not the tier-2 eval derivation). Prefer sources where
   `src_fps / 5` is an integer (25/30/50/60 fps → clean per-second
   patterns); 24/23.976 sources are admitted with 5-second pattern cycles
   (see §3) but capped at ~4 clips.
2. **Stratified diversity**: round-robin over (family × hint), picking the
   highest GT-track-count clip of each cell first (same heuristic as
   `reduce_dataset`), until 30. Target shape: ~6 mot (3 static / 3 moving),
   ~8 personpath22, ~4 meva, ~4 otw, ~4 bwc, ~2 raw_movies, ~2 uvg_vcm —
   adjusted by what passes the fps filter.
3. Clips must have **≥ 20 s of annotated GT** after any duration caps
   (gap patterns need room: a 0.8 s gap must recur dozens of times to
   measure fragmentation at gaps).
4. Exclude the known-empty-GT clips and `dropped_jitter` quarantine.

The selection is emitted as `cadence_manifest.json` (clip, corpus, tier-1
source path, src_fps, hint) — the single input to the builder.

## 3. Cadence variants

All variants deliver the SAME average 5 fps and the SAME total frame count
per clip (±1 frame at cycle boundaries); only the spacing differs. Patterns
are defined as **kept-source-frame indices within a repeating cycle**, so
burst tightness is automatically bounded by the source grid (a burst frame
cannot be closer than one source-frame period).

For a 30 fps source (cycle = 30 frames = 1 s; 5 kept per cycle):

| variant | kept indices per cycle | spacing pattern | character |
|---|---|---|---|
| `U`   | 0,6,12,18,24        | 200 ms uniform            | control (matches today's grid) |
| `J50` | 0,3,12,15,24        | 100/300 ms alternating    | ±50 % jitter, the "aliasing" regime |
| `B50` | 0,3,6,9,12          | 4×100 ms burst + 600 ms gap | 50 % duty cycle |
| `B17` | 0,1,2,3,4           | 4×33 ms burst + 867 ms gap  | maximal burst; the PM-shed / anchor-clamp regime |
| `G2`  | 0,6,12,18,24 cycle A; 0,12,24 cycle B alternating with 0,3,6,9,12,15,18 | alternating sparse/dense seconds | slow oscillation (VFR-like) |

For 25 fps sources scale indices by 25/30 (cycle 25, kept 5); for 50/60 fps
scale up (B17 gets genuinely tighter bursts — a deliberate bonus datapoint);
for 24/23.976 use a 5-second cycle of 120 frames keeping 24 (per-second
sub-patterns shifted to keep integer indices).

Rationale for the set: U is the paired control; J50 isolates *jitter*
(same max gap ≈ today's aliasing artefacts); B50/B17 isolate *gap length*
at fixed average (the deletion-gate and OF/CMC failure regime — B17's
0.87 s gap approaches `track_buffer_seconds`/2.2 s ÷ misses×dt territory);
G2 exercises the cadence-estimator tracking rate-of-change. Five variants ×
30 clips = 150 eval clips ≈ one ~5-minute eval round.

## 4. Build tooling

New `src/cadence_test.py` (builder + manifest emitter):

- **Input**: tier-1 source video + tier-1 native annotation (never tier-2 —
  we need the full frame grid to select from).
- **Frame selection**: ffmpeg `select` expression generated from the cycle
  pattern (`sum(eq(mod(n,C),k) for k in kept)`), scale ≤ 1280, **I+P h264
  `-bf 0`** (mp4-direct eval path requirement), `-an`.
- **Timestamps: TRUE source PTS preserved** (`-fps_mode vfr`, no setpts) —
  the opposite of the eval-lite derivation, deliberately: uneven real
  timestamps ARE the experiment. Sources are clean-CFR (post the July
  timestamp repairs) so kept-frame PTS are exact `k/src_fps`.
- **Annotations**: subset by the same kept-index sets (ordinal selection —
  the `frame_id`-dense path from `rewrite_annotation`), times kept at true
  PTS, `hint`/`box_convention`/`source_video` stamped as usual, plus a
  `cadence: {variant, cycle, kept, avg_fps}` provenance block.
- **Layout**: `/mldata/tracking/cadence_test/<variant>/{video,annotation}/`
  — experiment data, tier-2-class (regenerable from tier 1 + this plan),
  NOT registered in the corpus registry (it is not a corpus; the eval
  config points at it directly). `corpus_manifest check` does not cover it;
  the builder runs its own verification: per clip × variant, frame count,
  PTS set == expected grid, annotation frames == video frames.

## 5. Harness and measurement

- **Eval config** `track_search/cadence_eval.yaml` (eval-yaml form): the
  150 rows with `family: cadence_<variant>` and `group: <variant>` so the
  per-variant rollups fall out of the existing group machinery; per-clip
  `stream_hint` copied from the clip's canonical value.
- **Epsilon interval**: `min_interval: 0.001` (NOT `-1`, which defers to
  the production 0.18 s gate and would re-decimate the patterns).
- **Preflight (mandatory)**: run one clip per variant and assert the
  tracker's processed-frame count equals the container frame count. This
  catches any upstream gate (decoder `min_delta` constrain, B-frame/anchor
  policy, PM shed) silently re-decimating — if it fires, fix the leak
  before measuring anything.
- **Primary metrics**, all as **paired per-clip deltas vs the same clip's
  `U` variant**: fitness, MOTA, IDF1, IDsw, fragmentations. 30 pairs per
  variant → report mean delta, and a sign test for direction. Convention-
  permissive matching stays AUTO (identical across variants → cancels in
  the pairing).
- **Secondary diagnostics**: fragmentation *localisation* — classify each
  fragmentation/IDsw event by whether it lands within one frame of a
  pattern gap (attributes damage to gaps directly); NEW-confirm latency
  (time from first GT appearance to first emitted track) per variant.
- Also record the per-variant `Skip:` fraction — the motion tracker's skip
  behaviour is itself cadence-sensitive (cadence.md §4.3) and confounds
  detector exposure if it moves.

## 6. Phase 2 — evaluating the proposed changes

Each ubon_cstuff change lands **config-gated, default-off**, so a single
binding build can A/B every combination through eval yamls alone:

| gate | change (cadence.md §5) | expected signature in this harness |
|---|---|---|
| `deletion_rule: and` | P0-2 OR→AND | fewer false deletions in B17 gaps → frag/IDsw delta shrinks; ~neutral on U |
| `kf_fps_scale: auto` | P0-3 live cadence estimator | J50/G2 improve (correct Q per actual dt) |
| `ema_time_constants: true` | P0-4 | small; visible in long-gap ReID retention (B17 IDsw) |
| `assoc_dt_adaptive: true` | P1-6 d²-dominant + vbox·dt + appearance-heavier at large dt | the big one for B17/B50 match recovery after gaps |
| `of_cmc_gap_gate: true` | P1-6 | removes stale-flow damage at gap re-entry |

Procedure: baseline run (all gates off) → single-gate runs → best-combo
run, all against the same 150 clips; attribution = paired deltas vs the
all-off run per variant. Success criterion worth stating up front: the
B50/B17 deficit vs U should close materially (target: recover ≥ half of
the uniform-vs-bursty gap) with **no regression on U** (the production
cadence must not pay for robustness elsewhere).

## 7. Costs and risks

- Build: ~150 nvenc transcodes of ≤ 5-min clips — under an hour, resumable.
- Eval: ~5 min per full 150-clip round; the phase-2 matrix (~6 runs) is an
  afternoon including rebuilds.
- Risks: (a) upstream re-decimation invalidating the design — covered by
  the preflight; (b) burst frames are near-duplicates at 30 fps (33 ms
  apart) so detector evidence within a burst is correlated — that is the
  point of the burst regime, but it means B17 partly measures "5 looks at
  1 instant per second", which is the honest simulation of PM shedding;
  (c) `G2`'s estimator-tracking goal needs the estimator to exist (P0-1)
  before it can show anything — in phase 1 it just measures current-code
  sensitivity to rate oscillation.

## 8. Open questions (for MB)

1. Should a **1 fps uniform** variant be added (different average, breaks
   the same-average design, but anchors the "how bad can sparse get" curve
   the birth-latency discussion in cadence.md §4.4 needs)?
2. B17's 0.87 s max gap is well under `track_buffer_seconds` 2.2 s — add a
   `B17x3` (3-second cycle, 2.6 s gap) to actually cross the deletion
   boundary, or is that a separate "occlusion-like" experiment?
3. Are the five variants worth sweeping as a *cadence-sensitivity index*
   into the standard search report (a per-candidate robustness number), or
   is this strictly an offline investigation?
