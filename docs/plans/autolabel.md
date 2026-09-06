# Auto-labelling arbitrary MP4s: design notes

Goal: a new "importer" that takes any mp4 (CCTV, doorbell, phone footage — no
annotations) and produces TrackSet ground truth good enough to use for tracker
evaluation and parameter search, using a state-of-the-art offline
detection/tracking stack. Quality is the only priority; runtime is not.

## Why this is feasible at all

The system under test (uc_v11 and friends) is a *causal, real-time,
edge-constrained* tracker. An auto-labeller has none of those constraints:

- **Non-causal**: it can look at the whole video before deciding anything —
  associate through a 10-second occlusion because it has already seen the
  person re-emerge.
- **Unbounded compute**: transformer detectors at native resolution with
  tiling and test-time augmentation, per-object video segmentation, multiple
  passes. Seconds per frame is fine.
- **Global optimisation**: tracklet stitching and identity clustering over the
  entire timeline, instead of frame-by-frame greedy assignment.

That asymmetry is what makes pseudo-GT meaningfully better than the SUT's own
output. If the labeller were only "the same tracker but bigger", eval numbers
would be self-congratulatory.

## Design principles

1. **Decorrelate from the system under test.** The SUT is YOLO11-based
   (ubonpartners/ultralytics fork). The labeller must not be built on the same
   detector family or weights — shared blind spots (same misses, same ghost
   patterns) would silently inflate scores. Use transformer detection
   (DETR-family) and segmentation-driven association instead.
2. **Emit uncertainty, don't hide it.** Where the labeller is unsure (crowd
   blobs, reflections, posters/screens, sub-resolution people), emit class
   `other` ignore regions — the whole eval pipeline (compute_metrics,
   display_trackset, honest-FP fitness) already treats `other` as "don't
   score here". Forced guesses in ambiguous areas are what poisons pseudo-GT.
3. **Cache every stage.** Detections, masks, embeddings, tracklets each get a
   sidecar file keyed by (video hash, stage config hash). Iterating on
   association parameters must not re-run a 6-hour detection pass.
4. **Measure the labeller before trusting it.** We have real GT corpora
   (MOT17/20, PersonPath22, MEVA, OTW). Run the labeller on those and score it
   with our own compute_metrics. That gives a quantified error floor
   ("labeller achieves 0.9x IDF1 on MOT17-style scenes") and a regression
   harness for labeller changes.

## Proposed pipeline

### 1. Detection (per frame, dense)

- **Primary: Co-DINO / Co-DETR** (MMDetection; ~66 AP COCO) run at native
  resolution, every frame. Keep everything down to conf ≈ 0.05 — final
  thresholding happens after association, when track context exists.
- **Small-object pass: SAHI-style tiling** (2×2 with overlap) fused by NMS,
  for distant CCTV pedestrians the global pass misses.
- Classes: COCO person + vehicle classes (car/truck/bus/motorcycle/bicycle),
  mapped to the corpus scheme `["person","vehicle","other"]`.
- Optional cross-check: an open-vocabulary detector (Grounding-DINO family)
  once per N frames; disagreement (one detector fires confidently, the other
  is silent) marks *regions to flag for review or ignore*, not extra boxes.

### 2. Association — the hard part, and entirely off-the-shelf

Association quality is where pseudo-GT is won or lost, and nothing here
should be built from scratch. The current (2025/26) off-the-shelf landscape,
in order of relevance:

- **MASA** (CVPR'24): a *universal association module* — learned "match
  anything" appearance association that plugs onto any detector, zero-shot
  across domains. **Verified real & released** (~1.4k stars, maintained
  through 2025, mmdet-based). The most direct "don't reinvent association"
  option and the primary candidate.
- **SAM2MOT** (AAAI'26): detections are promoted into SAM2's streaming
  memory, which propagates a mask per object. Zero-shot SOTA *paradigm*
  (DanceTrack HOTA ~75.8 / IDF1 ~83.9, +12 HOTA over SUSHI there);
  identity persists through partial occlusion in a way box-IoU association
  fundamentally can't. **Caveat (checked July 2026): the official repo has
  no code** — README/assets only; the sole third-party reproduction is
  unvetted (0 stars). Treat as: (a) watch the repo for the release, (b) the
  paradigm is reproducible with modest glue on top of Meta's mature SAM2
  repo (box-prompted masklets from our cached detections + a trajectory
  manager per the paper), if the bake-off justifies it.
- **SUSHI** (CVPR'23): explicitly **offline** hierarchical graph tracking —
  associates over the whole clip via a recursive GNN rather than tracking
  forward. This is the right *family* for a non-causal labeller; it holds
  long-gap identity better than online methods. Research code, so treat as
  an evaluation candidate rather than the assumed backbone.
- **BoxMOT** (maintained library): packages the Kalman-filter lineage
  (BoostTrack++, Deep OC-SORT, BoT-SORT...). These are online, 2022-era
  paradigm, and *not* the backbone — their only role here is cheap ensemble
  diversity: where an independent second associator agrees with the
  primary, confidence is high; where they disagree, flag or ignore.
  Agreement between independent trackers is a far stronger signal than
  either tracker's own score.

Backbone choice should be empirical, not aesthetic: run MASA, a SAM2-glue
tracker (or SAM2MOT proper once its code lands), and (if the code
cooperates) SUSHI on our GT corpora (MOT17/20, PersonPath22, MEVA, OTW)
under the QA harness (§6) and let compute_metrics pick. Prior: MASA as
primary — it is the one verified-runnable modern associator today — with
SAM2-based masklets as the independent agreement channel.

**Offline stitching on top.** Whatever the backbone, finish with a global
non-causal pass: per-tracklet appearance embeddings (SOLIDER / CLIP-ReID at
high-quality frames), agglomerative tracklet merging over the full timeline
constrained by time/motion plausibility, Gaussian-smoothed interpolation
(GSI) across merged gaps, interpolated spans marked with reduced
confidence. This is glue-level code (a few hundred lines), not a new
tracker.

### 3. Per-track refinement

- Tight boxes from masks (visible-extent convention — matches PersonPath22
  "visible", OTW and MEVA; note MOT-style GT is more amodal, see Risks).
- Temporal box smoothing; flicker removal (tracks alive < ~0.3 s with low
  conf die); class decided by per-track vote, not per-frame.
- Birth/death trimming to the first/last frame with real mask support.

### 4. Enrichment to full TrackSet schema (optional but cheap relative to 1–2)

The trackset object record supports more than boxes, and the SUT is scored on
some of it:

- **face_points / subbox**: SCRFD or RetinaFace on person crops → face box +
  5 landmarks, associated to the parent person track.
- **pose_points**: ViTPose-H or Sapiens on person crops → 17-kp COCO pose in
  the flattened `[x, y, conf, ...]` layout.

These make the corpus usable for face/pose eval, not just person MOT.

### 5. Ignore-region generation

Automatically emit class `other` boxes for:

- dense crowd clusters where instance separation fails (many overlapping
  low-conf masks),
- detector-disagreement regions (step 1),
- people smaller than the eval's minimum size,
- static "person-like" false-positive magnets (posters, screens, mannequins,
  reflections): a track that never moves > ε for its whole life in a static
  camera is suspicious — flag for review or auto-ignore.

### 6. QA loop

- **Labeller benchmark**: score autolabels against real GT on MOT17/20,
  PersonPath22, MEVA, OTW subsets with track_test.compute_metrics. Gate any
  labeller change on these numbers (target before first corpus use: IDF1
  within ~10% of inter-annotator level on MOT17-val, no honest-FP anomalies).
- **Disagreement mining**: run the SUT on the new corpus; the frames where
  SUT and labeller disagree most are either SUT bugs (good — that's the
  point) or labeller errors (must be triaged early on).
- **Human spot-check**: display_trackset already renders GT with ignore
  regions; sample N random + N most-disagreeing clips per batch.

## Integration into this repo

Follow the established importer pattern, but keep the heavy stack out of the
main environment:

- `src/autolabel.py` — orchestration only: frame iteration, caching, stage
  sequencing, TrackSet assembly. Models run in a **separate conda env**
  (`autolabel`) spoken to via a small subprocess worker protocol (or sidecar
  files), because MMDetection/SAM2 pins will conflict with the ultralytics
  fork and Jetson-adjacent deps in `environment.yml`.
- `TrackSet.import_autolabel(video_path, config)` in
  `src/trackset_import.py` — thin: reads the cached final tracklets and fills
  frames/metadata like every other importer (normalized xyxy, dense frames at
  native fps; `objects_at_time` interpolation handles the rest).
- `convert_autolabel(src_folder, output_folder)` + `--autolabel` flag in
  `track.py` — walks a folder of mp4s, writes the usual
  `/mldata/tracking/<name>/{annotation,video}/` pairs (video copied, not
  re-encoded). Idempotent/resumable like convert_meva.
- Config in `/mldata/config/track/autolabel.yaml`: model choices/paths,
  thresholds, tiling, enrichment on/off — so labeller versions are
  reproducible and can be A/B'd with the QA harness. Stamp the config hash
  into `metadata` of every emitted trackset.
- Cache under `/mldata/autolabel_cache/<video_hash>/{det,masks,reid,tracks}/`.

### Hardware reality

Single RTX 3090 (24 GB). Co-DINO + SAM2-L + ViTPose don't fit resident
simultaneously — run stage-sequential per video (detect all frames → track →
enrich), which the caching design gives for free. Ballpark: 1–4 s/frame for
detection+tiling, similar for SAM2 with ~10 concurrent objects → a 5-minute
1080p clip ≈ 3–8 h. That is fine for building corpora overnight; the QA
benchmark subsets keep iteration cycles short.

## Alternatives considered

- **End-to-end transformer MOT (MOTRv2/v3, MOTIP)**: strong on benchmarks
  they're trained on, weaker zero-shot on arbitrary domains; the
  detector+association+stitching stack generalises better and is more
  debuggable stage-by-stage.
- **Just run a KF-family tracker on YOLO11x**: cheap, but correlated with
  the SUT and loses identity through long occlusions — exactly the cases we
  most want GT for.
- **Point tracking (CoTracker3)**: not a primary labeller, but a good future
  *verifier* (sample points inside a box, check they move coherently with the
  track).
- **Human-in-the-loop tools (CVAT + SAM assist)**: complementary; the
  auto-labeller's flagged/ignore regions are the natural work queue if manual
  correction is ever budgeted.

## Risks / open questions

1. **Box convention mismatch**: masks give visible extent; MOT17-style GT is
   partially amodal. Our own corpora are mixed (PersonPath22 has both
   variants). Decision: standardise pseudo-GT on *visible* extent, and rely
   on the QA benchmark to quantify the effect. Don't mix conventions inside
   one eval set.
2. **Night/IR and unusual domains** (ExDark/DarkFace-style footage is clearly
   of interest in /mldata): COCO-trained detectors degrade; the QA loop must
   include a low-light subset before trusting labels there. Open-vocab
   detectors + SAM2 degrade more gracefully than closed-set ones.
3. **fps/timing**: arbitrary mp4s have VFR and rotation metadata; reuse the
   OTW lesson — read via cv2/ffprobe, verify with a rendered spot-check frame
   as part of convert.
4. **Track identity across full-frame occlusion of the *camera*** (scene
   cuts, PTZ moves): detect via global-frame difference; split tracks at hard
   cuts rather than stitching across them.
5. **Faces**: given the face-blur lesson (EgoHumans), source videos must have
   real faces for the face/pose enrichment to be worth running.

## Suggested build order

1. **P0 — associator bake-off (this is the real first step)**: stand up
   MASA off-the-shelf and a SAM2 box-prompt glue tracker on a fixed
   detection set; score both (and their agreement-ensemble) against
   MOT17-val + PersonPath22 + 2 MEVA clips with compute_metrics. Try SUSHI
   on the MOT-shaped corpora; adopt SAM2MOT proper if/when its code is
   released. Output: a measured choice of association backbone, not an
   assumed one.
2. **P1**: plumbing — caching, TrackSet export, convert_autolabel, QA
   harness automation around the P0 winner; ReID tracklet stitching + GSI on
   top.
3. **P2**: ignore-region generation + disagreement mining + display_trackset
   review workflow.
4. **P3**: face/pose enrichment; corpus production runs.

References: [SAM2MOT (AAAI'25)](https://arxiv.org/abs/2504.04519),
[MOTRv2](https://arxiv.org/pdf/2211.09791),
[Seg2Track-SAM2](https://www.researchgate.net/publication/395527363_Seg2Track-SAM2_SAM2-based_Multi-object_Tracking_and_Segmentation_for_Zero-shot_Generalization),
[SAMOFT](https://arxiv.org/pdf/2605.09417).
