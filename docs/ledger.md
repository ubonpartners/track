# Ledger

Dated decisions and incidents, newest first. Code comments state the
rule and point here for the story (`see ledger YYYY-MM-DD <title>`).
One entry per decision: date, title, what was decided and why, the files
it touched. Keep entries short; link to a doc for the long version.

## 2026-09-06 Repo cleanup, stages 0 to 7

Plan and per-stage reviews: `docs/plans/repo_cleanup.md`. Decisions made
while executing it that a later reader may want to reverse:

- **The package stays `src`.** A package named `track` cannot coexist
  with `track.py` at the repo root; the rename is a separate decision
  once the shim entry point is gone. The console script is deferred
  for the same reason.
- **`search` may import `corpus`.** The registry consultation in
  `track_search._registry_check` (tiers spec section 4) is by design;
  the graph test allows the edge.
- **No compatibility forwarders on `TrackSet`** for the parsers or for
  `import_create`. No caller outside this repo used them (autolabel and
  edge_node checked); in-repo callers were switched in the same stage.
  `import_create(ts, ...)` lives in `src/tracker/run.py` because it
  creates the tracker.
- **Three shims stay until the autolabel repo switches**:
  `src/track_test.py` (compute_metrics, summary_string),
  `src/trackset.py` (TrackSet) and `src/corpus_manifest.py`
  (set_audit). Every other shim from stages 3 to 5 was deleted in
  stage 7.
- **`allows` and `set_file_source` are kept** although nothing in this
  repo calls them: both are the registry API the tiers spec publishes
  and the autolabel repo calls `allows` (eval/slices.py, tests).
- **`_FORCE_DICT` became `FORCE_DICT_RESULTS = False`** in
  `src/eval/runner.py`. It was a debug hook read through `globals()`
  since 2026-07-23 (6f51b45, packed-results parser) and nothing ever
  set it.
- **Deleted as unreferenced**: `search_test` (single-candidate wrapper,
  callers use `search_test_batch`), `result_string`,
  `object_class_remap`, `densify_sparse_gt` (see the antare entry
  below), `T2()` in the registry, and the two migrations below.
- **Media unification side effects** (stage 5, from its review): a
  probe that cannot express the frame rate (`0/0`) now raises in
  `derive.probe` and the antare importer instead of propagating
  `fps=0.0`; `native_fps` raises on an unreadable file where it used to
  return OpenCV's 0.0 (the only caller reports the clip as FAIL);
  `probe_audio` on an audio-only file raises (no caller passes one);
  an encoder falling back to the next one prints a one-line note with
  ffmpeg's last stderr line, since `check_call`'s live stderr is gone.
  Old-vs-new outputs compared by ffprobe signature (codec, packet
  counts, duration, first/last pts): bwc video92 through
  derive.transcode /4 (451 video + 2818 aac packets, 60.0 s) and with a
  20 s cap (154 + 967, 20.512 s), transcode_h264 (1803 + 2818), remux
  and the roundabouthd recipe on a synthetic clip (12 frames): all
  identical.
- **Smoke eval tolerance**: two runs of identical code differed by one
  ulp in one clip's MOTP (summation order). `tests/smoke_eval.py
  --compare` treats float differences under 1e-9 relative as noise and
  says so; anything larger is a defect.

## 2026-09-06 Transcodes preserve audio

Mark: eval media must keep audio when the source has it, since the
tracker runs audio analytics on the same file. Every ffmpeg recipe
(`src/corpus/media.py`) now copies AAC or re-encodes other codecs to
AAC, and passes `-an` only when the source has no audio. Before this
every recipe stripped audio. Corpora derived earlier (bwc-videotext
AAC, raw_movies AC3) have silent tier-2 media until re-derived; the
antare clips have no audio in any drop.

## 2026-09-06 antare_bwc replaced; derived at native 10 fps

The old antare_bwc corpus (escooter/justin, sparse 1 Hz GT plus the
autolabel densifier `densify_sparse_gt`) was retired as unusable. It
was replaced by 25 dense-GT clips from two drops, named
`<incident>-<bwc|fixed>-<NN>` for the multi-camera scenes. Mark asked
that the tier-2 transcode change only resolution and B-frames, so
`derive` grew a `--divisor` override (recorded in `derive_recipe.json`,
honoured by `check`); the bodycam spec would have halved 10 fps to 5.
Memory note: `antare-bwc-native-fps`.

## 2026-07-24 One objective config

Two eval configs were both called "canonical": the search yaml
(`track_search_v11_mc.yaml`, group-mean objective) and a 205-clip
`eval_ship_baseline.yaml` with two unweighted groups. They disagreed by
0.003 to 0.005 on the same runs and invalidated results three separate
times. Decision: exactly one objective config; `eval` with no path runs
it, and search and eval both read it so they cannot describe different
datasets. Every knob an eval run needs is a CLI override
(`--tracker-config`, `--results-location`, `--split`), never a copied
yaml. `eval <search yaml>` is allowed for one-off probes of the exact
search substrate (Mark's request) and prints a loud warning. Code:
`src/track_search.py` eval_track, `src/eval_compare.py`.

## 2026-07-24 Convention-permissive matching

GT corpora differ in box convention (visible extent vs fullbody through
occlusion). Mark's rulings for the matcher in `src/eval/matching.py`:
forgiveness is DIRECTIONAL (only the mismatch the GT convention
predicts is forgiven, so the optimiser cannot drift box extents
freely; an A/B that day showed symmetric forgiveness rewarding taller
boxes) and bounded by a plausibility aspect ratio (a forgiven box must
stay within a standing-human aspect; beyond h <= max_aspect*w plain
IoU applies). The convention comes from the corpus registry
(`box_convention` stamped per clip by derive). Same day's ruling on the
registry itself: cevo and cevo_april25 GT are visible-extent (the
fullbody seed had been migrated unaudited).

## 2026-07-24 Tier-2 eval spec

Mark's spec for what track evaluates ("nail it once and for all"):
tier 2 is a derived view of tier 1 with the longest side capped at
1280, frame rate decimated to the analytics grid the tracker config's
`min_time_delta_process` selects for the camera hint, I+P h264, the
annotation subset to retained frames with lite provenance, and a
conformance check (`corpus check`) that flags anything off spec. Spec:
`docs/specs/data_tiers_and_corpus_registry.md` section 6. Code:
`src/corpus/derive.py`.

## 2026-07-23 Single-shared eval

Mark's design for the eval runner (`src/eval/runner.py`
run_single_shared): one engine set, N concurrent tracker streams, CPU
pool scoring. Three measured facts recorded with it:

- Harvest any completed stream, no polling: each in-flight stream gets
  a thread that blocks in the C wait; the old harvest-oldest loop was
  head-of-line blocking (NVDEC burst to 84 percent then sat idle).
- Largest-first dispatch was tried and reverted: 793 s against 182 s
  unsorted. Sorting by annotation size co-schedules the giant-GT clips
  (MOT20, PersonPath22, 100 to 200 MB jsons) into one window and the
  memory pressure dwarfs the queue tail it was meant to save.
- `max_duration` is a real compute cap: the h264 is truncated at
  extraction with a duration-suffixed cache entry. The metrics-window
  default (1000 s) must not build `_t1000` variants of every clip; it
  once caused 306 pointless serial demuxes.

## 2026-07-23 Stale h264 cache

The elementary-stream cache (`src/tracker/upyc.py` h264_for_video) is
invalidated when the source mp4 is newer. Before that, a re-transcoded
lite video with a different frame rate against a stale .h264 played the
old frame set on the new clock and silently corrupted every timing;
it cost a night of search. The same day the preview stream key was
renamed to `preview_stream`; the old names are still trimmed so ancient
configs do not encode preview h264 for nothing.

## 2026-07-23 Capability registry seed

`CAPABILITIES_SEED` in `src/corpus/manifest.py` is the migration of the
autolabel repo's GT registry as audited that day; autolabel's
`eval/gt_audit.py` writes measured numbers back through `set_audit`.

## 2026-07-22 BDD100K frame mapping

Measured while importing the Kaggle BDD100K MOT subset
(`convert_bdd100k_kaggle`): label frameIndex k corresponds to video
time (k-1)/5 s, not k/5. A detector-vs-GT IoU sweep over 40 sampled
pedestrian boxes peaks at the -1 offset (mean best IoU 0.592 against
0.307), consistent with 1-based jpg numbering in the original 5 fps
extraction.

## 2026-07-23 `densify_sparse_gt` (8e3dd0f)

Autolabel motion transfer to densify the sparse 1 Hz antare GT. Its
only producer was retired on 2026-09-06 and the function deleted with
it (stage 7). The augment path (`augment_trackset_file`, MEVA/OTW
bystanders) is unrelated and stays.

## 2026-07-22 `estimate_bdd_time_offsets` (69a1d55)

BDD100K MOT keyframes were extracted at 5 fps with a per-clip offset
that the release does not record. The migration recovered each clip's
offset by detector agreement against the autolabel caches and
restamped `frame_time` in tier 1 (idempotent via
`metadata.time_offset_intervals`). It ran once on `bdd100k_mot`; the
corpus manifest carries the result. Deleted 2026-09-06 (stage 7); the
body is in git history at 69a1d55 if a re-import ever needs it.

## 2026-07-21 `fix_cevo25_vfr_times` (add2e37)

cevo_april25 cameras record variable frame rate; annotations carried
synthetic times from a rounded integer frame rate and drifted seconds
off by the end of a clip. The fix restamps GT `frame_time` from the
video's decoded pts. It is a live step of `convert_cevo`, not a
one-off, and lives in `src/corpus/importers.py`.

## 2026-07-19 `dofix` (977da71)

cevo yaml to json annotation rewrite with `original_video` repointed
from `/tracking/video` to `/tracking/cevo/video`, added with the move
of the importers into trackset_import.py. Nothing called it since.
Deleted 2026-09-06 (stage 7).
