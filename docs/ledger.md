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
