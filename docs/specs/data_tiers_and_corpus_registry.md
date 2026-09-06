# Data tiers and the shared corpus registry

Status: ADOPTED 2026-07-24. Owners: track repo (this spec, the manifest
format, all writers); autolabel repo (read-side enforcement, audit
measurements). Companion doc on the autolabel side:
`autolabel/docs/plans/data_tiers.md` (layout summary),
`autolabel/docs/plans/gt_rules_jul23.md` (the R1–R6 GT-usage rules the
registry enforces).

## 1. Why this exists

Until 2026-07-24 autolabel consumed videos directly from
`/mldata/tracking` — this repo's *import product*, which is mutable by
design (display transcodes, re-imports, eval-spec derivations). Two
incidents made the coupling concrete: the MEVA AVI→mp4 re-encode and
the RoundaboutHD mpeg4→h264 transcode both changed media under
autolabel and invalidated its detection caches from the outside.
Separately, the GT quality registry (which corpus may gate what) lived
as a Python dict inside autolabel, invisible to this repo's import and
eval code.

The fix is a tier split with explicit ownership, plus a single shared
registry both repos read.

## 2. The three tiers

```
tier 0   /mldata/downloaded_datasets/
         Raw acquisitions exactly as fetched (zips, raw YUV, AVI, PNG
         sequences, event streams). IMMUTABLE. Never consumed directly
         by either repo's runtime.

tier 1   /mldata/tracking_original/<corpus>/
             video/<clip>.mp4
             annotation/<clip>.json
             MANIFEST.json
         The canonical import: mezzanine media + GT tracksets +
         manifest. AUTOLABEL'S ONLY INPUT ROOT. Append-only: re-imports
         bump per-file `version` in the manifest; nothing is silently
         replaced. Mezzanine encoding policy: lossless remux where the
         source codec allows (CHIRLA: bit-identical stream); otherwise
         a PINNED recipe recorded in the manifest (RoundaboutHD: nvenc
         cq22 because 4K mpeg4-SP has no desktop hw decode; UVG: x264
         crf18 from raw YUV).

tier 2   /mldata/tracking/<corpus>/
         This repo's derived EVAL-SPEC view, produced by
         `corpus_manifest derive` (§6). Mutable, disposable,
         re-derivable at any time. NEVER an autolabel input.
```

Contracts:

1. Autolabel never reads tier 0 (format soup) or tier 2 (mutable).
2. Track never mutates tier 1 except through the two sanctioned flows:
   import (§5.1) and canonical GT passes (§5.3).
3. Track-side experiments wanting divergent media/GT overwrite their
   tier-2 copies, never tier 1.
4. Anything in tier 2 must be regenerable from tier 1 + the recorded
   recipe. If losing it would hurt, it belongs in tier 1.

## 3. MANIFEST.json — the shared registry

One per tier-1 corpus. Written ONLY by `src/corpus_manifest.py`
(schema + seed + all writers live there). Autolabel reads it through
`eval/gt_manifest.py` (a loader exposing the same
`corpus_for`/`allows` API its enforcement tests always had).

```jsonc
{
  "corpus": "roundabouthd",
  "license": "MIT (Bath 1574)",
  "source_root": "/mldata/downloaded_datasets/other/bath_1574",
  "import_recipe": "import_roundabouthd: mpeg4-SP 4K -> h264 nvenc cq22 ... + SCT GT",
  "gt_passes": [],                      // canonical GT mutations, append-only
  "generated": "2026-07-24 07:41:02",

  "capabilities": {                     // THE QUALITY/POLICY REGISTRY (§4)
    "box_convention": "visible",
    "completeness": "moving_vehicles_only(parked unlabelled: audit extra/f 1.93)",
    "density": "per_frame",
    "geometry": "medIoU 0.84 (tight)",
    "occlusion": "occl0 2.2%",
    "artifacts": null,
    "approved_uses": ["gate_association"],
    "audit": {                          // written by autolabel gt_audit --write-back
      "date": "2026-07-24",
      "tool": "autolabel eval/gt_audit.py",
      "measurements": { "roundabouthd": {"dt_ms": 66.7, "occl0": 0.022,
                                          "medIoU": 0.84, "extraPP": 1.93,
                                          "n_gt": 1024} }
    }
  },

  "files": {
    "video/roundabouthd_c001.mp4": {
      "sha256": "…", "bytes": 1937305198, "version": 1,
      "source": {                       // tier-0 origin (optional, §5.1)
        "path": "/mldata/downloaded_datasets/other/bath_1574/RoundaboutHD/imagesc001/video.mp4",
        "url": "https://researchdata.bath.ac.uk/1574/",
        "status": "present"             // present | deleted-refetchable | unknown-legacy
      }
    },
    "annotation/roundabouthd_c001.json": { "sha256": "…", "bytes": 18841339, "version": 1 }
  }
}
```

Field semantics:

- `files.*.version` — bumped whenever the hash changes on a `build`.
  A version bump without a matching `gt_passes` entry or import ledger
  note is a red flag.
- `files.*.source.status` — `present` (tier-0 bytes on disk),
  `deleted-refetchable` (tier-0 deleted to reclaim space; `url` +
  expected size retained — currently the UVG raw YUVs, ~28 GB each),
  `unknown-legacy` (pre-manifest imports: MOT release zips, cevo
  internal captures — recorded honestly rather than fabricated).
- `capabilities.audit` — measurement provenance; the descriptive
  fields (geometry/occlusion/completeness) should cite these numbers.

## 4. The capability registry (`capabilities`)

Answers two questions per corpus: *what are these labels?* and *what
may they be used for?* The `approved_uses` vocabulary:

| use               | meaning                                                    |
|-------------------|------------------------------------------------------------|
| screen            | may appear in fast screening sets (quickfast2)             |
| val               | may appear in the val gate set                             |
| frozen_test       | may appear in the frozen release testset                   |
| gate_detection    | may adjudicate detection changes                           |
| gate_association  | may adjudicate association/identity changes                |
| fp_gating         | its FP counts are trustworthy (complete labelling)         |
| recall_gating     | may adjudicate recall-adding changes                       |
| train_detector / train_reid / train_joiner | usable as training data           |

Rules of thumb encoded today (from the ratified GT rules R4–R6):
partial-GT corpora (JAAD selected-subjects, RoundaboutHD
parked-vehicles-unlabelled) never get `fp_gating`/`recall_gating`;
derived GT (augmented MEVA/OTW, bwc autolabel output) never gates
autolabel itself — `train_detector` only; BDD gates detection only
(58.8% of its GT is detector-invisible night/tiny).

Enforcement:

- `corpus_manifest build` REFUSES a corpus with no capabilities block —
  nothing enters tier 1 unclassified.
- `corpus_manifest verify` flags a missing block.
- Autolabel's test suite fails if any eval-set clip belongs to an
  unregistered corpus or one not approved for that set's use, and its
  tools consume the registry directly (e.g. the export-confidence
  calibration only counts corpora with `fp_gating`).
- This repo's eval/tuning SHOULD consult `allows(corpus, use)` the same
  way before adding corpora to its own comparisons (that is the point
  of sharing the file).

Policy changes (editing `approved_uses`) are deliberate acts: make the
edit via a build with updated capabilities, and ledger the reason in
autolabel's EXPERIMENTS.md. Drift protection is mechanical — a
narrowing change breaks autolabel's tests loudly.

## 5. Workflows

### 5.1 Importing a new dataset

1. Acquire into tier 0 (`/mldata/downloaded_datasets/...`). Never
   modify what you fetched.
2. Write/extend the importer (`src/trackset_import.py` `import_*` +
   `convert_*`). Converters default their `output_folder` to tier 1.
   Mezzanine policy: remux losslessly if the codec is mp4-compatible
   and sane; otherwise pin an explicit transcode recipe and put it in
   `CORPUS_INFO[corpus]["import_recipe"]`.
3. Declare capabilities: add the corpus to `CAPABILITIES_SEED` (or
   hand the block to `build` via an existing manifest). Unclassified
   corpora are refused.
4. `python -m src.corpus_manifest build <corpus>` — hashes everything,
   stamps capabilities + provenance.
5. Record per-file sources where known:
   `set_file_source(corpus, "video/x.mp4", {...})` (importers should
   do this inline as they convert).
6. `python -m src.corpus_manifest derive <corpus> --hint=static|bodycam`
   — produce the tier-2 eval-spec view for this repo (§6).
7. Autolabel side (whoever runs it): build detection caches, run
   `python -m eval.gt_audit --write-back` so measured quality lands in
   the manifest, and only then consider widening `approved_uses`.
   Adding the corpus to any eval set remains a deliberate,
   ledgered set-revision — never a side effect of import.

### 5.2 Re-importing / replacing media

Never overwrite silently. Re-run the converter (writes new bytes),
then `build` — changed files get `version+1`, and the change must be
ledgered. Note: autolabel's caches key on tier-1 path + content
fingerprint and FAIL CLOSED on content change, so a media re-import
implies autolabel cache rebuilds for that corpus (this is intended —
it is the incident class the tiers exist to make visible).

### 5.3 Canonical GT passes (tighten / densify / augment / consistency)

The ONE flow that mutates tier-1 annotations post-import. After the
pass: append a `gt_passes` entry (name + date), re-`build` (version
bumps record exactly which files changed). These passes come from the
autolabel bridge today (`src/autolabel_bridge.py`); they operate on
tier 1 and tier 2 follows at the next derive.

### 5.4 Integrity check

`python -m src.corpus_manifest verify <corpus> [...]` — full sha256
re-hash + missing/unmanifested/no-capabilities detection. Run after
any import or GT pass, and whenever tier-1 tampering is suspected
(e.g. before cutting a release). Autolabel's unit tests check layout
and membership only (hashing 87 GB is not a unit test).

### 5.5 Reclaiming tier-0 space

Per-file `source.status` makes deletion an informed decision: a tier-0
file whose tier-1 derivative is verified and whose source is
re-fetchable (`url` recorded) can be deleted and flipped to
`deleted-refetchable`. Practice so far: UVG raw YUVs deleted after
frame-exact verification; CHIRLA/Bath/MEVA sources retained.

## 6. Tier-2 derivation (eval-spec view)

`derive_tracking(corpus, hint, max_seconds)` produces what track.py
actually evaluates, per clip:

- resolution capped at 1280 (longest side), framerate decimated to the
  analytics grid selected by the tracker config's
  `min_time_delta_process` for the given camera-class `hint`
  (`static` | `bodycam`), I+P-only h264, audio preserved when the
  source has it (AAC copied, other codecs re-encoded to AAC; changed
  2026-09-06 from audio-stripped — corpora derived before that date
  have silent tier-2 media until re-derived);
- the annotation subset to the retained frames, with lite provenance
  and a `source_video` pointer back to tier 1;
- the recipe recorded in tier 2 as `derive_recipe.json`, so a bare
  `derive <corpus>` refresh reuses it;
- skip-if-current by mtime; hardlinked (same-inode) tier-2 files —
  the migration-day placeholder view — always re-derive;
- nvenc with libx264 fallback (intermittent nvenc session failures,
  exit 187).

Native-grid truth stays in tier 1: a tracker-config change (different
analytics grid) is a re-derive, never a re-autolabel. Eval ingests the
tier-2 mp4 directly via `run_on_mp4_file` (container pts drive timing)
for lite-provenance clips; B-framed full-rate mp4s keep the h264 path.

## 7. APIs

Track (`src/corpus_manifest.py` — sole writer):

| function | role |
|---|---|
| `build(corpus)` | hash + stamp manifest (refuses unclassified corpora) |
| `verify(corpus)` | full integrity check |
| `derive_tracking(corpus, hint, max_seconds)` | tier-2 eval-spec view |
| `load_capabilities(corpus)` / `allows(corpus, use)` | read side for THIS repo |
| `set_audit(corpus, audit)` | autolabel gt_audit write-back entry point |
| `set_file_source(corpus, rel, source)` | record tier-0 origin per file |

Autolabel (read-only + measurement):

- `eval/gt_manifest.py`: `MANIFEST` (loaded registry), `corpus_for(path)`,
  `allows(path, use)`, path resolver (`corpus_root/media_path/
  annotation_path/manifest_path`), `reload()`.
- `eval/gt_audit.py --write-back`: measures occl0/medIoU/extraPP/dt per
  corpus against detector unions and stamps them via `set_audit`.

## 8. Interactions and caveats

- **Autolabel cache identity** keys on tier-1 absolute path
  (`cache_dir_for`) + content fingerprint sidecars. Moving/renaming
  tier-1 files requires a cache-dir rename (migration pattern in
  autolabel's 2026-07-24 ledger); changing content invalidates
  fail-closed by design.
- **Hardlink inode sharing (migration day)**: tier-1 and tier-2 files
  initially share inodes. Replace-style writes (the norm, and
  everything `derive` does) diverge safely; an in-place edit on a
  tier-2 file would corrupt tier 1 — `verify` catches it, but don't do
  it. This caveat disappears as derives replace the placeholder links.
- **Encoder reproducibility**: pinned x264 recipes have reproduced
  byte-identical outputs in practice (same input/params/build), which
  made the mid-migration double-encode of UVG harmless. Do not RELY on
  this across ffmpeg upgrades; rely on manifest versions instead.
- **Licenses** are recorded per corpus in the manifest; MOT is
  CC BY-NC-SA (non-commercial) — anything productized must check the
  license field, not assume.

## 9. Current corpora (2026-07-24)

mot, personpath22, jaad, cevo, cevo_april25, bdd100k_mot, meva, otw,
chirla, roundabouthd, uvg_vcm, bwc-videotext — all manifested with
capabilities; audit measurements stamped for mot/personpath22/jaad/
meva/bdd100k_mot/chirla (+ roundabouthd/uvg via their audit rows).
Migration verified end-to-end: val120 bit-identical pre/post repoint
(122/122 clips), 563+3 cache dirs renamed, zero re-detection.
