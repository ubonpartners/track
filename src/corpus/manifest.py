"""Corpus MANIFEST.json for /mldata/tracking_original (the canonical
media+GT tier that autolabel consumes).

Layout contract:
  /mldata/downloaded_datasets/...      tier 0: raw acquisitions, immutable
  /mldata/tracking_original/<corpus>/  tier 1: canonical import (this
      video/ annotation/ MANIFEST.json  manifest). Append-only: re-imports
                                        bump `version` per clip, never
                                        overwrite silently.
  /mldata/tracking/<corpus>/           tier 2: tracker's derived eval-spec
                                        copies (src/corpus/derive.py
                                        derive_tracking) — mutable, never
                                        an autolabel input.

MANIFEST.json:
  {"corpus": ..., "license": ..., "source_root": ...,
   "import_recipe": ..., "generated": ...,
   "gt_passes": [...],                       # canonical GT mutations
   "files": {relpath: {"sha256": ..., "bytes": ..., "version": 1}}}

CLI:
  python -m src.corpus.manifest build <corpus> [...]   # (re)hash + write
  python -m src.corpus.manifest verify <corpus> [...]  # hash-check
  python -m src.corpus.manifest derive <corpus> [...]  # tier1 -> tier2

Moved verbatim from src/corpus_manifest.py (repo_cleanup.md stage 4d);
derive_tracking/check_tracking now live in src/corpus/derive.py.
"""
import hashlib
import json
import os
import sys
import time

import src.paths as paths


def T1():
    return paths.tier1()


# GT quality/capability registry — the SHARED authority both repos read
# (autolabel: eval/gt_manifest.py loader; track: load_capabilities).
# Declared at import, audit numbers written back by autolabel's
# eval/gt_audit.py via set_audit(). approved_uses vocabulary:
#   screen val frozen_test train_detector train_reid train_joiner
#   gate_detection gate_association fp_gating recall_gating
# Seed values = migration of autolabel's registry (ledger 2026-07-23 Capability registry seed).
CAPABILITIES_SEED = {
    "mot": {"box_convention": "fullbody",
            "completeness": "complete_with_ignore_regions",
            "density": "per_frame",
            "geometry": "loose(medIoU 0.71 mot17 / 0.86 mot20)",
            "occlusion": "labelled_through_occlusion(occl0 ~2.2%)",
            "artifacts": None,
            "approved_uses": ["screen", "val", "frozen_test", "train_joiner",
                              "gate_association", "gate_detection",
                              "fp_gating", "recall_gating"]},
    "personpath22": {"box_convention": "visible",
                     "completeness": "complete",
                     "density": "keyframe(~200ms)+interpolate",
                     "geometry": "medIoU 0.78",
                     "occlusion": "occl0 ~2.8%",
                     "artifacts": None,
                     "approved_uses": ["screen", "val", "frozen_test",
                                       "train_joiner", "gate_association",
                                       "gate_detection", "fp_gating",
                                       "recall_gating"]},
    "jaad": {"box_convention": "fullbody",
             "completeness": "selected_subjects_only",
             "density": "per_frame",
             "geometry": "medIoU 0.81",
             "occlusion": "occl0 ~2.7%",
             "artifacts": None,
             "approved_uses": ["screen", "val", "frozen_test",
                               "train_joiner", "gate_association"]},
    "cevo": {"box_convention": "visible",  # cevo GT is visible-extent (ledger 2026-07-24 Convention-permissive matching; importer comment concurs; fullbody seed was migrated unaudited)
             "completeness": "complete",
             "density": "per_frame", "geometry": "unaudited",
             "occlusion": "unaudited", "artifacts": None,
             "approved_uses": ["screen", "val", "frozen_test",
                               "train_joiner", "gate_association",
                               "gate_detection", "fp_gating",
                               "recall_gating"]},
    "cevo_april25": {"box_convention": "visible",  # same ruling as cevo (ledger 2026-07-24 Convention-permissive matching)
                     "completeness": "complete",
                     "density": "per_frame", "geometry": "unaudited",
                     "occlusion": "unaudited", "artifacts": None,
                     "approved_uses": ["screen", "val", "frozen_test",
                                       "train_joiner", "gate_association",
                                       "gate_detection", "fp_gating",
                                       "recall_gating"]},
    "bdd100k_mot": {"box_convention": "fullbody", "completeness": "complete",
                    "density": "keyframe(200ms, per-clip offset restamped)",
                    "geometry": "medIoU 0.43 (night/tiny: detector-limited)",
                    "occlusion": "occl0 58.8% (detector-invisible dominated)",
                    "artifacts": None,
                    "approved_uses": ["gate_detection"]},
    "meva": {"box_convention": "fullbody",
             "completeness": "derived(augmented+tightened; actors-only origin)",
             "density": "per_frame(densified)",
             "geometry": "tightened(2026-07-23)",
             "occlusion": "labelled_through_occlusion",
             "artifacts": "capture_corruption(~10s periodic, 30/50 clips)",
             "approved_uses": ["train_detector"]},
    "otw": {"box_convention": "visible",
            "completeness": "derived(augmented; actors-only origin)",
            "density": "per_frame(densified)",
            "geometry": "original_mot_style",
            "occlusion": "unaudited", "artifacts": None,
            "approved_uses": ["train_detector"]},
    "chirla": {"box_convention": "visible",
               "completeness": "complete(audit extraPP 0.00: no unlabelled people)",
               "density": "per_frame",
               "geometry": "medIoU 0.74 (reid-style, slightly loose)",
               "occlusion": "occl0 0.6%", "artifacts": None,
               "approved_uses": ["screen", "val", "frozen_test",
                                 "train_joiner", "gate_association",
                                 "gate_detection", "fp_gating",
                                 "recall_gating"]},
    "roundabouthd": {"box_convention": "visible",
                     "completeness": "moving_vehicles_only(parked unlabelled: "
                                     "audit extra/f 1.93)",
                     "density": "per_frame",
                     "geometry": "medIoU 0.84 (tight)",
                     "occlusion": "occl0 2.2%", "artifacts": None,
                     "approved_uses": ["gate_association"]},
    "uvg_vcm": {"box_convention": "visible",
                "completeness": "complete(professional dense labels)",
                "density": "per_frame",
                "geometry": "medIoU 0.79 person / 0.73 vehicle",
                "occlusion": "occl0 0.7% person / 4.4% vehicle; extraPP 0.01",
                "artifacts": "faces_blurred(JobFair)",
                "approved_uses": ["screen", "val", "gate_detection",
                                  "gate_association", "train_detector"]},
    "bwc-videotext": {"box_convention": "fullbody",
                      "completeness": "derived(pure autolabel output; no human GT)",
                      "density": "per_frame",
                      "geometry": "detector-tight",
                      "occlusion": "n/a", "artifacts": "night_strobe",
                      "approved_uses": ["train_detector"]},
    "antare_bwc": {"box_convention": "visible",  # eyeballed pub-garden f1 / market-halls f300: edge-clipped visible extents
                   "completeness": "complete(human MOT labels, every frame)",
                   "density": "per_frame",
                   "geometry": "unaudited(loose in places)",
                   "occlusion": "unaudited", "artifacts": None,
                   "approved_uses": ["screen", "val", "gate_association",
                                     "gate_detection"]},
    "raw_movies": {"box_convention": "fullbody",
                   "completeness": "derived(pure autolabel output + scene "
                                   "cuts; no human GT)",
                   "density": "per_frame",
                   "geometry": "detector-tight",
                   "occlusion": "n/a",
                   "artifacts": "edited_multi_shot(cuts)",
                   "approved_uses": []},
}


def corpus_info():
    """Licence / provenance per corpus (written into MANIFEST.json by build).
    A function so the tier-0 roots are resolved when build runs."""
    return {
        "mot": {"license": "CC BY-NC-SA 3.0 (MOTChallenge)",
                "source_root": "legacy import (MOTChallenge release zips)",
                "import_recipe": "convert_mot: seq images/video -> mp4 copy + GT"},
        "personpath22": {"license": "Apache-2.0 (dataset terms)",
                         "source_root": paths.downloads("other", "personpath22"),
                         "import_recipe": "import_personpath22: mp4 copy + keyframe GT"},
        "jaad": {"license": "MIT (JAAD terms)",
                 "source_root": paths.downloads("other", "JAAD"),
                 "import_recipe": "import_jaad: mp4 copy + CVAT XML GT"},
        "bdd100k_mot": {"license": "BSD-3 (BDD100K terms)",
                        "source_root": "kaggle bdd100k-mot drop",
                        "import_recipe": "convert_bdd100k_kaggle: mp4 + restamped keyframe GT"},
        "cevo": {"license": "internal", "source_root": "internal capture",
                 "import_recipe": "internal import"},
        "cevo_april25": {"license": "internal", "source_root": "internal capture",
                         "import_recipe": "internal import"},
        "meva": {"license": "Apache-2.0 (MEVA)",
                 "source_root": paths.downloads("other", "MEVA"),
                 "import_recipe": "import_meva: h264-avi remux -> mp4 (x264 crf18 "
                                  "fallback on broken pts) + KPF GT",
                 "gt_passes": ["tighten (2026-07-23)", "consistency (2026-07-23)",
                               "densify (2026-07-23)", "augment (2026-07-23)"]},
        "otw": {"license": "research (OTW terms)",
                "source_root": paths.downloads("other", "otw"),
                "import_recipe": "import_otw",
                "gt_passes": ["densify (2026-07-23)", "augment (2026-07-23)"]},
        "chirla": {"license": "CC BY 4.0",
                   "source_root": paths.downloads("other", "chirla"),
                   "import_recipe": "import_chirla: mpeg4-avi remux -> mp4 "
                                    "(bit-identical stream) + per-frame GT"},
        "roundabouthd": {"license": "MIT (Bath 1574)",
                         "source_root": paths.downloads("other", "bath_1574"),
                         "import_recipe": "import_roundabouthd: mpeg4-SP 4K -> "
                                          "h264 nvenc cq22 (no desktop hw decode "
                                          "for source codec) + SCT GT"},
        "uvg_vcm": {"license": "CC BY 4.0",
                    "source_root": paths.downloads("other", "uvg_vcm"),
                    "import_recipe": "import_uvg_vcm: raw yuv444p16 -> x264 crf18 "
                                     "yuv420p + COCO-class track GT"},
        "bwc-videotext": {"license": "internal",
                          "source_root": "internal bwc footage",
                          "import_recipe": "autolabel-labelled (no human GT)"},
        "antare_bwc": {"license": "internal (antare)",
                       "source_root": paths.downloads("antare") + "/ {individuals - body camera-"
                                      "20260902T102854Z-1-001, multiple views - body camera and "
                                      "fixed-20260906T050034Z-1-001}",
                       "import_recipe": "import_antare: mp4 copied unchanged + "
                                        "dense MOT GT (frame k == video frame k-1); "
                                        "per-clip camera hint in metadata (bodycam/static)"},
        "raw_movies": {"license": "unlicensed movie/trailer footage — "
                                  "INTERNAL EVAL ONLY, never train, never ship",
                       "source_root": paths.video("youtube"),
                       "import_recipe": "convert_raw_movies: autolabel with "
                                        "scene cuts (AV1 sources h264-transcoded)"},
    }


def _sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _files(root):
    for dp, _dirs, fns in os.walk(root):
        for fn in sorted(fns):
            if fn == "MANIFEST.json":
                continue
            p = os.path.join(dp, fn)
            yield os.path.relpath(p, root)


def build(corpus):
    root = os.path.join(T1(), corpus)
    mpath = os.path.join(root, "MANIFEST.json")
    old = json.load(open(mpath)) if os.path.isfile(mpath) else {"files": {}}
    info = corpus_info().get(corpus, {})
    caps = old.get("capabilities") or CAPABILITIES_SEED.get(corpus)
    if caps is None:
        raise SystemExit(
            f"{corpus}: no capabilities declared — a corpus cannot enter "
            "tier 1 unclassified. Add a capabilities block (see "
            "CAPABILITIES_SEED for the schema) at import.")
    man = {"corpus": corpus,
           "license": info.get("license", "UNKNOWN"),
           "source_root": info.get("source_root", "UNKNOWN"),
           "import_recipe": info.get("import_recipe", "UNKNOWN"),
           "gt_passes": info.get("gt_passes", old.get("gt_passes", [])),
           "capabilities": caps,
           "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "files": {}}
    for rel in _files(root):
        p = os.path.join(root, rel)
        sha = _sha256(p)
        prev = old["files"].get(rel)
        version = prev["version"] if prev and prev["sha256"] == sha else \
            (prev["version"] + 1 if prev else 1)
        man["files"][rel] = {"sha256": sha, "bytes": os.path.getsize(p),
                             "version": version}
        if prev and "source" in prev:
            man["files"][rel]["source"] = prev["source"]
    tmp = mpath + ".tmp"
    json.dump(man, open(tmp, "w"), indent=1)
    os.replace(tmp, mpath)
    print(f"{corpus}: {len(man['files'])} files manifested", flush=True)


def load_capabilities(corpus):
    """Read a corpus's capabilities block (both repos' entry point)."""
    mp = os.path.join(T1(), corpus, "MANIFEST.json")
    if not os.path.isfile(mp):
        return None
    return json.load(open(mp)).get("capabilities")


def allows(corpus, use):
    caps = load_capabilities(corpus)
    return bool(caps) and use in caps.get("approved_uses", [])


def set_audit(corpus, audit):
    """Autolabel's gt_audit write-back: merge measured numbers into
    capabilities.audit without rehashing files."""
    mp = os.path.join(T1(), corpus, "MANIFEST.json")
    man = json.load(open(mp))
    caps = man.setdefault("capabilities", {})
    caps["audit"] = dict(caps.get("audit", {}), **audit)
    tmp = mp + ".tmp"
    json.dump(man, open(tmp, "w"), indent=1)
    os.replace(tmp, mp)


def set_file_source(corpus, rel, source):
    """Importer hook: record a tier-1 file's tier-0 origin
    ({path,url,sha256|bytes,status: present|deleted-refetchable|
    unknown-legacy})."""
    mp = os.path.join(T1(), corpus, "MANIFEST.json")
    man = json.load(open(mp))
    if rel not in man["files"]:
        raise KeyError(f"{corpus}/{rel} not in manifest — build first")
    man["files"][rel]["source"] = source
    tmp = mp + ".tmp"
    json.dump(man, open(tmp, "w"), indent=1)
    os.replace(tmp, mp)


def verify(corpus):
    root = os.path.join(T1(), corpus)
    man = json.load(open(os.path.join(root, "MANIFEST.json")))
    bad = []
    for rel, rec in man["files"].items():
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            bad.append((rel, "MISSING"))
        elif os.path.getsize(p) != rec["bytes"] or _sha256(p) != rec["sha256"]:
            bad.append((rel, "HASH MISMATCH"))
    if not man.get("capabilities"):
        bad.append(("MANIFEST.json", "NO CAPABILITIES BLOCK"))
    extra = [r for r in _files(root) if r not in man["files"]]
    for rel in extra:
        bad.append((rel, "UNMANIFESTED"))
    if bad:
        for rel, why in bad:
            print(f"  {corpus}/{rel}: {why}")
        return False
    print(f"{corpus}: OK ({len(man['files'])} files)", flush=True)
    return True


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:]
                if a.startswith("--") and "=" in a)
    if len(args) < 2 or args[0] not in ("build", "verify", "derive", "check"):
        print(__doc__)
        sys.exit(2)
    from src.corpus.derive import check_tracking, derive_tracking   # lazy: derive imports this module
    ok = True
    for corpus in args[1:]:
        if args[0] == "derive":
            ms = opts.get("max-seconds")
            dv = opts.get("divisor")
            r = derive_tracking(corpus, hint=opts.get("hint"),
                                max_seconds=float(ms) if ms else None,
                                divisor=int(dv) if dv else None)
        elif args[0] == "check":
            r = check_tracking(corpus,
                               purge_legacy="purge-legacy" in
                               [a[2:] for a in sys.argv if a.startswith("--")])
        else:
            r = {"build": build, "verify": verify}[args[0]](corpus)
        ok = ok and (r is not False)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
