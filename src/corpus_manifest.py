"""Corpus MANIFEST.json for /mldata/tracking_original (the canonical
media+GT tier that autolabel consumes).

Layout contract:
  /mldata/downloaded_datasets/...      tier 0: raw acquisitions, immutable
  /mldata/tracking_original/<corpus>/  tier 1: canonical import (this
      video/ annotation/ MANIFEST.json  manifest). Append-only: re-imports
                                        bump `version` per clip, never
                                        overwrite silently.
  /mldata/tracking/<corpus>/           tier 2: tracker's derived copies
                                        (hardlinks by default, see
                                        derive_tracking) — mutable, never
                                        an autolabel input.

MANIFEST.json:
  {"corpus": ..., "license": ..., "source_root": ...,
   "import_recipe": ..., "generated": ...,
   "gt_passes": [...],                       # canonical GT mutations
   "files": {relpath: {"sha256": ..., "bytes": ..., "version": 1}}}

CLI:
  python -m src.corpus_manifest build <corpus> [...]   # (re)hash + write
  python -m src.corpus_manifest verify <corpus> [...]  # hash-check
  python -m src.corpus_manifest derive <corpus> [...]  # tier1 -> tier2
"""
import hashlib
import json
import os
import sys
import time

T1 = "/mldata/tracking_original"
T2 = "/mldata/tracking"

# GT quality/capability registry — the SHARED authority both repos read
# (autolabel: eval/gt_manifest.py loader; track: load_capabilities).
# Declared at import, audit numbers written back by autolabel's
# eval/gt_audit.py via set_audit(). approved_uses vocabulary:
#   screen val frozen_test train_detector train_reid train_joiner
#   gate_detection gate_association fp_gating recall_gating
# Seed values = migration of autolabel's registry (audit 2026-07-23).
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
    "cevo": {"box_convention": "visible",  # MB ruling 2026-07-24: cevo GT is visible-extent (importer comment concurs; fullbody seed was migrated unaudited)
             "completeness": "complete",
             "density": "per_frame", "geometry": "unaudited",
             "occlusion": "unaudited", "artifacts": None,
             "approved_uses": ["screen", "val", "frozen_test",
                               "train_joiner", "gate_association",
                               "gate_detection", "fp_gating",
                               "recall_gating"]},
    "cevo_april25": {"box_convention": "visible",  # same MB ruling as cevo
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
    "raw_movies": {"box_convention": "fullbody",
                   "completeness": "derived(pure autolabel output + scene "
                                   "cuts; no human GT)",
                   "density": "per_frame",
                   "geometry": "detector-tight",
                   "occlusion": "n/a",
                   "artifacts": "edited_multi_shot(cuts)",
                   "approved_uses": []},
}

CORPUS_INFO = {
    "mot": {"license": "CC BY-NC-SA 3.0 (MOTChallenge)",
            "source_root": "legacy import (MOTChallenge release zips)",
            "import_recipe": "convert_mot: seq images/video -> mp4 copy + GT"},
    "personpath22": {"license": "Apache-2.0 (dataset terms)",
                     "source_root": "/mldata/downloaded_datasets/other/personpath22",
                     "import_recipe": "import_personpath22: mp4 copy + keyframe GT"},
    "jaad": {"license": "MIT (JAAD terms)",
             "source_root": "/mldata/downloaded_datasets/other/JAAD",
             "import_recipe": "import_jaad: mp4 copy + CVAT XML GT"},
    "bdd100k_mot": {"license": "BSD-3 (BDD100K terms)",
                    "source_root": "kaggle bdd100k-mot drop",
                    "import_recipe": "convert_bdd100k_kaggle: mp4 + restamped keyframe GT"},
    "cevo": {"license": "internal", "source_root": "internal capture",
             "import_recipe": "internal import"},
    "cevo_april25": {"license": "internal", "source_root": "internal capture",
                     "import_recipe": "internal import"},
    "meva": {"license": "Apache-2.0 (MEVA)",
             "source_root": "/mldata/downloaded_datasets/other/MEVA",
             "import_recipe": "import_meva: h264-avi remux -> mp4 (x264 crf18 "
                              "fallback on broken pts) + KPF GT",
             "gt_passes": ["tighten (2026-07-23)", "consistency (2026-07-23)",
                           "densify (2026-07-23)", "augment (2026-07-23)"]},
    "otw": {"license": "research (OTW terms)",
            "source_root": "/mldata/downloaded_datasets/other/otw",
            "import_recipe": "import_otw",
            "gt_passes": ["densify (2026-07-23)", "augment (2026-07-23)"]},
    "chirla": {"license": "CC BY 4.0",
               "source_root": "/mldata/downloaded_datasets/other/chirla",
               "import_recipe": "import_chirla: mpeg4-avi remux -> mp4 "
                                "(bit-identical stream) + per-frame GT"},
    "roundabouthd": {"license": "MIT (Bath 1574)",
                     "source_root": "/mldata/downloaded_datasets/other/bath_1574",
                     "import_recipe": "import_roundabouthd: mpeg4-SP 4K -> "
                                      "h264 nvenc cq22 (no desktop hw decode "
                                      "for source codec) + SCT GT"},
    "uvg_vcm": {"license": "CC BY 4.0",
                "source_root": "/mldata/downloaded_datasets/other/uvg_vcm",
                "import_recipe": "import_uvg_vcm: raw yuv444p16 -> x264 crf18 "
                                 "yuv420p + COCO-class track GT"},
    "bwc-videotext": {"license": "internal",
                      "source_root": "internal bwc footage",
                      "import_recipe": "autolabel-labelled (no human GT)"},
    "raw_movies": {"license": "unlicensed movie/trailer footage — "
                              "INTERNAL EVAL ONLY, never train, never ship",
                   "source_root": "/mldata/video/youtube",
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
    root = os.path.join(T1, corpus)
    mpath = os.path.join(root, "MANIFEST.json")
    old = json.load(open(mpath)) if os.path.isfile(mpath) else {"files": {}}
    info = CORPUS_INFO.get(corpus, {})
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
    mp = os.path.join(T1, corpus, "MANIFEST.json")
    if not os.path.isfile(mp):
        return None
    return json.load(open(mp)).get("capabilities")


def allows(corpus, use):
    caps = load_capabilities(corpus)
    return bool(caps) and use in caps.get("approved_uses", [])


def set_audit(corpus, audit):
    """Autolabel's gt_audit write-back: merge measured numbers into
    capabilities.audit without rehashing files."""
    mp = os.path.join(T1, corpus, "MANIFEST.json")
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
    mp = os.path.join(T1, corpus, "MANIFEST.json")
    man = json.load(open(mp))
    if rel not in man["files"]:
        raise KeyError(f"{corpus}/{rel} not in manifest — build first")
    man["files"][rel]["source"] = source
    tmp = mp + ".tmp"
    json.dump(man, open(tmp, "w"), indent=1)
    os.replace(tmp, mp)


def verify(corpus):
    root = os.path.join(T1, corpus)
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


def derive_tracking(corpus, hint=None, max_seconds=None, hint_overrides=None):
    """tier 1 -> tier 2 EVAL-SPEC derivation (MB spec 2026-07-24): for
    every tier-1 video+annotation pair, produce in tier 2 the version
    track.py actually evaluates — resolution capped at 1280, framerate
    decimated to the analytics grid the tracker config's
    min_time_delta_process (per camera-class hint) selects, I+P-only
    h264, no audio — plus the annotation subset to the retained frames
    (native-grid truth stays in tier 1; a tracker-config change only
    needs a re-derive, never a re-autolabel). No video_lite/, no
    generated_h264/, no h264 cache: the tier-2 mp4 is the one eval
    artifact, ingested directly via run_on_mp4_file.

    The recipe (hint/max_seconds) is recorded in tier 2 as
    derive_recipe.json on first use, so a bare `derive <corpus>` refresh
    reuses it. Skip-if-current: a clip is re-derived when its tier-1
    video or annotation is newer than the tier-2 annotation."""
    import json as _json
    from src.dataset_lite import (divisor_from_config, min_delta_from_config,
                                  scale_dims, probe, transcode,
                                  rewrite_annotation)
    src = os.path.join(T1, corpus)
    dst = os.path.join(T2, corpus)
    recipe_path = os.path.join(dst, "derive_recipe.json")
    if hint is None and os.path.isfile(recipe_path):
        r = _json.load(open(recipe_path))
        hint, max_seconds = r["hint"], r.get("max_seconds")
        hint_overrides = hint_overrides or r.get("hint_overrides")
    if hint is None:
        print(f"{corpus}: no hint given and no {recipe_path}; "
              f"pass hint=static|bodycam", flush=True)
        return False
    hint_overrides = hint_overrides or {}
    os.makedirs(os.path.join(dst, "video"), exist_ok=True)
    os.makedirs(os.path.join(dst, "annotation"), exist_ok=True)
    with open(recipe_path, "w") as f:
        _json.dump({"hint": hint, "max_seconds": max_seconds,
                    "hint_overrides": hint_overrides}, f)
    done = skipped = missing = 0
    for name in sorted(os.listdir(os.path.join(src, "annotation"))):
        if not name.endswith(".json") or name.endswith(".meta.json"):
            continue
        stem = name[:-5]
        s_anno = os.path.join(src, "annotation", name)
        s_vid = os.path.join(src, "video", stem + ".mp4")
        d_anno = os.path.join(dst, "annotation", name)
        d_vid = os.path.join(dst, "video", stem + ".mp4")
        if not os.path.isfile(s_vid):
            missing += 1
            continue
        if (os.path.isfile(d_anno) and os.path.isfile(d_vid)
                # a hardlinked (same-inode) annotation is the migration's
                # placeholder view, not a derived one — always re-derive
                and not os.path.samefile(s_anno, d_anno)
                and os.path.getmtime(d_anno) >= os.path.getmtime(s_anno)
                and os.path.getmtime(d_anno) >= os.path.getmtime(s_vid)):
            skipped += 1
            continue
        clip_hint = hint_overrides.get(stem, hint)
        w, h, fps = probe(s_vid)
        divisor = divisor_from_config(fps, clip_hint)
        dims = scale_dims(w, h)
        tmp_vid = d_vid + f".part{os.getpid()}.mp4"
        transcode(s_vid, tmp_vid, divisor, dims, fps / divisor, max_seconds)
        os.replace(tmp_vid, d_vid)
        d = _json.load(open(s_anno))
        new, kept, dropped = rewrite_annotation(
            d, fps, divisor, dims, max_seconds, d_vid, hint=clip_hint,
            min_delta=min_delta_from_config(clip_hint))
        new["metadata"]["source_video"] = s_vid
        caps = load_capabilities(corpus)
        if caps and caps.get("box_convention"):
            # registry is the convention authority; stamp it per clip so
            # eval consumers never have to guess (convention-aware
            # matching reads this)
            new["metadata"]["box_convention"] = caps["box_convention"]
        tmp = d_anno + f".tmp{os.getpid()}"
        with open(tmp, "w") as f:
            _json.dump(new, f)
        os.replace(tmp, d_anno)
        done += 1
        print(f"  {stem}: {w}x{h}@{fps:.2f} /{divisor} -> "
              f"{dims[0]}x{dims[1]}@{fps / divisor:.2f} "
              f"({kept} frames kept, {dropped} dropped)", flush=True)
    print(f"{corpus}: {done} derived, {skipped} current, "
          f"{missing} without video -> {dst}", flush=True)


LEGACY_DIRS = ("video_lite", "generated_h264", "video_autolabel")


def check_tracking(corpus, purge_legacy=False):
    """Tier-2 spec conformance check (MB 'nail it once and for all',
    2026-07-24): every annotation carries lite provenance + hint +
    box_convention + a source_video that exists in tier 1 and an
    original_video that exists in tier 2; every video is <=1280 on the
    long side, B-frame-free, and on the analytics grid its hint selects
    from the tracker config. Legacy artifacts (video_lite/,
    generated_h264/ at any depth, video_autolabel/, *.json.meta.json,
    *.json.orig) are flagged — or deleted with purge_legacy=True.
    Returns False on any violation so this can gate CI."""
    import glob as _glob
    import shutil as _shutil
    import subprocess as _sp
    from src.dataset_lite import divisor_from_config
    root = os.path.join(T2, corpus)
    if not os.path.isdir(root):
        print(f"{corpus}: no tier-2 dir")
        return False
    problems = []
    purged = []
    # legacy artifacts at any depth
    for base, dirs, files in os.walk(root):
        for d in list(dirs):
            if d in LEGACY_DIRS:
                p = os.path.join(base, d)
                if purge_legacy:
                    _shutil.rmtree(p)
                    purged.append(p)
                    dirs.remove(d)
                else:
                    problems.append(f"legacy dir: {p}")
        for f in files:
            if f.endswith(".json.meta.json") or f.endswith(".json.orig"):
                p = os.path.join(base, f)
                if purge_legacy:
                    os.remove(p)
                    purged.append(p)
                else:
                    problems.append(f"legacy file: {p}")
    checked = 0
    for ap in sorted(_glob.glob(os.path.join(root, "annotation", "*.json"))):
        if ap.endswith(".meta.json"):
            continue
        md = (json.load(open(ap)).get("metadata") or {})
        stem = os.path.basename(ap)[:-5]
        for field in ("lite", "hint", "box_convention", "source_video"):
            if not md.get(field):
                problems.append(f"{stem}: missing metadata.{field}")
        sv = md.get("source_video", "")
        if sv and not os.path.isfile(sv):
            problems.append(f"{stem}: source_video missing on disk: {sv}")
        if sv and not sv.startswith(T1 + "/"):
            problems.append(f"{stem}: source_video not tier-1: {sv}")
        ov = md.get("original_video", "")
        if not (ov.startswith(os.path.join(root, "video") + "/")
                and os.path.isfile(ov)):
            problems.append(f"{stem}: original_video invalid: {ov}")
            continue
        out = _sp.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,avg_frame_rate,has_b_frames",
             "-of", "json", ov], capture_output=True, text=True)
        try:
            st = json.loads(out.stdout)["streams"][0]
        except (ValueError, KeyError, IndexError):
            problems.append(f"{stem}: unprobeable video {ov}")
            continue
        if max(st["width"], st["height"]) > 1280:
            problems.append(f"{stem}: video {st['width']}x{st['height']} "
                            f"exceeds 1280")
        if int(st.get("has_b_frames", 0)) != 0:
            problems.append(f"{stem}: video has B-frames")
        # measure fps from actual frame-stamp spacing, NOT avg_frame_rate:
        # container duration omits the last frame's display period, so the
        # frames/duration ratio runs 1/(n-1) hot — 1-2% on short clips,
        # which false-flags exactly the corpora with 5-15s clips
        # a ~25-frame window: single-interval measurement is bitten by
        # container-timebase quantization of non-integer grids (50fps/7)
        # on the first frame pair
        pts = _sp.run(
            ["ffprobe", "-v", "error", "-read_intervals", "%+#25",
             "-select_streams", "v:0", "-show_entries", "frame=pts_time",
             "-of", "csv=p=0", ov], capture_output=True, text=True).stdout
        ts = [float(x.rstrip(",")) for x in pts.split() if x.rstrip(",")]
        vfps = ((len(ts) - 1) / (ts[-1] - ts[0])
                if len(ts) >= 2 and ts[-1] > ts[0] else 0.0)
        L = md.get("lite") or {}
        src_fps, hint = L.get("source_fps"), md.get("hint")
        if src_fps and hint:
            want = src_fps / divisor_from_config(src_fps, hint)
            if abs(vfps - want) > 0.15:
                problems.append(f"{stem}: fps {vfps:.3f} != analytics grid "
                                f"{want:.3f} (hint {hint})")
            if abs(md.get("frame_rate", 0) - want) > 0.15:
                problems.append(f"{stem}: metadata frame_rate "
                                f"{md.get('frame_rate')} off grid {want:.3f}")
        checked += 1
    status = "CLEAN" if not problems else f"{len(problems)} PROBLEMS"
    print(f"{corpus}: {checked} clips checked, {len(purged)} legacy "
          f"artifacts purged, {status}", flush=True)
    for p in problems[:20]:
        print(f"  ! {p}", flush=True)
    if len(problems) > 20:
        print(f"  ... and {len(problems) - 20} more", flush=True)
    return not problems


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:]
                if a.startswith("--") and "=" in a)
    if len(args) < 2 or args[0] not in ("build", "verify", "derive", "check"):
        print(__doc__)
        sys.exit(2)
    ok = True
    for corpus in args[1:]:
        if args[0] == "derive":
            ms = opts.get("max-seconds")
            r = derive_tracking(corpus, hint=opts.get("hint"),
                                max_seconds=float(ms) if ms else None)
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
