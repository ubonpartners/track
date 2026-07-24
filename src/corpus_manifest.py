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
    man = {"corpus": corpus,
           "license": info.get("license", "UNKNOWN"),
           "source_root": info.get("source_root", "UNKNOWN"),
           "import_recipe": info.get("import_recipe", "UNKNOWN"),
           "gt_passes": info.get("gt_passes", old.get("gt_passes", [])),
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
    tmp = mpath + ".tmp"
    json.dump(man, open(tmp, "w"), indent=1)
    os.replace(tmp, mpath)
    print(f"{corpus}: {len(man['files'])} files manifested", flush=True)


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
    extra = [r for r in _files(root) if r not in man["files"]]
    for rel in extra:
        bad.append((rel, "UNMANIFESTED"))
    if bad:
        for rel, why in bad:
            print(f"  {corpus}/{rel}: {why}")
        return False
    print(f"{corpus}: OK ({len(man['files'])} files)", flush=True)
    return True


def derive_tracking(corpus, hint=None, max_seconds=None):
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
    if hint is None:
        print(f"{corpus}: no hint given and no {recipe_path}; "
              f"pass hint=static|bodycam", flush=True)
        return False
    os.makedirs(os.path.join(dst, "video"), exist_ok=True)
    os.makedirs(os.path.join(dst, "annotation"), exist_ok=True)
    with open(recipe_path, "w") as f:
        _json.dump({"hint": hint, "max_seconds": max_seconds}, f)
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
        w, h, fps = probe(s_vid)
        divisor = divisor_from_config(fps, hint)
        dims = scale_dims(w, h)
        tmp_vid = d_vid + f".part{os.getpid()}.mp4"
        transcode(s_vid, tmp_vid, divisor, dims, fps / divisor, max_seconds)
        os.replace(tmp_vid, d_vid)
        d = _json.load(open(s_anno))
        new, kept, dropped = rewrite_annotation(
            d, fps, divisor, dims, max_seconds, d_vid, hint=hint,
            min_delta=min_delta_from_config(hint))
        new["metadata"]["source_video"] = s_vid
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


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:]
                if a.startswith("--") and "=" in a)
    if len(args) < 2 or args[0] not in ("build", "verify", "derive"):
        print(__doc__)
        sys.exit(2)
    ok = True
    for corpus in args[1:]:
        if args[0] == "derive":
            ms = opts.get("max-seconds")
            r = derive_tracking(corpus, hint=opts.get("hint"),
                                max_seconds=float(ms) if ms else None)
        else:
            r = {"build": build, "verify": verify}[args[0]](corpus)
        ok = ok and (r is not False)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
