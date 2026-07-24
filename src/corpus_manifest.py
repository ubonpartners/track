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


def derive_tracking(corpus):
    """tier 1 -> tier 2 hardlink refresh. Tier 2 is the tracker's own
    space: it may replace these with divergent copies at any time; this
    just (re)establishes the default view."""
    src = os.path.join(T1, corpus)
    dst = os.path.join(T2, corpus)
    n = 0
    for rel in _files(src):
        s, d = os.path.join(src, rel), os.path.join(dst, rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if os.path.isfile(d) and os.path.samefile(s, d):
            continue
        if os.path.isfile(d):
            os.unlink(d)
        os.link(s, d)
        n += 1
    print(f"{corpus}: derived {n} new links into {dst}", flush=True)


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("build", "verify", "derive"):
        print(__doc__)
        sys.exit(2)
    fn = {"build": build, "verify": verify,
          "derive": derive_tracking}[sys.argv[1]]
    ok = True
    for corpus in sys.argv[2:]:
        r = fn(corpus)
        ok = ok and (r is not False)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
