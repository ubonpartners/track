"""One-off data migrations that ran once against tier 1 (kept for the
record until the ledger has their entries; deletion candidates,
repo_cleanup.md stage 7): the BDD100K time-offset restamp (added
2026-07-22, git 69a1d55) and the cevo yaml->json fix (added 2026-07-19,
git 977da71). Neither is called by any importer; the runs themselves
were never ledgered.

Moved verbatim from src/trackset_import.py (repo_cleanup.md stage 4b).
"""
import json
import os

import stuff

import src.paths as paths


def estimate_bdd_time_offsets(detector_cache_root=None):
    """Per-clip 5fps-extraction time offset for bdd100k_mot GT (run AFTER
    detection caches exist). The Kaggle subset's label frameIndex maps to
    video time with a PER-CLIP integer-interval offset (measured
    2026-07-22: 33 clips at -1, 12 at 0, 2 at -2, 1 at +1 — a global
    mapping cost up to 0.4 detector recall on misaligned clips and
    corrupted failure analyses). Chooses among 4 discrete hypotheses by
    detector agreement (person+vehicle best-IoU@0.5 rate, det_fill
    caches) and restamps frame_time in place; idempotent via
    metadata.time_offset_intervals. Alignment recovery of an acquisition
    artifact — boxes are never modified."""
    from src.autolabel_bridge import load_autolabel
    load_autolabel()          # puts the checkout on sys.path or raises with setup help
    from autolabel.pipeline import cache_dir_for
    from autolabel.detect import load_detections
    folder = paths.tier1("bdd100k_mot")
    anno = sorted(f for f in os.listdir(folder + "/annotation")
                  if f.endswith(".json"))
    for name in anno:
        ap = os.path.join(folder, "annotation", name)
        vp = os.path.join(folder, "video", name[:-5] + ".mp4")
        c = cache_dir_for(vp)
        p = os.path.join(c, "det_fill.npz")
        if not os.path.isfile(p):
            print(f"  skip {name}: no det cache")
            continue
        meta, det = load_detections(p)
        d = json.load(open(ap))
        H, W = d["metadata"]["height"], d["metadata"]["width"]
        pfps = meta["fps"]
        scores = {}
        for off in (-2, -1, 0, 1):
            tot = hit = 0
            for f in d["frames"][::3]:
                k = f["frame_id"]
                vfi = int(round(max(0.0, (k + off) / 5.0) * pfps))
                if vfi >= len(det):
                    continue
                b, s, l = det[vfi]
                for o in f["objects"].values():
                    cl = o.get("class")
                    if cl not in (0, 1):
                        continue
                    g = o["box"]
                    if (g[3] - g[1]) < 0.03:
                        continue
                    import numpy as np
                    gpx = np.array(g, float) * [W, H, W, H]
                    db = b[l == cl]
                    if not len(db):
                        continue
                    tot += 1
                    ix = np.clip(np.minimum(gpx[2], db[:, 2])
                                 - np.maximum(gpx[0], db[:, 0]), 0, None)
                    iy = np.clip(np.minimum(gpx[3], db[:, 3])
                                 - np.maximum(gpx[1], db[:, 1]), 0, None)
                    i = ix * iy
                    iou = i / np.maximum(
                        (gpx[2]-gpx[0])*(gpx[3]-gpx[1])
                        + (db[:, 2]-db[:, 0])*(db[:, 3]-db[:, 1]) - i, 1e-9)
                    if iou.max() >= 0.5:
                        hit += 1
            if tot >= 50:
                scores[off] = hit / tot
        if not scores:
            continue
        bo = max(scores, key=scores.get)
        if d["metadata"].get("time_offset_intervals") == bo:
            continue
        for f in d["frames"]:
            f["frame_time"] = round(max(0.0, (f["frame_id"] + bo) / 5.0), 6)
        d["metadata"]["time_offset_intervals"] = bo
        with open(ap + ".tmp", "w") as fh:
            json.dump(d, fh, indent=4)
        os.replace(ap + ".tmp", ap)
        print(f"  {name}: offset {bo} (agreement {max(scores.values()):.2f})")


def dofix():
    """cevo yaml -> json annotation fix: repoints original_video from the
    old /tracking/video layout to /tracking/cevo/video and rewrites each
    yaml as json. Added 2026-07-19 (git 977da71); nothing calls it."""
    dr=paths.tier2("cevo", "annotation")
    seqs=os.listdir(dr)
    for s in seqs:
        d=stuff.load_dictionary(dr+"/"+s)
        x=d["metadata"]["original_video"]
        x=x.replace("/tracking/video", "/tracking/cevo/video")
        d["metadata"]["original_video"]=x
        on=dr+"/"+s
        on=on.replace(".yaml",".json")
        with open(on, 'w') as json_file:
                json.dump(d, json_file, indent=4)
