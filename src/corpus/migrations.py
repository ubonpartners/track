"""One-off data migrations that ran once against tier 1 (kept for the
record until the ledger has their entries; deletion candidates,
repo_cleanup.md stage 7): BDD100K time-offset restamp, cevo_april25 VFR
restamp, the cevo yaml->json fix.

Moved verbatim from src/trackset_import.py (repo_cleanup.md stage 4b).
"""
import json
import os

import numpy as np
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


def fix_cevo25_vfr_times(folder=None):
    """Restamp cevo_april25 GT frame_times from the video's real decoded
    PTS. Most of these cameras record variable frame rate (intervals
    0.03-0.13s) but the annotations carried synthetic times from a
    ROUNDED integer frame_rate, drifting seconds off video time by end of
    clip (time-based scoring then matches tracker output against stale
    GT). Frame ids are 0-based and dense, one per decoded frame, so frame
    k's true time is the k-th display-ordered frame's PTS.

    B-frame caveat: two clips were muxed with pts=dts (decode-order
    stamps), so the decoder — which emits display order via picture-
    order-count — yields locally swapped stamps. The stamp multiset is
    still the display timeline (one frame per coded picture); sorting
    reconstructs it exactly. Idempotent; safe to re-run."""
    folder = folder or paths.tier1("cevo_april25")
    import av
    import numpy as np
    for name in sorted(os.listdir(folder + "/annotation")):
        if not name.endswith(".json"):
            continue
        ap = folder + "/annotation/" + name
        vp = folder + "/video/" + name[:-5] + ".mp4"
        if not os.path.isfile(vp):
            print("fix_cevo25_vfr_times: no video for", name)
            continue
        d = json.load(open(ap))
        times = []
        with av.open(vp) as c:
            stream = c.streams.video[0]
            tb = float(stream.time_base)
            for fr in c.decode(stream):
                t = fr.pts if fr.pts is not None else fr.dts
                times.append(float(t) * tb if t is not None else np.nan)
        pts = np.asarray(times, np.float64)
        if len(pts) > 1 and np.any(np.diff(pts) < 0):
            pts = np.sort(pts)  # broken pts=dts muxing (see docstring)
        pts = pts - pts[0]
        if len(pts) != len(d["frames"]):
            print(f"fix_cevo25_vfr_times: SKIP {name} "
                  f"(pts {len(pts)} != frames {len(d['frames'])})")
            continue
        dt = np.diff(pts)
        vfr = bool(len(dt) and np.median(dt) > 0
                   and np.max(np.abs(dt - np.median(dt)))
                   > 0.02 * np.median(dt))
        for f in d["frames"]:
            f["frame_time"] = round(float(pts[f["frame_id"]]), 6)
        d["metadata"]["frame_rate"] = round(
            float((len(pts) - 1) / max(pts[-1] - pts[0], 1e-9)), 6)
        d["metadata"]["vfr"] = vfr
        with open(ap, "w") as fh:
            json.dump(d, fh, indent=4)
        print("fix_cevo25_vfr_times: fixed", name, "vfr", vfr,
              "frame_rate ->", d["metadata"]["frame_rate"])


def dofix():
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
