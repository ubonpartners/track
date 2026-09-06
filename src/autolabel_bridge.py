"""Optional bridge to the autolabel pipeline (github.com/ubonpartners/autolabel).

track never hard-depends on autolabel: everything here imports it lazily
and raises a helpful RuntimeError when it is missing. Two capabilities:

1. augment_trackset_file(): add HIGH-CONFIDENCE autolabel tracks that are
   missing from a partially-annotated GT trackset (MEVA/OTW label only
   scripted-activity actors; real bystanders are unlabelled, which makes
   plain MOT scoring charge correct detections as FP). Existing GT tracks
   are never modified; candidates that duplicate GT are dropped.
2. autolabel_video(): run the full autolabel pipeline on one mp4 and
   return the trackset JSON (already in track's annotation format) — the
   basis of the generic "autolabel a folder of videos" importer.

Requirements when actually used: the autolabel checkout (sibling dir of
this repo, or $AUTOLABEL_PATH), its weights/ artifacts, and a GPU env
that can run its detectors (dataset-processor2; the rfdetr channel
spawns its own conda env internally).
"""
import json
import os
import sys
import time

import src.paths as paths
import traceback

_HELP = (
    "autolabel is required for this importer but could not be imported.\n"
    "  - clone github.com/ubonpartners/autolabel next to the track repo\n"
    "    (or set AUTOLABEL_PATH to the checkout)\n"
    "  - provision weights: autolabel/scripts/download_weights.sh\n"
    "  - run from a GPU env with its detector deps (dataset-processor2)\n"
    "Only autolabel-augmented importers need this; everything else in "
    "track works without it."
)


def _autolabel_root():
    cand = [paths.autolabel_repo(), paths.autolabel_sibling()]   # $AUTOLABEL_PATH first, sibling as fallback
    for c in cand:
        if c and os.path.isfile(os.path.join(c, "autolabel", "pipeline.py")):
            return c
    return None


def load_autolabel():
    """Import and return the autolabel package modules, or raise with a
    setup-help message. Import is deferred so track has no hard dep."""
    root = _autolabel_root()
    if root is None:
        raise RuntimeError(_HELP)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from autolabel import pipeline  # noqa: F401
        from autolabel.config import load_config  # noqa: F401
    except ImportError as e:
        raise RuntimeError(f"{_HELP}\n(import failed: {e})") from e
    from autolabel import pipeline as _p
    from autolabel import config as _c
    return _p, _c


def autolabel_video(video_path, out_json, convention="fullbody", cuts=False):
    """Full autolabel pipeline on one video -> trackset JSON path.
    Skips if out_json already exists (resume semantics). cuts=True enables
    autolabel's scene-cut detection (TransNetV2) for edited multi-shot
    content like movies; leave off for single-shot camera footage."""
    if os.path.isfile(out_json):
        # Existence alone is not completion: an interrupted direct JSON write
        # from older versions can leave a truncated file.
        with open(out_json) as fh:
            json.load(fh)
        return out_json
    pipeline, _config = load_autolabel()
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    tmp = out_json + f".tmp{os.getpid()}"
    kw = {}
    if cuts:
        cfg = _config.load_config()
        cfg.cuts = True
        kw["config"] = cfg
    try:
        pipeline.run(video_path, tmp, convention=convention, **kw)
        with open(tmp) as fh:
            json.load(fh)
        os.replace(tmp, out_json)
    finally:
        if os.path.isfile(tmp):
            os.unlink(tmp)
    return out_json


def _track_groups(doc, want_classes=(0, 1)):
    """autolabel export -> {track_id: [(frame_time, box, conf, cl)]}."""
    tracks = {}
    for f in doc["frames"]:
        t = float(f["frame_time"])
        for tid, o in f["objects"].items():
            if int(o["class"]) not in want_classes:
                continue
            tracks.setdefault(str(tid), []).append(
                (t, o["box"], float(o.get("conf", 1.0)), int(o["class"])))
    for v in tracks.values():
        v.sort(key=lambda x: x[0])
    return tracks


def augment_trackset_file(anno_path, video_path=None, min_conf=0.55,
                          min_seconds=1.0, dup_iou=0.5, max_dup_frac=0.3,
                          keyframe_fps=5.0, work_dir=None, verbose=True):
    """Augment a partially-annotated GT trackset IN PLACE with missing
    high-confidence autolabel tracks. Idempotent (metadata flag).

    A candidate autolabel track is added iff:
      - median box confidence >= min_conf and duration >= min_seconds;
      - the fraction of its sampled frames whose box matches ANY same-
        class GT box (time-interpolated via objects_at_time) at
        IoU >= dup_iou is <= max_dup_frac (no duplication of existing GT).
    Added tracks are written as sparse keyframes (~keyframe_fps) with
    their autolabel confidences and per-box "source": "autolabel".
    Existing GT objects are never touched. Returns #tracks added, or
    None if the file was already augmented.
    """
    import numpy as np
    import src.trackset as trackset

    doc = json.load(open(anno_path))
    meta = doc.get("metadata", {})
    if meta.get("autolabel_augmented"):
        return None
    if video_path is None:
        video_path = meta.get("original_video")
    if not (video_path and os.path.isfile(video_path)):
        raise FileNotFoundError(f"video for {anno_path} not found: "
                                f"{video_path}")

    # Keep datasets separate: basename-only keys collide when two corpora use
    # the same annotation stem.
    dataset = os.path.basename(os.path.dirname(
        os.path.dirname(os.path.abspath(anno_path))))
    wd = work_dir or paths.autolabel_cache("v1", "augment", dataset)
    al_json = os.path.join(
        wd, os.path.splitext(os.path.basename(anno_path))[0]
        + ".autolabel.json")
    autolabel_video(video_path, al_json,
                    convention=meta.get("box_convention", "fullbody"))
    al = json.load(open(al_json))

    gt = trackset.TrackSet(anno_path)
    gt_classes = gt.metadata["classes"]
    W, H = meta["width"], meta["height"]

    def iou(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        i = ix * iy
        u = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - i)
        return i / u if u > 0 else 0.0

    next_id = 1 + max(
        [int(t) for f in doc["frames"] for t in f["objects"]
         if str(t).lstrip("-").isdigit()] or [0])
    by_time = {round(f["frame_time"], 6): f for f in doc["frames"]}
    all_times = sorted(by_time)
    added = 0
    kf_dt = 1.0 / keyframe_fps

    for tid, obs in _track_groups(al).items():
        dur = obs[-1][0] - obs[0][0]
        confs = [o[2] for o in obs]
        cl = obs[0][3]
        if dur < min_seconds or float(np.median(confs)) < min_conf:
            continue
        if all_times and (obs[-1][0] < all_times[0] - 1.0
                          or obs[0][0] > all_times[-1] + 1.0):
            continue  # entirely outside the annotated window
        # duplication test vs interpolated same-class GT
        sample = obs[:: max(1, len(obs) // 25)]
        dup = tot = 0
        cls_name = ["person", "vehicle", "other"][cl]
        for t, box, conf, _cl in sample:
            gtob = gt.objects_at_time(t) or []
            tot += 1
            for g in gtob:
                if gt_classes[g.cl] != cls_name:
                    continue
                if iou(box, g.box) >= dup_iou:
                    dup += 1
                    break
        if tot == 0 or dup / tot > max_dup_frac:
            continue
        # add as sparse keyframes on (or between) existing GT frames
        last_t = -1e9
        wrote = 0
        for t, box, conf, _cl in obs:
            if t - last_t < kf_dt - 1e-6:
                continue
            last_t = t
            key = round(t, 6)
            fr = by_time.get(key)
            if fr is None:
                fr = {"frame_id": int(round(t * meta.get("frame_rate", 30))),
                      "frame_time": key, "objects": {}}
                by_time[key] = fr
            fr["objects"][str(next_id)] = {
                "box": [round(float(v), 4) for v in box],
                "class": cl, "conf": round(float(conf), 4),
                "source": "autolabel"}
            wrote += 1
        if wrote:
            added += 1
            next_id += 1

    doc["frames"] = [by_time[k] for k in sorted(by_time)]
    if "frame_times" in doc:
        doc["frame_times"] = sorted(by_time)
    # completeness floor: autolabel cannot reliably add tracks below the
    # measured detector knee, so added-track completeness only holds
    # above it (normalized height; ~58/1280). Per-class dict schema.
    fl = dict(meta.get("min_annotated_height") or {})
    fl["person"] = max(float(fl.get("person", 0.0)), 0.045)
    meta["min_annotated_height"] = fl
    meta.pop("min_annotated_person_height", None)
    meta["autolabel_augmented"] = {
        "tracks_added": added, "min_conf": min_conf,
        "dup_iou": dup_iou, "max_dup_frac": max_dup_frac,
        "source": os.path.basename(al_json)}
    doc["metadata"] = meta
    tmp = anno_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=4)
    os.replace(tmp, anno_path)
    # a frame entry claims FULL annotation: materialize the sparse
    # keyframes onto the existing frame grid so added tracks are present
    # on every frame they span (no per-frame blinking)
    densify_augmented_tracks(anno_path, verbose=False)
    if verbose:
        print(f"  augmented {os.path.basename(anno_path)}: "
              f"+{added} autolabel tracks")
    return added


def _augment_one(job):
    anno_path, kw = job
    t0 = time.time()
    try:
        return anno_path, augment_trackset_file(anno_path, **kw), \
            time.time() - t0, None
    except Exception:
        return anno_path, None, time.time() - t0, traceback.format_exc()


def augment_dataset(folder, limit=0, names=None, manifest=None, workers=None,
                    **kw):
    """Augment annotations in <folder>/annotation in place (skip
    already-augmented). Manifest runs default to two long-lived GPU workers;
    set AUTOLABEL_AUGMENT_WORKERS=1 for serial execution. Restrict scope with
    names=[stems] or manifest=path-to-reduced_N.json (see
    trackset_import.reduce_dataset)."""
    anno_dir = os.path.join(folder, "annotation")
    if manifest:
        names = json.load(open(manifest))
    if names is not None:
        names = [n if n.endswith(".json") else n + ".json" for n in names]
        names = [n for n in sorted(names)
                 if os.path.isfile(os.path.join(anno_dir, n))]
    else:
        names = sorted(f for f in os.listdir(anno_dir)
                       if f.endswith(".json"))
    # Manifest augmentation is the large offline corpus path. Two warmed
    # pipelines use ~26.5 GB on the 32 GB production GPU; three have OOM'd.
    # Keep importer-triggered augmentation serial unless explicitly opted in.
    if workers is None:
        workers = int(os.environ.get(
            "AUTOLABEL_AUGMENT_WORKERS", "2" if manifest else "1"))
    workers = max(1, int(workers))
    jobs = [(os.path.join(anno_dir, n), kw) for n in names]
    if limit:
        workers = 1  # preserve exact historical "N newly done" semantics

    done = skipped = failed = completed = 0
    t0 = time.time()

    def account(result):
        nonlocal done, skipped, failed, completed
        anno, r, dt, err = result
        completed += 1
        if err:
            failed += 1
            print(f"FAIL {os.path.basename(anno)} ({dt:.0f}s)\n{err}",
                  flush=True)
        elif r is None:
            skipped += 1
        else:
            done += 1
        elapsed = time.time() - t0
        eta = elapsed / max(completed, 1) * (len(jobs) - completed)
        print(f"augment progress {completed}/{len(jobs)} "
              f"done={done} skip={skipped} fail={failed} "
              f"last={dt:.0f}s eta={eta / 3600:.1f}h", flush=True)

    if workers == 1:
        for job in jobs:
            result = _augment_one(job)
            account(result)
            if limit and done >= limit:
                break
    else:
        import multiprocessing as mp
        # The documented entry point is `python -c`, for which spawn cannot
        # re-import __main__. The parent has not initialized CUDA here, so a
        # Linux fork pool is safe and each child owns its model stack.
        with mp.get_context("fork").Pool(workers) as pool:
            for result in pool.imap_unordered(_augment_one, jobs,
                                              chunksize=1):
                account(result)
    print(f"augment_dataset {folder}: {done} augmented, "
          f"{skipped} already done, {failed} failed")
    return {"done": done, "skipped": skipped, "failed": failed}


def tighten_trackset_file(anno_path, match_iou=0.4, min_conf=0.5,
                          time_tol=0.08, work_dir=None, verbose=True):
    """Tighten loose human-annotated GT boxes IN PLACE from the clip's
    autolabel output; identities, spans and classes are untouched.
    Idempotent (metadata flag). Returns #boxes tightened, or None if the
    file was already tightened.

    MEVA-style GT is actor-enclosing keyframe-interpolated geometry —
    measured on school.G336: median 1.30x the area of the matched
    detector box, 6% of pairs under IoU 0.5, i.e. misses under IoU-0.5
    scoring even with perfect tracking. Per GT frame, same-class GT and
    autolabel boxes (nearest autolabel frame within time_tol; covers
    stride-2 exports) are matched 1:1 greedily by descending IoU; a GT
    box is replaced iff its match has IoU >= match_iou and
    conf >= min_conf, and gets "source": "autolabel_tight". Unmatched
    frames keep the original loose box, so a track's geometry can
    alternate; consumers wanting only tightened geometry can filter on
    the per-box source. Boxes added by augmentation
    ("source": "autolabel") are already tight and skipped.
    """
    import bisect

    import numpy as np

    doc = json.load(open(anno_path))
    meta = doc.get("metadata", {})
    if meta.get("autolabel_tightened"):
        return None

    dataset = os.path.basename(os.path.dirname(
        os.path.dirname(os.path.abspath(anno_path))))
    wd = work_dir or paths.autolabel_cache("v1", "augment", dataset)
    al_json = os.path.join(
        wd, os.path.splitext(os.path.basename(anno_path))[0]
        + ".autolabel.json")
    if not os.path.isfile(al_json):
        raise FileNotFoundError(
            f"no autolabel output for {anno_path}: {al_json} "
            "(run augment_dataset / autolabel_video first)")
    al = json.load(open(al_json))
    al_frames = sorted(
        ((float(f["frame_time"]),
          [(np.asarray(o["box"], float), float(o.get("conf", 1.0)),
            int(o["class"]))
           for o in f["objects"].values()])
         for f in al["frames"]), key=lambda x: x[0])
    al_times = [t for t, _ in al_frames]

    def iou(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        i = ix * iy
        u = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - i)
        return i / u if u > 0 else 0.0

    tightened = total = 0
    for f in doc["frames"]:
        t = float(f["frame_time"])
        k = bisect.bisect_left(al_times, t)
        best = None
        for kk in (k - 1, k):
            if 0 <= kk < len(al_times) and abs(al_times[kk] - t) <= time_tol:
                if best is None or (abs(al_times[kk] - t)
                                    < abs(al_times[best] - t)):
                    best = kk
        gt_objs = [(tid, o) for tid, o in f["objects"].items()
                   if o.get("source") not in ("autolabel", "autolabel_tight")]
        total += len(gt_objs)
        if best is None or not gt_objs:
            continue
        cands = [(b, c, cl) for b, c, cl in al_frames[best][1]
                 if c >= min_conf]
        # greedy 1:1 by descending IoU (a detection must not tighten two
        # different GT people)
        pairs = []
        for gi, (tid, o) in enumerate(gt_objs):
            for ai, (b, c, cl) in enumerate(cands):
                if cl != int(o.get("class", 0)):
                    continue
                v = iou(o["box"], b)
                if v >= match_iou:
                    pairs.append((v, gi, ai))
        pairs.sort(reverse=True)
        used_g, used_a = set(), set()
        for v, gi, ai in pairs:
            if gi in used_g or ai in used_a:
                continue
            used_g.add(gi)
            used_a.add(ai)
            tid, o = gt_objs[gi]
            o["box"] = [round(float(x), 4) for x in cands[ai][0]]
            o["source"] = "autolabel_tight"
            tightened += 1

    meta["autolabel_tightened"] = {
        "boxes_tightened": tightened, "boxes_total": total,
        "match_iou": match_iou, "min_conf": min_conf,
        "time_tol": time_tol, "source": os.path.basename(al_json)}
    doc["metadata"] = meta
    tmp = anno_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=4)
    os.replace(tmp, anno_path)
    make_tight_consistent(anno_path, verbose=False)
    if verbose:
        print(f"  tightened {os.path.basename(anno_path)}: "
              f"{tightened}/{total} boxes")
    return tightened


def tighten_dataset(folder, limit=0, names=None, **kw):
    """Tighten annotations in <folder>/annotation in place (skip
    already-tightened; skip-and-count files with no autolabel output).
    Run AFTER augment_dataset — it produces the per-clip autolabel JSONs
    this reads. CPU-only and fast (~seconds per file)."""
    anno_dir = os.path.join(folder, "annotation")
    if names is not None:
        names = [n if n.endswith(".json") else n + ".json" for n in names]
        names = [n for n in sorted(names)
                 if os.path.isfile(os.path.join(anno_dir, n))]
    else:
        names = sorted(f for f in os.listdir(anno_dir)
                       if f.endswith(".json"))
    done = skipped = missing = failed = 0
    for n in names:
        try:
            r = tighten_trackset_file(os.path.join(anno_dir, n), **kw)
            if r is None:
                skipped += 1
            else:
                done += 1
                if limit and done >= limit:
                    break
        except FileNotFoundError:
            missing += 1
        except Exception:
            failed += 1
            print(f"FAIL {n}\n{traceback.format_exc()}", flush=True)
    print(f"tighten_dataset {folder}: {done} tightened, {skipped} already "
          f"done, {missing} no-autolabel, {failed} failed")
    return {"done": done, "skipped": skipped, "missing": missing,
            "failed": failed}


def make_tight_consistent(anno_path, verbose=True):
    """Remove per-frame loose/tight geometry alternation left by
    tightening IN PLACE. v1 tightening replaced only confidently-matched
    frames, so a track could flip between loose GT geometry and tight
    detector geometry frame to frame — visible as box "flashing",
    especially on small boxes (194/314 tracks mixed on school.G336).

    For every track that has at least one tightened box: interior runs
    of loose frames between tightened frames get linearly interpolated
    tight geometry; leading/trailing loose runs keep the loose box
    CENTER (approximately unbiased) with the nearest tightened box's
    width/height. Synthesized boxes get "source":
    "autolabel_tight_interp". Idempotent (flag inside
    metadata.autolabel_tightened); tighten_trackset_file calls this
    automatically, so standalone use is only needed for files tightened
    before the consistency pass existed. Returns #boxes rewritten, or
    None if already consistent or never tightened.
    """
    import numpy as np

    doc = json.load(open(anno_path))
    meta = doc.get("metadata", {})
    flag = meta.get("autolabel_tightened")
    if not flag or flag.get("consistent"):
        return None

    # per track: ordered (frame_obj, is_tight, time)
    tracks = {}
    for f in doc["frames"]:
        t = float(f["frame_time"])
        for tid, o in f["objects"].items():
            if o.get("source") == "autolabel":
                continue
            tracks.setdefault(tid, []).append((t, o))
    rewritten = 0
    for tid, obs in tracks.items():
        obs.sort(key=lambda x: x[0])
        tight = [i for i, (_, o) in enumerate(obs)
                 if o.get("source") in ("autolabel_tight",
                                        "autolabel_tight_interp")]
        if not tight:
            continue                       # fully loose track: consistent
        tset = set(tight)
        for i, (t, o) in enumerate(obs):
            if i in tset:
                continue
            prev = max((j for j in tight if j < i), default=None)
            nxt = min((j for j in tight if j > i), default=None)
            if prev is not None and nxt is not None:
                a = np.asarray(obs[prev][1]["box"], float)
                b = np.asarray(obs[nxt][1]["box"], float)
                ta, tb = obs[prev][0], obs[nxt][0]
                w = (t - ta) / max(tb - ta, 1e-9)
                box = a * (1 - w) + b * w
            else:
                near = prev if prev is not None else nxt
                nb = np.asarray(obs[near][1]["box"], float)
                lw, lh = nb[2] - nb[0], nb[3] - nb[1]
                g = np.asarray(o["box"], float)
                cx, cy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
                box = np.array([cx - lw / 2, cy - lh / 2,
                                cx + lw / 2, cy + lh / 2])
            o["box"] = [round(float(v), 4) for v in box]
            o["source"] = "autolabel_tight_interp"
            rewritten += 1
    flag["consistent"] = True
    flag["boxes_interp"] = flag.get("boxes_interp", 0) + rewritten
    doc["metadata"] = meta
    tmp = anno_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=4)
    os.replace(tmp, anno_path)
    if verbose:
        print(f"  consistent {os.path.basename(anno_path)}: "
              f"{rewritten} boxes interpolated")
    return rewritten


def _motion_transfer(Ha, Hb, Aa, Ab, At, w):
    """Warp lerp(Ha,Hb,w) by autolabel motion: center residual (relative
    to autolabel's own anchor lerp) plus size ratio. Box conventions
    cancel; zero residual == plain linear interpolation."""
    def cs(b):
        return ((b[0]+b[2])/2, (b[1]+b[3])/2, b[2]-b[0], b[3]-b[1])
    hxa, hya, hwa, hha = cs(Ha); hxb, hyb, hwb, hhb = cs(Hb)
    axa, aya, awa, aha = cs(Aa); axb, ayb, awb, ahb = cs(Ab)
    axt, ayt, awt, aht = cs(At)
    cx = (hxa + (hxb-hxa)*w) + (axt - (axa + (axb-axa)*w))
    cy = (hya + (hyb-hya)*w) + (ayt - (aya + (ayb-aya)*w))
    def size(h0, h1, a0, a1, at):
        base = h0 + (h1-h0)*w
        alin = a0 + (a1-a0)*w
        return base * (at/alin) if alin > 1e-6 and at > 1e-6 else base
    bw = size(hwa, hwb, awa, awb, awt)
    bh = size(hha, hhb, aha, ahb, aht)
    return [cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2]


def densify_sparse_gt(anno_path, anchor_iomin=0.55, max_gap=1.5,
                      work_dir=None, verbose=True):
    """Densify SPARSE HUMAN GT (metadata.sparse_gt — e.g. antare bodycam
    1Hz annotations) with autolabel in-betweens. Identity stays HUMAN;
    geometry between anchors comes from autolabel where it corroborates,
    linear interpolation otherwise:

      1. autolabel_video() on metadata.original_video (the retained-cadence
         chunk — full analytics framerate, single-shot so cuts=False);
      2. for each human track and consecutive anchor pair closer than
         max_gap: the autolabel track matching BOTH anchors (same mapped
         class, intersection-over-min-area >= anchor_iomin at each)
         supplies MOTION between them: the fill box is the lerp of the
         human anchors warped by autolabel's center residual and size
         ratio ("source": "autolabel_gtfill"). Motion transfer, not box
         copy: autolabel exports FULL-BODY boxes while MOT-style GT is
         visible-box (measured IoU median 0.35 on antare escooter — the
         visible pipeline path needs aux caches the current detector
         set no longer writes), so absolute geometry is unusable but
         relative motion is convention-free; IoMin likewise survives
         the fullbody-superset offset where plain IoU does not. Where
         autolabel motion is linear this reduces to exact linear interp;
      3. the dense grid = human frames + all matched in-between times, and
         EVERY live human track gets a box on EVERY grid frame in its span
         (linear fallback, "source": "interp") — a frame carrying only
         some tracks would break the reader's bracket interpolation for
         the absent ones;
      4. idempotent via metadata.sparse_gt.densified.

    Run augment_trackset_file afterwards for people the humans missed.
    Linear-only spans remain exactly as trustworthy as the old
    reader-interpolation — no better, marked so ("interp").
    """
    import bisect
    doc = json.load(open(anno_path))
    meta = doc.get("metadata", {})
    sparse = meta.get("sparse_gt")
    if not sparse or sparse.get("densified"):
        return None
    video = meta["original_video"]
    if work_dir is None:
        work_dir = os.path.join(os.path.dirname(anno_path), "autolabel_work")
    os.makedirs(work_dir, exist_ok=True)
    al_json = os.path.join(
        work_dir, os.path.splitext(os.path.basename(video))[0] + ".autolabel.json")
    autolabel_video(video, al_json, cuts=False)
    al = json.load(open(al_json))
    al_classes = al["metadata"]["classes"]
    gt_classes = meta["classes"]

    def iomin(a, b):
        ix = min(a[2], b[2]) - max(a[0], b[0])
        iy = min(a[3], b[3]) - max(a[1], b[1])
        if ix <= 0 or iy <= 0:
            return 0.0
        amin = min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))
        return ix * iy / amin if amin > 0 else 0.0

    # autolabel observations: tid -> sorted [(t, box, cls_name, conf)]
    al_tracks = {}
    for fr in al["frames"]:
        t = fr["frame_time"]
        for tid, o in (fr.get("objects") or {}).items():
            cls = al_classes[o["class"]] if o.get("class") is not None else "person"
            al_tracks.setdefault(tid, []).append(
                (t, o["box"], cls, o.get("conf", o.get("confidence", 1.0))))
    for obs in al_tracks.values():
        obs.sort(key=lambda x: x[0])

    def al_box_at(tid, t, max_bracket=0.5):
        """Autolabel track's box AT time t: exact/near sample, else lerp
        between the bracketing samples. A hard nearest-sample tolerance
        breaks when the GT anchor cadence is an odd multiple of the
        autolabel sampling stride (justin: anchors every 15 retained
        frames, autolabel every 2 -> alternate anchors sit half a grid
        step off and EVERY anchor pair failed -> 0 matches)."""
        import bisect
        obs = al_tracks[tid]
        times = [o[0] for o in obs]
        i = bisect.bisect_left(times, t)
        lo = obs[i - 1] if i > 0 else None
        hi = obs[i] if i < len(obs) else None
        for o in (lo, hi):
            if o and abs(o[0] - t) <= 0.02:
                return o
        if lo and hi and hi[0] - lo[0] <= max_bracket:
            w = (t - lo[0]) / (hi[0] - lo[0])
            box = [a * (1 - w) + b * w for a, b in zip(lo[1], hi[1])]
            return (t, box, lo[2], min(lo[3], hi[3]))
        return None

    # human observations: tid -> sorted [(t, rec)]
    frames = sorted(doc["frames"], key=lambda f: f["frame_time"])
    human = {}
    for fr in frames:
        for tid, o in (fr.get("objects") or {}).items():
            human.setdefault(tid, []).append((fr["frame_time"], o))
    for obs in human.values():
        obs.sort(key=lambda x: x[0])

    # matched fills: (tid, t) -> (box, conf)
    fills = {}
    n_pairs = n_matched = 0
    for tid, obs in human.items():
        for (ta, oa), (tb, ob) in zip(obs, obs[1:]):
            if not 0 < tb - ta <= max_gap:
                continue
            n_pairs += 1
            want_cls = gt_classes[oa["class"]]
            best = None  # (tid, A, B)
            best_score = anchor_iomin
            for al_tid in al_tracks:
                A = al_box_at(al_tid, ta)
                B = al_box_at(al_tid, tb)
                if not A or not B or A[2] != want_cls:
                    continue
                score = min(iomin(A[1], oa["box"]), iomin(B[1], ob["box"]))
                if score >= best_score:
                    best, best_score = (al_tid, A[1], B[1]), score
            if best is None:
                continue
            n_matched += 1
            al_tid, Aa, Ab = best
            Ha, Hb = oa["box"], ob["box"]
            for (t, box, _cls, conf) in al_tracks[al_tid]:
                if not ta < t < tb:
                    continue
                w = (t - ta) / (tb - ta)
                fills[(tid, round(t, 4))] = (
                    _motion_transfer(Ha, Hb, Aa, Ab, box, w),
                    conf, oa["class"])

    # dense grid = human times + fill times (fills landing within 20ms of
    # a human frame merge into it — 2ms twin frames confuse readers);
    # every live human track covered on every grid frame IT EXISTS ON
    human_times = sorted({round(f["frame_time"], 4) for f in frames})
    def _snap(t):
        i = bisect.bisect_left(human_times, t)
        for j in (i - 1, i):
            if 0 <= j < len(human_times) and abs(human_times[j] - t) <= 0.02:
                return human_times[j]
        return t
    fills = {(tid, _snap(t)): v for (tid, t), v in fills.items()}
    grid = sorted(set(human_times) | {t for (_tid, t) in fills})
    by_time = {round(f["frame_time"], 4): f for f in frames}
    added_fill = added_interp = 0
    for t in grid:
        fr = by_time.get(t)
        if fr is None:
            fr = {"frame_id": 0, "frame_time": t, "objects": {}}
            by_time[t] = fr
        for tid, obs in human.items():
            if tid in fr["objects"]:
                continue
            # only between CONSECUTIVE anchors <= max_gap apart: a human
            # track absent from in-between anchor frames is genuinely
            # absent (off-frame/occluded — annotators label every image),
            # and span-wide fill painted mid-air boxes across those gaps
            times_h = [o[0] for o in obs]
            i = bisect.bisect_right(times_h, t) - 1
            if not (0 <= i < len(obs) - 1):
                continue
            ta_, tb_ = times_h[i], times_h[i + 1]
            if not (ta_ < t < tb_) or tb_ - ta_ > max_gap:
                continue
            key = (tid, t)
            if key in fills:
                box, conf, cls = fills[key]
                fr["objects"][tid] = {"box": [round(v, 5) for v in box],
                                      "class": cls, "conf": round(conf, 4),
                                      "source": "autolabel_gtfill"}
                added_fill += 1
            else:
                (ta, oa), (tb, ob) = obs[i], obs[i + 1]
                w = (t - ta) / (tb - ta)
                box = [a * (1 - w) + b * w
                       for a, b in zip(oa["box"], ob["box"])]
                fr["objects"][tid] = {"box": [round(v, 5) for v in box],
                                      "class": oa["class"], "conf": 1.0,
                                      "source": "interp"}
                added_interp += 1

    out_frames = [by_time[t] for t in grid]
    for i, fr in enumerate(out_frames):
        fr["frame_id"] = i + 1
    doc["frames"] = out_frames
    sparse["densified"] = True
    sparse["anchor_pairs"] = n_pairs
    sparse["anchor_pairs_autolabel_matched"] = n_matched
    sparse["boxes_autolabel_fill"] = added_fill
    sparse["boxes_linear_interp"] = added_interp
    tmp = anno_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f)
    os.replace(tmp, anno_path)
    if verbose:
        print(f"  densified {os.path.basename(anno_path)}: "
              f"{n_matched}/{n_pairs} anchor pairs matched, "
              f"+{added_fill} autolabel boxes, +{added_interp} interp",
              flush=True)
    return added_fill


def densify_augmented_tracks(anno_path, max_gap=2.0, verbose=True):
    """Materialize augmentation-added tracks onto the GT frame grid IN
    PLACE. augment_trackset_file writes added tracks as sparse ~5fps
    keyframes ("the reader interpolates"), but on dense per-frame GT
    (MEVA: a box every 33ms) that convention mismatch makes added tracks
    blink — present one frame in six (X.....X.) to any per-frame
    consumer. For each consecutive keyframe pair closer than max_gap,
    every EXISTING annotation frame strictly between them gets a
    linearly interpolated box (box and conf); no new frames are created,
    and gaps wider than max_gap are left absent — a long gap is genuine
    occlusion, not keyframe spacing, and interpolating it would fabricate
    an unsupported trajectory. Idempotent (flag inside
    metadata.autolabel_augmented); augment_trackset_file calls this, so
    standalone use is only needed for files augmented before it existed.
    Returns #boxes added, or None if already dense (or never augmented).
    """
    import bisect

    import numpy as np

    doc = json.load(open(anno_path))
    meta = doc.get("metadata", {})
    flag = meta.get("autolabel_augmented")
    if not flag or flag.get("dense"):
        return None

    frames = sorted(doc["frames"], key=lambda f: f["frame_time"])
    times = [float(f["frame_time"]) for f in frames]
    tracks = {}
    for i, f in enumerate(frames):
        for tid, o in f["objects"].items():
            if o.get("source") == "autolabel":
                tracks.setdefault(tid, []).append((i, o))
    added = 0
    for tid, obs in tracks.items():
        for (ia, oa), (ib, ob) in zip(obs, obs[1:]):
            ta, tb = times[ia], times[ib]
            if not 0 < tb - ta <= max_gap or ib - ia < 2:
                continue
            a = np.asarray(oa["box"], float)
            b = np.asarray(ob["box"], float)
            ca, cb = float(oa.get("conf", 1.0)), float(ob.get("conf", 1.0))
            for k in range(ia + 1, ib):
                if tid in frames[k]["objects"]:
                    continue
                w = (times[k] - ta) / max(tb - ta, 1e-9)
                box = a * (1 - w) + b * w
                frames[k]["objects"][tid] = {
                    "box": [round(float(v), 4) for v in box],
                    "class": int(oa["class"]),
                    "conf": round(ca * (1 - w) + cb * w, 4),
                    "source": "autolabel"}
                added += 1
    flag["dense"] = True
    flag["boxes_densified"] = flag.get("boxes_densified", 0) + added
    doc["metadata"] = meta
    tmp = anno_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=4)
    os.replace(tmp, anno_path)
    if verbose:
        print(f"  densified {os.path.basename(anno_path)}: "
              f"+{added} interpolated boxes")
    return added
