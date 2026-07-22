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
    cand = [os.environ.get("AUTOLABEL_PATH")]
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cand.append(os.path.join(os.path.dirname(here), "autolabel"))
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


def autolabel_video(video_path, out_json, convention="fullbody"):
    """Full autolabel pipeline on one video -> trackset JSON path.
    Skips if out_json already exists (resume semantics)."""
    if os.path.isfile(out_json):
        # Existence alone is not completion: an interrupted direct JSON write
        # from older versions can leave a truncated file.
        with open(out_json) as fh:
            json.load(fh)
        return out_json
    pipeline, _config = load_autolabel()
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    tmp = out_json + f".tmp{os.getpid()}"
    try:
        pipeline.run(video_path, tmp, convention=convention)
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
    wd = work_dir or os.path.join("/mldata/autolabel_cache/v1/augment",
                                  dataset)
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
