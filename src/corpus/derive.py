"""Tier-2 derivation (repo_cleanup.md stage 4d): the eval-spec transcode +
annotation rewrite (was src/dataset_lite.py) and derive_tracking /
check_tracking (were in src/corpus_manifest.py).

Eval-spec view of a corpus (MB spec 2026-07-23/24):
  - longest side capped at 1280 (never upscaled; boxes are normalized so
    annotations are resolution-free — only metadata w/h changes);
  - framerate reduced by an INTEGER divisor N (keep every Nth frame) to the
    analytics grid the tracker config's min_time_delta_process selects for
    the camera hint (or a forced --divisor); retained frames restamped to
    an exact N/fps grid and the annotation subset to them
    (rewrite_annotation);
  - no B-frames, I+P only, GOP ~2s; audio preserved when the source has it;
  - optional hard duration cap (MEVA: 120s), plus <root>/video_autolabel/
    stream-copied native-rate clips for autolabel input when capped;
  - optional quarantine of clips with backward pts (OTW doorbell jitter)
    into <root>/dropped_jitter/.

AUTOLABEL ORDERING: auto-annotation always runs on the NATIVE-framerate
source; the fps drop is applied to the resulting annotation here. Time-
trimming is safe, rate-drop input is not.

CLIs: `python -m src.corpus.manifest derive|check <corpus>` is the tier
pipeline; `python -m src.corpus.derive --root <dir> --hint ...` is the
legacy in-place lite pass (process_dataset) on a <root>/annotation tree.
ffmpeg IS allowed here: offline dataset preparation, not the ai-node runtime.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys

from src.corpus.manifest import load_capabilities
import src.paths as paths
import src.corpus.media as media
from src.corpus.media import MP4_COPY_AUDIO, audio_args, probe_audio  # noqa: F401 (re-exported for the dataset_lite shim)
from src.corpus.media import has_backward_pts, scale_dims  # noqa: F401 (used here and re-exported)


def choose_divisor(src_fps, min_fps):
    """Largest integer N with src_fps/N >= min_fps (N>=1) — the LOWEST
    resulting framerate that still meets the floor."""
    if src_fps <= min_fps:
        return 1
    return max(1, int(math.floor(src_fps / float(min_fps) + 1e-9)))


def min_delta_from_config(hint, config_path=None):
    """min_time_delta_process for a camera class, read from the production
    tracker config. hint='bodycam' (any moving camera) reads the
    `(hint:bodycam)` variant when present; 'static' (or a hint with no
    variant) reads the base key. Imports derive their decimation from
    THIS so a config change only requires re-running the import."""
    import yaml
    cfg = yaml.safe_load(open(config_path or paths.tracker_config()))
    key = "min_time_delta_process"
    if hint and hint != "static":
        v = cfg.get(f"{key}(hint:{hint})")
        if v is not None:
            return float(v)
    return float(cfg[key])


def divisor_from_config(src_fps, hint, config_path=None):
    """Smallest integer N with N/src_fps >= min_time_delta_process: the
    decimated grid is then exactly the frame set the tracker's
    min-interval gate selects from the native stream (assuming clean CFR
    stamps), so the decimation is analytics-equivalent by construction —
    the gate passes every retained frame and no un-retained frame would
    ever have been processed."""
    delta = min_delta_from_config(hint, config_path)
    return max(1, int(math.ceil(delta * src_fps - 1e-9)))


def probe(video):
    """(width, height, fps) of the first video stream."""
    i = media.probe_video(video)
    if i.fps <= 0:
        raise ValueError(f"{video}: ffprobe cannot express the frame rate ({i.fps}); refusing to derive on it")
    return i.width, i.height, i.fps


def transcode(src, dst, divisor, dims, out_fps, max_seconds, dry_run=False):
    """The tier-2 eval-spec encode: keep every divisor-th frame, restamp
    onto a clean out_fps grid, scale to dims, I+P only, ~2 s GOP, audio
    preserved. NVENC first, x264 fallback (media.transcode). Written
    straight to dst: derive_tracking passes its own temp name."""
    vf = []
    if divisor > 1:
        vf.append(f"select='not(mod(n\\,{divisor}))'")
    # clean-CFR restamp onto the analytics grid: jittery/VFR source pts
    # can land 2-3% off-grid, inside the tracker gate's ~2.5-4% margin
    # (uc_v11 min_time_delta comment) — synthetic stamps make the grid
    # exact by construction; rewrite_annotation snaps GT times to match
    vf.append(f"setpts=N/({out_fps}*TB)")
    w, h = dims
    vf.append(f"scale={w}:{h}")
    gop = max(1, int(round(2 * out_fps)))
    pre = []
    if max_seconds is not None:
        pre = ["-t", str(max_seconds + 0.5)]  # pad half a frame period; the select+json caps exactly
    # NVENC session churn fails intermittently (exit 187) on long batch
    # runs; x264 keeps the same I+P contract (-bf 0)
    return media.transcode(src, dst, [media.NVENC_P4_CQ23, media.X264_VERYFAST_CRF23],
                           tmp=None, vf=vf, pre_input=pre, fps_mode="vfr",
                           gop=gop, bf=0, dry_run=dry_run)


def rewrite_annotation(d, src_fps, divisor, dims, max_seconds, lite_video,
                       hint=None, min_delta=None):
    """Subset an annotation dict to the retained frames + refresh metadata.
    Pure function (unit-tested); returns (new_dict, kept, dropped)."""
    md = dict(d.get("metadata") or {})
    frames_in = d.get("frames") or []
    # frame-selection index: the video transcode keeps frames by decode
    # ORDINAL (select not(mod(n,N))), so when the annotation's frame_ids
    # are dense+consecutive use the same ordinal — exact for VFR sources
    # (cevo_april25: pts-true times drift vs round(t*avg_fps) by frames).
    # Sparse/gappy ids (meva capture ticks) fall back to the time grid.
    ids = [fr.get("frame_id") for fr in frames_in]
    dense = (len(ids) > 1 and all(isinstance(i, int) for i in ids)
             and ids == list(range(ids[0], ids[0] + len(ids))))
    if dense:
        # The ordinal path additionally requires that annotation frames
        # BE the video frames (ordinal == video ordinal). Stride-N
        # annotations (raw_movies: autolabel ran every 2nd frame) have
        # dense ids too, but their ordinal is 1/N of the video ordinal —
        # using it halves every rewritten timestamp. Guard: the
        # annotation's implied rate must match src_fps (cevo's pts-true
        # VFR drift is a per-frame jitter, not a rate change, so it
        # still passes); otherwise fall back to the time grid.
        span = frames_in[-1].get("frame_time", 0.0) - frames_in[0].get("frame_time", 0.0)
        if span > 0:
            implied = (len(frames_in) - 1) / span
            if abs(implied - src_fps) > 0.2 * src_fps:
                dense = False
    kept, dropped = [], 0
    for k, fr in enumerate(frames_in):
        t = fr.get("frame_time", 0.0)
        if max_seconds is not None and t > max_seconds:
            dropped += 1
            continue
        idx = (ids[k] - ids[0]) if dense else int(round(t * src_fps))
        if idx % divisor != 0:
            dropped += 1
            continue
        fr = dict(fr)
        # snap to the same clean grid the transcode restamps the video to
        # (source pts jitter must not leak into eval timing)
        fr["frame_time"] = round(idx / src_fps, 6)
        kept.append(fr)
    md["frame_rate"] = src_fps / float(divisor)
    md["width"], md["height"] = dims
    md["original_video"] = lite_video
    if hint:
        # camera class: evals bind `(hint:x)` config variants through this
        md["hint"] = hint
    md["lite"] = {"source_fps": src_fps, "divisor": divisor,
                  "max_seconds": max_seconds,
                  "hint": hint, "min_time_delta": min_delta}
    out = dict(d)
    out["metadata"] = md
    out["frames"] = kept
    return out, len(kept), dropped


def process_dataset(root, min_fps=None, max_seconds=None, drop_jitter=False,
                    max_edge=1280, hint=None,
                    config_path=None):
    """hint mode (preferred): divisor per clip from the tracker config's
    min_time_delta_process for that camera class (analytics-equivalent
    decimation; re-run after a config change). min_fps mode (legacy):
    fixed framerate floor."""
    if (min_fps is None) == (hint is None):
        raise ValueError("pass exactly one of min_fps / hint")
    anno_dir = os.path.join(root, "annotation")
    lite_dir = os.path.join(root, "video_lite")
    os.makedirs(lite_dir, exist_ok=True)
    done = skipped = jittered = 0
    dropped_clips = []
    for name in sorted(os.listdir(anno_dir)):
        if not name.endswith(".json"):
            continue
        jpath = os.path.join(anno_dir, name)
        d = json.load(open(jpath))
        md = d.get("metadata") or {}
        if md.get("lite"):
            skipped += 1
            continue
        src = md.get("original_video")
        if not src or not os.path.isfile(src):
            print(f"  {name}: source video missing ({src}) — SKIPPED", flush=True)
            continue
        if drop_jitter and has_backward_pts(src):
            # broken stamps: quarantine video+annotation, never delete
            qdir = os.path.join(root, "dropped_jitter")
            os.makedirs(qdir, exist_ok=True)
            shutil.move(jpath, os.path.join(qdir, name))
            shutil.copy2(src, os.path.join(qdir, os.path.basename(src)))
            dropped_clips.append(name[:-5])
            jittered += 1
            print(f"  {name[:-5]}: BACKWARD PTS — dropped to {qdir}", flush=True)
            continue
        w, h, fps = probe(src)
        min_delta = None
        if hint is not None:
            min_delta = min_delta_from_config(hint, config_path)
            divisor = divisor_from_config(fps, hint, config_path)
        else:
            divisor = choose_divisor(fps, min_fps)
        dims = scale_dims(w, h, max_edge)
        out_fps = fps / divisor
        dst = os.path.join(lite_dir, os.path.splitext(os.path.basename(src))[0] + ".mp4")
        transcode(src, dst, divisor, dims, out_fps, max_seconds)
        if max_seconds is not None:
            # native-rate autolabel input for the trimmed window (see header)
            adir = os.path.join(root, "video_autolabel")
            os.makedirs(adir, exist_ok=True)
            adst = os.path.join(adir, os.path.basename(dst))
            if not os.path.exists(adst):
                subprocess.check_call(
                    ["ffmpeg", "-y", "-loglevel", "error", "-t", str(max_seconds),
                     "-i", src, "-c", "copy", adst])
        new, kept, droppedf = rewrite_annotation(d, fps, divisor, dims,
                                                 max_seconds, dst,
                                                 hint=hint,
                                                 min_delta=min_delta)
        orig = jpath + ".orig"
        if not os.path.exists(orig):
            shutil.copy2(jpath, orig)
        tmp = jpath + ".tmp"
        with open(tmp, "w") as f:
            json.dump(new, f)
        os.replace(tmp, jpath)
        done += 1
        print(f"  {name[:-5]}: {w}x{h}@{fps:.2f} /{divisor} -> {dims[0]}x{dims[1]}@{out_fps:.2f} "
              f"frames {kept} kept / {droppedf} dropped", flush=True)
    print(f"{root}: {done} transcoded, {skipped} already lite, "
          f"{jittered} dropped for jitter {dropped_clips or ''}", flush=True)
    return dropped_clips


def main():
    ap = argparse.ArgumentParser(description="In-place lite pass over <root>/annotation (legacy; the tier pipeline is src.corpus.manifest derive).")
    ap.add_argument("--root", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--hint", choices=["static", "bodycam"],
                   help="camera class: divisor derived from the tracker "
                        "config's min_time_delta_process (re-run after a "
                        "config change)")
    g.add_argument("--min-fps", type=float,
                   help="legacy fixed framerate floor")
    ap.add_argument("--config", default=None, metavar="YAML",
                    help="tracker config for --hint mode (default: the production config, see src/paths.py)")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--drop-jitter", action="store_true")
    ap.add_argument("--max-edge", type=int, default=1280)
    a = ap.parse_args()
    process_dataset(a.root, min_fps=a.min_fps, max_seconds=a.max_seconds,
                    drop_jitter=a.drop_jitter, max_edge=a.max_edge,
                    hint=a.hint, config_path=a.config)


def derive_tracking(corpus, hint=None, max_seconds=None, hint_overrides=None,
                    divisor=None):
    """tier 1 -> tier 2 EVAL-SPEC derivation (MB spec 2026-07-24): for
    every tier-1 video+annotation pair, produce in tier 2 the version
    track.py actually evaluates — resolution capped at 1280, framerate
    decimated to the analytics grid the tracker config's
    min_time_delta_process (per camera-class hint) selects, I+P-only
    h264, source audio preserved (AAC) — plus the annotation subset to the retained frames
    (native-grid truth stays in tier 1; a tracker-config change only
    needs a re-derive, never a re-autolabel). No video_lite/, no
    generated_h264/, no h264 cache: the tier-2 mp4 is the one eval
    artifact, ingested directly via run_on_mp4_file.

    `divisor` (optional) FORCES the framerate divisor instead of deriving
    it from the hint (e.g. 1 = keep the native frame timing; antare_bwc
    2026-09-06: 10 fps source kept at 10 fps by MB instruction, where the
    bodycam gate would have halved it). Recorded in the recipe so
    `check` judges the clip against the same grid.

    The recipe (hint/max_seconds) is recorded in tier 2 as
    derive_recipe.json on first use, so a bare `derive <corpus>` refresh
    reuses it. Skip-if-current: a clip is re-derived when its tier-1
    video or annotation is newer than the tier-2 annotation."""
    import json as _json
    src = os.path.join(paths.tier1(), corpus)
    dst = os.path.join(paths.tier2(), corpus)
    recipe_path = os.path.join(dst, "derive_recipe.json")
    if hint is None and os.path.isfile(recipe_path):
        r = _json.load(open(recipe_path))
        hint, max_seconds = r["hint"], r.get("max_seconds")
        hint_overrides = hint_overrides or r.get("hint_overrides")
        divisor = divisor if divisor is not None else r.get("divisor")
    if hint is None:
        print(f"{corpus}: no hint given and no {recipe_path}; "
              f"pass hint=static|bodycam", flush=True)
        return False
    hint_overrides = hint_overrides or {}
    os.makedirs(os.path.join(dst, "video"), exist_ok=True)
    os.makedirs(os.path.join(dst, "annotation"), exist_ok=True)
    with open(recipe_path, "w") as f:
        _json.dump({"hint": hint, "max_seconds": max_seconds,
                    "hint_overrides": hint_overrides, "divisor": divisor}, f)
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
        # per-clip camera class: explicit override > tier-1 metadata.hint
        # (importer-declared, e.g. antare_bwc mixes body-worn and fixed
        # cameras in one corpus) > corpus default
        d = _json.load(open(s_anno))
        clip_hint = hint_overrides.get(stem) or (d.get("metadata") or {}).get("hint") or hint
        w, h, fps = probe(s_vid)
        clip_div = divisor or divisor_from_config(fps, clip_hint)
        dims = scale_dims(w, h)
        tmp_vid = d_vid + f".part{os.getpid()}.mp4"
        transcode(s_vid, tmp_vid, clip_div, dims, fps / clip_div, max_seconds)
        os.replace(tmp_vid, d_vid)
        new, kept, dropped = rewrite_annotation(
            d, fps, clip_div, dims, max_seconds, d_vid, hint=clip_hint,
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
        print(f"  {stem} [{clip_hint}]: {w}x{h}@{fps:.2f} /{clip_div} -> "
              f"{dims[0]}x{dims[1]}@{fps / clip_div:.2f} "
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
    root = os.path.join(paths.tier2(), corpus)
    if not os.path.isdir(root):
        print(f"{corpus}: no tier-2 dir")
        return False
    problems = []
    purged = []
    forced_div = None
    rp = os.path.join(root, "derive_recipe.json")
    if os.path.isfile(rp):
        forced_div = json.load(open(rp)).get("divisor")
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
            # .json.meta.json sidecars are the eval scheduler's own
            # mtime-validated cache — regenerated on every eval touch, so
            # flagging them is permanent noise on an active box
            if f.endswith(".json.orig"):
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
        doc = json.load(open(ap))
        md = doc.get("metadata") or {}
        stem = os.path.basename(ap)[:-5]
        if not any(f.get("objects") for f in doc.get("frames") or []):
            # empty GT scores -inf in eval and poisons aggregates; either
            # the source has no labels (jaad selected-subjects) or a
            # duration cap emptied it (meva G419 starts at 120.9s) —
            # both must be excluded from eval sets
            problems.append(f"{stem}: annotation has zero GT boxes")
        for field in ("lite", "hint", "box_convention", "source_video"):
            if not md.get(field):
                problems.append(f"{stem}: missing metadata.{field}")
        sv = md.get("source_video", "")
        if sv and not os.path.isfile(sv):
            problems.append(f"{stem}: source_video missing on disk: {sv}")
        if sv and not sv.startswith(paths.tier1() + "/"):
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
            want = src_fps / (forced_div or divisor_from_config(src_fps, hint))
            if abs(vfps - want) > 0.15:
                problems.append(f"{stem}: fps {vfps:.3f} != analytics grid "
                                f"{want:.3f} (hint {hint}"
                                f"{f', forced divisor {forced_div}' if forced_div else ''})")
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


if __name__ == "__main__":
    sys.exit(main())
