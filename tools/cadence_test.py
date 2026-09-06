# Cadence test set builder — docs/plans/cadence_test.md (MB 2026-07-28).
#
# Builds, from TIER-1 sources (native framerate; never tier-2), a set of
# eval clips whose average analytics rate is constant (target_fps) while
# the cadence pattern varies. Frame selection is by source-frame index;
# selected frames keep their TRUE source PTS (the unevenness IS the
# experiment). Annotations are resampled from the tier-1 GT at the kept
# frame times via TrackSet interpolation (works for both per-frame and
# keyframe-density GT).
#
# Output: /mldata/tracking/cadence_test/<variant>/{video,annotation}/
# plus cadence_manifest.json (selection) and cadence_eval.yaml (harness).
# Experiment data: tier-2-class, NOT in the corpus registry; regenerable
# from tier 1 + this script.
#
# Extensibility: patterns are (name -> generator(p)) entries returning
# kept-indices + cycle length for p = src_fps/target_fps; add lower
# target rates or new cadences by adding entries / changing target_fps.
import argparse
import json
import math
import os
import re
import subprocess
import sys

import yaml

T1 = os.environ.get("TRACK_TIER1") or "/mldata/tracking_original"   # research tool: mirrors src.paths.tier1()
OUT_ROOT = "/mldata/tracking/cadence_test"
SEARCH_YAML = "/mldata/config/track/search/track_search_v11_mc.yaml"
TARGET_FPS = 5.0
MIN_SRC_FPS = 23.9
MAX_CLIPS = 30
MIN_DURATION_S = 20.0


def _r(x):
    return int(round(x))


def pattern_uniform(p):
    """Nearest-to-grid at the target period: exact uniform when p is an
    integer; the practical 'gear' cadence otherwise (this IS Pgear on
    non-integer-ratio sources — the only constructible 'uniform')."""
    n_ticks = 5
    cycle = _r(n_ticks * p)
    kept = sorted({min(cycle - 1, _r(k * p)) for k in range(n_ticks)})
    return kept, cycle


def pattern_j50(p):
    """Alternating short/long spacing (~0.5p / ~1.5p)."""
    cycle = _r(2 * p)
    return sorted({0, min(cycle - 1, _r(0.5 * p))}), cycle


def pattern_b50(p):
    """5 frames at half spacing, then a gap to the next cycle."""
    r = max(1, _r(0.5 * p))
    cycle = _r(5 * p)
    return sorted({min(cycle - 1, k * r) for k in range(5)}), cycle


def pattern_b17(p):
    """5 consecutive source frames then a long gap."""
    cycle = _r(5 * p)
    return list(range(5)), cycle


def pattern_g2(p):
    """Sparse second (3 frames) alternating with dense second (7)."""
    half = _r(5 * p)
    a = {0, min(half - 1, _r(2 * p)), min(half - 1, _r(4 * p))}
    step = max(1, _r(0.5 * p))
    b = {half + k * step for k in range(7) if half + k * step < 2 * half}
    return sorted(a | b), 2 * half


def pattern_fpt(p):
    """Today's first-past-threshold decimator: every ceil(p)-th frame —
    cadence-clean but a RATE error on non-integer p (e.g. 4.8 fps from
    24). Only meaningful on non-integer-ratio sources."""
    step = math.ceil(p - 1e-9)
    cycle = 5 * step
    return [k * step for k in range(5)], cycle


def pattern_b17x3(p):
    """15 consecutive source frames then a ~2.5 s gap (3 s cycle, same
    5 fps average) — the long-gap regime that CROSSES
    track_buffer_seconds (2.2 s), where the deletion rule actually
    binds (phase-2 finding: sub-buffer gaps never fire it)."""
    cycle = _r(15 * p)
    return list(range(15)), cycle


PATTERNS = {
    "U": pattern_uniform,      # control (== Pgear on non-integer sources)
    "J50": pattern_j50,
    "B50": pattern_b50,
    "B17": pattern_b17,
    "B17x3": pattern_b17x3,
    "G2": pattern_g2,
}
NONINTEGER_EXTRA = {"Pfpt": pattern_fpt}


def probe(video):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate",
         "-show_entries", "format=duration", "-of", "json", video],
        capture_output=True, text=True)
    j = json.loads(out.stdout)
    st = j["streams"][0]
    num, den = st["avg_frame_rate"].split("/")
    fps = float(num) / float(den)
    return int(st["width"]), int(st["height"]), fps, float(j["format"]["duration"])


def select_clips(max_clips=MAX_CLIPS):
    """Stratified selection over the search config's train+val rows:
    tier-1 source fps >= MIN_SRC_FPS, duration >= MIN_DURATION_S,
    round-robin over (family, hint) by gt_tracks desc."""
    raw = open(SEARCH_YAML).read()
    cfg = yaml.safe_load(raw)
    gt_counts = dict(re.findall(r"path: (\S+\.json) \}.*?gt_tracks=(\d+)", raw))
    rows = []
    for key, row in (cfg.get("datasets") or {}).items():
        if row.get("split") not in ("train", "val"):
            continue
        path = row["path"]
        corpus = path.split("/")[3]
        stem = os.path.basename(path)[:-5]
        src = f"{T1}/{corpus}/video/{stem}.mp4"
        if not os.path.isfile(src):
            continue
        rows.append({
            "key": key, "corpus": corpus, "stem": stem, "source": src,
            "t1_anno": f"{T1}/{corpus}/annotation/{stem}.json",
            "hint": "bodycam" if row.get("stream_hint") else "static",
            "gt_tracks": int(gt_counts.get(path, 0)),
        })
    # probe lazily, best-first inside each stratum
    strata = {}
    for r in rows:
        strata.setdefault((r["corpus"], r["hint"]), []).append(r)
    for v in strata.values():
        v.sort(key=lambda r: -r["gt_tracks"])
    picked = []
    exhausted = set()
    while len(picked) < max_clips and len(exhausted) < len(strata):
        for skey in sorted(strata, key=lambda k: -strata[k][0]["gt_tracks"]
                           if strata[k] else 0):
            if len(picked) >= max_clips:
                break
            bucket = strata[skey]
            while bucket:
                cand = bucket.pop(0)
                if cand["gt_tracks"] < 1:
                    continue          # zero-GT clips carry no signal
                w, h, fps, dur = probe(cand["source"])
                if fps < MIN_SRC_FPS or dur < MIN_DURATION_S:
                    continue
                cand.update({"src_fps": round(fps, 4), "width": w,
                             "height": h, "duration": round(dur, 1)})
                picked.append(cand)
                break
            if not bucket:
                exhausted.add(skey)
    return picked


def build_clip(clip, variant, gen, target_fps=TARGET_FPS, max_edge=1280):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for src.*
    import src.trackset as trackset
    from src.dataset_lite import scale_dims
    from src.corpus_manifest import load_capabilities

    p = clip["src_fps"] / target_fps
    kept, cycle = gen(p)
    vdir = f"{OUT_ROOT}/{variant}/video"
    adir = f"{OUT_ROOT}/{variant}/annotation"
    os.makedirs(vdir, exist_ok=True)
    os.makedirs(adir, exist_ok=True)
    out_v = f"{vdir}/{clip['stem']}.mp4"
    out_a = f"{adir}/{clip['stem']}.json"
    if os.path.isfile(out_v) and os.path.isfile(out_a):
        return "skip"
    dims = scale_dims(clip["width"], clip["height"], max_edge)
    sel = "+".join(f"eq(mod(n\\,{cycle})\\,{k})" for k in kept)
    tmp = out_v + f".part{os.getpid()}.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", clip["source"],
           "-vf", f"select='{sel}',scale={dims[0]}:{dims[1]}",
           "-fps_mode", "vfr", "-an",
           "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23",
           "-g", "10", "-bf", "0", tmp]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:  # nvenc flake fallback
        cmd[cmd.index("h264_nvenc")] = "libx264"
        cmd[cmd.index("-cq")] = "-crf"
        subprocess.check_call([c if c != "p4" else "veryfast" for c in cmd])
    os.replace(tmp, out_v)

    # kept frame times on the true source grid
    n_src = int(clip["duration"] * clip["src_fps"])
    times = [(c * cycle + k) / clip["src_fps"]
             for c in range(n_src // cycle + 1) for k in kept
             if (c * cycle + k) < n_src]
    gt = trackset.TrackSet(clip["t1_anno"])
    caps = load_capabilities(clip["corpus"]) or {}
    frames = []
    for i, t in enumerate(times):
        objs = {}
        for o in (gt.objects_at_time(t) or []):
            objs[str(o.track_id)] = {
                "box": [round(float(v), 4) for v in o.box],
                "class": int(o.cl), "conf": round(float(o.confidence), 4)}
        frames.append({"frame_id": i, "frame_time": round(t, 6),
                       "objects": objs})
    md = dict(gt.metadata)
    md.update({
        "frame_rate": target_fps, "width": dims[0], "height": dims[1],
        "original_video": out_v, "source_video": clip["source"],
        "hint": clip["hint"],
        "box_convention": caps.get("box_convention"),
        "lite": {"source_fps": clip["src_fps"], "divisor": None,
                 "max_seconds": None, "hint": clip["hint"],
                 "min_time_delta": None},
        "cadence": {"variant": variant, "cycle_frames": cycle,
                    "kept": kept, "target_fps": target_fps,
                    "actual_avg_fps": round(
                        len(kept) * clip["src_fps"] / cycle, 4)},
    })
    tmpa = out_a + f".tmp{os.getpid()}"
    json.dump({"metadata": md, "frames": frames}, open(tmpa, "w"))
    os.replace(tmpa, out_a)

    # verify: container frame count vs expectation
    got = int(subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams",
         "v:0", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
         out_v], capture_output=True, text=True).stdout.strip().rstrip(","))
    if abs(got - len(times)) > len(kept):
        raise RuntimeError(
            f"{variant}/{clip['stem']}: frame count {got} != ~{len(times)}")
    bframes = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=has_b_frames", "-of", "csv=p=0", out_v],
        capture_output=True, text=True).stdout.strip().rstrip(",")
    if bframes != "0":
        raise RuntimeError(
            f"{variant}/{clip['stem']}: B-FRAMES present (has_b_frames="
            f"{bframes}) — violates the I+P mp4-direct contract")
    return "built"


def build_all(manifest):
    for clip in manifest:
        integer_ratio = abs(clip["src_fps"] / TARGET_FPS
                            - round(clip["src_fps"] / TARGET_FPS)) < 1e-3
        pats = dict(PATTERNS)
        if not integer_ratio:
            pats.update(NONINTEGER_EXTRA)
        for variant, gen in pats.items():
            st = build_clip(clip, variant, gen)
            print(f"{variant}/{clip['stem']}: {st}", flush=True)


def write_eval_config(manifest):
    cfg = yaml.safe_load(open(SEARCH_YAML))
    sc = dict(cfg["tests"]["search_config"])
    sc["min_interval"] = 0.001   # epsilon: process every delivered frame
    datasets = {}
    for clip in manifest:
        integer_ratio = abs(clip["src_fps"] / TARGET_FPS
                            - round(clip["src_fps"] / TARGET_FPS)) < 1e-3
        variants = list(PATTERNS) + ([] if integer_ratio
                                     else list(NONINTEGER_EXTRA))
        for v in variants:
            row = {"split": "train", "family": f"cadence_{v}",
                   "group": v,
                   "path": f"{OUT_ROOT}/{v}/annotation/{clip['stem']}.json"}
            if clip["hint"] == "bodycam":
                row["stream_hint"] = "bodycam"
            datasets[f"{v}_{clip['key']}"] = row
    out = {"tests": {"search_config": sc}, "datasets": datasets,
           "result_log_file_path": "/mldata/results/cadence_test",
           "num_workers": "auto", "sort_key": "fitness"}
    path = "/mldata/config/track/search/cadence_eval.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    print(f"eval config -> {path} ({len(datasets)} rows)", flush=True)


def analyze(log_path):
    """Paired per-clip fitness deltas vs the U control from an eval log."""
    txt = re.sub(r"\x1b\[[0-9;]*m", "", open(log_path, errors="ignore").read())
    clip = {}
    variants = ["Pfpt"] + [v for v in PATTERNS if v != "U"] + ["U"]
    for ln in txt.splitlines():
        m = re.match(r"([A-Za-z]\S*)\s+search_config\s", ln)
        if not m:
            continue
        try:
            val = float(ln.split()[-1])
        except ValueError:
            continue
        name = m.group(1)
        for v in variants:
            if name.startswith(v):
                clip[(v, name[len(v):].lstrip("_"))] = val
                break
    u = {k: s2 for (v, k), s2 in clip.items() if v == "U" and abs(s2) < 10}
    out = {}
    print(f"{'variant':6s} {'n':>3s} {'meanD':>8s} {'medianD':>8s} {'worse':>7s}")
    for v in variants:
        if v == "U":
            continue
        ds = sorted(clip[(v, k)] - u[k] for (vv, k) in clip
                    if vv == v and k in u and abs(clip[(v, k)]) < 10)
        if not ds:
            continue
        n = len(ds)
        out[v] = {"n": n, "mean": sum(ds) / n, "median": ds[n // 2],
                  "worse": sum(1 for d in ds if d < 0)}
        print(f"{v:6s} {n:3d} {out[v]['mean']:+8.4f} {out[v]['median']:+8.4f} "
              f"{out[v]['worse']:3d}/{n:<3d}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--select-only", action="store_true")
    ap.add_argument("--analyze", default=None, metavar="EVAL_LOG")
    a = ap.parse_args()
    if a.analyze:
        analyze(a.analyze)
        return
    mpath = f"{OUT_ROOT}/cadence_manifest.json"
    if os.path.isfile(mpath):
        manifest = json.load(open(mpath))
        print(f"using existing manifest ({len(manifest)} clips)")
    else:
        manifest = select_clips()
        os.makedirs(OUT_ROOT, exist_ok=True)
        json.dump(manifest, open(mpath, "w"), indent=1)
        print(f"selected {len(manifest)} clips -> {mpath}")
    for c in manifest:
        print(f"  {c['corpus']:14s} {c['stem'][:34]:36s} "
              f"{c['src_fps']:7.3f}fps {c['duration']:6.1f}s "
              f"{c['hint']:7s} gt={c['gt_tracks']}")
    if a.select_only:
        return
    build_all(manifest)
    write_eval_config(manifest)


if __name__ == "__main__":
    main()
