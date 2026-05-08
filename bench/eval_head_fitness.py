"""Closed-loop fitness eval — Phase 20a.

Replaces val combined AUC as the model-selection metric. Runs the
actual C tracker on a diverse held-out clip subset and returns
aggregate fitness:

    fitness = mota - 0.0005 * fp_tracks - 0.002 * fp_per_frame

Use this when comparing trained heads — it predicts deployment
fitness within ~0.005, replacing the open-loop val AUC that has
consistently disagreed with deployment outcomes.

Two subsets shipped:
- `eval_subset_diverse.json` — 29 clips, ~3-4 min sequential, full
  family + density coverage. Use for final evaluation / commit decisions.
- `eval_subset_fast.json` — ~20 clips, drops the most-expensive 40%
  per family (where match-cost is O(N²) in objs/frame). Use for
  in-training-loop selection.

Speed levers:
- `--workers N`: parallel clip evaluation via ProcessPoolExecutor.
  4-8 workers cuts wall time ~4× since the per-clip work is mostly
  C-tracker time (releases GIL).
- `--frame-cap-sec T`: pass `end_time=T` to `import_create` so each
  clip stops tracking at T seconds. Caps the tail of long/dense
  clips like MOT20-05 (663 frames, 222 obj/frame avg → 64% of 29-
  clip cost). Pair with `--workers` for fast in-loop eval.

Eval params match the user's search bench:
- match_iou=0.45
- eval_min_framerate=9.9
- eval_rate_divisor=1
"""
from __future__ import annotations
import argparse, json, os, sys, tempfile, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.trackset as ts
import src.track_test as tt


DIVERSE_SUBSET_PATH = Path(__file__).resolve().parent / "eval_subset_diverse.json"
FAST_SUBSET_PATH    = Path(__file__).resolve().parent / "eval_subset_fast.json"
FULL_SUBSET_PATH    = Path(__file__).resolve().parent / "eval_subset_full178.json"
DEFAULT_DATASET_CFG = Path(__file__).resolve().parent / "pair_log_config_v6_with_pp22.yaml"

# User's search-bench eval params (Phase 18 confirmed match).
EVAL_PARAMS = dict(eval_rate_divisor=1, eval_min_framerate=9.9, match_iou=0.45)


def load_subset(name: str = "diverse") -> list:
    p = {"diverse": DIVERSE_SUBSET_PATH, "fast": FAST_SUBSET_PATH,
         "full":    FULL_SUBSET_PATH}[name]
    return json.loads(p.read_text())["clips"]


def load_dataset_paths(dataset_cfg=DEFAULT_DATASET_CFG) -> dict:
    cfg = yaml.safe_load(open(dataset_cfg))
    return {name: info["trackset"] for name, info in cfg["dataset"].items()}


_LOAD_FAIL_PATTERNS = (
    "failed to load",
    "in_dim mismatch",
    "in_dim out of range",
    "short header",
)


def _run_one_clip(args):
    """Worker: run tracker on one clip and return its raw metrics.
    On a tracker exception (e.g., trackset assertion on non-monotonic frame
    times in a few PP22 clips), return a sentinel so the bench can continue
    on the remaining clips. Skipped clips contribute zero to the aggregate.

    We also capture C-runtime stderr at fd-level via a temp file and scan
    it for known load-failure patterns. The C tracker's `log_error` writes
    to fd 2 directly and is otherwise invisible to Python — without this
    capture, a silently-disabled state head produces a "no head" bench
    that *looks* like a normal result. The §20.19 retraction was caused
    exactly by this failure mode."""
    clip, path, yaml_path, frame_cap_sec = args
    import tempfile
    saved_fd = None
    err_path = None
    err_text = ""
    try:
        # Redirect fd 2 to a temp file so we capture C-side log_error output.
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".stderr",
                                          prefix=f"clip_{clip}_", delete=False) as ef:
            err_path = ef.name
        sys.stderr.flush()
        saved_fd = os.dup(2)
        ef_fd = os.open(err_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(ef_fd, 2)
        os.close(ef_fd)
        try:
            gt = ts.TrackSet(path)
            out = ts.TrackSet()
            end_time = frame_cap_sec if frame_cap_sec else 100000
            out.import_create(gt, track_min_interval=0.199, debug=False,
                              config_file=yaml_path, debug_enable=False, params=None,
                              end_time=end_time)
            m = tt.compute_metrics(gt, out, frame_metrics=False, show_pbar=False,
                                    metrics="python", **EVAL_PARAMS)
        finally:
            sys.stderr.flush()
            os.dup2(saved_fd, 2)
            os.close(saved_fd); saved_fd = None
            try:
                with open(err_path, "r", errors="replace") as f: err_text = f.read()
            except OSError: pass
    except Exception as e:
        if saved_fd is not None:
            try: os.dup2(saved_fd, 2); os.close(saved_fd)
            except OSError: pass
        return clip, {"_skip_reason": f"{type(e).__name__}: {str(e)[:200]}",
                       "_stderr": err_text[:1000]}
    finally:
        if err_path:
            try: os.unlink(err_path)
            except OSError: pass

    # Detect the silent-disable failure mode: any known head-load error
    # pattern in C-runtime stderr → mark as load-failed so the parent can
    # abort rather than silently producing a "no head" bench.
    for pat in _LOAD_FAIL_PATTERNS:
        if pat in err_text:
            return clip, {"_load_fail": err_text[:500]}

    return clip, {
        "num_objects":         float(m["num_objects"]),
        "num_false_positives": float(m["num_false_positives"]),
        "num_misses":          float(m["num_misses"]),
        "num_switches":        float(m["num_switches"]),
        "num_frames":          float(m["num_frames"]),
        "fp_tracks":           int(m["fp_tracks"]),
        "mota":                float(m["mota"]),
        "fitness":             float(m["fitness"]),
    }


def eval_config(yaml_path: str, clips: list[str] | None = None,
                workers: int = 1, frame_cap_sec: float | None = None,
                verbose: bool = False) -> dict:
    """Run the tracker on the eval subset, return aggregate fitness.

    Args:
        yaml_path: tracker config YAML path.
        clips: optional list of clip names; defaults to diverse subset.
        workers: parallelism for clip evaluation (1 = sequential).
        frame_cap_sec: stop tracking each clip at T seconds (helps with
            dense clips like MOT20-05). None = no cap.
        verbose: print per-clip progress to stderr.
    """
    if clips is None:
        clips = load_subset("diverse")
    paths = load_dataset_paths()

    work = []
    for clip in clips:
        path = paths.get(clip)
        if path is None or not os.path.isfile(path):
            if verbose: print(f"  [warn] {clip}: no path, skipping", file=sys.stderr)
            continue
        work.append((clip, path, yaml_path, frame_cap_sec))

    raw = {}
    skipped = {}
    def _ingest(clip, m):
        if "_load_fail" in m:
            # Abort loud — never silently produce a "no head" bench.
            raise RuntimeError(
                f"Tracker config produced a state-head load failure "
                f"on clip {clip!r}. The C runtime was disabled rather than "
                f"using the head you intended; results would be invalid. "
                f"Captured stderr:\n{m['_load_fail']}")
        if "_skip_reason" in m:
            skipped[clip] = m["_skip_reason"]
            print(f"  [warn] skip {clip}: {m['_skip_reason']}", file=sys.stderr)
        else:
            raw[clip] = m
            if verbose: print(f"  [{clip}] fit={m['fitness']:.4f}", file=sys.stderr)

    if workers <= 1:
        for w in work:
            clip, m = _run_one_clip(w)
            _ingest(clip, m)
    else:
        # Use 'spawn' to avoid CUDA + fork incompatibility (TensorRT engines
        # don't survive a fork from a parent that touched CUDA).
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {ex.submit(_run_one_clip, w): w[0] for w in work}
            for fut in as_completed(futs):
                clip, m = fut.result()
                _ingest(clip, m)

    # Aggregate fitness across selected clips.
    def aggregate(keys):
        if not keys: return None
        tot_obj = sum(raw[k]["num_objects"] for k in keys)
        tot_fp  = sum(raw[k]["num_false_positives"] for k in keys)
        tot_fn  = sum(raw[k]["num_misses"] for k in keys)
        tot_sw  = sum(raw[k]["num_switches"] for k in keys)
        tot_frm = sum(raw[k]["num_frames"] for k in keys)
        tot_fpt = sum(raw[k]["fp_tracks"] for k in keys)
        if tot_obj == 0 or tot_frm == 0: return None
        mota = 1.0 - (tot_fp + tot_fn + tot_sw) / tot_obj
        fp_pf = tot_fp / tot_frm
        return {
            "mota":         mota,
            "fp_tracks":    int(tot_fpt),
            "fp_per_frame": fp_pf,
            "fitness":      mota - 0.0005 * tot_fpt - 0.002 * fp_pf,
            "num_clips":    len(keys),
        }

    overall = aggregate(list(raw.keys()))
    fams = {"MOT17": [], "MOT20": [], "PP22": [], "UKof": [], "INof": []}
    for k in raw:
        for f in fams:
            if k.startswith(f):
                fams[f].append(k); break
    per_family = {f: aggregate(keys) for f, keys in fams.items() if keys}

    return {
        "overall": overall,
        "per_family": per_family,
        "raw": raw,
        "skipped": skipped,
    }


def make_yaml_with_state_bin(state_bin: str, base_yaml: str | None = None) -> str:
    """Produce a temp YAML with nn_state_path pointing at `state_bin`,
    other params from `base_yaml` (defaults to current /mldata prod)."""
    if base_yaml is None:
        base_yaml = "/mldata/config/track/trackers/uc_v11.yaml"
    cfg = yaml.safe_load(open(base_yaml))
    cfg["utrack"]["nn_state_path"] = state_bin
    fd, tmp = tempfile.mkstemp(suffix=".yaml", prefix="eval_head_")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(cfg, f)
    return tmp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="full tracker config YAML")
    p.add_argument("--state-bin", help="state head .bin; will be plugged into base config")
    p.add_argument("--base", default=None,
                   help="base YAML when --state-bin used (default: live prod)")
    p.add_argument("--subset", default="diverse", choices=["diverse", "fast", "full"],
                   help="eval subset: 'diverse' (29 clips, ~3 min seq) or "
                        "'fast' (~20 clips, drops expensive ones) or "
                        "'full' (178 clips from pair_log_config_v6_with_pp22.yaml — "
                        "for final validation; ~8 min per run with 4 workers)")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel clip eval workers (1 = sequential)")
    p.add_argument("--frame-cap-sec", type=float, default=None,
                   help="cap each clip's tracking at T seconds (default: no cap)")
    p.add_argument("--out", default=None, help="write JSON result to this path")
    p.add_argument("--runs", type=int, default=1,
                   help="number of repeat runs to average across (default 1). "
                        "FP-thread ordering and float accumulation in the C "
                        "tracker give ~±0.001 fitness noise per single run "
                        "(fp_tracks is integer/deterministic; mota and FP/fr "
                        "fluctuate). For comparing heads where the gap is "
                        "<0.005 fitness, use --runs 5 and read the reported "
                        "mean ± std. Each run is independent; total wall time "
                        "scales linearly.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.config and args.state_bin:
        sys.exit("Specify either --config OR --state-bin, not both")
    if args.config:
        cfg = args.config
    elif args.state_bin:
        cfg = make_yaml_with_state_bin(args.state_bin, args.base)
        print(f"Built temp YAML with nn_state_path={args.state_bin}", file=sys.stderr)
    else:
        sys.exit("Need --config or --state-bin")

    clips = load_subset(args.subset)
    import time
    t0 = time.time()
    n_runs = max(1, int(args.runs))
    runs = []
    for ri in range(n_runs):
        run_t0 = time.time()
        result = eval_config(cfg, clips=clips, workers=args.workers,
                              frame_cap_sec=args.frame_cap_sec, verbose=not args.quiet)
        run_dt = time.time() - run_t0
        ov = result["overall"]
        if n_runs > 1:
            print(f"  [run {ri+1}/{n_runs}] mota={ov['mota']:.4f} fit={ov['fitness']:.4f} "
                  f"FPTr={ov['fp_tracks']} FP/fr={ov['fp_per_frame']:.3f}  ({run_dt:.0f}s)",
                  file=sys.stderr)
        runs.append(result)
    elapsed = time.time() - t0
    print(f"\n[eval_head_fitness] {len(clips)} clips, workers={args.workers}, "
          f"frame_cap_sec={args.frame_cap_sec}, runs={n_runs}, "
          f"elapsed={elapsed:.1f}s", file=sys.stderr)

    if n_runs > 1:
        # Aggregate across runs: mean ± std on fitness, mota, fp_per_frame.
        # fp_tracks is integer; report mean and the set of distinct values.
        import statistics
        fits  = [r["overall"]["fitness"] for r in runs]
        motas = [r["overall"]["mota"] for r in runs]
        fpfs  = [r["overall"]["fp_per_frame"] for r in runs]
        fpts  = [r["overall"]["fp_tracks"] for r in runs]
        agg = {
            "n_runs":           n_runs,
            "fitness_mean":     statistics.mean(fits),
            "fitness_stdev":    statistics.stdev(fits) if n_runs > 1 else 0.0,
            "fitness_min":      min(fits),
            "fitness_max":      max(fits),
            "mota_mean":        statistics.mean(motas),
            "mota_stdev":       statistics.stdev(motas) if n_runs > 1 else 0.0,
            "fp_per_frame_mean": statistics.mean(fpfs),
            "fp_tracks_mean":   sum(fpts) / n_runs,
            "fp_tracks_values": fpts,
        }
        result_out = {"runs": runs, "aggregate": agg}
        print(f"\n[multi-run aggregate]", file=sys.stderr)
        print(f"  fitness  = {agg['fitness_mean']:.4f} ± {agg['fitness_stdev']:.4f} "
              f"(min {agg['fitness_min']:.4f}, max {agg['fitness_max']:.4f})", file=sys.stderr)
        print(f"  mota     = {agg['mota_mean']:.4f} ± {agg['mota_stdev']:.4f}", file=sys.stderr)
        print(f"  FPTr     = {agg['fp_tracks_mean']:.1f} (values {agg['fp_tracks_values']})",
              file=sys.stderr)
        print(f"  FP/frame = {agg['fp_per_frame_mean']:.3f}", file=sys.stderr)
    else:
        result_out = runs[0]

    out = json.dumps(result_out, indent=2)
    if args.out:
        Path(args.out).write_text(out)
    print(out)


if __name__ == "__main__":
    main()
