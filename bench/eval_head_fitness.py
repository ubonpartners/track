"""Closed-loop fitness eval — Phase 20a.

Replaces val combined AUC as the model-selection metric. Runs the
actual C tracker on a diverse held-out 29-clip subset (see
`bench/eval_subset_diverse.json`) and returns aggregate fitness:

    fitness = mota - 0.0005 * fp_tracks - 0.002 * fp_per_frame

Use this when comparing trained heads — it predicts deployment
fitness within ~0.005, replacing the open-loop val AUC that has
consistently disagreed with deployment outcomes (Phase 11/14/17/19
session findings).

Eval params match the user's search bench
(`/mldata/config/track/search/track_search_v11.yaml`):
- match_iou=0.45
- eval_min_framerate=9.9
- eval_rate_divisor=1

Usage:
    python -m bench.eval_head_fitness --config /tmp/foo.yaml
    python -m bench.eval_head_fitness --state-bin /mldata/.../nn_state_v14.bin

Returns:
    {fitness, mota, fp_tracks, fp_per_frame, ...} as JSON to stdout
"""
from __future__ import annotations
import argparse, json, os, sys, tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.trackset as ts
import src.track_test as tt


SUBSET_PATH = Path(__file__).resolve().parent / "eval_subset_diverse.json"
DEFAULT_DATASET_CFG = Path(__file__).resolve().parent / "pair_log_config_v6_with_pp22.yaml"

# User's search-bench eval params (Phase 18 confirmed match).
EVAL_PARAMS = dict(eval_rate_divisor=1, eval_min_framerate=9.9, match_iou=0.45)


def load_subset() -> list:
    """Return the 29-clip stratified subset (clip_name strings)."""
    return json.loads(SUBSET_PATH.read_text())["clips"]


def load_dataset_paths(dataset_cfg=DEFAULT_DATASET_CFG) -> dict:
    """Map clip_name -> trackset path, from pair_log_config_v6_with_pp22.yaml."""
    cfg = yaml.safe_load(open(dataset_cfg))
    return {name: info["trackset"] for name, info in cfg["dataset"].items()}


def eval_config(yaml_path: str, clips: list[str] | None = None) -> dict:
    """Run the tracker on the eval subset, return aggregate fitness.

    Args:
        yaml_path: tracker config YAML path (must be loadable by import_create).
        clips: optional list of clip names; defaults to the diverse 29-clip subset.

    Returns:
        Dict with: fitness, mota, fp_tracks, fp_per_frame, num_clips,
                   per_family aggregates, per_clip raw counts.
    """
    if clips is None:
        clips = load_subset()
    paths = load_dataset_paths()

    raw = {}
    for clip in clips:
        path = paths.get(clip)
        if path is None or not os.path.isfile(path):
            print(f"  [warn] {clip}: no path, skipping", file=sys.stderr)
            continue
        gt = ts.TrackSet(path)
        out = ts.TrackSet()
        out.import_create(gt, track_min_interval=0.199, debug=False,
                          config_file=yaml_path, debug_enable=False, params=None)
        m = tt.compute_metrics(gt, out, frame_metrics=False, show_pbar=False,
                                metrics="python", **EVAL_PARAMS)
        raw[clip] = {
            "num_objects":         float(m["num_objects"]),
            "num_false_positives": float(m["num_false_positives"]),
            "num_misses":          float(m["num_misses"]),
            "num_switches":        float(m["num_switches"]),
            "num_frames":          float(m["num_frames"]),
            "fp_tracks":           int(m["fp_tracks"]),
            "mota":                float(m["mota"]),
            "fitness":             float(m["fitness"]),
        }

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
    p.add_argument("--out", default=None, help="write JSON result to this path")
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

    result = eval_config(cfg)
    out = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(out)
    print(out)


if __name__ == "__main__":
    main()
