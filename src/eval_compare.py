"""THE canonical comparator for tracker A/Bs. (See also: track.py --eval.)

Reads the eval tool's OWN persisted rollups (results-*.json written by
_write_eval_summary_json when the eval yaml sets `results_location`) and
prints them verbatim — groupmean (the search-weighted objective; on the
val split, `fitness_multi` here IS the track_search score), the group
breakdown, and per-clip deltas of the tool's per-clip metrics.

It performs NO aggregation of its own beyond subtracting per-clip values
between runs. If a needed view is missing, extend the eval summary
writer or this file — do not write one-off analysis scripts.

Usage:
    python -m src.eval_compare RUN_DIR [RUN_DIR ...] [--metric fitness_multi]
        [--clips N]

RUN_DIR is a results_location directory (the newest results-*.json in it
is used). With 2+ runs, the first is the baseline for deltas.
"""
import argparse
import glob
import json
import math
import os


HEADLINE = ("fitness_multi", "fitness", "idf1", "mota",
            "switch_per_obj", "frag_per_obj", "fp_tracks")


def load(run_dir):
    paths = sorted(glob.glob(os.path.join(run_dir, "results-*.json")))
    if not paths:
        raise SystemExit(f"no results-*.json in {run_dir} — did the eval "
                         f"yaml set results_location?")
    d = json.load(open(paths[-1]))
    if len(d["tests"]) != 1:
        raise SystemExit(f"{paths[-1]} has {len(d['tests'])} tests; "
                         f"comparator expects single-test evals")
    return os.path.basename(paths[-1]), next(iter(d["tests"].values()))


def fmt(v):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "    --"
    return f"{v:+.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="results_location dirs")
    ap.add_argument("--metric", default="fitness_multi",
                    help="per-clip delta metric (default fitness_multi)")
    ap.add_argument("--clips", type=int, default=6,
                    help="biggest per-clip movers to list per run")
    a = ap.parse_args()

    loaded = [(r, *load(r)) for r in a.runs]

    print("== groupmean (the search-weighted objective; on a val-split run "
          "fitness_multi = track_search score) ==")
    hdr = "run".ljust(28) + "".join(k.rjust(15) for k in HEADLINE)
    print(hdr)
    for run, fname, t in loaded:
        gm = t.get("groupmean") or {}
        print(os.path.basename(run.rstrip("/")).ljust(28)
              + "".join(fmt(gm.get(k)).rjust(15) for k in HEADLINE)
              + f"   [{fname}]")

    print("\n== groups (fitness_multi) ==")
    all_groups = sorted({g for _, _, t in loaded for g in t["groups"]
                         if not g.endswith("_mean")})
    print("run".ljust(28) + "".join(g.rjust(12) for g in all_groups))
    for run, _, t in loaded:
        print(os.path.basename(run.rstrip("/")).ljust(28)
              + "".join(fmt((t["groups"].get(g) or {}).get("fitness_multi"))
                        .rjust(12) for g in all_groups))

    if len(loaded) > 1:
        base = loaded[0][2]["clips"]
        for run, _, t in loaded[1:]:
            common = [c for c in base if c in t["clips"]]
            deltas = []
            for c in common:
                x = base[c].get(a.metric)
                y = t["clips"][c].get(a.metric)
                if x is None or y is None:
                    continue
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                deltas.append((y - x, c, x, y))
            deltas.sort()
            n_up = sum(1 for d in deltas if d[0] > 0)
            n_dn = sum(1 for d in deltas if d[0] < 0)
            print(f"\n== per-clip {a.metric}: {os.path.basename(run.rstrip('/'))}"
                  f" vs baseline ==  n={len(deltas)} better={n_up} worse={n_dn}")
            for d, c, x, y in deltas[:a.clips] + deltas[-a.clips:]:
                print(f"  {c:36s} {x:+.3f} -> {y:+.3f}  ({d:+.3f})")


if __name__ == "__main__":
    main()
