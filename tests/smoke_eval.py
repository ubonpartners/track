#!/usr/bin/env python
"""Three-clip smoke eval (repo_cleanup.md section 3).

Builds a one-off eval yaml FROM the objective search config at run time
(so it cannot drift from the objective), runs it on three antare clips
that cover a static camera, a moving camera and a clip whose tier-1
source is 4K (all three are evaluated on their 1280x720 tier-2 copies),
and writes the eval tool's own results-*.json plus a provenance.json
(sha256 of the objective and tracker yamls, git rev, dirty files) into
--out. The run goes through the same shared-stream runner the objective
uses (single_shared_streams is carried over from the objective yaml).

Compare two stages EXACTLY (every clip, every metric):

    python tests/smoke_eval.py --compare <out of stage N-1> <out of stage N>

exits 1 and prints every differing cell. Repeat runs of the same code
agree to the printed precision (exact-shape detector batching), so any
difference is a defect, not noise; --compare also prints the provenance
diff so a config edit under the repo's feet is visible.

Usage:
    python tests/smoke_eval.py --out /mldata/results/cleanup/stage0
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

import yaml

OBJECTIVE = "/mldata/config/track/search/track_search_v11_mc.yaml"
CLIPS = ("antare_knife_drawn_fixed_06",     # static, 1080p source
         "antare_knife_drawn_bwc_04",       # moving (bodycam), 1080p source
         "antare_refused_entry_fixed_03")   # static, 4K source (1280x720 in tier 2)
# keys copied from the objective. single_shared_streams selects the
# shared-stream runner (track_test.py) that every real search/eval uses;
# without it the eval takes the multiprocess path and the smoke would be
# blind to regressions in the path that matters.
COPY_KEYS = ("result_test_opt_key", "result_dataset_opt_key",
             "result_dataset_opt_param", "fitness_weights", "tests", "classes",
             "columns", "sort_key", "single_shared_streams")


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_yaml(out_dir, objective=OBJECTIVE, clips=CLIPS):
    c = yaml.safe_load(open(objective))
    e = {k: c[k] for k in COPY_KEYS if k in c}
    assert "single_shared_streams" in e, "objective no longer sets single_shared_streams"
    e["datasets"] = {k: c["datasets"][k] for k in clips}
    e["num_workers"] = 1
    e["results_location"] = out_dir
    return e


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def provenance(objective=OBJECTIVE):
    tracker_cfg = yaml.safe_load(open(objective))["tests"]["search_config"]["config"]
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                                    text=True).stdout.strip()
    return {"objective_yaml": objective, "objective_sha256": _sha(objective),
            "tracker_config": tracker_cfg, "tracker_config_sha256": _sha(tracker_cfg),
            "git_head": git("rev-parse", "HEAD"),
            "git_dirty": git("status", "--short").split("\n") if git("status", "--short") else []}


def load_rows(out_dir):
    paths = sorted(glob.glob(os.path.join(out_dir, "results-*.json")))
    if not paths:
        raise SystemExit(f"no results-*.json in {out_dir}")
    d = json.load(open(paths[-1]))
    (test,) = d["tests"].values()
    return paths[-1], test


def compare(a_dir, b_dir):
    """Exact cell-by-cell diff of every clip and rollup metric. Returns the
    list of differing cells (empty = identical)."""
    pa, ta = load_rows(a_dir)
    pb, tb = load_rows(b_dir)
    diffs = []
    for section in ("clips", "groups", "overall", "groupmean", "arithmean"):
        va, vb = ta.get(section), tb.get(section)
        if section in ("clips", "groups"):
            for key in sorted(set(va) | set(vb)):
                ra, rb = va.get(key), vb.get(key)
                if ra is None or rb is None:
                    diffs.append((section, key, "<missing>", ra is None, rb is None)); continue
                for m in sorted(set(ra) | set(rb)):
                    if ra.get(m) != rb.get(m):
                        diffs.append((section, key, m, ra.get(m), rb.get(m)))
        else:
            for m in sorted(set(va or {}) | set(vb or {})):
                if (va or {}).get(m) != (vb or {}).get(m):
                    diffs.append((section, "", m, (va or {}).get(m), (vb or {}).get(m)))
    print(f"A: {pa}\nB: {pb}")
    for f in ("provenance.json",):
        qa, qb = os.path.join(a_dir, f), os.path.join(b_dir, f)
        if os.path.isfile(qa) and os.path.isfile(qb):
            ja, jb = json.load(open(qa)), json.load(open(qb))
            for k in ("objective_sha256", "tracker_config_sha256", "git_head"):
                flag = "" if ja.get(k) == jb.get(k) else "   <-- DIFFERS"
                print(f"  {k}: {str(ja.get(k))[:12]} vs {str(jb.get(k))[:12]}{flag}")
    if diffs:
        print(f"{len(diffs)} differing cells:")
        for d in diffs:
            print("  ", *d)
    else:
        print("identical: every clip and rollup metric matches")
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="results dir (eval writes results-*.json here)")
    ap.add_argument("--compare", nargs=2, metavar="DIR", help="exact-diff two results dirs")
    a = ap.parse_args()
    if a.compare:
        sys.exit(1 if compare(*a.compare) else 0)
    if not a.out:
        ap.error("--out or --compare required")
    os.makedirs(a.out, exist_ok=True)
    ypath = os.path.join(a.out, "smoke.yaml")
    with open(ypath, "w") as f:
        yaml.safe_dump(build_yaml(a.out), f)
    with open(os.path.join(a.out, "provenance.json"), "w") as f:
        json.dump(provenance(), f, indent=1)
    cmd = [sys.executable, os.path.join(ROOT, "track.py"), "--eval", ypath,
           "--eval-split", "both"]
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
