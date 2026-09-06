"""Run the (resolution x rate) eval grid that quality_table.py consumes.

One eval per operating point, on the val split, results into
/mldata/tracking/results/qtab/<point>/ where `python -m tools.quality_table`
expects them. Points (see quality_table.py):

    grid_r{1,2,3}_{640,512,416,320}    plain drop of non-analytics frames
    gridm_r{2,3}_{640,512,416,320}     non-analytics frames delivered as
                                       MOTION/NVOF carry frames instead

Resolution is imposed with `src.cli --pm`; the pm index is looked up by
RES CAP in the CURRENT pm_table (ubon_cstuff include/pm_controller.h rows are
(res_cap, rate) pairs now — the index/cap mapping is no longer 1:1, so this
table must match the header). Rate is imposed exactly with
debug_analytics_mask ("1"=every frame, "10"=every 2nd, "100"=every 3rd),
written into a DERIVED eval yaml (the objective config with the mask and, for
gridm, min_time_delta_motion added). Deriving a yaml is correct here — these
runs are deliberately NOT the objective; eval prints its not-the-objective
warning for each, which is right.

Usage:
    python -m tools.quality_grid [--only REGEX] [--dry-run]
    python -m tools.quality_table          # then build the table from the runs

Rebuild this grid whenever the tracker config or eval data changes — the
table is a property of BOTH. ~20 val evals run back to back; expect hours.
"""
import argparse
import os
import re
import subprocess
import sys

import yaml
import src.paths as paths

OBJECTIVE = paths.search_yaml()
QTAB = paths.tier2("results", "qtab")

# res cap -> pm index in the CURRENT pm_table ((res,rate) rows; pick the first
# row with that cap — eval streams are non-realtime so only the cap half acts).
PM_FOR_RES = {640: 0, 512: 1, 416: 3, 320: 5}
RATE_MASK = {"r1": "1", "r2": "10", "r3": "100"}
# gridm: masked-out frames become MOTION/NVOF carry frames (what
# performance.skip_mode: motion delivers live). Any small positive value
# enables the MOTION frame class; 0 disables it (track_stream.c).
MOTION_CARRY_DELTA = 0.01


def derived_yaml(base, mask, carry, path):
    doc = yaml.safe_load(open(base))
    for t in doc.get("tests", {}).values():
        t["debug_analytics_mask"] = mask
        if carry:
            t["min_time_delta_motion"] = MOTION_CARRY_DELTA
    yaml.safe_dump(doc, open(path, "w"), sort_keys=False)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="regex over point names")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    points = []
    for res in (640, 512, 416, 320):
        for r in ("r1", "r2", "r3"):
            points.append((f"grid_{r}_{res}", res, r, False))
    for res in (640, 512, 416, 320):
        for r in ("r2", "r3"):  # carry is a no-op at full rate
            points.append((f"gridm_{r}_{res}", res, r, True))
    if a.only:
        points = [p for p in points if re.search(a.only, p[0])]

    cfg_dir = os.path.join(QTAB, "cfg")
    os.makedirs(cfg_dir, exist_ok=True)
    for name, res, r, carry in points:
        out = os.path.join(QTAB, name)
        ycfg = derived_yaml(OBJECTIVE, RATE_MASK[r], carry,
                            os.path.join(cfg_dir, f"{name}.yaml"))
        cmd = [sys.executable, "-m", "src.cli", "--pm", str(PM_FOR_RES[res]),
               "eval", ycfg, "--split", "val", "--results-location", out]
        print("::", name, "->", " ".join(cmd), flush=True)
        if a.dry_run:
            continue
        rc = subprocess.call(cmd)
        if rc != 0:
            sys.exit(f"{name} failed rc={rc}")
    print("grid complete; now: python -m tools.quality_table")


if __name__ == "__main__":
    main()
