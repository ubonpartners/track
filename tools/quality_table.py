"""Build the quality table that capacity_curve.py looks up.

The realtime question is "how many streams at what quality", and the PM
controller answers the streams half by choosing an operating point: a detector
resolution cap and an analytics rate. This tool measures the quality half —
tracking quality at each such operating point — so the two can be joined.

Inputs are the eval tool's OWN persisted rollups (results-*.json, written when
the eval yaml sets results_location), one run per grid point:

    grid_r{1,2,3}_{640,512,416,320}    resolution x rate, PM shed = plain drop
    gridm_r{2,3}_{640,512,416,320}     same, with the MOTION/NVOF carry on
                                       (min_time_delta_motion), i.e. what
                                       performance.skip_mode: motion delivers

r1/r2/r3 are analytics rates 1, 1/2, 1/3, imposed by debug_analytics_mask so
the rate is exact rather than load-dependent. The carry axis only exists below
rate 1 — at full rate there are no non-analytics frames to carry through.

Quality is indexed by CONTENT TYPE, because the degradation that is cheapest
for one content type is not cheapest for another (static CCTV collapses when
detector resolution drops; handheld and dashcam footage barely notices). The
content type of a clip is read from the source tag the eval puts in its clip
key; the mapping below is explicit and exhaustive, and an unrecognised clip is
a hard error rather than a silent omission.

Usage:
    python -m tools.quality_table [--qtab DIR] [--out PATH]
"""
import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

import yaml

QTAB = "/mldata/tracking/results/qtab"
TRACKING = "/mldata/tracking"
OUT = "/mldata/config/track/quality_table.yaml"

# Clip-key prefix -> content type. The eval already tags every clip with its
# source corpus in the key it reports, so the source is read off the data rather
# than guessed: PP22_ is PersonPath22, bwc_ is the bodycam corpus, video_ is
# JAAD, and so on. Checked exhaustively — a clip matching no prefix is a hard
# error, not a silent omission, because a whole missing family would quietly
# skew the per-content rows.
PREFIX_CONTENT = (
    ("bwc_",    "bodycam"),
    ("video_",  "dashcam_jaad"),
    ("bdd_",    "dashcam_bdd"),
    ("PP22_",   "handheld_crowd"),
    ("otw_",    "doorway"),
    ("MEVA_",   "cctv_static"),
    ("MOT",     "cctv_dense"),
    ("INof_",   "office_indoor"),
    ("UKof_",   "office_indoor"),
    ("movie_",  "movie"),
)

RATE = {"r1": "1.0000", "r2": "0.5000", "r3": "0.3333"}


def content_of(clip):
    for pre, content in PREFIX_CONTENT:
        if clip.startswith(pre):
            return content
    return None


def load_point(qtab, name):
    paths = sorted(glob.glob(os.path.join(qtab, name, "results-*.json")))
    if not paths:
        return None
    d = json.load(open(paths[-1]))
    if len(d["tests"]) != 1:
        sys.exit(f"{paths[-1]}: expected a single test")
    return next(iter(d["tests"].values()))


def summarise(test, unmapped, dropped):
    """One grid point -> {group: quality}."""
    out = {}
    # ALL is the eval's own search-weighted objective, so the table's headline
    # number is the same score the tracker search optimises.
    gm = test.get("groupmean") or {}
    out["ALL"] = gm.get("fitness_multi")
    for g, vals in (test.get("groups") or {}).items():
        if g.endswith("_mean"):
            continue
        v = (vals or {}).get("fitness_multi")
        if v is not None:
            out[f"group_{g}"] = v
    # per-content-type: unweighted mean over that type's clips
    buckets = defaultdict(list)
    for clip, vals in (test.get("clips") or {}).items():
        content = content_of(clip)
        if content is None:
            unmapped.add(clip)
            continue
        v = (vals or {}).get("fitness_multi")
        # A clip with no scoreable GT for the weighted classes yields nan/-inf.
        # One such clip must not poison its whole content type's mean, so drop
        # it here and count it — the per-content row is then a mean over the
        # clips that actually scored.
        if v is None or not math.isfinite(v):
            dropped[content] = dropped.get(content, 0) + 1
            continue
        buckets[content].append(v)
    for content, vs in buckets.items():
        out[content] = sum(vs) / len(vs)
    return out


def build(qtab):
    unmapped = set()
    dropped = {}
    table = defaultdict(dict)         # no carry
    table_carry = defaultdict(dict)   # MOTION/NVOF carry on
    missing = []
    for res in (640, 512, 416, 320):
        for r in ("r1", "r2", "r3"):
            t = load_point(qtab, f"grid_{r}_{res}")
            if t is None:
                missing.append(f"grid_{r}_{res}")
            else:
                table[str(res)][RATE[r]] = summarise(t, unmapped, dropped)
            if r == "r1":
                # rate 1 has no skipped frames, so carry is a no-op by
                # construction; reuse the same measurement rather than
                # implying a separate one exists.
                if t is not None:
                    table_carry[str(res)][RATE[r]] = table[str(res)][RATE[r]]
                continue
            tm = load_point(qtab, f"gridm_{r}_{res}")
            if tm is None:
                missing.append(f"gridm_{r}_{res}")
            else:
                table_carry[str(res)][RATE[r]] = summarise(tm, unmapped, dropped)
    return dict(table), dict(table_carry), missing, unmapped, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qtab", default=QTAB)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    table, table_carry, missing, unmapped, dropped = build(a.qtab)
    if dropped:
        print("unscoreable clips excluded from per-content means (summed over "
              "all grid points): "
              + ", ".join(f"{k}={v}" for k, v in sorted(dropped.items())),
              file=sys.stderr)
    if missing:
        print(f"WARNING: {len(missing)} grid points missing: "
              f"{', '.join(missing)}", file=sys.stderr)
    if unmapped:
        sys.exit(f"{len(unmapped)} clips match no content prefix: "
                 f"{', '.join(sorted(unmapped)[:10])}. Add them to "
                 f"PREFIX_CONTENT — an unmapped family would skew the "
                 f"per-content rows.")

    doc = {
        "note": "quality (fitness_multi) by detector resolution cap x analytics"
                " rate, indexed by CONTENT TYPE (per-clip means) plus the"
                " eval's own groups and ALL (the search-weighted objective)."
                " table = PM shed drops the frame; table_motion_carry = shed"
                " frames are delivered as MOTION/NVOF carry frames"
                " (performance.skip_mode: motion). Rate 1.0 is shared: with no"
                " skipped frames the carry cannot apply.",
        "source": "python -m tools.quality_table, from track.py --eval"
                  " grid_*/gridm_* runs on the val split",
        "table": table,
        "table_motion_carry": table_carry,
    }
    with open(a.out, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=True, default_flow_style=False)
    print(f"wrote {a.out}")

    groups = sorted({g for r in table.values() for c in r.values() for g in c})
    print(f"{len(table)} resolutions x {len(next(iter(table.values())))} rates,"
          f" {len(groups)} groups")
    print("\nALL, by resolution x rate (carry off / carry on):")
    print("      " + "".join(f"{r:>18s}" for r in ("1.0000", "0.5000", "0.3333")))
    for res in ("640", "512", "416", "320"):
        line = f"{res:>5s} "
        for rate in ("1.0000", "0.5000", "0.3333"):
            a_ = (table.get(res, {}).get(rate) or {}).get("ALL")
            b_ = (table_carry.get(res, {}).get(rate) or {}).get("ALL")
            line += f"  {a_:.4f}/{b_:.4f}" if a_ and b_ else f"{'--':>18s}"
        print(line)


if __name__ == "__main__":
    main()
