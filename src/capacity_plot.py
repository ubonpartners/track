"""Render the capacity curve — quality vs concurrent streams — as a PNG.

The chart everyone remembers: one line per policy CSV, shaded gap + per-N
deltas between the first two. Joins the same two halves as capacity_curve
(quality_table.yaml x rt_benchmark CSVs); this is just the picture.

Usage:
    python -m src.capacity_plot --csv "new=rt_new.csv" [--csv "old=rt_old.csv"] \
        [--group ALL] [--table /mldata/config/track/quality_table.yaml] \
        [--title "..."] [-o capacity.png]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from src.capacity_curve import curve, parse_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--table", default="/mldata/config/track/quality_table.yaml")
    ap.add_argument("--group", default="ALL")
    ap.add_argument("--title", default="Jetson Orin Nano — quality vs concurrent streams")
    ap.add_argument("-o", "--out", default="capacity.png")
    a = ap.parse_args()

    doc = yaml.safe_load(open(a.table))
    table, table_carry = doc["table"], doc.get("table_motion_carry")

    curves = {}
    for spec in a.csv:
        name, _, path = spec.partition("=")
        curves[name] = curve(parse_csv(path), table, table_carry, a.group)

    fig, ax = plt.subplots(figsize=(10, 6))
    styles = [dict(color="tab:blue", lw=2.5, marker="o", ms=9),
              dict(color="gray", lw=2, ls="--", marker="o", ms=9)]
    for (name, c), st in zip(curves.items(), styles + [{}] * len(curves)):
        ax.plot([p["streams"] for p in c], [p["quality"] for p in c],
                label=name, **st)

    if len(curves) >= 2:  # shade + annotate the gap between the first two
        (n1, c1), (n2, c2) = list(curves.items())[:2]
        common = sorted({p["streams"] for p in c1} & {p["streams"] for p in c2})
        q1 = {p["streams"]: p["quality"] for p in c1}
        q2 = {p["streams"]: p["quality"] for p in c2}
        ax.fill_between(common, [q2[n] for n in common], [q1[n] for n in common],
                        alpha=0.15, color="tab:blue")
        for n in common:
            d = q1[n] - q2[n]
            if abs(d) >= 0.01:
                ax.annotate(f"{d:+.3f}", (n, (q1[n] + q2[n]) / 2),
                            color="tab:blue", ha="center", fontsize=11)

    ax.set_xlabel("concurrent streams")
    ax.set_ylabel(f"quality (fitness_multi, {a.group})")
    ax.set_title(a.title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
