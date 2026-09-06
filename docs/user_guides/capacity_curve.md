# Remaking the capacity curve (quality vs concurrent streams)

The "quality vs N streams" chart is the product performance metric: at N
streams the Jetson's PM controller settles on an operating-point mix, and that
mix implies a tracking quality. The chart joins two independently-measured
halves. **Both halves go stale**: the quality table whenever the tracker
config / eval data / detector changes (it was measured WITH a specific
`uc_v11.yaml`), the sweep whenever runtime performance or the pm_table
changes. Rebuild both when in doubt — each is a one-command, walk-away job.

## 1. Quality table (x86, offline) — PM operating point → quality

~20 val evals, one per (resolution cap × analytics rate [× motion-carry])
point, then a rollup:

```bash
python -m tools.quality_grid            # runs the grid into /mldata/tracking/results/qtab/ (hours)
python -m tools.quality_table           # rolls up into /mldata/config/track/quality_table.yaml
```

Mechanics (all in `tools/quality_grid.py`): resolution is imposed with
`--pm <idx>` where idx is looked up by res cap in the CURRENT `pm_table`
(`ubon_cstuff include/pm_controller.h` — rows are (res,rate) pairs, so the
index↔cap mapping is not 1:1 and the script's `PM_FOR_RES` must match the
header). Rate is imposed exactly via `debug_analytics_mask` in a derived eval
yaml; `gridm_*` points add `min_time_delta_motion` so masked frames become
MOTION/NVOF carry frames (`performance.skip_mode: motion`'s offline twin).

Prove-knobs-live check before trusting a fresh grid: a `gridm_*` run must
show MOTION frames in its log / a lower quality delta vs its `grid_*` twin at
320; `--pm` must change the reported infer resolution in the eval output.

## 2. Stream sweep (Jetson, online) — #streams → PM mix

```bash
# on the Jetson, in ubon_cstuff (build rt_benchmark first — older binaries
# emit only 4 of the 6 pm columns):
cmake --build build --target rt_benchmark -j6
./tools/jetson_pm_sweep.sh rt_new.csv                      # current config
./tools/jetson_pm_sweep.sh rt_old.csv /tmp/old_policy.yaml # optional comparison policy
```

~35 min per policy (8 points × 30 s). The CSV rows carry the realized PM
distribution (one column per pm_table row), force-skip rate and the
motion-carry fraction — the complete operating point.

## 3. Combine and plot

```bash
scp jetson:.../rt_new.csv .
python -m tools.capacity_curve --csv "pm_table (new)=rt_new.csv" --group ALL   # numbers
python -m tools.capacity_plot  --csv "pm_table (new)=rt_new.csv" \
                             --csv "old policy=rt_old.csv" -o capacity.png   # the chart
```

`--group` selects the content type (ALL = search-weighted objective; also
static/moving/movie/bodycam/dashcam_* etc. from the table) — per-content
curves are where degrade-policy tradeoffs actually show.

## Consistency rules (each has bitten before)

- The pm_table in `capacity_curve.py` (`PM_TABLE`) and the `PM_FOR_RES` map in
  `quality_grid.py` MUST match `include/pm_controller.h`. A 4-column CSV is
  read as the legacy resolution-only ladder; 6 columns as the current table.
- Quality table and sweep must measure the SAME tracker config (the promoted
  `uc_v11.yaml`); a table built under an old tracker silently miscolours a new
  sweep.
- The eval grid deliberately runs derived yamls — track.py's "not the
  objective" warning per run is expected, not a problem.
- `force_skip_rate` in the CSV is the realized skip INCLUDING the table rows'
  own rate dithering; never multiply row rates in again (capacity_curve
  already does this correctly).
