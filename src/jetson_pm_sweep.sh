#!/bin/bash
# The Jetson half of the capacity curve: measure the #streams -> PM-mix mapping.
# Run ON the Jetson (or via ssh) from the ubon_cstuff checkout. One policy per
# invocation; the tracker yaml decides the policy under test.
#
#   ./src/jetson_pm_sweep.sh OUT.csv [TRACKER_YAML]
#
# Produces an rt_benchmark CSV (streams,...,pm0..pmN,motion_percent) that
# src.capacity_curve / src.capacity_plot consume. 25 s measure + 5 s warmup
# per point, N=8..64 step 8 => ~35 min. Build rt_benchmark first
# (cmake --build build --target rt_benchmark) so the CSV has ALL pm_table
# columns (older binaries emitted only 4 of 6).
set -e
OUT="${1:?usage: jetson_pm_sweep.sh OUT.csv [TRACKER_YAML]}"
YAML="${2:-}"
ARGS=(--csv --min 8 --max 64 --step 8 --runtime 25 --warmup 5)
[ -n "$YAML" ] && ARGS+=(--tracker-yaml "$YAML")
./build/rt_benchmark "${ARGS[@]}" | tee "$OUT"
echo "# wrote $OUT" >&2
