#!/bin/bash
# Bootstrap recipe — a deterministic, no-prior-weights training pipeline
# for the match-cost and state-head NNs.
#
# Breaks the circular dependency where every retrain inherits the prior
# NNs' bias via the pair-log. Three iterations:
#
#   iter 0: pair-log generated with NO NNs loaded (legacy heuristic
#           match + legacy state machine).
#   iter 1: train match + state on iter-0 pair-log. → match_iter1, state_iter1.
#   iter 2: regen pair-log with iter1 NNs loaded. Retrain match + state.
#           → match_iter2, state_iter2.
#   iter 3: regen pair-log with iter2 NNs loaded. Retrain match + state.
#           → match_iter3, state_iter3. STOP.
#
# In our empirical runs, iter1→iter2 closed most of the gap to shipped.
# Iter 3 included to see if a further refinement step helps (open
# question — DAgger has no convergence guarantee).
#
# All inputs are pinned: tracker config, dataset, hyperparameters. Rerun
# anytime to reproduce. Provenance is self-describing — no archaeology.
#
# Usage:
#   bench/bootstrap_recipe.sh                   # run end-to-end (~3-4h)
#   bench/bootstrap_recipe.sh --stop-after 1    # stop after iter 1
#   bench/bootstrap_recipe.sh --stop-after 2    # stop after iter 2 (old default)
#   bench/bootstrap_recipe.sh --start-at 2      # iter 2+iter 3 (assumes iter 1 done)
#   bench/bootstrap_recipe.sh --start-at 3      # iter 3 only (assumes iter 1+2 done)

set -euo pipefail

cd /home/mark/stuff/ubonpartners/track

START_AT=1
STOP_AFTER=3
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-at)  START_AT="$2"; shift 2;;
        --stop-after) STOP_AFTER="$2"; shift 2;;
        # Back-compat aliases:
        --iter1-only) STOP_AFTER=1; shift;;
        --skip-iter1) START_AT=2; shift;;
        -h|--help) sed -n '2,32p' "$0"; exit 0;;
        *) echo "Unknown arg: $1" >&2; exit 2;;
    esac
done

if ! [[ "$START_AT"  =~ ^[1-3]$ ]] || ! [[ "$STOP_AFTER" =~ ^[1-3]$ ]]; then
    echo "ERROR: --start-at and --stop-after must be 1, 2, or 3" >&2
    exit 2
fi
if [[ "$START_AT" -gt "$STOP_AFTER" ]]; then
    echo "ERROR: --start-at ($START_AT) > --stop-after ($STOP_AFTER)" >&2
    exit 2
fi

ITER0_CONFIG=bench/pair_log_config_iter0_noNN_jaad.yaml
EVAL_BASE=bench/eval_base_bootstrap.yaml

if [[ ! -f "$ITER0_CONFIG" || ! -f "$EVAL_BASE" ]]; then
    echo "ERROR: required configs missing — $ITER0_CONFIG / $EVAL_BASE" >&2
    exit 2
fi

LOG=/tmp/bootstrap_recipe_$(date +%Y%m%d_%H%M%S).log
echo "Bootstrap recipe log: $LOG"
echo "============================================================" | tee "$LOG"
echo "bootstrap_recipe.sh started at $(date)"                       | tee -a "$LOG"
echo "  iter0 config: $ITER0_CONFIG"                                | tee -a "$LOG"
echo "  eval base:    $EVAL_BASE"                                   | tee -a "$LOG"
echo "  start_at:     iter $START_AT"                               | tee -a "$LOG"
echo "  stop_after:   iter $STOP_AFTER"                             | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

# Helper: build a pair-log config from a template, baking in NN paths.
# $1 = prev-iter match .bin path  $2 = prev-iter state .bin path
# $3 = analysis name              $4 = output config path
make_iter_config() {
    local match_bin="$1"
    local state_bin="$2"
    local name="$3"
    local out="$4"
    python3 - <<PY
import yaml
cfg = yaml.safe_load(open('$ITER0_CONFIG'))
cfg['tracker_config_overrides']['utrack']['nn_path']       = '$match_bin'
cfg['tracker_config_overrides']['utrack']['nn_state_path'] = '$state_bin'
cfg['tracker_config_overrides']['utrack']['nn_lambda']     = 0.05
cfg['analysis_name']                                       = '$name'
cfg['output_root']                                         = '/mldata/track_analysis_runs/$name'
cfg['module_params']['pair_logger']['output_dir']          = '/mldata/track_analysis_runs/$name/pair_log'
yaml.safe_dump(cfg, open('$out', 'w'))
print('Wrote $out')
print('  using match :', '$match_bin')
print('  using state :', '$state_bin')
PY
}

# Helper: assert iter outputs exist, abort otherwise.
require_iter_outputs() {
    local n="$1"
    local match="bench/data/iter${n}/nn_match_iter${n}.bin"
    local state="bench/data/iter${n}/state_head_iter${n}_pw0.5.bin"
    if [[ ! -f "$match" || ! -f "$state" ]]; then
        echo "ERROR: iter${n} outputs missing — need $match and $state" >&2
        exit 1
    fi
}

# ----------------------------------------------------------------------
# Iter 1: train on no-NN pair-log.
# ----------------------------------------------------------------------
if [[ "$START_AT" -le 1 && "$STOP_AFTER" -ge 1 ]]; then
    echo ""                                                          | tee -a "$LOG"
    echo "##### ITER 1: train on no-NN pair-log #####"               | tee -a "$LOG"
    echo "started at $(date)"                                        | tee -a "$LOG"
    bench/run_pipeline.sh \
        --config "$ITER0_CONFIG" \
        --tag iter1 \
        --eval-base "$EVAL_BASE" \
        2>&1 | tee -a "$LOG"
    echo "iter 1 done at $(date)"                                    | tee -a "$LOG"
fi
require_iter_outputs 1

# ----------------------------------------------------------------------
# Iter 2: pair-log with iter1 NNs → retrain match + state.
# ----------------------------------------------------------------------
if [[ "$START_AT" -le 2 && "$STOP_AFTER" -ge 2 ]]; then
    ITER2_CONFIG=bench/pair_log_config_iter2_iter1NN.yaml
    make_iter_config \
        bench/data/iter1/nn_match_iter1.bin \
        bench/data/iter1/state_head_iter1_pw0.5.bin \
        pair_log_iter2_iter1NN \
        "$ITER2_CONFIG"

    echo ""                                                          | tee -a "$LOG"
    echo "##### ITER 2: train on iter-1-NN pair-log #####"           | tee -a "$LOG"
    echo "started at $(date)"                                        | tee -a "$LOG"
    bench/run_pipeline.sh \
        --config "$ITER2_CONFIG" \
        --tag iter2 \
        --eval-base "$EVAL_BASE" \
        2>&1 | tee -a "$LOG"
    echo "iter 2 done at $(date)"                                    | tee -a "$LOG"
fi
[[ "$STOP_AFTER" -ge 2 ]] && require_iter_outputs 2

# ----------------------------------------------------------------------
# Iter 3: pair-log with iter2 NNs → retrain match + state.
# Open question: does another DAgger pass help, or are we already at
# the head's representational ceiling? Run-and-see.
# ----------------------------------------------------------------------
if [[ "$START_AT" -le 3 && "$STOP_AFTER" -ge 3 ]]; then
    ITER3_CONFIG=bench/pair_log_config_iter3_iter2NN.yaml
    make_iter_config \
        bench/data/iter2/nn_match_iter2.bin \
        bench/data/iter2/state_head_iter2_pw0.5.bin \
        pair_log_iter3_iter2NN \
        "$ITER3_CONFIG"

    echo ""                                                          | tee -a "$LOG"
    echo "##### ITER 3: train on iter-2-NN pair-log #####"           | tee -a "$LOG"
    echo "started at $(date)"                                        | tee -a "$LOG"
    bench/run_pipeline.sh \
        --config "$ITER3_CONFIG" \
        --tag iter3 \
        --eval-base "$EVAL_BASE" \
        2>&1 | tee -a "$LOG"
    echo "iter 3 done at $(date)"                                    | tee -a "$LOG"
fi

# ----------------------------------------------------------------------
# Summary table — fitness across all iterations + references.
# ----------------------------------------------------------------------
echo ""                                                              | tee -a "$LOG"
echo "##### FINAL SUMMARY #####"                                     | tee -a "$LOG"
python3 - 2>&1 | tee -a "$LOG" <<'PY'
import json, os
rows = [
    ('no-NN baseline (full)',          '/tmp/eval_full_nonn.json'),
    ('iter1 full      (newM + newS)',  '/tmp/run_pipeline_iter1/full.json'),
    ('iter1 match-only',               '/tmp/run_pipeline_iter1/match_only.json'),
    ('iter1 state-only',               '/tmp/run_pipeline_iter1/state_only.json'),
    ('iter2 full      (newM + newS)',  '/tmp/run_pipeline_iter2/full.json'),
    ('iter2 match-only',               '/tmp/run_pipeline_iter2/match_only.json'),
    ('iter2 state-only',               '/tmp/run_pipeline_iter2/state_only.json'),
    ('iter3 full      (newM + newS)',  '/tmp/run_pipeline_iter3/full.json'),
    ('iter3 match-only',               '/tmp/run_pipeline_iter3/match_only.json'),
    ('iter3 state-only',               '/tmp/run_pipeline_iter3/state_only.json'),
    ('prod default    (shipped M+S)',  '/tmp/eval_full_prod.json'),
]
print(f"{'config':42s}  {'fitness':>8}  {'MOTA':>8}  {'fp_tracks':>10}")
for tag, p in rows:
    if not os.path.exists(p):
        print(f"{tag:42s}  (missing: {p})"); continue
    d = json.load(open(p))
    a = d.get('aggregate') or d.get('overall') or {}
    fm = a.get('fitness_mean') or a.get('fitness')
    mm = a.get('mota_mean')    or a.get('mota')
    ft = a.get('fp_tracks_mean') or a.get('fp_tracks')
    print(f"{tag:42s}  {fm:>8.4f}  {mm:>8.4f}  {ft:>10}")
PY

echo "============================================================"  | tee -a "$LOG"
echo "bootstrap_recipe.sh DONE at $(date)"                           | tee -a "$LOG"
echo "Full log: $LOG"
