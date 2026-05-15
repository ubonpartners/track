#!/bin/bash
# Clean recipe runner for {pair-log → match-cost → state-corpus →
# state-head → eval}. One parameterised script that replaces the
# per-experiment bash scripts that used to live in /tmp/skip_roi_exp/.
#
# Usage:
#   ml/orchestration/run_pipeline.sh --config ml/configs/pair_log_config_v19_clean.yaml \
#                         --tag v19 \
#                         [--no-regen]          # reuse cached ubtrk2 + pair_log
#                         [--no-eval]           # skip eval step
#                         [--pw 0.5]            # state head pos_weight
#                         [--epochs-match 25]   # train_phase3 epochs
#                         [--epochs-state 16]   # train_state_head epochs
#                         [--fp-boost 1.0]      # build_state_corpus fp_boost (DEFAULT 1.0 — DO NOT change without measuring)
#
# Steps run sequentially; if any step fails, the script aborts. All
# inputs are validated up-front: missing files / unknown yaml keys /
# dead-under-NN knobs are surfaced before any GPU work starts.

set -euo pipefail

# ---- arg parse ----
CONFIG=""
TAG=""
REGEN=1
DO_EVAL=1
PW=0.5
EPOCHS_MATCH=25
EPOCHS_STATE=16
FP_BOOST=1.0
SEED=0
EVAL_BASE=ml/configs/eval_base_uc_v11.yaml

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)         CONFIG="$2"; shift 2;;
        --tag)            TAG="$2"; shift 2;;
        --no-regen)       REGEN=0; shift;;
        --no-eval)        DO_EVAL=0; shift;;
        --pw)             PW="$2"; shift 2;;
        --epochs-match)   EPOCHS_MATCH="$2"; shift 2;;
        --epochs-state)   EPOCHS_STATE="$2"; shift 2;;
        --fp-boost)       FP_BOOST="$2"; shift 2;;
        --seed)           SEED="$2"; shift 2;;
        --eval-base)      EVAL_BASE="$2"; shift 2;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *) echo "Unknown arg: $1" >&2; exit 2;;
    esac
done

# ---- arg validation ----
if [[ -z "$CONFIG" || -z "$TAG" ]]; then
    echo "ERROR: --config and --tag are both required" >&2
    exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: --config not found: $CONFIG" >&2
    exit 2
fi
if [[ ! -f "$EVAL_BASE" ]]; then
    echo "ERROR: --eval-base not found: $EVAL_BASE" >&2
    exit 2
fi

# Tag must be a simple identifier to avoid surprises in paths.
if [[ ! "$TAG" =~ ^[a-zA-Z0-9_]+$ ]]; then
    echo "ERROR: --tag must be [a-zA-Z0-9_]+ (got '$TAG')" >&2
    exit 2
fi
# fp-boost default warning. The flag exists for measured experiments
# but the canonical recipe uses 1.0; anything else MUST be a deliberate
# decision since the trainer bug-fix on 2026-05-12 made this knob real.
if [[ "$FP_BOOST" != "1.0" ]]; then
    echo "WARNING: --fp-boost=$FP_BOOST (canonical recipe uses 1.0)." >&2
    echo "         Justify this in your run notes." >&2
fi

# ---- derived paths ----
DDIR="ml/data/${TAG}"
# pair-log output_root must match what the yaml declares — parse it
# directly so the launcher and the engine agree on disk layout.
PAIRLOGDIR=$(python -c "
import yaml, sys
cfg = yaml.safe_load(open('$CONFIG'))
root = cfg.get('output_root')
if not root:
    sys.exit('output_root missing from $CONFIG')
print(root)
")
PHASE3_PT="${DDIR}/phase3_${TAG}.pt"
PHASE3_BIN="${DDIR}/nn_match_${TAG}.bin"
STATE_CORPUS_PREFIX="${DDIR}/state_corpus_${TAG}"
STATE_HEAD_PT="${DDIR}/state_head_${TAG}_pw${PW}.pt"
STATE_HEAD_BIN="${DDIR}/state_head_${TAG}_pw${PW}.bin"
EVAL_OUTDIR="/tmp/run_pipeline_${TAG}"

mkdir -p "$DDIR" "$EVAL_OUTDIR"

echo "===================================================="
echo "  run_pipeline.sh  tag=${TAG}  config=${CONFIG}"
echo "    pair-log dir : ${PAIRLOGDIR}"
echo "    data dir     : ${DDIR}"
echo "    regen        : ${REGEN}  do_eval : ${DO_EVAL}"
echo "    fp_boost     : ${FP_BOOST}"
echo "    pw           : ${PW}"
echo "    eval base    : ${EVAL_BASE}"
echo "===================================================="

# ---- Step 1: pair-log generation ----
echo "=== [1/6] pair-log generation ==="
if [[ "$REGEN" -eq 1 ]]; then
    python track_analysis.py --config "$CONFIG" --force-regen --split all
else
    # Without --force-regen, cached ubtrk2 are reused. Useful when only
    # the metric (pair_logger) changed, not the tracker code.
    python track_analysis.py --config "$CONFIG" --split all
fi

PAIRLOG_NPZ_COUNT=$(ls "${PAIRLOGDIR}/pair_log/"*.npz 2>/dev/null | wc -l)
if [[ "$PAIRLOG_NPZ_COUNT" -lt 100 ]]; then
    echo "ERROR: pair-log produced only $PAIRLOG_NPZ_COUNT npz (expected ≥100)" >&2
    exit 1
fi
echo "  pair-log npz count: $PAIRLOG_NPZ_COUNT"

# ---- Step 2: build pair dataset (3 splits) ----
echo "=== [2/6] build pair dataset ==="
for split in train val test; do
    python -m ml.data_prep.build_pair_dataset \
        --pair-log-dir "${PAIRLOGDIR}/pair_log" \
        --analysis-yaml "$CONFIG" \
        --split "$split" \
        --out "${DDIR}/pairs_${split}.npz" \
        --comment "${TAG} — from ${CONFIG}"
    if [[ ! -s "${DDIR}/pairs_${split}.npz" ]]; then
        echo "ERROR: pairs_${split}.npz missing or zero-size" >&2
        exit 1
    fi
done

# ---- Step 3: train match-cost (phase3) ----
echo "=== [3/6] train match-cost ==="
python -m ml.train.train_phase3 \
    --epochs "$EPOCHS_MATCH" \
    --data_dir "$DDIR" \
    --save "$PHASE3_PT" \
    --seed "$SEED" \
    --comment "${TAG} match-cost"
if [[ ! -f "$PHASE3_PT" ]]; then
    echo "ERROR: phase3 .pt not produced" >&2; exit 1
fi
python -m ml.export.export_phase3 --in "$PHASE3_PT" --out "$PHASE3_BIN"
if [[ ! -f "$PHASE3_BIN" ]]; then
    echo "ERROR: phase3 .bin not produced" >&2; exit 1
fi

# ---- Step 4: state corpus build ----
echo "=== [4/6] state corpus build ==="
python -m ml.data_prep.build_state_corpus \
    --pair-log-dir "$PAIRLOGDIR" \
    --gt-config "$CONFIG" \
    --label-driven \
    --phase3-model "$PHASE3_PT" \
    --fitness-fp-boost "$FP_BOOST" \
    --out "$STATE_CORPUS_PREFIX" \
    --comment "${TAG} state corpus — phase3=${PHASE3_PT} fp_boost=${FP_BOOST}"
for split in train val test; do
    if [[ ! -s "${STATE_CORPUS_PREFIX}_${split}.npz" ]]; then
        echo "ERROR: state_corpus_${TAG}_${split}.npz missing or zero-size" >&2
        exit 1
    fi
done

# ---- Step 5: train state head ----
echo "=== [5/6] train state head pw=${PW} ==="
python -m ml.train.train_state_head_decoupled \
    --train "${STATE_CORPUS_PREFIX}_train.npz" \
    --val   "${STATE_CORPUS_PREFIX}_val.npz" \
    --save  "$STATE_HEAD_PT" \
    --epochs "$EPOCHS_STATE" \
    --seed "$SEED" \
    --hidden 64 \
    --pos-weight "$PW" \
    --comment "${TAG} state head pw=${PW}"
if [[ ! -f "$STATE_HEAD_PT" ]]; then
    echo "ERROR: state head .pt not produced" >&2; exit 1
fi
python -m ml.export.export_decoupled_head --in "$STATE_HEAD_PT" --out "$STATE_HEAD_BIN"
if [[ ! -f "$STATE_HEAD_BIN" ]]; then
    echo "ERROR: state head .bin not produced" >&2; exit 1
fi

# ---- Step 6: eval (full subset, 3 configs: full / match-only / state-only) ----
if [[ "$DO_EVAL" -eq 0 ]]; then
    echo "=== [6/6] eval SKIPPED (--no-eval) ==="
    echo "==== run_pipeline.sh tag=${TAG} done ===="
    exit 0
fi

echo "=== [6/6] eval ==="
# Build the three eval yamls from EVAL_BASE
python -c "
import yaml
base = yaml.safe_load(open('$EVAL_BASE'))
# Strip flags that pre-dated this clean recipe — fresh eval should
# mirror the canonical config exactly.
base.pop('include_pending_tracks_in_roi', None)
base.pop('match_clip_to_roi', None)
ut = base.setdefault('utrack', {})
for k in ('match_clip_to_roi', 'include_pending_tracks_in_roi'):
    ut.pop(k, None)

# full: ${TAG} match + ${TAG} state
cfg = yaml.safe_load(yaml.safe_dump(base))
cfg['utrack']['nn_path']       = '$PHASE3_BIN'
cfg['utrack']['nn_state_path'] = '$STATE_HEAD_BIN'
yaml.safe_dump(cfg, open('$EVAL_OUTDIR/eval_full.yaml', 'w'))

# match-only: new match + shipped state
cfg = yaml.safe_load(yaml.safe_dump(base))
cfg['utrack']['nn_path']       = '$PHASE3_BIN'
cfg['utrack']['nn_state_path'] = '/mldata/config/track/trackers/nn_state_v20_pw05.bin'
yaml.safe_dump(cfg, open('$EVAL_OUTDIR/eval_match_only.yaml', 'w'))

# state-only: shipped match + new state
cfg = yaml.safe_load(yaml.safe_dump(base))
cfg['utrack']['nn_path']       = '/mldata/config/track/trackers/nn_match_v10_face.bin'
cfg['utrack']['nn_state_path'] = '$STATE_HEAD_BIN'
yaml.safe_dump(cfg, open('$EVAL_OUTDIR/eval_state_only.yaml', 'w'))
"

print_metrics() {
    local label="$1"; local path="$2"
    python -c "
import json
d = json.load(open('$path'))
a = d.get('aggregate') or d['overall']
fm = a.get('fitness_mean') or a.get('fitness')
fs = a.get('fitness_stdev', 0.0)
mm = a.get('mota_mean') or a.get('mota')
ft = a.get('fp_tracks_mean') or a.get('fp_tracks')
print(f'$label: fitness={fm:.4f}+/-{fs:.4f}  MOTA={mm:.4f}  fp_tracks={ft}')"
}

for variant in full match_only state_only; do
    echo "--- eval ${TAG}_${variant} ---"
    python -m ml.eval.eval_head_fitness \
        --config "$EVAL_OUTDIR/eval_${variant}.yaml" \
        --subset full --workers 2 --runs 1 \
        --out "$EVAL_OUTDIR/${variant}.json" --quiet
    print_metrics "${TAG}_${variant}" "$EVAL_OUTDIR/${variant}.json"
done

echo "==== run_pipeline.sh tag=${TAG} done ===="
