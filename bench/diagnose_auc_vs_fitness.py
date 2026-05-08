"""Diagnostic — AUC vs aggregate fitness across trained heads.

Demonstrates concretely why the offline pipeline is broken: take the
trained heads from this session (v14, v15, v16, v17, v18, v19b, v20),
read their best val combined AUC from saved logs, and run them through
the closed-loop fitness eval (`bench/eval_head_fitness.py`) on the
diverse 29-clip subset.

Output: a table showing val AUC ranking ≠ fitness ranking. This is
the smoking gun that motivates Phase 20a.
"""
from __future__ import annotations
import json, os, re, sys, tempfile
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.eval_head_fitness import eval_config, make_yaml_with_state_bin

# Heads to compare. We pick a representative spread.
HEADS = [
    # label              .bin path                                                  log path (for AUC)
    ("v14",              "/mldata/config/track/trackers/nn_state_v14.bin",          None),
    ("v15_p1.0",         "bench/data/state_head_v15_p1.0.bin",                      "/tmp/joint_retrain/state_head_v15_p1.0.log"),
    ("v15_p2.0",         "bench/data/state_head_v15_p2.0.bin",                      "/tmp/joint_retrain/state_head_v15_p2.0.log"),
    ("v16_p2.0",         "bench/data/state_head_v16_p2.0.bin",                      "/tmp/joint_retrain/state_head_v16_p2.0.log"),
    ("v17_p3.0",         "bench/data/state_head_v17_p3.0.bin",                      "/tmp/joint_retrain/state_head_v17_p3.0.log"),
    ("v18_p3.0",         "bench/data/state_head_v18_p3.0.bin",                      "/tmp/joint_retrain/state_head_v18_p3.0.log"),
    ("v19b_fb20",        "bench/data/state_head_v19b_fb20_0.bin",                   "/tmp/joint_retrain/state_head_v19b_fb20_0.log"),
    ("v20_fb5",          "bench/data/state_head_v20_fb5_0.bin",                     "/tmp/joint_retrain/state_head_v20_fb5_0.log"),
]

def parse_auc(log_path):
    if not log_path or not os.path.isfile(log_path): return None
    txt = open(log_path).read()
    m = re.search(r"best combined AUC = ([\d.]+)", txt)
    return float(m.group(1)) if m else None

# Use current live as the base (everything except nn_state_path).
BASE_YAML = "/mldata/config/track/trackers/uc_v11.yaml"

results = []
for label, bin_path, log_path in HEADS:
    if not os.path.isfile(bin_path):
        print(f"[skip] {label}: missing .bin at {bin_path}", file=sys.stderr)
        continue
    auc = parse_auc(log_path)
    print(f"\n=== {label} (.bin={bin_path}) ===", flush=True)
    yaml_path = make_yaml_with_state_bin(bin_path, BASE_YAML)
    res = eval_config(yaml_path)
    o = res["overall"]
    print(f"  val AUC: {auc!s}  agg_fit: {o['fitness']:.4f}  mota: {o['mota']:.4f}  FPTr: {o['fp_tracks']}  FP/fr: {o['fp_per_frame']:.3f}")
    results.append({
        "label": label, "bin": bin_path, "val_auc": auc,
        "agg_fitness": o["fitness"], "agg_mota": o["mota"],
        "fp_tracks": o["fp_tracks"], "fp_per_frame": o["fp_per_frame"],
        "per_family": res["per_family"],
    })

# Output the disagreement table.
print(f"\n\n=== AUC vs aggregate fitness on 29-clip diverse subset ===")
print(f"  {'head':14s}  {'val_AUC':>8s}  {'agg_fit':>8s}  {'mota':>7s}  {'FPTr':>5s}  {'AUC_rank':>8s}  {'fit_rank':>8s}")
auc_sorted   = sorted(results, key=lambda r: -(r["val_auc"] or 0))
fit_sorted   = sorted(results, key=lambda r: -r["agg_fitness"])
auc_rank = {r["label"]: i+1 for i, r in enumerate(auc_sorted)}
fit_rank = {r["label"]: i+1 for i, r in enumerate(fit_sorted)}
for r in fit_sorted:
    auc_s = f"{r['val_auc']:.4f}" if r["val_auc"] is not None else "  n/a "
    print(f"  {r['label']:14s}  {auc_s:>8s}  {r['agg_fitness']:8.4f}  {r['agg_mota']:7.4f}  {r['fp_tracks']:5d}  {auc_rank[r['label']]:>8d}  {fit_rank[r['label']]:>8d}")

# Save results for future iteration.
out = Path("/tmp/joint_retrain/diagnose_auc_vs_fitness.json")
out.write_text(json.dumps(results, indent=2))
print(f"\nSaved -> {out}")
