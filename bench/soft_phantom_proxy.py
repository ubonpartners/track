"""Smooth proxy for `fp_tracks` (Phase 20.12).

User feedback on Phase 20.7/20.8b: the integer-counted `fp_tracks`
metric is too noisy for selection — a single phantom flipping in/out
of existence changes fitness by 0.0005, comparable to the gap between
working heads. They proposed: smooth via "how close was the promote
decision was" — i.e., for each phantom-track that the head would
emit, weight by the head's confidence at the promote moment, so
borderline emissions (σ≈0.51) count less than confident ones (σ≈0.99).

This module implements an offline approximation of that idea using
the val corpus. For each (seq, track_id) where the track is a phantom
(no GT match anywhere), we compute the max σ_promote over its
UNCONFIRMED-state rows. That's "the most committed the head was to
emitting this phantom on the legacy state-distribution". Sum over
phantoms = `soft_phantom_emit_proxy`. Smooth, differentiable in head
parameters, computes in seconds.

Caveat (per Phase 20.8b): val-corpus behaviour does not perfectly
predict deployment behaviour — the val distribution is biased to the
legacy operating point. So this proxy is for SCREENING, not final
selection. We still want diverse-29 deployment fitness for shipping
decisions. Worth checking is whether the proxy's *ranking* across
candidates correlates with deployment fitness ranking.

Usage:
    python -m bench.soft_phantom_proxy \\
        --val bench/data/state_corpus_v9_val.npz \\
        bench/data/state_head_v14.pt \\
        bench/data/state_head_v23.pt \\
        bench/data/state_head_v24_h64.pt \\
        ... \\
        [--diverse-fitness-json /tmp/joint_retrain/diverse_*.json]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.train_state_head import StateHead, build_input_matrix


def compute_proxies(pt_path: str, val_npz_path: str) -> dict:
    """Returns soft phantom/real-emit proxies for the head at pt_path."""
    ck = torch.load(pt_path, weights_only=False, map_location="cpu")
    in_dim = ck["in_dim"]; hidden = ck["hidden"]; e_dim = ck["e_dim"]
    backbone_layers = ck.get("backbone_layers", 2)

    data = np.load(val_npz_path, allow_pickle=False)
    rec = data["records"]
    X = build_input_matrix(rec)
    if in_dim == 32 and X.shape[1] == 37:
        # 32-dim head was trained with --no-history; drop history columns.
        X = np.concatenate([X[:, :14], X[:, 19:]], axis=1)
    elif X.shape[1] != in_dim:
        raise SystemExit(f"in_dim mismatch: head={in_dim}, corpus={X.shape[1]}")

    model = StateHead(in_dim=in_dim, hidden=hidden, e_dim=e_dim,
                      dropout=0.0, backbone_layers=backbone_layers)
    model.load_state_dict(ck["state_dict"]); model.eval()

    with torch.no_grad():
        pl, _, _, _ = model(torch.from_numpy(X))
    sig = torch.sigmoid(pl).numpy()

    valid_p = rec["valid_promote"].astype(bool)
    seq = rec["sequence"].astype(str)
    tid = rec["track_id"].astype(np.int64)
    gid = rec["gt_id_now"]

    keys = np.char.add(seq, np.char.add("#", tid.astype(str)))
    _, t_idx = np.unique(keys, return_inverse=True)
    n_tracks = int(t_idx.max()) + 1

    # Per-track: ever_matched (any gid >= 0).
    is_phantom = np.ones(n_tracks, dtype=bool)
    if (gid >= 0).any():
        is_phantom[np.unique(t_idx[gid >= 0])] = False

    # Per-track: max σ over UNCONFIRMED rows (only valid_promote=1).
    max_sig = np.full(n_tracks, -1.0, dtype=np.float32)
    for i in range(len(rec)):
        if not valid_p[i]: continue
        t = t_idx[i]
        if sig[i] > max_sig[t]:
            max_sig[t] = sig[i]
    # Drop tracks with no UNCONFIRMED rows (immediate-confirm shortcut).
    has_unc = max_sig >= 0
    max_sig = max_sig[has_unc]
    is_phantom = is_phantom[has_unc]

    n_phantom = int(is_phantom.sum())
    n_real = int((~is_phantom).sum())
    # Threshold-soft "would emit" — σ at the most-confident frame.
    soft_phantom_emit = float(max_sig[is_phantom].sum())
    soft_real_emit    = float(max_sig[~is_phantom].sum())
    # Hard threshold-0.5 mimics the C runtime's deterministic decision.
    hard_phantom_emit = int((max_sig[is_phantom] >= 0.5).sum())
    hard_real_emit    = int((max_sig[~is_phantom] >= 0.5).sum())
    return {
        "n_phantom_tracks":   n_phantom,
        "n_real_tracks":      n_real,
        "soft_phantom_emit":  soft_phantom_emit,
        "soft_real_emit":     soft_real_emit,
        "hard_phantom_emit":  hard_phantom_emit,
        "hard_real_emit":     hard_real_emit,
        "soft_phantom_rate":  soft_phantom_emit / max(1, n_phantom),
        "real_emit_rate":     soft_real_emit / max(1, n_real),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("heads", nargs="+", help=".pt checkpoint paths")
    p.add_argument("--val", required=True, help="val corpus .npz")
    p.add_argument("--diverse-fitness-json", default=None,
                   help="optional path to a JSON like /tmp/joint_retrain/diverse_v23.json "
                        "OR a directory containing such files; deployment fitness will be "
                        "joined into the output table when filename matches.")
    args = p.parse_args()

    # Optional: load deployment fitness (from prior eval_head_fitness runs).
    fit_lookup = {}
    if args.diverse_fitness_json:
        p_jf = Path(args.diverse_fitness_json)
        candidates = [p_jf] if p_jf.is_file() else list(p_jf.glob("diverse_*.json"))
        for jf in candidates:
            try:
                d = json.loads(jf.read_text())["overall"]
                # Map filename stem to fitness; user will name them after head tags.
                fit_lookup[jf.stem] = d.get("fitness")
            except Exception:
                pass

    rows = []
    for pt in args.heads:
        try:
            r = compute_proxies(pt, args.val)
        except Exception as e:
            print(f"  [skip] {pt}: {e}", file=sys.stderr); continue
        r["head"] = Path(pt).stem
        # Best-effort: pull deployment fitness if a matching diverse_*.json exists.
        deploy_fit = None
        for k, v in fit_lookup.items():
            if r["head"] in k or k.replace("diverse_", "") in r["head"]:
                deploy_fit = v; break
        r["deploy_fitness"] = deploy_fit
        rows.append(r)

    # Print sorted by soft_phantom_emit (lower = better).
    rows.sort(key=lambda r: r["soft_phantom_emit"])
    print(f"\n{'head':>30s}  {'soft_phantom':>12s}  {'hard_phantom':>12s}  "
          f"{'real_rate':>9s}  {'deploy_fit':>10s}")
    for r in rows:
        df = f"{r['deploy_fitness']:.4f}" if r["deploy_fitness"] is not None else "  n/a "
        print(f"  {r['head']:>28s}  {r['soft_phantom_emit']:12.2f}  "
              f"{r['hard_phantom_emit']:12d}  {r['real_emit_rate']:9.4f}  {df:>10s}")


if __name__ == "__main__":
    main()
