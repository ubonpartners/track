"""Measure the decoupled head's calibration on the val corpus.

Two questions:

1. Reliability of `p_TP`: among rows the head says "p_TP=p", how often
   does the track actually turn out to be TP? Bin and report.
2. Implied c_FP_track: derive the cost-rule coefficient from empirical
   data instead of guessing. Specifically — sweep c_FP_track over a
   small grid using the offline state-machine sim, score each on a
   bench-fitness-shaped proxy, and report the arg-max.

The val-optimal c_FP_track from #2 is the operating point that lands
on the Pareto front for THIS head's calibration, no sweeping in the
loop required after this once.

Run:
    python -m bench.calibrate_head \\
        --head bench/data/state_head_dc_v5.pt \\
        --val  bench/data/state_corpus_v15ld_val.npz
"""
from __future__ import annotations
import argparse
import numpy as np
import torch

from bench.train_state_head_gru        import (
    build_input_matrix_no_state, group_rows_by_track, pack_batch as _pack_batch_4)
from bench.train_state_head_decoupled  import DecoupledGRUHead
from bench.train_bayesian_head         import compute_track_labels
from bench.eval_decoupled_offline      import cost_decision, UNCONFIRMED, TRACKED, LOST


def head_forward_all_rows(model, X, groups, device, batch_size=256):
    """Run the GRU head over the val corpus, return p_TP, μ_TP, μ_FP per row.
    `groups` is a list of np.ndarray of row-indices, one per track."""
    model.eval()
    p_TP   = np.zeros(len(X), dtype=np.float32)
    mu_TP  = np.zeros(len(X), dtype=np.float32)
    mu_FP  = np.zeros(len(X), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(groups), batch_size):
            row_lists = groups[s:s+batch_size]
            T_b = max(len(r) for r in row_lists)
            B = len(row_lists)
            in_dim = X.shape[1]
            xb = np.zeros((B, T_b, in_dim), dtype=np.float32)
            lens = np.zeros(B, dtype=np.int32)
            for i, rows in enumerate(row_lists):
                xb[i, :len(rows)] = X[rows]
                lens[i] = len(rows)
            xt = torch.from_numpy(xb).to(device)
            llr_logit, mu_tp_log, mu_fp_log, _h = model(xt)
            llr_logit = llr_logit.cpu().numpy()
            mu_tp_log = mu_tp_log.cpu().numpy()
            mu_fp_log = mu_fp_log.cpu().numpy()
            for i, rows in enumerate(row_lists):
                L = int(lens[i])
                ll  = llr_logit[i, :L]
                mt  = np.expm1(mu_tp_log[i, :L])
                mf  = np.expm1(mu_fp_log[i, :L])
                p   = 1.0 / (1.0 + np.exp(-ll))
                for k, ridx in enumerate(rows):
                    p_TP[ridx]  = p[k]
                    mu_TP[ridx] = max(0.0, mt[k])
                    mu_FP[ridx] = max(0.0, mf[k])
    return p_TP, mu_TP, mu_FP


def reliability_curve(p_pred, label, n_bins=10):
    """Return (bin_centers, empirical_rate, count_per_bin)."""
    bins = np.linspace(0, 1, n_bins + 1)
    centers, emp, counts = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (p_pred >= lo) & (p_pred < hi if i < n_bins-1 else p_pred <= hi)
        n = int(m.sum())
        if n == 0:
            continue
        centers.append(0.5 * (lo + hi))
        emp.append(float(label[m].mean()))
        counts.append(n)
    return np.array(centers), np.array(emp), np.array(counts)


def simulate_with_cost(p_TP, mu_TP, mu_FP, matched, is_TP_track,
                       groups, c_MOTA, c_FP_track, c_FP_frame,
                       match_rate_TP=0.95):
    """Run the offline sim track-by-track with the given cost rule.
    Returns track-level outcomes."""
    promoted_TP = 0
    promoted_FP = 0
    not_promoted_TP = 0
    not_promoted_FP = 0
    tp_frames_when_TRACKED = 0  # MOTA gain proxy
    fp_frames_when_TRACKED = 0  # fp_per_frame proxy
    total_TP_rows = 0
    total_FP_rows = 0

    class C: pass
    c = C()
    c.c_MOTA = c_MOTA
    c.c_FP_track = c_FP_track
    c.c_FP_frame = c_FP_frame
    c.match_rate_TP = match_rate_TP

    for rows in groups:
        state = UNCONFIRMED
        ever_promoted = False
        is_TP = bool(is_TP_track[rows[0]])
        for r in rows:
            m = bool(matched[r])
            ns = cost_decision(
                float(p_TP[r]), float(mu_TP[r]), float(mu_FP[r]),
                state, m,
                c_MOTA=c.c_MOTA, c_FP_track=c.c_FP_track,
                c_FP_frame=c.c_FP_frame, match_rate_TP=c.match_rate_TP)
            state = ns
            if state == TRACKED:
                ever_promoted = True
                if is_TP: tp_frames_when_TRACKED += 1
                else:     fp_frames_when_TRACKED += 1
            if is_TP: total_TP_rows += 1
            else:     total_FP_rows += 1
        if is_TP:
            if ever_promoted: promoted_TP += 1
            else:             not_promoted_TP += 1
        else:
            if ever_promoted: promoted_FP += 1
            else:             not_promoted_FP += 1

    n_total_rows = max(1, total_TP_rows + total_FP_rows)
    mota_proxy = tp_frames_when_TRACKED / max(1, total_TP_rows)
    fp_per_frame = fp_frames_when_TRACKED / n_total_rows
    # Bench fitness shape:
    fitness_proxy = mota_proxy - 0.005 * promoted_FP - 0.002 * fp_per_frame
    return {
        "promoted_TP": promoted_TP,
        "promoted_FP": promoted_FP,
        "not_promoted_TP": not_promoted_TP,
        "not_promoted_FP": not_promoted_FP,
        "tp_coverage": tp_frames_when_TRACKED / max(1, total_TP_rows),
        "fp_exposure": fp_frames_when_TRACKED / max(1, total_FP_rows),
        "mota_proxy": mota_proxy,
        "fitness_proxy": fitness_proxy,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", required=True)
    p.add_argument("--val",  required=True)
    p.add_argument("--n-bins", type=int, default=10)
    args = p.parse_args()

    print(f"loading head from {args.head}", flush=True)
    ckpt = torch.load(args.head, map_location="cpu", weights_only=False)
    model = DecoupledGRUHead(in_dim=int(ckpt["in_dim"]),
                             hidden=int(ckpt["hidden"]))
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"loading {args.val}", flush=True)
    rec = np.load(args.val, allow_pickle=False)["records"]
    X = build_input_matrix_no_state(rec)
    print(f"  {len(X)} rows, in_dim={X.shape[1]}", flush=True)

    print("grouping by track ...", flush=True)
    groups = group_rows_by_track(rec)
    print(f"  {len(groups)} tracks", flush=True)

    print("running head forward over corpus ...", flush=True)
    p_TP, mu_TP, mu_FP = head_forward_all_rows(model, X, groups, device)

    is_TP_track, _rem = compute_track_labels(rec)
    is_TP_track = is_TP_track.astype(np.float32)
    is_TP_now    = (rec["gt_id_now"] != -1).astype(np.float32)
    matched      = (rec["matched"] != 0).astype(bool)

    # ---- Reliability of p_TP per row (per-frame is_TP_now label) ----
    print("\n=== Reliability: p_TP vs per-frame is_TP_now ===", flush=True)
    centers, emp, counts = reliability_curve(p_TP, is_TP_now, args.n_bins)
    print(f"  {'bin_mid':>8}  {'predicted':>10}  {'empirical':>10}  {'n':>10}  {'gap':>8}")
    for c_, e_, n_ in zip(centers, emp, counts):
        gap = e_ - c_
        print(f"  {c_:8.3f}  {c_:10.3f}  {e_:10.3f}  {n_:10d}  {gap:+8.3f}")

    # ---- Reliability vs track-level label (for "is this track promotable?") ----
    print("\n=== Reliability: p_TP vs track-level is_TP ===", flush=True)
    centers, emp, counts = reliability_curve(p_TP, is_TP_track, args.n_bins)
    print(f"  {'bin_mid':>8}  {'predicted':>10}  {'empirical':>10}  {'n':>10}  {'gap':>8}")
    for c_, e_, n_ in zip(centers, emp, counts):
        gap = e_ - c_
        print(f"  {c_:8.3f}  {c_:10.3f}  {e_:10.3f}  {n_:10d}  {gap:+8.3f}")

    # ---- Sweep c_FP_track on offline sim, find argmax fitness ----
    print("\n=== c_FP_track sweep on offline sim (val) ===", flush=True)
    print(f"  {'c_FP_track':>12}  {'prom_TP':>8}  {'prom_FP':>8}  "
          f"{'tpcov':>6}  {'fpexp':>6}  {'fitness':>9}")
    grid = [0.001, 0.005, 0.010, 0.025, 0.05, 0.10, 0.20]
    best = None
    for cft in grid:
        out = simulate_with_cost(
            p_TP, mu_TP, mu_FP, matched, is_TP_track, groups,
            c_MOTA=0.001, c_FP_track=cft, c_FP_frame=0.002)
        print(f"  {cft:12.4f}  {out['promoted_TP']:8d}  {out['promoted_FP']:8d}  "
              f"{out['tp_coverage']:6.3f}  {out['fp_exposure']:6.3f}  "
              f"{out['fitness_proxy']:9.4f}")
        if best is None or out["fitness_proxy"] > best[0]:
            best = (out["fitness_proxy"], cft, out)
    print(f"\nval-optimal c_FP_track = {best[1]:.4f}  fitness_proxy={best[0]:.4f}")
    print(f"  (vs the bench formula's literal 0.005 — "
          f"the gap is the head's miscalibration)")


if __name__ == "__main__":
    main()
