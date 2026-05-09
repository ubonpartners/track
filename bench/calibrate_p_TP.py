"""Build a calibration map for raw GRU p_TP, save it, and re-run the
offline cost-rule sim to measure the fitness lift.

Why this exists
---------------
The cost rule in the runtime treats `p_TP` as a probability in
expected-value arithmetic:

    ΔF(promote) = p_TP * c_MOTA * μ_TP * match_rate
                 − (1 − p_TP) * (c_FP_track + c_FP_frame * μ_FP)

But the GRU head's output is just `sigmoid(linear(hidden_state))` —
not a calibrated probability. v8 reliability shows a U-shape:
well-calibrated at the extremes but massively over-confident in the
0.4–0.7 mid-range (head says ~0.5, empirical per-frame is_TP_now is
~0.07). Plugging an over-confident value into the cost rule sends
ΔF(promote) the wrong sign on borderline tracks, and that's the
offline/online gap we keep papering over with c_FP_track sweeps.

This script:
  1. Forward-passes the val corpus through the head.
  2. Fits a calibrator g: p_raw -> p_calibrated against the per-frame
     is_TP_now label, using one of:
       - isotonic   (sklearn IsotonicRegression, monotone non-parametric)
       - platt      (sigmoid fit on logits, parametric, robust to small bins)
       - piecewise  (legacy bin-then-PAV used by the original starter)
  3. Reports Brier score and ECE for raw vs calibrated.
  4. Sweeps c_FP_track on the offline state-machine sim for both raw
     and calibrated p_TP, and reports fitness_proxy at each.
  5. Saves the calibrator as a small numpy table (xs, ys) suitable for
     a `np.interp` lookup at runtime.

Usage:
    python -m bench.calibrate_p_TP \\
        --head bench/data/state_head_dc_v8.pt \\
        --val  bench/data/state_corpus_v9_val.npz \\
        --out  bench/data/calib_v8.npz \\
        --method isotonic
"""
from __future__ import annotations
import argparse
from typing import Tuple

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from bench.train_state_head_gru        import build_input_matrix_no_state, group_rows_by_track
from bench.train_state_head_decoupled  import DecoupledGRUHead
from bench.eval_decoupled_offline      import cost_decision, UNCONFIRMED, TRACKED, LOST


# ---------------------------------------------------------------------------
# Forward pass over the corpus (same shape as eval_decoupled_offline).
# ---------------------------------------------------------------------------
def head_forward_all(model, X, groups, device, in_dim, batch=64):
    """Run the GRU head over every track. Returns (p_raw, mu_TP, mu_FP)
    arrays of length len(X). p_raw is sigmoid of the LLR logit."""
    p_raw = np.zeros(len(X), dtype=np.float32)
    mu_TP = np.zeros(len(X), dtype=np.float32)
    mu_FP = np.zeros(len(X), dtype=np.float32)
    order = sorted(range(len(groups)), key=lambda i: len(groups[i]))
    with torch.no_grad():
        for s in range(0, len(groups), batch):
            picks = order[s:s+batch]
            T_max = max(len(groups[g]) for g in picks)
            B = len(picks)
            X_b = np.zeros((B, T_max, in_dim), dtype=np.float32)
            for bi, g in enumerate(picks):
                rows = groups[g]
                X_b[bi, :len(rows)] = X[rows]
            Xt = torch.from_numpy(X_b).to(device)
            llr_l, mtp_l, mfp_l, _ = model(Xt)
            p_b  = torch.sigmoid(llr_l).cpu().numpy()
            mtp  = np.expm1(np.clip(mtp_l.cpu().numpy(), 0, 10))
            mfp  = np.expm1(np.clip(mfp_l.cpu().numpy(), 0, 10))
            for bi, g in enumerate(picks):
                rows = groups[g]
                L = len(rows)
                p_raw[rows] = p_b[bi, :L]
                mu_TP[rows] = np.maximum(0.0, mtp[bi, :L])
                mu_FP[rows] = np.maximum(0.0, mfp[bi, :L])
    return p_raw, mu_TP, mu_FP


# ---------------------------------------------------------------------------
# Calibrators. Each returns (xs, ys) — a piecewise-linear lookup table
# that the runtime can apply with `np.interp(p_raw, xs, ys)`. We use a
# common discretization grid (n_grid points on [0,1]) for all methods so
# the on-disk format is identical regardless of fitting method.
# ---------------------------------------------------------------------------
def _emit_lookup_table(predict_fn, n_grid: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a calibrator at evenly-spaced points on [0,1] and clamp to
    a non-decreasing, [0,1]-valued sequence. Returns (xs, ys) float32."""
    xs = np.linspace(0.0, 1.0, n_grid).astype(np.float32)
    ys = np.clip(predict_fn(xs), 0.0, 1.0).astype(np.float32)
    # Enforce monotonicity numerically (sklearn isotonic is already monotone
    # but Platt is too — this is just a paranoid safety net for piecewise).
    ys = np.maximum.accumulate(ys)
    return xs, ys


def fit_isotonic(p_raw: np.ndarray, label: np.ndarray, n_grid: int = 256):
    """Monotone non-parametric calibration via sklearn's PAV implementation."""
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip",
                             increasing=True)
    iso.fit(p_raw.astype(np.float64), label.astype(np.float64))
    return _emit_lookup_table(lambda xs: iso.predict(xs.astype(np.float64)),
                              n_grid=n_grid)


def fit_platt(p_raw: np.ndarray, label: np.ndarray, n_grid: int = 256):
    """Platt scaling: fit a 1-D logistic regression on the raw logit
    (= logit(p_raw)) against the label, then resample on a [0,1] grid.

    Robust to bins with few samples — only 2 parameters are fit."""
    eps = 1e-6
    p_clip = np.clip(p_raw.astype(np.float64), eps, 1.0 - eps)
    logit = np.log(p_clip / (1.0 - p_clip)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    lr.fit(logit, label.astype(np.int32))

    def _predict(xs: np.ndarray) -> np.ndarray:
        x = np.clip(xs.astype(np.float64), eps, 1.0 - eps)
        z = np.log(x / (1.0 - x)).reshape(-1, 1)
        return lr.predict_proba(z)[:, 1]

    return _emit_lookup_table(_predict, n_grid=n_grid)


def fit_piecewise(p_raw: np.ndarray, label: np.ndarray, n_bins: int = 20,
                  n_grid: int = 256):
    """Legacy bin-then-PAV calibration. Kept for parity with the original
    starter and so we can A/B against isotonic on small corpora."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = np.zeros(n_bins, dtype=np.float64)
    emp     = np.zeros(n_bins, dtype=np.float64)
    counts  = np.zeros(n_bins, dtype=np.int64)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        if i < n_bins - 1:
            m = (p_raw >= lo) & (p_raw < hi)
        else:
            m = (p_raw >= lo) & (p_raw <= hi)
        n = int(m.sum())
        counts[i] = n
        centers[i] = 0.5 * (lo + hi)
        emp[i] = float(label[m].mean()) if n > 0 else np.nan

    keep = ~np.isnan(emp)
    knot_x = centers[keep]
    knot_y = emp[keep]
    knot_w = counts[keep].astype(np.float64)

    if knot_x.size == 0:
        # Degenerate corpus — fall back to the identity map.
        knot_x = np.array([0.0, 1.0])
        knot_y = np.array([0.0, 1.0])
        knot_w = np.array([1.0, 1.0])

    # Anchor endpoints to keep np.interp coverage on [0,1].
    if knot_x[0] > 1e-6:
        knot_x = np.concatenate([[0.0], knot_x])
        knot_y = np.concatenate([[knot_y[0]], knot_y])
        knot_w = np.concatenate([[knot_w[0]], knot_w])
    if knot_x[-1] < 1.0 - 1e-6:
        knot_x = np.concatenate([knot_x, [1.0]])
        knot_y = np.concatenate([knot_y, [knot_y[-1]]])
        knot_w = np.concatenate([knot_w, [knot_w[-1]]])

    # Use sklearn's weighted isotonic fit on the bin centers — same monotone
    # guarantees as the manual PAV loop, but battle-tested.
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip",
                             increasing=True)
    iso.fit(knot_x, knot_y, sample_weight=knot_w)
    return _emit_lookup_table(lambda xs: iso.predict(xs.astype(np.float64)),
                              n_grid=n_grid)


def remap(p_raw: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Apply the saved calibration table. Matches the runtime lookup."""
    return np.interp(p_raw, xs, ys).astype(np.float32)


# ---------------------------------------------------------------------------
# Calibration metrics: Brier score + Expected Calibration Error.
# ---------------------------------------------------------------------------
def brier_score(p: np.ndarray, label: np.ndarray) -> float:
    return float(np.mean((p.astype(np.float64) - label.astype(np.float64)) ** 2))


def expected_calibration_error(p: np.ndarray, label: np.ndarray,
                               n_bins: int = 15) -> float:
    """ECE = sum_b (n_b / N) * |mean(p_b) - mean(label_b)|. Equal-width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    N = len(p)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        n = int(m.sum())
        if n == 0:
            continue
        ece += (n / N) * abs(float(p[m].mean()) - float(label[m].mean()))
    return float(ece)


def reliability_table(p: np.ndarray, label: np.ndarray, n_bins: int = 10):
    """Return a (centers, predicted_mean, empirical_mean, count) tuple
    for printing a small reliability table."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, pred_mean, emp_mean, counts = [], [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        n = int(m.sum())
        if n == 0:
            continue
        centers.append(0.5 * (lo + hi))
        pred_mean.append(float(p[m].mean()))
        emp_mean.append(float(label[m].mean()))
        counts.append(n)
    return (np.asarray(centers), np.asarray(pred_mean),
            np.asarray(emp_mean), np.asarray(counts))


# ---------------------------------------------------------------------------
# Offline state-machine sim — same shape as eval_decoupled_offline.main().
# ---------------------------------------------------------------------------
def simulate(p_TP, mu_TP, mu_FP, matched, gt, groups,
             c_MOTA, c_FP_track, c_FP_frame, match_rate_TP=0.95):
    n_rows = len(p_TP)
    bayes_state = np.full(n_rows, -1, dtype=np.int8)
    is_TP_track = np.zeros(len(groups), dtype=bool)
    promote_k = np.full(len(groups), -1, dtype=np.int32)
    for gi, rows in enumerate(groups):
        is_TP_track[gi] = bool(np.any(gt[rows] != -1))
        state = UNCONFIRMED
        for k_idx, r in enumerate(rows):
            state = cost_decision(
                float(p_TP[r]), float(mu_TP[r]), float(mu_FP[r]),
                state, bool(matched[r]),
                c_MOTA=c_MOTA, c_FP_track=c_FP_track,
                c_FP_frame=c_FP_frame, match_rate_TP=match_rate_TP)
            bayes_state[r] = state
            if promote_k[gi] < 0 and state == TRACKED:
                promote_k[gi] = k_idx
    promoted = promote_k >= 0
    fp_wrong_promote = int(((~is_TP_track) & promoted).sum())
    aligned = (gt != -1)
    is_TRACKED = bayes_state == TRACKED
    n_tp_frames = int((is_TRACKED & aligned).sum())
    n_fp_frames = int((is_TRACKED & ~aligned).sum())
    mota_proxy = n_tp_frames / max(1, int(aligned.sum()))
    fp_per_frame = n_fp_frames / max(1, n_rows)
    # Same shape as eval_decoupled_offline.main(): mota - 5e-4*fp_tracks - 2e-3*fp_per_frame
    fitness = mota_proxy - 0.0005 * fp_wrong_promote - 0.002 * fp_per_frame
    return {
        "promoted_TP": int((is_TP_track & promoted).sum()),
        "promoted_FP": fp_wrong_promote,
        "tpcov": n_tp_frames / max(1, int(aligned.sum())),
        "fpexp": n_fp_frames / max(1, int((~aligned).sum())),
        "mota_proxy": mota_proxy,
        "fp_per_frame": fp_per_frame,
        "fitness_proxy": fitness,
    }


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", required=True)
    p.add_argument("--val",  required=True)
    p.add_argument("--out",  required=True)
    p.add_argument("--method", choices=("isotonic", "platt", "piecewise"),
                   default="isotonic",
                   help="Calibration method. isotonic=sklearn PAV (default), "
                        "platt=2-parameter sigmoid, piecewise=bin-then-PAV.")
    p.add_argument("--n-bins", type=int, default=20,
                   help="Bin count for the piecewise method and the printed reliability tables.")
    p.add_argument("--n-grid", type=int, default=256,
                   help="Grid size of the saved (xs, ys) lookup table.")
    p.add_argument("--c-mota",     type=float, default=1e-3)
    p.add_argument("--c-fp-frame", type=float, default=2e-3)
    p.add_argument("--match-rate-tp", type=float, default=0.95)
    args = p.parse_args()

    # ---- Load head and corpus ---------------------------------------------
    print(f"loading head {args.head}", flush=True)
    ckpt = torch.load(args.head, map_location="cpu", weights_only=False)
    in_dim = int(ckpt["in_dim"]); hidden = int(ckpt["hidden"])
    model = DecoupledGRUHead(in_dim=in_dim, hidden=hidden)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"loading {args.val}", flush=True)
    rec = np.load(args.val, allow_pickle=False)["records"]
    X = build_input_matrix_no_state(rec)
    groups = group_rows_by_track(rec)
    matched = rec["matched"].astype(bool)
    gt = rec["gt_id_now"]
    is_TP_now = (gt != -1).astype(np.float32)
    print(f"  {len(rec)} rows, {len(groups)} tracks, in_dim={in_dim}", flush=True)

    # ---- Forward pass -----------------------------------------------------
    print("forward pass over corpus ...", flush=True)
    p_raw, mu_TP, mu_FP = head_forward_all(model, X, groups, device, in_dim)

    # ---- Fit calibrator ---------------------------------------------------
    print(f"fitting calibrator: method={args.method}", flush=True)
    if args.method == "isotonic":
        xs, ys = fit_isotonic(p_raw, is_TP_now, n_grid=args.n_grid)
    elif args.method == "platt":
        xs, ys = fit_platt(p_raw, is_TP_now, n_grid=args.n_grid)
    else:  # piecewise
        xs, ys = fit_piecewise(p_raw, is_TP_now, n_bins=args.n_bins,
                               n_grid=args.n_grid)
    print(f"  saved-table size: {len(xs)} knots on [{xs[0]:.3f}, {xs[-1]:.3f}]",
          flush=True)

    np.savez(args.out, xs=xs, ys=ys, method=np.array(args.method),
             head_path=np.array(args.head), val_path=np.array(args.val))
    print(f"saved -> {args.out}", flush=True)

    # ---- Apply to val and report calibration metrics ----------------------
    p_cal = remap(p_raw, xs, ys)

    print(f"\np_raw stats: min={p_raw.min():.3f} mean={p_raw.mean():.3f} "
          f"max={p_raw.max():.3f}")
    print(f"p_cal stats: min={p_cal.min():.3f} mean={p_cal.mean():.3f} "
          f"max={p_cal.max():.3f}")
    print(f"label rate (per-frame is_TP_now): {is_TP_now.mean():.3f}")

    brier_raw = brier_score(p_raw, is_TP_now)
    brier_cal = brier_score(p_cal, is_TP_now)
    ece_raw   = expected_calibration_error(p_raw, is_TP_now, n_bins=15)
    ece_cal   = expected_calibration_error(p_cal, is_TP_now, n_bins=15)
    print(f"\n=== Calibration metrics (per-frame is_TP_now label) ===")
    print(f"  {'metric':<15} {'raw':>10} {'cal':>10} {'delta':>10}")
    print(f"  {'Brier score':<15} {brier_raw:10.5f} {brier_cal:10.5f} "
          f"{brier_cal - brier_raw:+10.5f}  (lower is better)")
    print(f"  {'ECE (15 bins)':<15} {ece_raw:10.5f} {ece_cal:10.5f} "
          f"{ece_cal - ece_raw:+10.5f}  (lower is better)")

    # ---- Reliability table (raw vs cal) -----------------------------------
    print(f"\n=== Reliability table (raw, {args.n_bins} equal-width bins) ===")
    print(f"  {'bin_mid':>8}  {'pred_mean':>10}  {'emp_mean':>10}  {'count':>10}")
    c_, pm_, em_, cn_ = reliability_table(p_raw, is_TP_now, n_bins=args.n_bins)
    for ci, pi, ei, ni in zip(c_, pm_, em_, cn_):
        print(f"  {ci:8.3f}  {pi:10.3f}  {ei:10.3f}  {ni:10d}")

    print(f"\n=== Reliability table (calibrated, {args.n_bins} equal-width bins) ===")
    print(f"  {'bin_mid':>8}  {'pred_mean':>10}  {'emp_mean':>10}  {'count':>10}")
    c_, pm_, em_, cn_ = reliability_table(p_cal, is_TP_now, n_bins=args.n_bins)
    for ci, pi, ei, ni in zip(c_, pm_, em_, cn_):
        print(f"  {ci:8.3f}  {pi:10.3f}  {ei:10.3f}  {ni:10d}")

    # ---- c_FP_track sweep, raw vs calibrated ------------------------------
    print("\n=== c_FP_track sweep on offline sim (raw vs calibrated) ===")
    print(f"{'cFP':>7}  {'mode':>5}  {'prom_TP':>7}  {'prom_FP':>7}  "
          f"{'tpcov':>6}  {'fpexp':>6}  {'mota':>6}  {'fitness':>9}")
    grid = [0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.200]
    best_raw = best_cal = None
    for cft in grid:
        out_raw = simulate(p_raw, mu_TP, mu_FP, matched, gt, groups,
                           c_MOTA=args.c_mota, c_FP_track=cft,
                           c_FP_frame=args.c_fp_frame,
                           match_rate_TP=args.match_rate_tp)
        out_cal = simulate(p_cal, mu_TP, mu_FP, matched, gt, groups,
                           c_MOTA=args.c_mota, c_FP_track=cft,
                           c_FP_frame=args.c_fp_frame,
                           match_rate_TP=args.match_rate_tp)
        for tag, o in [("raw", out_raw), ("cal", out_cal)]:
            print(f"{cft:7.4f}  {tag:>5}  {o['promoted_TP']:7d}  "
                  f"{o['promoted_FP']:7d}  {o['tpcov']:6.3f}  "
                  f"{o['fpexp']:6.3f}  {o['mota_proxy']:6.3f}  "
                  f"{o['fitness_proxy']:9.4f}")
        if best_raw is None or out_raw["fitness_proxy"] > best_raw[0]:
            best_raw = (out_raw["fitness_proxy"], cft)
        if best_cal is None or out_cal["fitness_proxy"] > best_cal[0]:
            best_cal = (out_cal["fitness_proxy"], cft)

    print(f"\nbest raw: c_FP_track={best_raw[1]:.4f}  fitness={best_raw[0]:.4f}")
    print(f"best cal: c_FP_track={best_cal[1]:.4f}  fitness={best_cal[0]:.4f}")
    print(f"calibration lift @ argmax: {best_cal[0] - best_raw[0]:+.4f}")


if __name__ == "__main__":
    main()
