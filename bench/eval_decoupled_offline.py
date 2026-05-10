"""Offline closed-loop sim for the decoupled head — exact mirror of
what the C runtime does, in Python, so we can measure the head's
behaviour without touching the C tracker.

The runtime's decision is purely cost-based (no thresholds):

    ΔF(promote) = p_TP · c_MOTA · μ_TP · match_rate
                 − p_FP · (c_FP_track + c_FP_frame · μ_FP)

    ΔF(demote)  = p_FP · c_FP_frame · μ_FP
                 − p_TP · c_MOTA · μ_TP · match_rate

A transition fires when ΔF for that move is > 0. State codes:
0=UNCONFIRMED, 1=TRACKED, 2=LOST.

We also track the runtime-relevant tallies the user asked for:

  TP-tracks correctly promoted   = TP tracks that ever entered TRACKED
  TP-tracks NEVER promoted       = false negatives (recall failure)
  FP-tracks incorrectly promoted = TP-rule misfired on a non-real track
  FP-tracks correctly suppressed = stayed UNCONFIRMED forever

  Promotion latency (TP only)    = first k_idx where state became TRACKED
  TP coverage                    = fraction of TP track frames spent TRACKED
  FP exposure                    = fraction of FP track frames spent TRACKED

Usage:
    python -m bench.eval_decoupled_offline \\
        --head bench/data/state_head_dc_v1.pt \\
        --val  bench/data/state_corpus_v13_val.npz
"""
from __future__ import annotations
import argparse, json
import numpy as np
import torch

from bench.train_state_head_decoupled import (
    build_input_matrix_no_state, group_rows_by_track,
    compute_track_labels, DecoupledGRUHead)


UNCONFIRMED, TRACKED, LOST = 0, 1, 2


def cost_decision(p_TP: float, mu_TP: float, mu_FP: float,
                  current_state: int, matched: bool,
                  c_MOTA: float, c_FP_track: float, c_FP_frame: float,
                  match_rate_TP: float) -> int:
    """Return desired next state. Pure expected-value cost rule.

    Mirrors the C runtime in `utrack.c` (BAYESIAN mode, around line 1280):
      - delta_promote (UNCONFIRMED→TRACKED) charges c_FP_track for entering.
      - delta_repromote (LOST→TRACKED) does NOT charge c_FP_track again
        — the track already paid on first entry. Symmetric inverse of
        delta_demote.
      - delta_demote is unchanged.

    Earlier versions of this function used delta_promote for both
    transitions, which double-charged c_FP_track on demote→re-promote
    cycles and inflated the with-demote baseline in offline experiments.
    Fixed 2026-05-09."""
    p_FP = 1.0 - p_TP
    gain_promote = p_TP * c_MOTA * mu_TP * match_rate_TP
    loss_promote_unc = p_FP * (c_FP_track + c_FP_frame * mu_FP)
    delta_promote   = gain_promote - loss_promote_unc
    delta_repromote = gain_promote - p_FP * c_FP_frame * mu_FP
    delta_demote    = p_FP * c_FP_frame * mu_FP - gain_promote

    if current_state == UNCONFIRMED:
        if matched and delta_promote > 0:
            return TRACKED
        return UNCONFIRMED
    elif current_state == TRACKED:
        if not matched and delta_demote > 0:
            return LOST
        return TRACKED
    elif current_state == LOST:
        if matched and delta_repromote > 0:
            return TRACKED
        return LOST
    return current_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", required=True)
    p.add_argument("--val",  required=True)
    p.add_argument("--c-mota",      type=float, default=1e-3)
    p.add_argument("--c-fp-track",  type=float, default=5e-4)
    p.add_argument("--c-fp-frame",  type=float, default=2e-3)
    p.add_argument("--match-rate-tp", type=float, default=0.95)
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    print(f"loading head from {args.head}", flush=True)
    ckpt = torch.load(args.head, map_location="cpu", weights_only=False)
    in_dim = int(ckpt["in_dim"]); hidden = int(ckpt["hidden"])
    model = DecoupledGRUHead(in_dim=in_dim, hidden=hidden)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"loading {args.val}", flush=True)
    rec = np.load(args.val, allow_pickle=False)["records"]
    # Auto-detect from in_dim whether the head expects scene-aggregate features.
    expect_scene = (in_dim == 25)
    X = build_input_matrix_no_state(rec, with_scene=expect_scene)
    if X.shape[1] != in_dim:
        raise SystemExit(
            f"feature width mismatch: head wants {in_dim}, builder produced "
            f"{X.shape[1]} (with_scene={expect_scene})"
        )
    is_TP_row, _ = compute_track_labels(rec)
    matched = rec["matched"].astype(bool)
    gt = rec["gt_id_now"]

    groups = group_rows_by_track(rec)
    n_tracks = len(groups)
    print(f"  {n_tracks} tracks, val rows: {len(rec)}", flush=True)

    # Forward pass batched over tracks of similar length.
    print("forward pass + state-machine simulation ...", flush=True)
    bayes_state = np.full(len(rec), -1, dtype=np.int8)
    p_TP_per_row = np.full(len(rec), np.nan, dtype=np.float32)
    promote_k = np.full(n_tracks, -1, dtype=np.int32)
    is_TP_track = np.zeros(n_tracks, dtype=bool)

    order = sorted(range(n_tracks), key=lambda i: len(groups[i]))
    B = 64
    with torch.no_grad():
        for i in range(0, n_tracks, B):
            picks = order[i:i+B]
            T_max = max(len(groups[g]) for g in picks)
            Bp = len(picks)
            X_b = np.zeros((Bp, T_max, in_dim), dtype=np.float32)
            for bi, g in enumerate(picks):
                rows = groups[g]
                X_b[bi, :len(rows)] = X[rows]
            Xt = torch.from_numpy(X_b).to(device)
            llr_l, mtp_l, mfp_l, _ = model(Xt)
            p_TP_b = torch.sigmoid(llr_l).cpu().numpy()
            mu_TP_b = np.expm1(np.clip(mtp_l.cpu().numpy(), 0, 10))
            mu_FP_b = np.expm1(np.clip(mfp_l.cpu().numpy(), 0, 10))

            for bi, g in enumerate(picks):
                rows = groups[g]
                L = len(rows)
                is_TP_track[g] = bool(np.any(gt[rows] != -1))

                state = UNCONFIRMED
                for k_idx in range(L):
                    r = rows[k_idx]
                    p_TP_per_row[r] = p_TP_b[bi, k_idx]
                    state = cost_decision(
                        float(p_TP_b[bi, k_idx]),
                        float(mu_TP_b[bi, k_idx]),
                        float(mu_FP_b[bi, k_idx]),
                        state,
                        bool(matched[r]),
                        c_MOTA=args.c_mota,
                        c_FP_track=args.c_fp_track,
                        c_FP_frame=args.c_fp_frame,
                        match_rate_TP=args.match_rate_tp,
                    )
                    bayes_state[r] = state
                    if promote_k[g] < 0 and state == TRACKED:
                        promote_k[g] = k_idx

    # ==================================================================
    # Confusion matrix at the *track-level* promote decision.
    # ==================================================================
    promoted = promote_k >= 0
    tp_correct_promote   = int(((is_TP_track) & ( promoted)).sum())  # TP and promoted
    tp_missed_promote    = int(((is_TP_track) & (~promoted)).sum())  # TP, never promoted
    fp_wrong_promote     = int(((~is_TP_track) & ( promoted)).sum()) # FP, promoted
    fp_correct_suppress  = int(((~is_TP_track) & (~promoted)).sum())

    print(f"\n=== Track-level promote confusion matrix ===")
    print(f"  TP tracks total     : {is_TP_track.sum()}")
    print(f"  FP tracks total     : {(~is_TP_track).sum()}")
    print(f"  TP correctly promoted   : {tp_correct_promote:6d}  "
          f"({tp_correct_promote / max(1, int(is_TP_track.sum())):6.2%})")
    print(f"  TP missed (no promote)  : {tp_missed_promote:6d}  "
          f"({tp_missed_promote / max(1, int(is_TP_track.sum())):6.2%})")
    print(f"  FP wrongly promoted     : {fp_wrong_promote:6d}  "
          f"({fp_wrong_promote / max(1, int((~is_TP_track).sum())):6.2%})")
    print(f"  FP correctly suppressed : {fp_correct_suppress:6d}  "
          f"({fp_correct_suppress / max(1, int((~is_TP_track).sum())):6.2%})")

    # Promotion latency on TP tracks that did promote.
    pk_tp = promote_k[is_TP_track & promoted]
    print(f"\n=== Promotion latency (TP-promoted only) ===")
    if len(pk_tp):
        print(f"  median k = {np.median(pk_tp):.0f}, mean = {pk_tp.mean():.1f}, "
              f"p90 = {np.percentile(pk_tp, 90):.0f}, max = {pk_tp.max()}")
    else:
        print(f"  (no TP tracks ever promoted)")

    # Coverage / exposure: fraction of frames spent TRACKED.
    n_tp_rows = sum(len(groups[g]) for g in range(n_tracks) if is_TP_track[g])
    n_fp_rows = sum(len(groups[g]) for g in range(n_tracks) if not is_TP_track[g])
    n_tp_tracked = int(((bayes_state == TRACKED) & (is_TP_row > 0.5)).sum())
    n_fp_tracked = int(((bayes_state == TRACKED) & (is_TP_row <= 0.5)).sum())
    print(f"\n=== Frame-level exposure ===")
    print(f"  TP rows total : {n_tp_rows:7d}  TRACKED : {n_tp_tracked:7d}  "
          f"({n_tp_tracked / max(1, n_tp_rows):6.2%})  ← TP coverage")
    print(f"  FP rows total : {n_fp_rows:7d}  TRACKED : {n_fp_tracked:7d}  "
          f"({n_fp_tracked / max(1, n_fp_rows):6.2%})  ← FP exposure")

    # Cost-rule fitness proxy. Same shape as eval_bayesian_head_offline.
    aligned = (gt != -1)
    is_TRACKED = bayes_state == TRACKED
    n_gt_aligned = int(aligned.sum())
    n_tp_frames  = int((is_TRACKED & aligned).sum())
    n_fp_frames  = int((is_TRACKED & ~aligned).sum())
    mota_proxy   = n_tp_frames / max(1, n_gt_aligned)
    fp_per_frame = n_fp_frames / max(1, len(rec))
    fitness_proxy = mota_proxy - 0.0005 * fp_wrong_promote - 0.002 * fp_per_frame

    print(f"\n=== Synthetic-fitness proxy ===")
    print(f"  mota_proxy    : {mota_proxy:.4f}  (TP-tracked frames / GT-aligned frames)")
    print(f"  fp_per_frame  : {fp_per_frame:.4f}")
    print(f"  fitness_proxy : {fitness_proxy:.4f}  "
          f"(= mota - 5e-4*fp_tracks - 2e-3*fp_per_frame)")

    if args.out_json:
        json.dump({
            "n_tracks": n_tracks,
            "tp_correct_promote":  tp_correct_promote,
            "tp_missed_promote":   tp_missed_promote,
            "fp_wrong_promote":    fp_wrong_promote,
            "fp_correct_suppress": fp_correct_suppress,
            "tp_coverage":  n_tp_tracked / max(1, n_tp_rows),
            "fp_exposure":  n_fp_tracked / max(1, n_fp_rows),
            "promote_latency_median": float(np.median(pk_tp)) if len(pk_tp) else None,
            "mota_proxy": mota_proxy,
            "fp_per_frame": fp_per_frame,
            "fitness_proxy": fitness_proxy,
        }, open(args.out_json, "w"), indent=2)
        print(f"  saved {args.out_json}")


if __name__ == "__main__":
    main()
