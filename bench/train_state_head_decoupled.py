"""Decoupled state head: GRU + 3 outputs (p_TP, μ_TP, μ_FP).

Design (per user requirement, 2026-05-09):

  The NN's job is ONE thing: given the track's history so far, predict
    1. P(track is real, not FP)        — sigmoid(llr_logit)
    2. expected remaining lifetime if TP — μ_TP, in seconds
    3. expected remaining lifetime if FP — μ_FP, in seconds

  That is ALL. The NN does not know about runtime states (UNCONFIRMED /
  TRACKED / LOST), does not produce promote / demote decisions, has no
  thresholds baked in. Each output is the unconditional posterior given
  the track's frame sequence.

  The decision logic — promote / demote / drop — lives in the C runtime
  as a pure expected-eval-fitness cost rule:

      ΔF(promote)  = p_TP · c_MOTA · μ_TP · match_rate
                    − p_FP · (c_FP_track + c_FP_frame · μ_FP)

  with all coefficients in yaml. The runtime fires transitions when ΔF
  changes sign, never against a fixed threshold.

  This is the framework the user has asked for repeatedly; previous
  attempts (4-head MLP + threshold baking; 4-head GRU; LLR-only Bayesian
  prototype) all conflated the NN with state-machine framing. This file
  is the clean separation.

Architecture:

  Input: 20-dim per-frame feature vector (no prior_state OH — the GRU's
         hidden state replaces it).
  GRU(hidden=H), single layer, batch_first.
  Three Linear(H → 1) heads:
    llr_logit   → BCE against per-track is_TP (broadcast across frames)
    mu_tp_log   → MSE against log1p(remaining_lifetime), gated to is_TP=1 rows
    mu_fp_log   → MSE against log1p(remaining_lifetime), gated to is_TP=0 rows

  Hidden state h₀ = 0 at frame 0 of each track. The runtime carries
  it forward frame-to-frame in utdet_t.

Single-track inference (the "separately testable" requirement):

  bench/infer_single_track.py loads a checkpoint, takes a (sequence,
  track_id) pair from a corpus, walks it through the GRU statefully,
  and prints the per-frame (p_TP, μ_TP, μ_FP) trajectory.

Usage:
    python -m bench.train_state_head_decoupled \\
        --train bench/data/state_corpus_v13_train.npz \\
        --val   bench/data/state_corpus_v13_val.npz \\
        --save  bench/data/state_head_dc_v1.pt \\
        --epochs 30 --hidden 32 --t-max 64
"""
from __future__ import annotations
import argparse
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bench.train_state_head_gru import (
    build_input_matrix_no_state, group_rows_by_track, pack_batch as _pack_batch_4)
from bench.train_bayesian_head import compute_track_labels


def _compute_match_fraction(rec: np.ndarray) -> np.ndarray:
    """Per-row label = fraction of THIS track's rows that GT-align,
    restricted to the prefix ending at the last GT-aligned row.

    Phase 27 finding: binary track-level is_TP (=any-row-aligns) makes
    the head a binary 'ever matched?' classifier; tracks with low match
    rate (e.g. 27%) sit at p_TP=1.0 and the cost rule's demote rule
    never fires.

    Phase 28 finding: a *whole-track* match-fraction includes the long
    LOST tail at the end of TP tracks (object left the scene), which
    dominates the average — TP tracks end up with median match-frac
    ~0.2-0.3, head saturates at 0.91, the cost rule never reaches
    its promote threshold for brief TP tracks (10 frames, 7 matches
    → p_TP plateaus at 0.48). Fix: truncate to the last matched row
    so TP tracks with clean matching get fractions near 1.0 and only
    mid-track occlusion (real signal) lowers the label. FP tracks
    still get 0.0 (no matched row anywhere). The head's output range
    becomes meaningful for the cost rule's threshold mathematics."""
    seq = rec["sequence"].astype(str)
    tid = rec["track_id"]
    fi  = rec["frame_idx"]
    gt  = rec["gt_id_now"]
    n = len(rec)
    out = np.zeros(n, dtype=np.float32)
    order = np.lexsort((fi, tid, seq))
    i = 0
    while i < n:
        s_i, t_i = seq[order[i]], tid[order[i]]
        j = i
        while j < n and seq[order[j]] == s_i and tid[order[j]] == t_i:
            j += 1
        grp = order[i:j]
        aligned = (gt[grp] != -1)
        # Find last GT-aligned row in this track (in frame order).
        if aligned.any():
            last_aligned = int(np.where(aligned)[0].max())
            # Match fraction over the prefix ending at last GT-aligned row.
            prefix = aligned[:last_aligned + 1]
            frac = float(prefix.mean())  # in (0, 1]
        else:
            # No GT alignment anywhere → pure FP track.
            frac = 0.0
        out[grp] = frac
        i = j
    return out


def compute_per_frame_labels(rec: np.ndarray):
    """Per-row labels for the decoupled head:
      - is_TP_now: 1 if THIS frame's gt_id_now != -1, else 0. The head's
        predicted p_TP at frame k is supervised against the question
        'is this track currently aligned with a real object at frame k?',
        not 'will this track ever align with anything?'. This is what
        makes the head correctly drop p_TP for tracks that have stopped
        being detected — without this, a TP track gets labelled is_TP=1
        even on its 200th unmatched frame and the head learns to ignore
        the time_since_det / num_missed signals.

      - rem_life_TP: seconds until the track's last GT-aligned row from
        the current row's frame_time, clamped at 0 if no future GT
        alignment. Replaces the old track-level remaining_lifetime,
        which double-counted post-loss frames as 'lifetime'.
      - rem_life_FP: max remaining time in any state (FP tracks don't
        align with GT, so 'lifetime' is just exposure).
    """
    seq = rec["sequence"].astype(str)
    tid = rec["track_id"]
    fi  = rec["frame_idx"]
    ft  = rec["frame_time"].astype(np.float64)
    gt  = rec["gt_id_now"]

    n = len(rec)
    is_TP_now    = (gt != -1).astype(np.float32)
    rem_life_TP  = np.zeros(n, dtype=np.float32)
    rem_life_FP  = np.zeros(n, dtype=np.float32)

    order = np.lexsort((fi, tid, seq))
    i = 0
    while i < n:
        s_i, t_i = seq[order[i]], tid[order[i]]
        j = i
        while j < n and seq[order[j]] == s_i and tid[order[j]] == t_i:
            j += 1
        group = order[i:j]
        ft_g  = ft[group]
        # Last frame of THIS track that still aligns with GT.
        gt_aligned = gt[group] != -1
        if gt_aligned.any():
            last_aligned_t = float(ft_g[gt_aligned].max())
            rem_life_TP[group] = np.maximum(0.0, last_aligned_t - ft_g).astype(np.float32)
        # Last frame in any state — FP exposure window.
        last_t = float(ft_g.max())
        rem_life_FP[group] = np.maximum(0.0, last_t - ft_g).astype(np.float32)
        i = j

    return is_TP_now, rem_life_TP, rem_life_FP


class DecoupledGRUHead(nn.Module):
    """GRU + 3 output heads. Hidden state IS the per-track belief."""
    def __init__(self, in_dim: int = 20, hidden: int = 32):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden,
                          num_layers=1, batch_first=True)
        self.llr     = nn.Linear(hidden, 1)
        self.mu_tp   = nn.Linear(hidden, 1)
        self.mu_fp   = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, h0: torch.Tensor = None):
        """x: (B, T, in_dim); h0 optional (1, B, hidden).
        Returns llr_logit, mu_tp_log, mu_fp_log each (B, T), and final h (1, B, hidden)."""
        out, h_T = self.gru(x, h0)  # (B, T, hidden)
        return (
            self.llr(out).squeeze(-1),
            self.mu_tp(out).squeeze(-1),
            self.mu_fp(out).squeeze(-1),
            h_T,
        )


def pack_decoupled_batch(track_groups: List[np.ndarray],
                         picks: List[int],
                         X: np.ndarray,
                         is_TP_row: np.ndarray,
                         logrem_row: np.ndarray,
                         t_max: int,
                         train: bool,
                         rng: np.random.Generator):
    """Pack a list of (track-group) indices into padded (B, T, ...) tensors.

    Like train_state_head_gru.pack_batch but emits the labels we need:
    is_TP_per_row (broadcast track-level) and log1p(remaining_lifetime)
    per row. Mask out padded positions so loss isn't computed there.
    """
    B = len(picks)
    starts: List[int] = []
    lens: List[int] = []
    for gi in picks:
        rows = track_groups[gi]
        L = len(rows)
        if L <= t_max:
            starts.append(0); lens.append(L)
        else:
            s = int(rng.integers(0, L - t_max + 1)) if train else 0
            starts.append(s); lens.append(t_max)
    T_b = max(lens) if lens else 1

    in_dim = X.shape[1]
    X_b      = np.zeros((B, T_b, in_dim), dtype=np.float32)
    pad      = np.zeros((B, T_b),         dtype=bool)
    is_TP_b  = np.zeros((B, T_b),         dtype=np.float32)
    logrem_b = np.zeros((B, T_b),         dtype=np.float32)

    for bi, (gi, s, L) in enumerate(zip(picks, starts, lens)):
        rows = track_groups[gi][s:s + L]
        X_b[bi, :L]      = X[rows]
        pad[bi, :L]      = True
        is_TP_b[bi, :L]  = is_TP_row[rows]
        logrem_b[bi, :L] = logrem_row[rows]

    return (torch.from_numpy(X_b),
            torch.from_numpy(is_TP_b),
            torch.from_numpy(logrem_b),
            torch.from_numpy(pad))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val",   required=True)
    p.add_argument("--save",  default="bench/data/state_head_dc_v1.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--t-max", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--lambda-llr",      type=float, default=1.0,
                   help="weight on BCE(llr_logit, is_TP)")
    p.add_argument("--lambda-lifetime", type=float, default=0.3,
                   help="weight on lifetime MSE (sum of TP and FP gated MSEs)")
    p.add_argument("--rebalance",       action="store_true",
                   help="apply class-balance pos_weight=(1-pi)/pi. Without "
                        "this flag (the default), pos_weight=1 and the head "
                        "is trained to predict P(TP | X) under the training "
                        "distribution's natural class prior. The cost rule "
                        "downstream uses p_TP as an absolute probability, so "
                        "calibration to the actual prior is what we want — "
                        "rebalancing to 50/50 produces under-estimated p_TP.")
    p.add_argument("--label-mode",
                   choices=["per-frame", "track-level", "match-fraction"],
                   default="per-frame",
                   help="per-frame: is_TP_now per row (gt_id_now != -1). "
                        "track-level: 1 broadcast across all rows of a track "
                        "if it ever matched GT. ⚠ Phase 27 finding: this "
                        "makes the head a binary 'ever matched?' classifier — "
                        "tracks with 27%% match rate get p_TP=1.0. "
                        "match-fraction: continuous label = fraction of frames "
                        "the track is GT-aligned. The cost rule's p_TP becomes "
                        "'expected match rate' which correctly down-weights "
                        "tracks that match only occasionally.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"loading {args.train}", flush=True)
    rec_tr = np.load(args.train, allow_pickle=False)["records"]
    print(f"loading {args.val}",   flush=True)
    rec_va = np.load(args.val,   allow_pickle=False)["records"]

    X_tr = build_input_matrix_no_state(rec_tr)
    X_va = build_input_matrix_no_state(rec_va)
    in_dim = X_tr.shape[1]
    print(f"train rows: {len(rec_tr)}  val rows: {len(rec_va)}  in_dim={in_dim}", flush=True)

    print(f"computing labels (mode={args.label_mode}) ...", flush=True)
    is_TP_tr_pf, rem_tp_tr, rem_fp_tr = compute_per_frame_labels(rec_tr)
    is_TP_va_pf, rem_tp_va, rem_fp_va = compute_per_frame_labels(rec_va)
    if args.label_mode == "per-frame":
        is_TP_tr = is_TP_tr_pf
        is_TP_va = is_TP_va_pf
    elif args.label_mode == "track-level":
        # Track-level: broadcast is_TP across all rows of a track. Use the
        # legacy compute_track_labels (returns is_TP per row, broadcast).
        is_TP_tr, _ = compute_track_labels(rec_tr)
        is_TP_va, _ = compute_track_labels(rec_va)
    else:  # match-fraction
        # Continuous label: per-track fraction of frames matched to GT.
        # Broadcast across rows of the track. Trains the head to predict
        # 'expected match rate' instead of binary TP/FP, so a track with
        # 27% match rate gets label=0.27 (not 1.0). The cost rule then
        # correctly hesitates on it.
        is_TP_tr = _compute_match_fraction(rec_tr)
        is_TP_va = _compute_match_fraction(rec_va)
    logrem_tp_tr = np.log1p(rem_tp_tr).astype(np.float32)
    logrem_fp_tr = np.log1p(rem_fp_tr).astype(np.float32)
    logrem_tp_va = np.log1p(rem_tp_va).astype(np.float32)
    logrem_fp_va = np.log1p(rem_fp_va).astype(np.float32)
    # Combined per-row lifetime label: rem_TP when is_TP_now=1, else rem_FP.
    # The training loss masks one or the other — see the pack function.
    logrem_tr = np.where(is_TP_tr > 0.5, logrem_tp_tr, logrem_fp_tr).astype(np.float32)
    logrem_va = np.where(is_TP_va > 0.5, logrem_tp_va, logrem_fp_va).astype(np.float32)
    rem_tr, rem_va = rem_tp_tr, rem_tp_va
    print(f"  train: per-frame TP rate = {is_TP_tr.mean():.3f}  "
          f"rem_TP mean={rem_tp_tr.mean():.2f}s max={rem_tp_tr.max():.2f}s "
          f"rem_FP mean={rem_fp_tr.mean():.2f}s max={rem_fp_tr.max():.2f}s",
          flush=True)
    print(f"  val:   per-frame TP rate = {is_TP_va.mean():.3f}", flush=True)

    print("grouping by (sequence, track_id) ...", flush=True)
    groups_tr = group_rows_by_track(rec_tr)
    groups_va = group_rows_by_track(rec_va)
    lens_tr = np.array([len(g) for g in groups_tr])
    lens_va = np.array([len(g) for g in groups_va])
    print(f"  train: {len(groups_tr)} tracks, len mean={lens_tr.mean():.1f} "
          f"median={np.median(lens_tr):.0f} max={lens_tr.max()}", flush=True)
    print(f"  val:   {len(groups_va)} tracks, len mean={lens_va.mean():.1f} "
          f"median={np.median(lens_va):.0f} max={lens_va.max()}", flush=True)

    # Class-balance setup. Default: pos_weight=1 so the head is calibrated
    # to the training distribution's natural prior (P(TP|X) under π=99%).
    # The cost rule downstream uses p_TP as an absolute probability — if
    # we rebalanced loss to 50/50 the head would output P(TP|X, π=0.5),
    # which when used absolutely says creation-frame TPs have P(TP)≈0.3
    # and the cost rule under-promotes everything.
    n_pos = float((is_TP_tr > 0.5).sum())
    n_neg = float((is_TP_tr <= 0.5).sum())
    if args.rebalance:
        pos_weight_val = n_neg / max(1.0, n_pos)
    else:
        pos_weight_val = 1.0
    print(f"is_TP class balance: pos={int(n_pos)} neg={int(n_neg)} → "
          f"pos_weight={pos_weight_val:.4f} (rebalance={args.rebalance})",
          flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecoupledGRUHead(in_dim=in_dim, hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: GRU(in={in_dim}, hidden={args.hidden}) + 3 heads, "
          f"{n_params} params, device={device}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    pos_weight = torch.tensor([pos_weight_val], device=device)

    best_score = -float("inf")
    best_state = None
    bs = args.batch_size

    for ep in range(args.epochs):
        # ---- training -----------------------------------------------------
        model.train()
        perm = rng.permutation(len(groups_tr))
        ep_llr  = 0.0
        ep_life = 0.0
        n_b = 0
        for i in range(0, len(perm), bs):
            picks = perm[i:i+bs].tolist()
            X_b, isTP_b, logrem_b, pad = pack_decoupled_batch(
                groups_tr, picks, X_tr, is_TP_tr, logrem_tr,
                t_max=args.t_max, train=True, rng=rng)
            X_b      = X_b.to(device, non_blocking=True)
            isTP_b   = isTP_b.to(device, non_blocking=True)
            logrem_b = logrem_b.to(device, non_blocking=True)
            pad      = pad.to(device, non_blocking=True)

            llr_l, mtp_l, mfp_l, _ = model(X_b)

            # BCE on llr_logit: per-frame label is the (broadcast) track-
            # level is_TP. Mask out pads.
            bce_per = F.binary_cross_entropy_with_logits(
                llr_l, isTP_b, pos_weight=pos_weight, reduction="none")
            llr_loss = (bce_per * pad.float()).sum() / pad.float().sum().clamp(min=1.0)

            # Lifetime: gated MSE. The μ_TP target is "remaining matched-
            # seconds from this row"; for tail rows past the last GT-aligned
            # frame the target is 0, which conflicts with mid-track-occlusion
            # rows where time_since_det looks similar but real μ_TP > 0. Mask
            # out tail rows from the μ_TP loss (logrem_b == 0 for TP tracks
            # iff we're at-or-past last_aligned_t) so the head only learns
            # μ_TP from rows where it's a meaningful positive target.
            tp_mask = pad & (isTP_b > 0.5) & (logrem_b > 0)
            fp_mask = pad & (isTP_b <= 0.5)
            life_loss = X_b.new_zeros(())
            if tp_mask.any():
                d = (mtp_l - logrem_b) * tp_mask.float()
                life_loss = life_loss + (d * d).sum() / tp_mask.float().sum().clamp(min=1.0)
            if fp_mask.any():
                d = (mfp_l - logrem_b) * fp_mask.float()
                life_loss = life_loss + (d * d).sum() / fp_mask.float().sum().clamp(min=1.0)

            loss = args.lambda_llr * llr_loss + args.lambda_lifetime * life_loss
            opt.zero_grad(); loss.backward(); opt.step()
            ep_llr  += float(llr_loss.item())
            ep_life += float(life_loss.item())
            n_b += 1

        # ---- validation: full-track walk, AUC + lifetime MAE -------------
        model.eval()
        all_llr  = []
        all_mtp  = []
        all_mfp  = []
        all_isTP = []
        all_rem  = []
        with torch.no_grad():
            order_va = np.arange(len(groups_va))
            eval_bs = max(8, bs // 4)
            for i in range(0, len(order_va), eval_bs):
                picks = order_va[i:i+eval_bs].tolist()
                T_picks = max(len(groups_va[gi]) for gi in picks)
                X_b, isTP_b, logrem_b, pad = pack_decoupled_batch(
                    groups_va, picks, X_va, is_TP_va, logrem_va,
                    t_max=T_picks, train=False, rng=rng)
                X_b = X_b.to(device, non_blocking=True)
                llr_l, mtp_l, mfp_l, _ = model(X_b)

                pad_np = pad.numpy()
                all_llr.append(llr_l.cpu().numpy()[pad_np])
                all_mtp.append(mtp_l.cpu().numpy()[pad_np])
                all_mfp.append(mfp_l.cpu().numpy()[pad_np])
                all_isTP.append(isTP_b.numpy()[pad_np])
                all_rem.append(logrem_b.numpy()[pad_np])
        L = np.concatenate(all_llr)
        Y = np.concatenate(all_isTP)
        rem_log = np.concatenate(all_rem)
        mtp_pred = np.expm1(np.clip(np.concatenate(all_mtp), 0, 10))
        mfp_pred = np.expm1(np.clip(np.concatenate(all_mfp), 0, 10))
        rem_true = np.expm1(rem_log)

        from sklearn.metrics import roc_auc_score
        # AUC needs binary labels; threshold continuous labels at 0.5
        # so match-fraction targets above 0.5 count as positives. Same
        # answer for binary 0/1 labels.
        Y_bin = (Y > 0.5).astype(np.int32)
        auc = (float(roc_auc_score(Y_bin, L))
               if (Y_bin.min() != Y_bin.max()) else float("nan"))
        tp_msk = Y > 0.5
        fp_msk = ~tp_msk
        mae_tp = float(np.mean(np.abs(mtp_pred[tp_msk] - rem_true[tp_msk]))) if tp_msk.any() else float("nan")
        mae_fp = float(np.mean(np.abs(mfp_pred[fp_msk] - rem_true[fp_msk]))) if fp_msk.any() else float("nan")

        # Composite: weight AUC most, lifetime errors lightly.
        score = auc - 0.05 * (mae_tp + mae_fp)
        print(f"ep {ep:3d}  L_llr={ep_llr/max(1,n_b):.4f} "
              f"L_life={ep_life/max(1,n_b):.4f}  "
              f"AUC={auc:.4f}  MAE_TP={mae_tp:.2f}s MAE_FP={mae_fp:.2f}s  "
              f"score={score:.4f}", flush=True)
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nbest score = {best_score:.4f}", flush=True)
    save = {
        "state_dict": best_state,
        "in_dim": in_dim, "hidden": args.hidden,
        "model_kind": "decoupled_gru_v1",
        "feature_layout": "build_input_matrix-23dim minus prior_state OH (=20dim)",
        "n_outputs": 3,
        "output_names": ["llr_logit", "mu_tp_log", "mu_fp_log"],
    }
    torch.save(save, args.save)
    print(f"saved {args.save}", flush=True)


if __name__ == "__main__":
    main()
