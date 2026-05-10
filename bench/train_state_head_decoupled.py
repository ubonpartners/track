"""Decoupled state head: GRU(in_dim=19, hidden=H) + 3 outputs.

  llr_logit  → sigmoid → p_TP   (probability the track is real, not FP)
  mu_tp_log  → expm1   → μ_TP   (expected matched-lifetime if TP, seconds)
  mu_fp_log  → expm1   → μ_FP   (expected exposure-lifetime if FP, seconds)

The NN does not know about runtime states (UNCONFIRMED / TRACKED / LOST),
does not produce promote/demote decisions, has no thresholds baked in.
The decision logic — promote / repromote — lives in the C runtime as a
pure expected-eval-fitness cost rule:

    ΔF(promote)  = p_TP · c_MOTA · μ_TP · match_rate
                  − p_FP · (c_FP_track + c_FP_frame · μ_FP)

Hidden state h₀ = 0 at frame 0 of each track. The runtime carries it
forward frame-to-frame in utdet_t.h_state.

Per-track label: continuous match-fraction (fraction of frames the
track GT-aligns over the prefix ending at its last GT-aligned row).
The cost rule's p_TP becomes "expected match rate" which correctly
down-weights tracks that match only occasionally. Pure FP tracks
(no GT alignment anywhere) get label 0.0.

Usage:
    python -m bench.train_state_head_decoupled \\
        --train bench/data/state_corpus_v17_train.npz \\
        --val   bench/data/state_corpus_v17_val.npz \\
        --save  bench/data/state_head_dc_v15.pt \\
        --epochs 30 --hidden 32
"""
from __future__ import annotations
import argparse
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Feature builder (state-agnostic, 19-dim) ----------------------

# MUST match the C-side utrack_state_build_features in
# ubon_cstuff/src/track/utrack/utrack_state.c. Mismatch is fatal at .bin
# load time (UTRACK_NN_STATE_GRU_IN_DIM = 19).
INPUT_DIM = 19


def build_input_matrix_no_state(rec: np.ndarray) -> np.ndarray:
    """(N, 19) float32 — state-agnostic per-frame feature vector.

    Layout (same as utrack_state.c):
      [ 0]  matched
      [ 1]  log1p(observations)
      [ 2]  log1p(num_missed)
      [ 3]  time_since_det
      [ 4]  log1p(scene_density)
      [ 5]  det_conf
      [ 6]  prev_det_conf
      [ 7]  phase3_pair_score
      [ 8]  near_edge
      [ 9]  det_w
      [10]  det_h
      [11]  log_aspect (clamped [-3, 3] in the C builder; corpus pre-clamps)
      [12]  log_pose_kp
      [13]  ema_match_x_conf
      [14]  log_sum_det_conf
      [15]  min_match_score
      [16]  mean_match_score
      [17]  n_strong_matches
      [18]  log1p(time_since_creation)
    """
    n = len(rec)
    def _f(name):
        return rec[name].astype(np.float32).reshape(-1, 1)
    cols = [
        _f("matched"),
        np.log1p(_f("observations")),
        np.log1p(_f("num_missed")),
        _f("time_since_det"),
        np.log1p(_f("scene_density")),
        _f("det_conf"),
        _f("prev_det_conf"),
        _f("phase3_pair_score"),
        _f("near_edge"),
        _f("det_w"),
        _f("det_h"),
        _f("log_aspect"),
        _f("log_pose_kp"),
        _f("ema_match_x_conf"),
        _f("log_sum_det_conf"),
        _f("min_match_score"),
        _f("mean_match_score"),
        _f("n_strong_matches"),
        np.log1p(_f("time_since_creation")),
    ]
    return np.ascontiguousarray(np.concatenate(cols, axis=1))


# ---------- Per-track grouping --------------------------------------------

def group_rows_by_track(rec: np.ndarray) -> List[np.ndarray]:
    """Return a list of arrays, each containing the row indices (into rec)
    for a single (sequence, track_id), ordered by frame_idx, deduplicated
    by frame_idx. Determinism guaranteed by stable lexsort.
    """
    seq = rec["sequence"].astype(str)
    tid = rec["track_id"]
    fi  = rec["frame_idx"]
    order = np.lexsort((fi, tid, seq))

    groups: List[np.ndarray] = []
    n = len(order)
    i = 0
    while i < n:
        s_i = seq[order[i]]; t_i = tid[order[i]]
        j = i
        while j < n and seq[order[j]] == s_i and tid[order[j]] == t_i:
            j += 1
        block = order[i:j]
        fis = fi[block]
        keep_pos = np.concatenate([[True], fis[1:] != fis[:-1]])
        groups.append(block[keep_pos])
        i = j
    return groups


# ---------- Labels --------------------------------------------------------

def compute_track_labels(rec: np.ndarray):
    """Per-row track-level binary label, broadcast across all rows of a track:
    is_TP=1 iff the track ever GT-aligned (any frame's gt_id_now != -1).

    Also returns per-row remaining_lifetime: seconds until the track's
    last frame in the corpus (used for FP tracks; TP tracks use
    `compute_match_fraction_and_lifetimes`).

    Used by infer_single_track.py for truth display and by
    eval_decoupled_offline.py for per-row TP/FP grouping.
    """
    seq = rec["sequence"].astype(str)
    tid = rec["track_id"]
    fi  = rec["frame_idx"]
    ft  = rec["frame_time"].astype(np.float64)
    gt  = rec["gt_id_now"]
    n = len(rec)

    is_TP = np.zeros(n, dtype=np.float32)
    rem   = np.zeros(n, dtype=np.float32)

    order = np.lexsort((fi, tid, seq))
    i = 0
    while i < n:
        s_i, t_i = seq[order[i]], tid[order[i]]
        j = i
        while j < n and seq[order[j]] == s_i and tid[order[j]] == t_i:
            j += 1
        grp = order[i:j]
        ft_g = ft[grp]
        track_is_TP = float((gt[grp] != -1).any())
        is_TP[grp] = track_is_TP
        last_t = float(ft_g.max())
        rem[grp] = np.maximum(0.0, last_t - ft_g).astype(np.float32)
        i = j
    return is_TP, rem


def compute_match_fraction_and_lifetimes(rec: np.ndarray):
    """Returns (label, rem_TP, rem_FP):
      label  — per-row continuous label = fraction of THIS track's rows
               that GT-align, restricted to the prefix ending at the
               last GT-aligned row. Pure FP tracks get 0.0.
      rem_TP — per-row remaining matched-lifetime: seconds from this row
               to the track's *last GT-aligned* row, clamped at 0.
      rem_FP — per-row exposure-lifetime: seconds to the track's last
               row in any state.

    The trainer uses (label, log1p(rem_TP) for TP-flagged rows,
    log1p(rem_FP) for FP-flagged rows).
    """
    seq = rec["sequence"].astype(str)
    tid = rec["track_id"]
    fi  = rec["frame_idx"]
    ft  = rec["frame_time"].astype(np.float64)
    gt  = rec["gt_id_now"]
    n = len(rec)

    label   = np.zeros(n, dtype=np.float32)
    rem_TP  = np.zeros(n, dtype=np.float32)
    rem_FP  = np.zeros(n, dtype=np.float32)

    order = np.lexsort((fi, tid, seq))
    i = 0
    while i < n:
        s_i, t_i = seq[order[i]], tid[order[i]]
        j = i
        while j < n and seq[order[j]] == s_i and tid[order[j]] == t_i:
            j += 1
        grp = order[i:j]
        ft_g = ft[grp]
        aligned = (gt[grp] != -1)

        if aligned.any():
            # match fraction over the prefix ending at the last aligned row
            last_aligned = int(np.where(aligned)[0].max())
            frac = float(aligned[:last_aligned + 1].mean())
            last_aligned_t = float(ft_g[last_aligned])
            rem_TP[grp] = np.maximum(0.0, last_aligned_t - ft_g).astype(np.float32)
        else:
            frac = 0.0
        label[grp] = frac

        last_t = float(ft_g.max())
        rem_FP[grp] = np.maximum(0.0, last_t - ft_g).astype(np.float32)
        i = j
    return label, rem_TP, rem_FP


# ---------- Model ---------------------------------------------------------

class DecoupledGRUHead(nn.Module):
    """GRU(in=19, hidden=H) + 3 linear heads (llr / mu_tp / mu_fp).

    Hidden state IS the per-track belief.
    """
    def __init__(self, in_dim: int = INPUT_DIM, hidden: int = 32):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden,
                          num_layers=1, batch_first=True)
        self.llr   = nn.Linear(hidden, 1)
        self.mu_tp = nn.Linear(hidden, 1)
        self.mu_fp = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, h0: torch.Tensor = None):
        """x: (B, T, in_dim); h0 optional (1, B, hidden).
        Returns (llr_logit, mu_tp_log, mu_fp_log, h_T) — heads (B, T) each."""
        out, h_T = self.gru(x, h0)
        return (
            self.llr(out).squeeze(-1),
            self.mu_tp(out).squeeze(-1),
            self.mu_fp(out).squeeze(-1),
            h_T,
        )


# ---------- Batch packing -------------------------------------------------

def pack_decoupled_batch(track_groups: List[np.ndarray],
                         picks: List[int],
                         X: np.ndarray,
                         label_row: np.ndarray,
                         logrem_row: np.ndarray,
                         t_max: int,
                         train: bool,
                         rng: np.random.Generator):
    """Pack picked tracks into padded (B, T, ...) tensors.

    label_row    — per-row training label (match-fraction, broadcast).
    logrem_row   — per-row log1p(remaining_lifetime). For TP-flagged rows
                   this is log1p(rem_TP); for FP-flagged it's log1p(rem_FP).
    t_max        — max sequence length in the batch (random crop in train).
    """
    B = len(picks)
    starts: List[int] = []
    lens: List[int] = []
    for gi in picks:
        L = len(track_groups[gi])
        if L <= t_max:
            starts.append(0); lens.append(L)
        else:
            s = int(rng.integers(0, L - t_max + 1)) if train else 0
            starts.append(s); lens.append(t_max)
    T_b = max(lens) if lens else 1

    in_dim = X.shape[1]
    X_b      = np.zeros((B, T_b, in_dim), dtype=np.float32)
    pad      = np.zeros((B, T_b),         dtype=bool)
    label_b  = np.zeros((B, T_b),         dtype=np.float32)
    logrem_b = np.zeros((B, T_b),         dtype=np.float32)

    for bi, (gi, s, L) in enumerate(zip(picks, starts, lens)):
        rows = track_groups[gi][s:s + L]
        X_b[bi, :L]      = X[rows]
        pad[bi, :L]      = True
        label_b[bi, :L]  = label_row[rows]
        logrem_b[bi, :L] = logrem_row[rows]

    return (torch.from_numpy(X_b),
            torch.from_numpy(label_b),
            torch.from_numpy(logrem_b),
            torch.from_numpy(pad))


# ---------- Trainer -------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val",   required=True)
    p.add_argument("--save",  required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--t-max", type=int, default=64,
                   help="max sequence length per batch row (random crop in train)")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--lambda-llr",      type=float, default=1.0)
    p.add_argument("--lambda-lifetime", type=float, default=0.3)
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

    print("computing labels (match-fraction) ...", flush=True)
    label_tr, rem_tp_tr, rem_fp_tr = compute_match_fraction_and_lifetimes(rec_tr)
    label_va, rem_tp_va, rem_fp_va = compute_match_fraction_and_lifetimes(rec_va)
    logrem_tp_tr = np.log1p(rem_tp_tr).astype(np.float32)
    logrem_fp_tr = np.log1p(rem_fp_tr).astype(np.float32)
    logrem_tp_va = np.log1p(rem_tp_va).astype(np.float32)
    logrem_fp_va = np.log1p(rem_fp_va).astype(np.float32)
    # Combined per-row lifetime label: rem_TP when label > 0.5, else rem_FP.
    # The training loss masks one or the other based on label_b > 0.5.
    logrem_tr = np.where(label_tr > 0.5, logrem_tp_tr, logrem_fp_tr).astype(np.float32)
    logrem_va = np.where(label_va > 0.5, logrem_tp_va, logrem_fp_va).astype(np.float32)
    print(f"  train: label mean={label_tr.mean():.3f}  "
          f"rem_TP mean={rem_tp_tr.mean():.2f}s rem_FP mean={rem_fp_tr.mean():.2f}s",
          flush=True)
    print(f"  val:   label mean={label_va.mean():.3f}", flush=True)

    print("grouping by (sequence, track_id) ...", flush=True)
    groups_tr = group_rows_by_track(rec_tr)
    groups_va = group_rows_by_track(rec_va)
    lens_tr = np.array([len(g) for g in groups_tr])
    lens_va = np.array([len(g) for g in groups_va])
    print(f"  train: {len(groups_tr)} tracks, len mean={lens_tr.mean():.1f} "
          f"median={np.median(lens_tr):.0f} max={lens_tr.max()}", flush=True)
    print(f"  val:   {len(groups_va)} tracks, len mean={lens_va.mean():.1f} "
          f"median={np.median(lens_va):.0f} max={lens_va.max()}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecoupledGRUHead(in_dim=in_dim, hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: GRU(in={in_dim}, hidden={args.hidden}) + 3 heads, "
          f"{n_params} params, device={device}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # No class-balance rebalancing: the cost rule downstream uses p_TP as
    # an absolute probability. Rebalancing to 50/50 produces under-estimated
    # p_TP under the true π and the cost rule under-promotes everything.
    pos_weight = torch.tensor([1.0], device=device)

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
            X_b, label_b, logrem_b, pad = pack_decoupled_batch(
                groups_tr, picks, X_tr, label_tr, logrem_tr,
                t_max=args.t_max, train=True, rng=rng)
            X_b      = X_b.to(device, non_blocking=True)
            label_b  = label_b.to(device, non_blocking=True)
            logrem_b = logrem_b.to(device, non_blocking=True)
            pad      = pad.to(device, non_blocking=True)

            llr_l, mtp_l, mfp_l, _ = model(X_b)

            # BCE on llr_logit against the (broadcast) match-fraction label.
            bce_per = F.binary_cross_entropy_with_logits(
                llr_l, label_b, pos_weight=pos_weight, reduction="none")
            llr_loss = (bce_per * pad.float()).sum() / pad.float().sum().clamp(min=1.0)

            # Lifetime: gated MSE.
            tp_mask = pad & (label_b > 0.5)
            fp_mask = pad & (label_b <= 0.5)
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
        all_lab  = []
        all_rem  = []
        with torch.no_grad():
            order_va = np.arange(len(groups_va))
            eval_bs = max(8, bs // 4)
            for i in range(0, len(order_va), eval_bs):
                picks = order_va[i:i+eval_bs].tolist()
                T_picks = max(len(groups_va[gi]) for gi in picks)
                X_b, label_b, logrem_b, pad = pack_decoupled_batch(
                    groups_va, picks, X_va, label_va, logrem_va,
                    t_max=T_picks, train=False, rng=rng)
                X_b = X_b.to(device, non_blocking=True)
                llr_l, mtp_l, mfp_l, _ = model(X_b)

                pad_np = pad.numpy()
                all_llr.append(llr_l.cpu().numpy()[pad_np])
                all_mtp.append(mtp_l.cpu().numpy()[pad_np])
                all_mfp.append(mfp_l.cpu().numpy()[pad_np])
                all_lab.append(label_b.numpy()[pad_np])
                all_rem.append(logrem_b.numpy()[pad_np])
        L = np.concatenate(all_llr)
        Y = np.concatenate(all_lab)
        rem_log = np.concatenate(all_rem)
        mtp_pred = np.expm1(np.clip(np.concatenate(all_mtp), 0, 10))
        mfp_pred = np.expm1(np.clip(np.concatenate(all_mfp), 0, 10))
        rem_true = np.expm1(rem_log)

        from sklearn.metrics import roc_auc_score
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
    from bench._artefact_meta import make_pt_meta
    hparams = {
        "in_dim": in_dim, "hidden": args.hidden, "n_outputs": 3,
        "best_score": float(best_score),
    }
    save = {
        "state_dict": best_state,
        "in_dim": in_dim, "hidden": args.hidden,
        "model_kind": "decoupled_gru_v1",
        "feature_layout": "decoupled-19dim (matched, log_obs..pose, hist[5], log_t_creation)",
        "n_outputs": 3,
        "output_names": ["llr_logit", "mu_tp_log", "mu_fp_log"],
        "_meta": make_pt_meta(
            artefact_kind="state_head_decoupled",
            args=args, hparams=hparams,
            dataset_info={"corpus": getattr(args, "corpus", None)},
        ),
    }
    torch.save(save, args.save)
    print(f"saved {args.save}", flush=True)


if __name__ == "__main__":
    main()
