"""GRU state-head trainer — drops `prior_state` from input.

Motivation (Phase 22 finding): the MLP head trained with the 3-OH
`prior_state` input took the trivial shortcut of using state as a
near-label (corpus replay had `prior_state==1` correlated almost
perfectly with `is_TP`). At deployment every track starts at
`prior_state=0` and never receives positive LLR. The cost-rule
Bayesian framework collapses for that reason — not because per-frame
LLR is the wrong abstraction, but because the head was never asked to
read history features.

This trainer takes the opposite design choice:

  * Drop `prior_state` (3 dims) from the input → in_dim 23 → 20.
  * Replace MLP backbone with a GRU. The hidden state IS the track's
    learned belief — analogue of accumulated log-odds + maturity, but
    trained end-to-end against deployment objectives instead of
    hand-engineered.
  * Train per-track BPTT on padded sequences (T_max default 64; longer
    tracks are sampled with random contiguous windows).
  * Same 4-head output as v41 so the .bin slot layout is preserved
    (export script will need a GRU-aware variant — that's Phase B).

Loss machinery (cost-weighted BCE per head, per-row valid masks)
matches `train_state_head.py` so offline AUCs are directly comparable.

Usage:
    python -m bench.train_state_head_gru \
        --train bench/data/state_corpus_v13_train.npz \
        --val   bench/data/state_corpus_v13_val.npz \
        --save  bench/data/state_head_gru_v1.pt \
        --epochs 30 --hidden 32 --t-max 64
"""
from __future__ import annotations
import argparse, json, os, sys
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bench.train_state_head import build_input_matrix, per_head_auc


def build_input_matrix_no_state(rec: np.ndarray,
                                 *, with_scene: bool = False) -> np.ndarray:
    """State-agnostic feature matrix for the decoupled head.

    Drops ALL state-derived features. The decoupled NN's whole purpose
    is to produce (p_TP, μ_TP, μ_FP) from track history alone — it must
    not see the runtime/corpus state machine, or its predictions become
    policy-dependent and break under deployment-time covariate shift
    (Phase 25 finding: head trained on a corpus that promotes at obs=2
    sees no UNCONFIRMED+obs=10 data, so when the cost rule defers
    promotion it queries the head OOD).

    Drops:
      • prior_state one-hot[3] — the runtime's current state
      • time_in_state (last col of the 23-dim layout) — encodes when
        the corpus's replay policy last transitioned the track. State-
        equivalent under a different name; deployment with a different
        policy produces different time_in_state distributions for the
        same underlying track features.

    Kept (all state-INVARIANT functions of detection history):
      matched, log1p(observations), log1p(num_missed), time_since_det,
      log1p(scene_density), det_conf, prev_det_conf, pair_score,
      near_edge, det_w, det_h, log_aspect, log_pose_kp,
      ema_match_x_conf, log_sum_det_conf, min/mean_match_score,
      n_strong_matches, log1p(time_since_creation).

    Result: (N, 19) float32 by default. When `with_scene=True`, appends
    Phase 29's 6 per-scene features at indices 19..24:
      scene_promote_rate, scene_mean_det_conf_TRACKED,
      scene_mean_det_conf_unmatched, log1p(scene_track_density_smooth),
      log1p(scene_mean_alive_track_age), det_conf_minus_scene_TP_avg
    Result: (N, 25) float32. Layout matches the C-side's
    UTRACK_NN_STATE_GRU_IN_DIM_V2 (=25) feature builder.
    """
    X = build_input_matrix(rec)  # (N, 23): [3 OH | 1 matched | 7 num | 5 spatial | 5 hist | 2 age]
    # Keep [3..21]: drop the prior_state OH (cols 0..2) AND drop the
    # trailing time_in_state column (col 22).
    base = X[:, 3:22]
    if not with_scene:
        return np.ascontiguousarray(base)

    # Phase 29 scene-stat features. Bootstrap defaults applied if the
    # corpus dtype lacks them (older .npz files):
    n = len(rec)
    def _maybe(name, default):
        if name in rec.dtype.names:
            return rec[name].astype(np.float32).reshape(-1, 1)
        return np.full((n, 1), default, dtype=np.float32)
    scene_cols = [
        _maybe("scene_promote_rate", 0.5),
        _maybe("scene_mean_det_conf_TRACKED", 0.7),
        _maybe("scene_mean_det_conf_unmatched", 0.3),
        np.log1p(_maybe("scene_track_density_smooth", 5.0)),
        np.log1p(_maybe("scene_mean_alive_track_age", 5.0)),
        _maybe("det_conf_minus_scene_TP_avg", 0.0),
    ]
    return np.ascontiguousarray(np.concatenate([base] + scene_cols, axis=1))


def group_rows_by_track(rec: np.ndarray) -> List[np.ndarray]:
    """Return a list of arrays, each containing the row indices (into rec)
    for a single (sequence, track_id), ordered by frame_idx, deduplicated
    by frame_idx (corpus may emit multiple state-pass rows per frame).

    Dedup keeps the first row encountered for each frame_idx; sort by
    frame_idx is stable so this is deterministic given the input order.
    """
    seq = rec["sequence"].astype(str)
    tid = rec["track_id"]
    fi  = rec["frame_idx"]

    # lexsort: primary key sequence, secondary track_id, tertiary frame_idx
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
        # Dedup by frame_idx (block is already sorted by frame_idx).
        fis = fi[block]
        keep_pos = np.concatenate([[True], fis[1:] != fis[:-1]])
        groups.append(block[keep_pos])
        i = j
    return groups


class GRUStateHead(nn.Module):
    """Single-layer GRU + 4-head linear projection.

    Hidden state per track is the GRU's running hidden state. New tracks
    start with h=0. Inference unrolls one step per frame; training
    unrolls full sequences via BPTT.

    Output heads match v41 layout (promote, demote, drop_unconfirmed,
    drop_lost). When the trainer is run with --no-drop-heads the latter
    two heads still exist in the parameter tensor (and in the .bin) for
    slot compatibility but receive no gradient.
    """
    def __init__(self, in_dim: int = 20, hidden: int = 32):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden,
                          num_layers=1, batch_first=True)
        self.promote          = nn.Linear(hidden, 1)
        self.demote           = nn.Linear(hidden, 1)
        self.drop_unconfirmed = nn.Linear(hidden, 1)
        self.drop_lost        = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, h0: torch.Tensor = None):
        """x: (B, T, in_dim); h0: (1, B, hidden) or None.
        Returns 4 logits each (B, T) and final hidden (1, B, hidden)."""
        out, h_T = self.gru(x, h0)  # out: (B, T, hidden)
        return (
            self.promote(out).squeeze(-1),
            self.demote(out).squeeze(-1),
            self.drop_unconfirmed(out).squeeze(-1),
            self.drop_lost(out).squeeze(-1),
            h_T,
        )


def pack_batch(track_groups: List[np.ndarray],
               picks: List[int],
               X: np.ndarray,
               labels: Dict[str, np.ndarray],
               valids: Dict[str, np.ndarray],
               t_max: int,
               train: bool,
               rng: np.random.Generator) -> Tuple[
                   torch.Tensor,
                   Dict[str, torch.Tensor],
                   Dict[str, torch.Tensor],
                   torch.Tensor,
               ]:
    """Pack a list of track-group indices into padded (B, T, ...) tensors.

    During training (train=True), tracks longer than t_max are sampled
    at a random contiguous window of length t_max. During eval, tracks
    are truncated to t_max (head/tail walk done outside this function).

    Returns:
      X_b:     (B, T_b, in_dim)   padded inputs
      lbl_b:   {head: (B, T_b)}   padded labels
      msk_b:   {head: (B, T_b)}   padded valid masks (0 on pad rows)
      pad_b:   (B, T_b) bool      True where row is real, False where pad
    """
    B = len(picks)
    # First decide each track's effective length and start offset.
    starts: List[int] = []
    lens:   List[int] = []
    for gi in picks:
        rows = track_groups[gi]
        L = len(rows)
        if L <= t_max:
            starts.append(0); lens.append(L)
        else:
            if train:
                s = int(rng.integers(0, L - t_max + 1))
            else:
                s = 0  # eval truncates to head; full-track walk is offline
            starts.append(s); lens.append(t_max)
    T_b = max(lens) if lens else 1

    in_dim = X.shape[1]
    X_b = np.zeros((B, T_b, in_dim), dtype=np.float32)
    pad = np.zeros((B, T_b), dtype=bool)
    lbl_b = {k: np.zeros((B, T_b), dtype=np.float32) for k in labels}
    msk_b = {k: np.zeros((B, T_b), dtype=np.float32) for k in valids}

    for bi, (gi, s, L) in enumerate(zip(picks, starts, lens)):
        rows = track_groups[gi][s:s + L]
        X_b[bi, :L] = X[rows]
        pad[bi, :L] = True
        for k in labels:
            lbl_b[k][bi, :L] = labels[k][rows]
        for k in valids:
            msk_b[k][bi, :L] = valids[k][rows]

    return (
        torch.from_numpy(X_b),
        {k: torch.from_numpy(v) for k, v in lbl_b.items()},
        {k: torch.from_numpy(v) for k, v in msk_b.items()},
        torch.from_numpy(pad),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val",   required=True)
    p.add_argument("--save",  default="bench/data/state_head_gru_v1.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256,
                   help="number of tracks per batch (sequence batch)")
    p.add_argument("--t-max", type=int, default=64,
                   help="max sequence length per training step")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--cr-promote",          type=float, default=0.1)
    p.add_argument("--cr-demote",           type=float, default=1.0)
    p.add_argument("--cr-drop-unconfirmed", type=float, default=0.2)
    p.add_argument("--cr-drop-lost",        type=float, default=0.1)
    p.add_argument("--no-drop-heads", action="store_true",
                   help="zero gradient on drop_unconfirmed/drop_lost heads")
    p.add_argument("--with-scene", action="store_true",
                   help="Append the 6 Phase 29 per-scene features → in_dim=25")
    p.add_argument("--filter-matched-only", action="store_true",
                   help="invalidate valid_promote on matched=0 rows")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"loading {args.train}", flush=True)
    rec_tr_data = np.load(args.train, allow_pickle=False)
    rec_tr = rec_tr_data["records"]
    print(f"loading {args.val}", flush=True)
    rec_va_data = np.load(args.val, allow_pickle=False)
    rec_va = rec_va_data["records"]

    X_tr = build_input_matrix_no_state(rec_tr, with_scene=args.with_scene)
    X_va = build_input_matrix_no_state(rec_va, with_scene=args.with_scene)
    in_dim = X_tr.shape[1]

    if args.filter_matched_only:
        for r in (rec_tr, rec_va):
            unmatched = ~r["matched"].astype(bool)
            vp = r["valid_promote"].copy()
            vp[unmatched] = 0
            r["valid_promote"] = vp

    # Per-head valid masks. drop heads use prior_state-conditional masks
    # to mirror train_state_head.py semantics.
    UNCONFIRMED, _, LOST = 0, 1, 2
    def split_drop(rec):
        prior = rec["prior_state"].astype(np.int32)
        valid_drop = rec["valid_drop"].astype(np.int32)
        return {
            "valid_drop_unconfirmed": (valid_drop & (prior == UNCONFIRMED)).astype(np.int32),
            "valid_drop_lost":        (valid_drop & (prior == LOST)).astype(np.int32),
        }
    extra_tr = split_drop(rec_tr)
    extra_va = split_drop(rec_va)

    labels_tr = {
        "promote":          rec_tr["promote_label"].astype(np.float32),
        "demote":           rec_tr["demote_label"].astype(np.float32),
        "drop_unconfirmed": rec_tr["drop_label"].astype(np.float32),
        "drop_lost":        rec_tr["drop_label"].astype(np.float32),
    }
    valids_tr = {
        "promote":          rec_tr["valid_promote"].astype(np.float32),
        "demote":           rec_tr["valid_demote"].astype(np.float32),
        "drop_unconfirmed": extra_tr["valid_drop_unconfirmed"].astype(np.float32),
        "drop_lost":        extra_tr["valid_drop_lost"].astype(np.float32),
    }
    labels_va = {
        "promote":          rec_va["promote_label"].astype(np.float32),
        "demote":           rec_va["demote_label"].astype(np.float32),
        "drop_unconfirmed": rec_va["drop_label"].astype(np.float32),
        "drop_lost":        rec_va["drop_label"].astype(np.float32),
    }
    valids_va = {
        "promote":          rec_va["valid_promote"].astype(np.float32),
        "demote":           rec_va["valid_demote"].astype(np.float32),
        "drop_unconfirmed": extra_va["valid_drop_unconfirmed"].astype(np.float32),
        "drop_lost":        extra_va["valid_drop_lost"].astype(np.float32),
    }

    print(f"train: {len(rec_tr)} rows, val: {len(rec_va)} rows, in_dim={in_dim}", flush=True)
    print("grouping rows by (sequence, track_id) ...", flush=True)
    groups_tr = group_rows_by_track(rec_tr)
    groups_va = group_rows_by_track(rec_va)
    lens_tr = np.array([len(g) for g in groups_tr])
    lens_va = np.array([len(g) for g in groups_va])
    print(f"  train: {len(groups_tr)} tracks, "
          f"len mean={lens_tr.mean():.1f} median={np.median(lens_tr):.0f} max={lens_tr.max()}",
          flush=True)
    print(f"  val:   {len(groups_va)} tracks, "
          f"len mean={lens_va.mean():.1f} median={np.median(lens_va):.0f} max={lens_va.max()}",
          flush=True)

    # ---- pos_weights from class balance × cost ratio ---------------------
    cost_ratio = {
        "promote":          args.cr_promote,
        "demote":           args.cr_demote,
        "drop_unconfirmed": args.cr_drop_unconfirmed,
        "drop_lost":        args.cr_drop_lost,
    }
    cls_pw = {}
    for k in cost_ratio:
        v = valids_tr[k] > 0
        if v.sum() == 0:
            cls_pw[k] = 1.0; continue
        n_pos = float(labels_tr[k][v].sum())
        n_neg = float(v.sum() - n_pos)
        cls_pw[k] = n_neg / max(1.0, n_pos)
    pw = {k: cls_pw[k] * cost_ratio[k] for k in cost_ratio}
    print("class-balance pos_weight: " +
          " ".join(f"{k}={cls_pw[k]:.2f}" for k in cost_ratio), flush=True)
    print("effective pos_weight:     " +
          " ".join(f"{k}={pw[k]:.3f}" for k in cost_ratio), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUStateHead(in_dim=in_dim, hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: GRU(in={in_dim}, hidden={args.hidden}) + 4 heads, "
          f"{n_params} params, device={device}", flush=True)

    pw_t = {k: torch.tensor([pw[k]], device=device) for k in cost_ratio}
    head_keys = list(cost_ratio.keys())
    if args.no_drop_heads:
        train_keys = ["promote", "demote"]
    else:
        train_keys = head_keys

    # ---- training loop ---------------------------------------------------
    best_auc = -1.0
    best_state = None
    bs = args.batch_size

    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(len(groups_tr))
        ep_losses = {k: 0.0 for k in train_keys}
        n_b = 0
        for i in range(0, len(perm), bs):
            picks = perm[i:i+bs].tolist()
            X_b, lbl_b, msk_b, pad = pack_batch(
                groups_tr, picks, X_tr, labels_tr, valids_tr,
                t_max=args.t_max, train=True, rng=rng)
            X_b = X_b.to(device, non_blocking=True)
            for k in lbl_b: lbl_b[k] = lbl_b[k].to(device, non_blocking=True)
            for k in msk_b: msk_b[k] = msk_b[k].to(device, non_blocking=True)

            promote_l, demote_l, drop_u_l, drop_l_l, _ = model(X_b)
            logits = {
                "promote": promote_l, "demote": demote_l,
                "drop_unconfirmed": drop_u_l, "drop_lost": drop_l_l,
            }

            total_loss = X_b.new_zeros(())
            for k in train_keys:
                m = msk_b[k]
                if m.sum() < 1: continue
                # masked BCE: weight per-row valid mask, sum over valid
                bce = F.binary_cross_entropy_with_logits(
                    logits[k], lbl_b[k], pos_weight=pw_t[k], reduction="none")
                kl = (bce * m).sum() / m.sum().clamp(min=1.0)
                ep_losses[k] += float(kl.item())
                total_loss = total_loss + kl
            opt.zero_grad(); total_loss.backward(); opt.step()
            n_b += 1

        # ---- validation: walk full tracks (no t_max truncation) ---------
        model.eval()
        all_logits = {k: [] for k in head_keys}
        all_labels = {k: [] for k in head_keys}
        all_masks  = {k: [] for k in head_keys}
        with torch.no_grad():
            # batch tracks for eval too — full-length, but small batch since
            # max length can be very large. Pad to longest in each batch.
            eval_bs = max(8, bs // 4)
            order_va = np.arange(len(groups_va))
            for i in range(0, len(order_va), eval_bs):
                picks = order_va[i:i+eval_bs].tolist()
                # use t_max=full so pack truncates to longest in pick
                # but we need per-track full length here — set t_max big.
                T_picks = max(len(groups_va[gi]) for gi in picks)
                X_b, lbl_b, msk_b, pad = pack_batch(
                    groups_va, picks, X_va, labels_va, valids_va,
                    t_max=T_picks, train=False, rng=rng)
                X_b = X_b.to(device, non_blocking=True)
                p_l, d_l, du_l, dl_l, _ = model(X_b)
                preds = {"promote": p_l, "demote": d_l,
                         "drop_unconfirmed": du_l, "drop_lost": dl_l}
                pad = pad.to(device)
                for k in head_keys:
                    flat_pred = preds[k][pad].cpu().numpy()
                    flat_lbl  = lbl_b[k].to(device)[pad].cpu().numpy()
                    flat_msk  = msk_b[k].to(device)[pad].cpu().numpy()
                    all_logits[k].append(flat_pred)
                    all_labels[k].append(flat_lbl)
                    all_masks[k].append(flat_msk)

        ep_metrics = {}
        for k in head_keys:
            l = np.concatenate(all_logits[k])
            y = np.concatenate(all_labels[k])
            m = np.concatenate(all_masks[k]).astype(np.int32)
            ep_metrics[k] = per_head_auc(l, y, m)

        ep_loss_str = " ".join(f"{k[:3]}={ep_losses[k]/max(1,n_b):.4f}"
                               for k in train_keys)
        ep_auc_str = " ".join(f"{k[:3]}={ep_metrics[k]:.4f}"
                              for k in head_keys)
        # composite: promote + demote AUCs (ignore drop heads if disabled)
        if args.no_drop_heads:
            score = float(np.nanmean([ep_metrics["promote"], ep_metrics["demote"]]))
        else:
            score = float(np.nanmean([ep_metrics[k] for k in head_keys]))
        print(f"ep {ep:3d}  L({ep_loss_str})  AUC({ep_auc_str})  score={score:.4f}",
              flush=True)
        if score > best_auc:
            best_auc = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nbest score = {best_auc:.4f}", flush=True)
    save = {
        "state_dict": best_state,
        "in_dim": in_dim, "hidden": args.hidden,
        "model_kind": "gru_v1",
        "feature_layout": ("v2-25dim+scene" if args.with_scene
                           else "build_input_matrix-23dim minus prior_state OH (=19dim)"),
        "no_drop_heads": args.no_drop_heads,
    }
    torch.save(save, args.save)
    print(f"saved {args.save}", flush=True)


if __name__ == "__main__":
    main()
