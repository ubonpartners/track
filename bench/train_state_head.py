"""
Train the unified state-transition head (UTRACK_NN.md follow-on).

Three binary classifiers share an MLP backbone:
  promote (UNCONFIRMED → TRACKED)
  demote  (TRACKED     → LOST)
  drop    (UNCONFIRMED|LOST → REMOVED)

Each example contributes loss only on the heads where its prior_state makes
the head valid (mask via valid_promote / valid_demote / valid_drop in the
records). Per-head class-weighted BCE handles imbalance.

Inputs (assembled in build_input_matrix), in order:
  prior_state one-hot[3], matched, log1p(observations), log1p(num_missed),
  time_since_det, log1p(scene_density), det_conf, prev_det_conf,
  phase3_pair_score,
  near_edge, det_w, det_h, log_aspect, log_pose_kp,
  e_track[16]   → 32 dims (e_track zeros if Phase 3 replay disabled).

Usage:
  python -m bench.train_state_head \\
      --train bench/data/state_corpus_train.npz \\
      --val   bench/data/state_corpus_val.npz \\
      --save  bench/data/state_head_v1.pt \\
      --epochs 50
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


N_E_TRACK = 16
# 3 prior_state + 9 scalar trackers + 5 spatial/quality + 5 history-agg + 16 e_track = 37
# History-agg fields (added 2026-05-07): ema_match_x_conf,
# log_sum_det_conf, min_match_score, mean_match_score, n_strong_matches.
# v1 in_dim was 27, v2 was 32, v3 is 37.
N_INPUT = 3 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 5 + 5 + N_E_TRACK   # = 37


def build_input_matrix(rec: np.ndarray) -> np.ndarray:
    """(N, 32) float32. Order matches the C-side runtime feeder
    (utrack_build_state_features in utrack.c). Don't reorder without
    bumping UTRACK_NN_STATE_IN_DIM and the C builder.

    The 5 spatial / detection-quality features (near_edge through
    log_pose_kp) were added 2026-05-05 to give the heads context that
    e_track's 16-dim compression of the Phase 3 obs features couldn't
    preserve at full fidelity (in particular: edge proximity, raw
    detection size, and the keypoint-visibility count)."""
    n = len(rec)
    state_oh = np.zeros((n, 3), dtype=np.float32)
    state_oh[np.arange(n), np.clip(rec["prior_state"].astype(np.int32), 0, 2)] = 1.0
    cols = [
        state_oh,
        rec["matched"].astype(np.float32).reshape(-1, 1),
        np.log1p(rec["observations"].astype(np.float32)).reshape(-1, 1),
        np.log1p(rec["num_missed"].astype(np.float32)).reshape(-1, 1),
        rec["time_since_det"].astype(np.float32).reshape(-1, 1),
        np.log1p(rec["scene_density"].astype(np.float32)).reshape(-1, 1),
        rec["det_conf"].astype(np.float32).reshape(-1, 1),
        rec["prev_det_conf"].astype(np.float32).reshape(-1, 1),
        rec["phase3_pair_score"].astype(np.float32).reshape(-1, 1),
        rec["near_edge"].astype(np.float32).reshape(-1, 1),
        rec["det_w"].astype(np.float32).reshape(-1, 1),
        rec["det_h"].astype(np.float32).reshape(-1, 1),
        rec["log_aspect"].astype(np.float32).reshape(-1, 1),
        rec["log_pose_kp"].astype(np.float32).reshape(-1, 1),
        # v3 (history aggregates) — fall back to zeros for old corpora
        # that don't have these fields.
        (rec["ema_match_x_conf"] if "ema_match_x_conf" in rec.dtype.names
         else np.zeros(len(rec), np.float32)).astype(np.float32).reshape(-1, 1),
        (rec["log_sum_det_conf"] if "log_sum_det_conf" in rec.dtype.names
         else np.zeros(len(rec), np.float32)).astype(np.float32).reshape(-1, 1),
        (rec["min_match_score"] if "min_match_score" in rec.dtype.names
         else np.zeros(len(rec), np.float32)).astype(np.float32).reshape(-1, 1),
        (rec["mean_match_score"] if "mean_match_score" in rec.dtype.names
         else np.zeros(len(rec), np.float32)).astype(np.float32).reshape(-1, 1),
        (rec["n_strong_matches"] if "n_strong_matches" in rec.dtype.names
         else np.zeros(len(rec), np.float32)).astype(np.float32).reshape(-1, 1),
        rec["e_track"].astype(np.float32),
    ]
    return np.concatenate(cols, axis=1)


def load_split(path: str, zero_e_track: bool = False,
               zero_spatial: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    rec = data["records"]
    X = build_input_matrix(rec)
    if zero_spatial:
        # near_edge..log_pose_kp are columns 9..13 (0-indexed) — see
        # build_input_matrix's column order.
        X[:, 9:14] = 0.0
    if zero_e_track:
        # e_track is the trailing 16 dims.
        X[:, -N_E_TRACK:] = 0.0
    return rec, X


class StateHead(nn.Module):
    """Shared backbone with four sigmoid heads.

    Drop is split into UNCONFIRMED-drop and LOST-drop because the two
    transitions have very different costs (UNCONFIRMED is unexposed →
    cheap to drop; LOST may recover → expensive to drop). Splitting at
    training time lets each head's bias settle to the cost-correct
    operating point rather than relying on a post-hoc threshold.

    Backbone shape:
      backbone_layers=2 (default)  in_dim → hidden → e_dim
      backbone_layers=3            in_dim → hidden → hidden → e_dim
      backbone_layers=1            in_dim → e_dim   (hidden ignored)
    Each Linear is followed by ReLU and (optional) Dropout. Export
    code (export_state_head.py) only knows how to serialise the
    backbone_layers=2 form — deeper backbones are training-time
    experiments only.
    """
    def __init__(self, in_dim: int = N_INPUT, hidden: int = 32, e_dim: int = 16,
                 dropout: float = 0.0, backbone_layers: int = 2):
        super().__init__()
        layers = []
        if backbone_layers <= 1:
            layers += [nn.Linear(in_dim, e_dim), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        else:
            layers += [nn.Linear(in_dim, hidden), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            for _ in range(backbone_layers - 2):
                layers += [nn.Linear(hidden, hidden), nn.ReLU(inplace=True)]
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            layers += [nn.Linear(hidden, e_dim), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)
        self.promote          = nn.Linear(e_dim, 1)
        self.demote           = nn.Linear(e_dim, 1)
        self.drop_unconfirmed = nn.Linear(e_dim, 1)
        self.drop_lost        = nn.Linear(e_dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        return (
            self.promote(h).squeeze(-1),
            self.demote(h).squeeze(-1),
            self.drop_unconfirmed(h).squeeze(-1),
            self.drop_lost(h).squeeze(-1),
        )


def per_head_auc(logits: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
    """Standard binary AUC over examples where mask=1."""
    keep = mask.astype(bool)
    if keep.sum() == 0 or labels[keep].sum() == 0 or labels[keep].sum() == keep.sum():
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels[keep], logits[keep]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val",   required=True)
    p.add_argument("--save",  default="bench/data/state_head_v1.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--e-dim", type=int, default=16)
    # Cost ratios (cost_FN / cost_FP) per head — multiplied with the
    # class-balance pos_weight. Defaults match the priority-derived
    # analysis in the doc; override to sweep operating points.
    p.add_argument("--cr-promote",          type=float, default=0.1)
    p.add_argument("--cr-demote",           type=float, default=1.0)
    p.add_argument("--cr-drop-unconfirmed", type=float, default=0.2)
    p.add_argument("--cr-drop-lost",        type=float, default=0.1)
    p.add_argument("--dropout",             type=float, default=0.0,
                   help="dropout p applied after each backbone ReLU")
    p.add_argument("--backbone-layers",     type=int, default=2,
                   help="number of Linear layers in backbone (2 = current)")
    p.add_argument("--zero-e-track",        action="store_true",
                   help="zero out the 16-dim e_track input (ablation)")
    p.add_argument("--zero-spatial",        action="store_true",
                   help="zero out near_edge/det_w/det_h/log_aspect/log_pose_kp (ablation)")
    p.add_argument("--no-history",          action="store_true",
                   help="drop the 5 history-aggregate input columns "
                        "(ema_match_x_conf, log_sum_det_conf, "
                        "min/mean_match_score, n_strong_matches). "
                        "Trains a 32-dim head that's drop-in deployable "
                        "with the current C runtime (UTRACK_NN_STATE_IN_DIM=32).")
    p.add_argument("--fitness-fp-boost",    type=float, default=1.0,
                   help="Per-example weight multiplier on examples whose "
                        "track never matches any GT (i.e., the track would "
                        "count in fp_tracks for the user's fitness function "
                        "`mota - 0.0005*fp_tracks - 0.002*fp_per_frame`). "
                        "Default 1.0 = no change. Computed at train-time from "
                        "the corpus's gt_id_now field — no corpus rebuild needed.")
    # Phase 20b — closed-loop fitness selection.
    p.add_argument("--fitness-select",      action="store_true",
                   help="At end of training, evaluate the top-K AUC checkpoints "
                        "by running the actual C tracker on a held-out clip "
                        "subset (bench/eval_subset_fast.json by default), and "
                        "save the BEST-BY-FITNESS checkpoint as args.save. "
                        "This replaces 'best val combined AUC' as the model-"
                        "selection signal — see Phase 20a/b in "
                        "STATE_HEAD_RETRAIN_PLAN.md. Adds ~1-2 min total to "
                        "training time (parallel eval on the fast 20-clip "
                        "subset).")
    p.add_argument("--fitness-top-k",       type=int, default=3,
                   help="number of top-AUC checkpoints to evaluate by fitness "
                        "(only used with --fitness-select). Default 3.")
    p.add_argument("--fitness-subset",      default="fast", choices=["fast","diverse"],
                   help="which clip subset to eval against. 'fast'≈30s/eval, "
                        "'diverse'≈80s/eval but covers MOT20-05 etc.")
    p.add_argument("--fitness-workers",     type=int, default=4,
                   help="parallel workers for fitness eval (default 4).")
    # Phase 20c — sequence-level (track-batched) surrogate loss.
    p.add_argument("--track-loss-enable",   action="store_true",
                   help="Enable Phase 20c sequence-level loss. Switches "
                        "training to track-batched sampling (each batch = T "
                        "tracks' UNCONFIRMED rows) and adds two terms to the "
                        "BCE: phantom_loss = E_track[P(emit)*is_phantom], "
                        "miss_loss = E_track[(1-P(emit))*is_real]. P(emit) per "
                        "track = 1 - exp(-Σ softplus(promote_logit_i)) over "
                        "the track's UNCONFIRMED frames in the batch — i.e., "
                        "1 - Π(1-sigmoid). This penalises 'promote at f1, drop "
                        "at f2 on the same phantom track' the same way "
                        "fp_tracks does, which per-frame BCE cannot. See "
                        "Phase 20c in STATE_HEAD_RETRAIN_PLAN.md.")
    p.add_argument("--lambda-phantom",      type=float, default=1.0,
                   help="weight on per-track phantom_loss (only with "
                        "--track-loss-enable). Default 1.0.")
    p.add_argument("--lambda-miss",         type=float, default=0.5,
                   help="weight on per-track miss_loss (only with "
                        "--track-loss-enable). Default 0.5 (lower than phantom "
                        "because miss already dominates BCE on real tracks).")
    p.add_argument("--tracks-per-batch",    type=int, default=64,
                   help="tracks per batch in track-batched mode. With avg ~124 "
                        "UNCONFIRMED frames/track, 64 ≈ 8K rows/batch — same "
                        "magnitude as random-row --batch-size 8192.")
    p.add_argument("--summary-out",         type=str, default=None,
                   help="if set, write JSON {best_score, best_aucs, params} to this path")
    # Phase 20.21 — corpus rebalancing: 98% of UNCONFIRMED rows are
    # absent-track artefacts (matched=0, det_conf=0). They contribute
    # zero discriminative signal. Filtering to matched=1 only means the
    # promote-head training is dominated by the 1.7% rows where the
    # head actually has to decide.
    p.add_argument("--filter-matched-only",        action="store_true",
                   help="Drop training rows where matched=0 (i.e., absent-track "
                        "artefacts emitted by the corpus builder for tracks alive "
                        "but not seen this frame). 98% of UNCONFIRMED rows are "
                        "trivial matched=0 'no det → don't promote' examples; "
                        "filtering rebalances the loss toward the 1.7% matched=1 "
                        "rows that contain the actual hard discriminations the "
                        "head needs to learn for low-conf track-rescue. "
                        "Reduces train set ~50× — typical sign-of-life is whether "
                        "history-feature weight norms increase vs an unfiltered "
                        "v29 baseline.")
    p.add_argument("--seed",                type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rec_tr, X_tr = load_split(args.train, zero_e_track=args.zero_e_track,
                              zero_spatial=args.zero_spatial)
    rec_va, X_va = load_split(args.val,   zero_e_track=args.zero_e_track,
                              zero_spatial=args.zero_spatial)

    if getattr(args, "filter_matched_only", False):
        # Per-head fix: invalidate valid_promote on matched=0 rows so the
        # promote head only trains on examples it'll actually see at
        # deployment (the C runtime calls promote_head only on matched
        # detections at match time). Demote/drop heads keep their
        # matched=0 rows because drop_unc/drop_lost ARE supposed to fire
        # on absent tracks, and demote fires on TRACKED state regardless.
        # Naive row-filtering would kill drop-head training.
        for r in (rec_tr, rec_va):
            unmatched = ~r["matched"].astype(bool)
            n_promote_before = int(r["valid_promote"].sum())
            # Mutate via a copy since rec dtype is structured.
            vp = r["valid_promote"].copy()
            vp[unmatched] = 0
            r["valid_promote"] = vp
            n_promote_after = int(r["valid_promote"].sum())
            print(f"--filter-matched-only: valid_promote {n_promote_before} -> "
                  f"{n_promote_after} ({n_promote_after/max(1,n_promote_before)*100:.1f}% kept)")
    if args.no_history:
        # Drop the 5 history-aggregate columns (14:19) so the trained head
        # is drop-in deployable with the C runtime's UTRACK_NN_STATE_IN_DIM=32.
        # Layout from build_input_matrix: [0:9] base, [9:14] spatial,
        # [14:19] history, [19:35] e_track.
        X_tr = np.concatenate([X_tr[:, :14], X_tr[:, 19:]], axis=1)
        X_va = np.concatenate([X_va[:, :14], X_va[:, 19:]], axis=1)
    print(f"train: {len(rec_tr)} examples; val: {len(rec_va)}  (in_dim={X_tr.shape[1]})")
    if args.zero_e_track or args.zero_spatial or args.no_history:
        print(f"ablation: zero_e_track={args.zero_e_track} "
              f"zero_spatial={args.zero_spatial} no_history={args.no_history}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StateHead(in_dim=X_tr.shape[1], hidden=args.hidden, e_dim=args.e_dim,
                      dropout=args.dropout, backbone_layers=args.backbone_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    print(f"model: {sum(p.numel() for p in model.parameters())} params, device={device}")

    # ---- Cost-weighted BCE -------------------------------------------------
    # Per-head pos_weight is the product of two factors:
    #   - class_balance = n_neg / n_pos          (corrects label imbalance)
    #   - cost_ratio    = cost_FN / cost_FP      (encodes deployment priorities)
    # Together they make threshold=0.5 the cost-correct decision boundary.
    # The cost ratios below come from cost analysis of each head's two
    # error modes against the user-stated priority order (FPTr ≫ SW >
    # FP-on-good-track ≈ FRG):
    #
    #   promote          cFN=1,   cFP=10   pos_weight = cls * 0.1
    #     (false-promote = phantom exposed → FPTr;
    #      false-no-promote = real track delayed exposure → mild FN)
    #   demote           cFN=1,   cFP=1    pos_weight = cls * 1.0
    #     (false-demote = brief LOST blink, ID preserved → FRG-like;
    #      false-no-demote = TRACKED phantom emits frames briefly → mild)
    #   drop_unconfirmed cFN=1,   cFP=5    pos_weight = cls * 0.2
    #     (false-no-drop = phantom kept alive 1 frame, dies via existing
    #       num_missed rule anyway → low cost;
    #      false-drop = real track killed before confirm; either lost
    #       entirely or re-formed as a new ID → moderate cost)
    #   drop_lost        cFN=0.5, cFP=5    pos_weight = cls * 0.1
    #     (false-drop = recoverable removed → SW or FRG;
    #      false-no-drop = LOST stays around, no exposure cost)
    cost_ratio = {
        "promote":          args.cr_promote,
        "demote":           args.cr_demote,
        "drop_unconfirmed": args.cr_drop_unconfirmed,
        "drop_lost":        args.cr_drop_lost,
    }

    # Build derived valid masks for the two drop heads from prior_state.
    UNCONFIRMED, _, LOST = 0, 1, 2
    prior_tr = rec_tr["prior_state"].astype(np.int32)
    prior_va = rec_va["prior_state"].astype(np.int32)

    def add_split_drop_masks(rec, prior):
        valid_drop = rec["valid_drop"].astype(np.int32)
        return {
            "valid_drop_unconfirmed": (valid_drop & (prior == UNCONFIRMED)).astype(np.int32),
            "valid_drop_lost":        (valid_drop & (prior == LOST)).astype(np.int32),
        }
    extra_tr = add_split_drop_masks(rec_tr, prior_tr)
    extra_va = add_split_drop_masks(rec_va, prior_va)

    def class_balance_pos_weight(label, valid):
        v = valid.astype(bool)
        if v.sum() == 0:
            return 1.0
        n_pos = int(label[v].sum())
        n_neg = int(v.sum() - n_pos)
        return n_neg / max(1, n_pos)

    cls_pw = {
        "promote":          class_balance_pos_weight(rec_tr["promote_label"], rec_tr["valid_promote"]),
        "demote":           class_balance_pos_weight(rec_tr["demote_label"],  rec_tr["valid_demote"]),
        "drop_unconfirmed": class_balance_pos_weight(rec_tr["drop_label"],    extra_tr["valid_drop_unconfirmed"]),
        "drop_lost":        class_balance_pos_weight(rec_tr["drop_label"],    extra_tr["valid_drop_lost"]),
    }
    pw = {k: cls_pw[k] * cost_ratio[k] for k in cost_ratio}
    print("class-balance pos_weight: " +
          " ".join(f"{k}={cls_pw[k]:.2f}" for k in cost_ratio))
    print("cost ratio (cFN/cFP):     " +
          " ".join(f"{k}={cost_ratio[k]:.2f}" for k in cost_ratio))
    print("effective pos_weight:     " +
          " ".join(f"{k}={pw[k]:.3f}" for k in cost_ratio))

    X_tr_t = torch.from_numpy(X_tr).to(device)
    X_va_t = torch.from_numpy(X_va).to(device)

    def labels_and_masks(rec, extra):
        return {
            "promote": (
                torch.from_numpy(rec["promote_label"].astype(np.float32)).to(device),
                torch.from_numpy(rec["valid_promote"].astype(np.float32)).to(device),
            ),
            "demote":  (
                torch.from_numpy(rec["demote_label"].astype(np.float32)).to(device),
                torch.from_numpy(rec["valid_demote"].astype(np.float32)).to(device),
            ),
            "drop_unconfirmed": (
                torch.from_numpy(rec["drop_label"].astype(np.float32)).to(device),
                torch.from_numpy(extra["valid_drop_unconfirmed"].astype(np.float32)).to(device),
            ),
            "drop_lost": (
                torch.from_numpy(rec["drop_label"].astype(np.float32)).to(device),
                torch.from_numpy(extra["valid_drop_lost"].astype(np.float32)).to(device),
            ),
        }
    lm_tr = labels_and_masks(rec_tr, extra_tr)
    lm_va = labels_and_masks(rec_va, extra_va)

    # Sample weights — written by the label-driven corpus builder. Older
    # corpora that lack the field are treated as all-1.0.
    if "weight" in rec_tr.dtype.names:
        sw_tr_np = rec_tr["weight"].astype(np.float32).copy()
    else:
        sw_tr_np = np.ones(len(rec_tr), dtype=np.float32)

    # Fitness-shaped FP-track boost. A track is FP iff it never has a
    # matched GT id across any of its frames in this sequence — exactly
    # what counts as `fp_tracks` in src/track_test.py:fitness_score
    # (penalty 0.0005 each, plus 0.002 per FP per frame). Up-weighting
    # examples on those tracks pushes the loss to track deployment fitness
    # rather than just per-head AUC.
    if args.fitness_fp_boost != 1.0 and "gt_id_now" in rec_tr.dtype.names:
        from collections import defaultdict
        seq_arr = rec_tr["sequence"]; tid_arr = rec_tr["track_id"]
        gid_arr = rec_tr["gt_id_now"]
        ever_matched: Dict[Tuple[str, int], bool] = defaultdict(bool)
        # Pass 1: any GT match per (seq, tid)?
        for i in range(len(rec_tr)):
            if gid_arr[i] >= 0:
                key = (str(seq_arr[i]), int(tid_arr[i]))
                ever_matched[key] = True
        # Pass 2: boost FP-track examples in-place.
        n_fp = 0
        for i in range(len(rec_tr)):
            key = (str(seq_arr[i]), int(tid_arr[i]))
            if not ever_matched.get(key, False):
                sw_tr_np[i] *= args.fitness_fp_boost
                n_fp += 1
        print(f"fitness FP-track boost: ×{args.fitness_fp_boost} on "
              f"{n_fp} examples ({n_fp/len(rec_tr)*100:.1f}% of train)")
    sw_tr = torch.from_numpy(sw_tr_np).to(device)

    # Phase 20c — sequence-level loss prep. Group rows by (sequence, track_id)
    # so each batch can carry all of a track's UNCONFIRMED rows and we can
    # compute the correct per-track P(emit) via scatter_add.
    track_loss_enabled = bool(getattr(args, "track_loss_enable", False))
    if track_loss_enabled:
        if "gt_id_now" not in rec_tr.dtype.names:
            sys.exit("--track-loss-enable requires corpus with gt_id_now field")
        seq_arr = rec_tr["sequence"].astype(str)
        tid_arr = rec_tr["track_id"].astype(np.int64)
        gid_arr = rec_tr["gt_id_now"]
        keys = np.char.add(seq_arr, np.char.add("#", tid_arr.astype(str)))
        _, track_idx_per_row = np.unique(keys, return_inverse=True)
        n_tracks_all = int(track_idx_per_row.max()) + 1

        # is_phantom[t] = 1 iff track t never matches GT (matches the
        # `fp_tracks` definition in src/track_test.py).
        is_phantom_arr = np.ones(n_tracks_all, dtype=np.float32)
        matched_mask = (gid_arr >= 0)
        if matched_mask.any():
            is_phantom_arr[np.unique(track_idx_per_row[matched_mask])] = 0.0

        # Per-track row indices — *all* rows of a track (any state). The
        # demote/drop_lost heads need TRACKED/LOST rows for their BCE; the
        # sequence loss masks itself down to valid_promote=1 (UNCONFIRMED).
        # Tracks with zero UNCONFIRMED rows are dropped from the trainable set
        # because they can't contribute to the sequence loss; BCE on those
        # rows still happens via the random-row path when --track-loss is off.
        valid_p_arr = rec_tr["valid_promote"].astype(bool)
        rows_by_track_all = [[] for _ in range(n_tracks_all)]
        n_unc_by_track = np.zeros(n_tracks_all, dtype=np.int32)
        for i in range(len(rec_tr)):
            t = track_idx_per_row[i]
            rows_by_track_all[t].append(i)
            if valid_p_arr[i]:
                n_unc_by_track[t] += 1
        kept = [t for t in range(n_tracks_all) if n_unc_by_track[t] > 0]
        track_all_rows = [np.asarray(rows_by_track_all[t], dtype=np.int64) for t in kept]
        track_phantom = is_phantom_arr[kept].astype(np.float32)
        n_phantom = int(track_phantom.sum())
        total_unc = int(n_unc_by_track[kept].sum())
        total_rows = sum(len(r) for r in track_all_rows)
        print(f"[track-loss] {len(kept)} trainable tracks "
              f"({n_phantom} phantom, {len(kept)-n_phantom} real); "
              f"{total_rows} rows total ({total_unc} UNCONFIRMED, "
              f"{total_unc/max(1,len(kept)):.1f}/track)")
        track_phantom_t = torch.from_numpy(track_phantom).to(device)

    head_names = ["promote", "demote", "drop_unconfirmed", "drop_lost"]

    bs = args.batch_size
    best_score = -1.0
    best_state = None
    best_aucs = {k: float("nan") for k in head_names}
    # Phase 20b: keep top-K candidates by AUC for end-of-training fitness eval.
    # Each entry: dict(epoch, score, aucs, state_dict_cpu).
    candidates = []
    K = max(1, int(getattr(args, "fitness_top_k", 3))) if getattr(args, "fitness_select", False) else 0
    def _build_track_batch(t_indices: np.ndarray):
        """Returns (flat_row_idx, within_batch_track_idx) — concatenated all-state
        rows for the given tracks plus a per-row 0..T-1 index used by scatter_add."""
        rows = [track_all_rows[t] for t in t_indices]
        flat = np.concatenate(rows)
        within = np.concatenate([np.full(len(r), bi, dtype=np.int64)
                                  for bi, r in enumerate(rows)])
        return flat, within

    for ep in range(args.epochs):
        model.train()
        ep_loss = 0.0
        ep_phantom = 0.0
        ep_miss = 0.0
        n_batches = 0
        if track_loss_enabled:
            T = max(1, int(args.tracks_per_batch))
            tperm = np.random.permutation(len(track_all_rows))
            for i in range(0, len(tperm), T):
                t_batch = tperm[i:i+T]
                flat_rows, within = _build_track_batch(t_batch)
                idx = torch.from_numpy(flat_rows).to(device)
                within_t = torch.from_numpy(within).to(device)
                phantom_b = track_phantom_t[torch.from_numpy(t_batch).to(device)]
                xb = X_tr_t[idx]
                pl, dl, rl_u, rl_l = model(xb)
                logits = {"promote": pl, "demote": dl,
                          "drop_unconfirmed": rl_u, "drop_lost": rl_l}
                loss = 0.0
                sw_b = sw_tr[idx]
                for name in head_names:
                    lbl, mask = lm_tr[name]
                    lbl_b = lbl[idx]; mask_b = mask[idx]
                    eff = mask_b * sw_b
                    if eff.sum() < 1e-6:
                        continue
                    pos_w = torch.tensor([pw[name]], device=device)
                    bce = F.binary_cross_entropy_with_logits(
                        logits[name], lbl_b, pos_weight=pos_w, reduction="none",
                    )
                    loss = loss + (bce * eff).sum() / eff.sum()
                # Sequence loss on the promote head. P(track emitted) =
                # 1 - Π_i (1 - sigmoid(pl_i)) over the track's UNCONFIRMED
                # rows. Numerically stable via softplus identity:
                # log(1 - sigmoid(x)) = -softplus(x). Masked to valid_promote
                # rows so TRACKED/LOST states (which the promote head doesn't
                # gate) don't leak into the per-track product.
                n_tb = len(t_batch)
                _, valid_p_mask = lm_tr["promote"]
                vp_b = valid_p_mask[idx]
                sum_softplus = torch.zeros(n_tb, device=device)
                sum_softplus.scatter_add_(0, within_t, F.softplus(pl) * vp_b)
                # log_p_not_emit = -sum_softplus  →  p_emit = 1 - exp(-S).
                p_emit = -torch.expm1(-sum_softplus)
                n_phantom_b = phantom_b.sum().clamp(min=1.0)
                n_real_b    = (1.0 - phantom_b).sum().clamp(min=1.0)
                phantom_loss = (p_emit * phantom_b).sum() / n_phantom_b
                miss_loss    = ((1.0 - p_emit) * (1.0 - phantom_b)).sum() / n_real_b
                loss = (loss
                        + args.lambda_phantom * phantom_loss
                        + args.lambda_miss * miss_loss)
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += float(loss.item())
                ep_phantom += float(phantom_loss.item())
                ep_miss += float(miss_loss.item())
                n_batches += 1
        else:
            perm = torch.randperm(len(X_tr_t), device=device)
            for i in range(0, len(perm), bs):
                idx = perm[i:i+bs]
                xb = X_tr_t[idx]
                pl, dl, rl_u, rl_l = model(xb)
                logits = {"promote": pl, "demote": dl,
                          "drop_unconfirmed": rl_u, "drop_lost": rl_l}
                loss = 0.0
                sw_b = sw_tr[idx]
                for name in head_names:
                    lbl, mask = lm_tr[name]
                    lbl_b = lbl[idx]; mask_b = mask[idx]
                    eff = mask_b * sw_b
                    if eff.sum() < 1e-6:
                        continue
                    pos_w = torch.tensor([pw[name]], device=device)
                    bce = F.binary_cross_entropy_with_logits(
                        logits[name], lbl_b, pos_weight=pos_w, reduction="none",
                    )
                    loss = loss + (bce * eff).sum() / eff.sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += float(loss.item())
                n_batches += 1

        # Validation AUCs (per head, masked appropriately).
        model.eval()
        with torch.no_grad():
            pl, dl, rl_u, rl_l = model(X_va_t)
            arr = {
                "promote":          pl.cpu().numpy(),
                "demote":           dl.cpu().numpy(),
                "drop_unconfirmed": rl_u.cpu().numpy(),
                "drop_lost":        rl_l.cpu().numpy(),
            }
        aucs = {}
        aucs["promote"]          = per_head_auc(arr["promote"],          rec_va["promote_label"], rec_va["valid_promote"])
        aucs["demote"]           = per_head_auc(arr["demote"],           rec_va["demote_label"],  rec_va["valid_demote"])
        aucs["drop_unconfirmed"] = per_head_auc(arr["drop_unconfirmed"], rec_va["drop_label"],    extra_va["valid_drop_unconfirmed"])
        aucs["drop_lost"]        = per_head_auc(arr["drop_lost"],        rec_va["drop_label"],    extra_va["valid_drop_lost"])
        vals = [aucs[k] for k in head_names]
        score = float(np.mean(vals)) if not any(np.isnan(vals)) else -1.0
        seq_str = ""
        if track_loss_enabled:
            seq_str = (f"  L_phantom={ep_phantom/max(1,n_batches):.4f} "
                       f"L_miss={ep_miss/max(1,n_batches):.4f}")
        print(f"ep {ep:3d}  loss={ep_loss/max(1,n_batches):.4f}{seq_str}  AUC " +
              " ".join(f"{k[:4]}={aucs[k]:.4f}" for k in head_names))
        if score > best_score:
            best_score = score
            best_aucs = {k: float(aucs[k]) for k in head_names}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        # Maintain top-K candidates by AUC for fitness-select.
        if K > 0:
            sd_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            candidates.append({"epoch": ep, "score": score,
                                "aucs": {k: float(aucs[k]) for k in head_names},
                                "state": sd_cpu})
            candidates.sort(key=lambda c: -c["score"])
            candidates[:] = candidates[:K]

    print(f"\nbest combined AUC = {best_score:.4f}")

    # Phase 20b — fitness-select: pick best-by-fitness across top-K AUC
    # candidates instead of best-by-AUC. The diagnostic in Phase 20.1 showed
    # AUC ranking can disagree with fitness ranking by 6 positions on a
    # 7-head set, so this is the load-bearing fix.
    selected_score = best_score
    selected_state = best_state
    selected_aucs  = best_aucs
    selected_fitness = None
    fitness_table = []
    if K > 0 and candidates:
        import subprocess, tempfile, time
        from bench.eval_head_fitness import eval_config, load_subset, make_yaml_with_state_bin
        clips = load_subset(args.fitness_subset)
        print(f"\n[fitness-select] running fitness eval on top-{len(candidates)} "
              f"AUC candidates (subset={args.fitness_subset}, "
              f"workers={args.fitness_workers}) ...")
        for c in candidates:
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                pt_path = f.name
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
                bin_path = f.name
            torch.save({
                "state_dict": c["state"],
                "in_dim": X_tr.shape[1],
                "hidden": args.hidden,
                "e_dim": args.e_dim,
                "dropout": args.dropout,
                "backbone_layers": args.backbone_layers,
                "best_score": c["score"],
            }, pt_path)
            r = subprocess.run(
                [sys.executable, "-m", "bench.export_state_head",
                 "--in", pt_path, "--out", bin_path],
                capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  ep {c['epoch']:3d}  AUC={c['score']:.4f}  EXPORT FAILED: {r.stderr.strip()[:200]}")
                os.unlink(pt_path);
                if os.path.exists(bin_path): os.unlink(bin_path)
                continue
            # Build temp YAML and run fitness eval.
            from bench.eval_head_fitness import make_yaml_with_state_bin
            yaml_path = make_yaml_with_state_bin(bin_path)
            t0 = time.time()
            res = eval_config(yaml_path, clips=clips,
                               workers=args.fitness_workers)
            dt = time.time() - t0
            fit = res["overall"]["fitness"]
            print(f"  ep {c['epoch']:3d}  AUC={c['score']:.4f}  "
                  f"fitness={fit:.4f}  ({dt:.1f}s)")
            fitness_table.append({
                "epoch": c["epoch"], "auc": c["score"],
                "fitness": fit, "overall": res["overall"],
                "state": c["state"],
            })
            os.unlink(pt_path); os.unlink(bin_path); os.unlink(yaml_path)
        if fitness_table:
            fitness_table.sort(key=lambda r: -r["fitness"])
            best = fitness_table[0]
            selected_state = best["state"]
            selected_score = best["auc"]
            selected_fitness = best["fitness"]
            print(f"\n[fitness-select] picked epoch {best['epoch']} "
                  f"(AUC {best['auc']:.4f}, fitness {best['fitness']:.4f}) "
                  f"as the saved checkpoint.")
            # Note when AUC and fitness disagree on the winner.
            auc_winner = max(fitness_table, key=lambda r: r["auc"])
            if auc_winner["epoch"] != best["epoch"]:
                print(f"[fitness-select] note: highest-AUC candidate was "
                      f"epoch {auc_winner['epoch']} (AUC {auc_winner['auc']:.4f}, "
                      f"fitness {auc_winner['fitness']:.4f}) — fitness selection "
                      f"chose differently.")

    if selected_state is not None:
        model.load_state_dict(selected_state)
        torch.save({
            "state_dict": model.state_dict(),
            "in_dim": X_tr.shape[1],
            "hidden": args.hidden,
            "e_dim": args.e_dim,
            "dropout": args.dropout,
            "backbone_layers": args.backbone_layers,
            "best_score": selected_score,
            "best_fitness": selected_fitness,
            "fitness_subset": args.fitness_subset if K > 0 else None,
        }, args.save)
        print(f"saved → {args.save}")

    if args.summary_out:
        import json
        summary = {
            "best_score": best_score,
            "best_aucs": best_aucs,
            "params": {
                "epochs": args.epochs, "batch_size": args.batch_size,
                "lr": args.lr, "wd": args.wd,
                "hidden": args.hidden, "e_dim": args.e_dim,
                "dropout": args.dropout, "backbone_layers": args.backbone_layers,
                "cr_promote": args.cr_promote, "cr_demote": args.cr_demote,
                "cr_drop_unconfirmed": args.cr_drop_unconfirmed,
                "cr_drop_lost": args.cr_drop_lost,
                "zero_e_track": args.zero_e_track, "zero_spatial": args.zero_spatial,
                "seed": args.seed,
            },
            "save": args.save,
        }
        with open(args.summary_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"summary → {args.summary_out}")


if __name__ == "__main__":
    main()
