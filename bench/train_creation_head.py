"""
Train the creation head: 7-dim det features → 3-way softmax over
{discard, start_unconfirmed, start_tracked}.

This NN replaces param_new_track_thr + param_immediate_confirm_thr in
utrack.c. Cost-weighted cross-entropy mirrors the priority order:
- FP-track-creation (predict tracked or unconfirmed when label=discard) is
  costly: pollutes downstream matching, manufactures ID switches.
- Missed-track-creation (predict discard when label=unconfirmed/tracked)
  is also costly: causes a dropped real track that no downstream NN can
  recover.
- Wrong-confirm-shortcut (predict tracked when label=unconfirmed) is
  cheap: just adds a tiny obs-2 wait.

The cost ratio knobs are CLI flags; the runtime always uses argmax of
the 3-way softmax (no threshold knob).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn

from bench.build_creation_corpus import (
    CREATION_FEATURE_NAMES,
    EXAMPLE_DTYPE,
    N_FEATURES,
)


class CreationHead(nn.Module):
    def __init__(self, in_dim: int = N_FEATURES, hidden: int = 32,
                 layers: int = 2, n_classes: int = 3):
        super().__init__()
        ls: List[nn.Module] = []
        d = in_dim
        for _ in range(layers):
            ls.append(nn.Linear(d, hidden))
            ls.append(nn.ReLU(inplace=True))
            d = hidden
        ls.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*ls)

    def forward(self, x):  # (B, F) → (B, 3)
        return self.net(x)


def _build_xy(arr):
    x = arr["features"].astype(np.float32)
    y = arr["label"].astype(np.int64)
    return x, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val",   required=True)
    p.add_argument("--save",  default="bench/data/creation_head_v1.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    # cost weights per (true_label, pred_label) — but we use class-weighted CE
    # which captures the leading dimension well enough. Defaults: discards
    # are usually majority class so we up-weight unconfirmed/tracked to
    # ensure recall on those.
    p.add_argument("--w-discard",     type=float, default=1.0)
    p.add_argument("--w-unconfirmed", type=float, default=2.0)
    p.add_argument("--w-tracked",     type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tr = np.load(args.train, allow_pickle=True)["records"]
    va = np.load(args.val, allow_pickle=True)["records"]

    print(f"train: {len(tr)} examples  val: {len(va)}")
    for name, a in (("train", tr), ("val", va)):
        n_disc = int((a["label"] == 0).sum())
        n_unc  = int((a["label"] == 1).sum())
        n_trk  = int((a["label"] == 2).sum())
        n = len(a)
        print(f"  {name:5s}: discard={n_disc} ({100*n_disc/n:.1f}%)  "
              f"unconfirmed={n_unc} ({100*n_unc/n:.1f}%)  "
              f"tracked={n_trk} ({100*n_trk/n:.1f}%)")

    x_tr, y_tr = _build_xy(tr)
    x_va, y_va = _build_xy(va)

    # Standardise features using train stats
    mean = x_tr.mean(axis=0); std = x_tr.std(axis=0)
    std[std < 1e-6] = 1.0
    x_tr = (x_tr - mean) / std
    x_va = (x_va - mean) / std

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CreationHead(in_dim=N_FEATURES, hidden=args.hidden,
                         layers=args.layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  CreationHead params: {n_params}  hidden={args.hidden} "
          f"layers={args.layers}  device={device}")

    weights = torch.tensor([args.w_discard, args.w_unconfirmed, args.w_tracked],
                           dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)

    x_tr_g = torch.from_numpy(x_tr).to(device)
    y_tr_g = torch.from_numpy(y_tr).to(device)
    x_va_g = torch.from_numpy(x_va).to(device)
    y_va_g = torch.from_numpy(y_va).to(device)

    n = x_tr_g.shape[0]
    bs = args.batch_size

    print(f"\nepochs={args.epochs} bs={bs}")
    print(f"{'epoch':>5s}  {'tr_loss':>8s}  {'va_acc':>8s}  "
          f"{'va_prec_unc':>11s}  {'va_prec_trk':>11s}  "
          f"{'va_rec_unc':>10s}  {'va_rec_trk':>10s}")

    best_score = -1.0
    best_state = None

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            xb = x_tr_g[idx]; yb = y_tr_g[idx]
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach()) * xb.shape[0]

        model.eval()
        with torch.no_grad():
            logits_va = model(x_va_g)
            pred_va = logits_va.argmax(dim=1)
            acc = float((pred_va == y_va_g).float().mean())
            # per-class precision / recall
            def prc(c):
                p_mask = (pred_va == c)
                t_mask = (y_va_g == c)
                tp = float(((p_mask & t_mask).float()).sum())
                fp = float(((p_mask & ~t_mask).float()).sum())
                fn = float(((~p_mask & t_mask).float()).sum())
                prec = tp / max(1.0, tp + fp)
                rec  = tp / max(1.0, tp + fn)
                return prec, rec
            p_unc, r_unc = prc(1)
            p_trk, r_trk = prc(2)

        # Score: balanced — emphasise recall on unconfirmed/tracked
        # (those are the "real tracks" we want to keep)
        score = 0.5 * (r_unc + r_trk) + 0.25 * acc

        marker = "  ★" if score > best_score else ""
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"{ep:5d}  {total_loss/n:8.5f}  {acc:8.4f}  "
              f"{p_unc:11.4f}  {p_trk:11.4f}  "
              f"{r_unc:10.4f}  {r_trk:10.4f}{marker}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save
    torch.save({
        "state_dict": model.state_dict(),
        "feature_mean": mean.tolist(),
        "feature_std":  std.tolist(),
        "feature_names": list(CREATION_FEATURE_NAMES),
        "hidden": args.hidden,
        "layers": args.layers,
        "in_dim": N_FEATURES,
        "n_classes": 3,
        "w_class": [args.w_discard, args.w_unconfirmed, args.w_tracked],
        "best_score": float(best_score),
    }, args.save)
    print(f"\nsaved → {args.save}")
    print(f"best score = {best_score:.4f}")

    # Confusion matrix at best epoch
    model.eval()
    with torch.no_grad():
        logits_va = model(x_va_g)
        pred_va = logits_va.argmax(dim=1).cpu().numpy()
        true_va = y_va_g.cpu().numpy()
    print("\nConfusion matrix (rows=true, cols=pred), classes 0=discard 1=unconfirmed 2=tracked:")
    for i in range(3):
        row = [int(((true_va == i) & (pred_va == j)).sum()) for j in range(3)]
        print(f"  true={i}: {row}")


if __name__ == "__main__":
    main()
