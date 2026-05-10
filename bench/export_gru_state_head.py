"""Export a GRU state-head .pt to the v3 .bin format.

The .bin layout (see ubon_cstuff/src/track/utrack/nn_state.h, "Binary
format v3" comment) is:

  u32 magic 'USHT', u32 version 3
  u32 in_dim, u32 hidden, u32 n_outputs (=4)
  f32 W_ih [3·hidden × in_dim]   (PyTorch GRU concat order: r||z||n)
  f32 b_ih [3·hidden]
  f32 W_hh [3·hidden × hidden]
  f32 b_hh [3·hidden]
  f32 promote_W[hidden],          promote_b[1]
  f32 demote_W[hidden],           demote_b[1]
  f32 drop_unconfirmed_W[hidden], drop_unconfirmed_b[1]
  f32 drop_lost_W[hidden],        drop_lost_b[1]

PyTorch nn.GRU stores weight_ih_l0 in shape (3*hidden, in_dim) with rows
in order (r, z, n) — exactly the layout the C runtime expects. Same for
weight_hh_l0 (3*hidden, hidden) and biases. So the export is just a
flat .tobytes().

Usage:
    python -m bench.export_gru_state_head \\
        --in  bench/data/state_head_gru_v1.pt \\
        --out bench/data/state_head_gru_v1.bin
"""
from __future__ import annotations
import argparse, math, struct
import numpy as np
import torch


MAGIC   = 0x55534854  # 'USHT'
VERSION = 3


def _bias_shift_for_threshold(thr: float) -> float:
    """Return Δbias such that sigmoid(logit + Δbias) > 0.5 ⟺ sigmoid(logit) > thr.

    Δbias = -logit(thr). The C runtime fires each head at a fixed
    cutoff of 0.5, so we shift biases here to encode the deployment
    operating point. Same trick as bench/export_state_head.py.
    """
    thr = max(1e-6, min(1.0 - 1e-6, float(thr)))
    return -math.log(thr / (1.0 - thr))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in",  dest="src", required=True)
    p.add_argument("--out", dest="dst", required=True)
    # Defaults match bench/export_state_head.py — same operating point
    # v41 ships at, so the GRU is a drop-in calibration comparison.
    p.add_argument("--thr-promote",          type=float, default=0.9)
    p.add_argument("--thr-demote",           type=float, default=0.9)
    p.add_argument("--thr-drop-unconfirmed", type=float, default=0.5)
    p.add_argument("--thr-drop-lost",        type=float, default=0.95)
    args = p.parse_args()

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    in_dim = int(ckpt["in_dim"])
    hidden = int(ckpt["hidden"])
    print(f"  in_dim={in_dim} hidden={hidden}")
    if in_dim not in (19, 20, 25):
        print(f"  WARNING: GRU head expected in_dim ∈ {{19,20,25}}, "
              f"got {in_dim}; runtime will reject")

    def f32(t):
        return t.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)

    # PyTorch GRU layout: weight_ih_l0 shape (3*H, in_dim); concat r||z||n.
    W_ih = f32(sd["gru.weight_ih_l0"])
    b_ih = f32(sd["gru.bias_ih_l0"])
    W_hh = f32(sd["gru.weight_hh_l0"])
    b_hh = f32(sd["gru.bias_hh_l0"])
    assert W_ih.size == 3 * hidden * in_dim, f"W_ih size {W_ih.size}"
    assert b_ih.size == 3 * hidden,           f"b_ih size {b_ih.size}"
    assert W_hh.size == 3 * hidden * hidden,  f"W_hh size {W_hh.size}"
    assert b_hh.size == 3 * hidden,           f"b_hh size {b_hh.size}"

    thrs = {
        "promote":          args.thr_promote,
        "demote":           args.thr_demote,
        "drop_unconfirmed": args.thr_drop_unconfirmed,
        "drop_lost":        args.thr_drop_lost,
    }
    head_keys = ["promote", "demote", "drop_unconfirmed", "drop_lost"]
    head_bytes = []
    for k in head_keys:
        W = f32(sd[f"{k}.weight"])
        b = f32(sd[f"{k}.bias"])
        assert W.size == hidden, f"{k}.weight size {W.size}"
        assert b.size == 1,      f"{k}.bias size {b.size}"
        # Bake threshold into bias: shift by -logit(thr) so the runtime's
        # fixed 0.5 cutoff is equivalent to the trained head's `thr`.
        shift = _bias_shift_for_threshold(thrs[k])
        b = b + np.float32(shift)
        print(f"  {k}: thr={thrs[k]:.3f} bias_shift={shift:+.4f}")
        head_bytes.append((W, b))

    with open(args.dst, "wb") as f:
        f.write(struct.pack("<II", MAGIC, VERSION))
        f.write(struct.pack("<III", in_dim, hidden, 4))
        f.write(W_ih.tobytes())
        f.write(b_ih.tobytes())
        f.write(W_hh.tobytes())
        f.write(b_hh.tobytes())
        for k, (W, b) in zip(head_keys, head_bytes):
            f.write(W.tobytes())
            f.write(b.tobytes())

    print(f"wrote {args.dst}")
    print(f"  GRU(in={in_dim}, hidden={hidden}) + 4 heads")


if __name__ == "__main__":
    main()
