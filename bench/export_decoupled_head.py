"""Export a decoupled GRU state-head .pt to .bin v3 (n_outputs=3).

The decoupled head outputs raw (p_TP, μ_TP, μ_FP) without thresholds.
The C runtime applies sigmoid/expm1 in utrack.c's Bayesian-mode path
and feeds (p_TP, μ_TP, μ_FP) directly into the cost-rule decision.

Layout (see nn_state.h "Binary format v3"):
  u32 magic 'USHT', u32 version 3
  u32 in_dim, u32 hidden, u32 n_outputs (=3)
  f32 W_ih[3·hidden × in_dim]  (PyTorch GRU r||z||n)
  f32 b_ih[3·hidden]
  f32 W_hh[3·hidden × hidden]
  f32 b_hh[3·hidden]
  f32 llr_W[hidden],    llr_b[1]      ← slot 0
  f32 mu_tp_W[hidden],  mu_tp_b[1]    ← slot 1
  f32 mu_fp_W[hidden],  mu_fp_b[1]    ← slot 2

Usage:
    python -m bench.export_decoupled_head \\
        --in  bench/data/state_head_dc_v3.pt \\
        --out bench/data/state_head_dc_v3.bin
"""
from __future__ import annotations
import argparse, struct
import numpy as np
import torch


MAGIC   = 0x55534854   # 'USHT'
VERSION = 3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in",  dest="src", required=True)
    p.add_argument("--out", dest="dst", required=True)
    args = p.parse_args()

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    in_dim = int(ckpt["in_dim"])
    hidden = int(ckpt["hidden"])
    print(f"  in_dim={in_dim} hidden={hidden} model_kind={ckpt.get('model_kind')}")
    # Runtime accepts in_dim==19 only (UTRACK_NN_STATE_GRU_IN_DIM in nn_state.h).
    if in_dim != 19:
        raise SystemExit(
            f"in_dim={in_dim} but C runtime requires 19; refusing to export."
        )

    def f32(t):
        return t.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)

    W_ih = f32(sd["gru.weight_ih_l0"])
    b_ih = f32(sd["gru.bias_ih_l0"])
    W_hh = f32(sd["gru.weight_hh_l0"])
    b_hh = f32(sd["gru.bias_hh_l0"])
    assert W_ih.size == 3 * hidden * in_dim
    assert b_ih.size == 3 * hidden
    assert W_hh.size == 3 * hidden * hidden
    assert b_hh.size == 3 * hidden

    # Slot order: 0=llr, 1=mu_tp, 2=mu_fp. Module attrs are llr / mu_tp / mu_fp.
    head_keys = ["llr", "mu_tp", "mu_fp"]
    head_bytes = []
    for k in head_keys:
        W = f32(sd[f"{k}.weight"])
        b = f32(sd[f"{k}.bias"])
        assert W.size == hidden, f"{k}.weight size {W.size}"
        assert b.size == 1,      f"{k}.bias size {b.size}"
        head_bytes.append((W, b))

    from bench._artefact_meta import make_pt_meta, bin_trailer, write_meta_sidecar
    pt_meta = ckpt.get("_meta")
    bin_meta = make_pt_meta(
        artefact_kind="state_head_decoupled_bin",
        args=args,
        hparams={
            "in_dim": in_dim, "hidden": hidden, "n_outputs": 3,
            "magic": "USHT", "version": VERSION,
            "head_order": head_keys,
        },
        dataset_info={"source_pt": args.src, "source_pt_meta": pt_meta},
    )

    with open(args.dst, "wb") as f:
        f.write(struct.pack("<II",  MAGIC, VERSION))
        f.write(struct.pack("<III", in_dim, hidden, 3))
        f.write(W_ih.tobytes())
        f.write(b_ih.tobytes())
        f.write(W_hh.tobytes())
        f.write(b_hh.tobytes())
        for k, (W, b) in zip(head_keys, head_bytes):
            f.write(W.tobytes())
            f.write(b.tobytes())
        f.write(bin_trailer(bin_meta))
    write_meta_sidecar(bin_meta, args.dst)

    print(f"wrote {args.dst}")
    print(f"  GRU(in={in_dim}, hidden={hidden}) + 3 heads (llr, mu_tp, mu_fp)")


if __name__ == "__main__":
    main()
