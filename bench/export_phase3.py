"""
Export a trained Phase 3 (two-tower + per-track accumulator) model to a
binary format consumable by the C tracker.

Format (little-endian):
  u32  magic = 0x55503350  ('UP3P' = Utrack Phase-3 Pack)
  u32  version = 1
  u32  obs_in        — input dim of f_obs
  u32  det_in        — input dim of g_det
  u32  pair_in       — pair feature dim into head
  u32  e_dim         — embedding dim (output of each tower)
  u32  tower_hidden  — hidden width of each tower
  u32  head_hidden_0, head_hidden_1  — head hidden widths
  f32  alpha         — EMA coefficient for accumulator updates
  f32  lambda        — multiplier on residual when added to pre_thr_score
  u32  no_skip_flag  — 1 if head must NOT consume current obs's f_obs(); 0 if it does
  --- f_obs MLP (3 layers: in → hidden, hidden → e_dim) ---
  f32  W0[hidden × obs_in], B0[hidden]
  f32  W1[e_dim × hidden],   B1[e_dim]
  --- g_det MLP (same shape) ---
  f32  W0[hidden × det_in],  B0[hidden]
  f32  W1[e_dim × hidden],   B1[e_dim]
  --- h MLP (3 hidden layers + linear out: in_h → 32 → 16 → 1) ---
  f32  W0[head_hidden_0 × (2*e_dim + pair_in)], B0[head_hidden_0]
  f32  W1[head_hidden_1 × head_hidden_0],       B1[head_hidden_1]
  f32  W2[1 × head_hidden_1],                   B2[1]
  --- Normalisation ---
  u32  n_obs_norm = obs_in
  f32  obs_mean[obs_in], obs_std[obs_in]
  u32  n_det_norm = det_in
  f32  det_mean[det_in], det_std[det_in]
  u32  n_pair_norm = pair_in
  f32  pair_mean[pair_in], pair_std[pair_in]

Total weight count for default config (obs_in=12, det_in=6, pair_in=15,
e_dim=16, tower_hidden=24, head_h0=32, head_h1=16):
  f_obs: 12*24+24 + 16*24+16 = 712
  g_det:  6*24+24 + 16*24+16 = 568
  h:    47*32+32 + 16*32+16 + 1*16+1 = 2081
  norm: 2*(12+6+15) = 66
  total ~= 3361 + 66 = 3427 floats

C-side runtime cost (per frame, T tracks × D detections):
  - precompute g_det once per detection:        D * (6*24 + 24*16)         =  D * 528
  - precompute f_obs only on matched pairs:     M * (12*24 + 24*16)        =  M * 672
  - per-pair score (the hot inner loop):        T*D * (47*32 + 32*16 + 16) =  T*D * 2080
  - per-match accumulator update:               M * (e_dim adds)           =  M * 16

  Compared to Phase 2 prod (T*D * 1296), this is ~1.6× the per-pair work.
  Mitigations available if the head is the bottleneck: shrink head, drop
  one hidden layer (47 → 16 → 1 = 786 MACs), or keep the int8 SIMD path
  (Phase 4) on the table.
"""
from __future__ import annotations

import argparse
import struct

import numpy as np
import torch

from bench.train_phase3 import TwoTower


MAGIC = 0x55503350  # 'UP3P'
VERSION = 1


def write_mlp(out, weights: list[np.ndarray], biases: list[np.ndarray]):
    """Write a sequence of (W, b) float32 layers. W is (out, in) row-major."""
    for W, b in zip(weights, biases):
        out.write(W.astype(np.float32).tobytes(order="C"))
        out.write(b.astype(np.float32).tobytes(order="C"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--no_skip", action="store_true")
    args = p.parse_args()

    ckpt = torch.load(args.inp, map_location="cpu", weights_only=False)
    obs_in = ckpt["obs_in"]; det_in = ckpt["det_in"]; pair_in = ckpt["pair_in"]
    e_dim = ckpt["e_dim"]; tower_hidden = ckpt["tower_hidden"]
    alpha = ckpt["alpha"]; lam = ckpt["lambda"]

    model = TwoTower(obs_in=obs_in, det_in=det_in, pair_in=pair_in,
                     e_dim=e_dim, hidden=tower_hidden)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Pull out weights from each Sequential. `MLP` was 3 layers in code:
    # Linear, ReLU, Linear (no ReLU on output).
    def extract_mlp(mlp_seq: torch.nn.Sequential):
        Ws, Bs = [], []
        for layer in mlp_seq:
            if isinstance(layer, torch.nn.Linear):
                Ws.append(layer.weight.detach().numpy())
                Bs.append(layer.bias.detach().numpy())
        return Ws, Bs

    fobs_W, fobs_B = extract_mlp(model.f_obs.net)
    gdet_W, gdet_B = extract_mlp(model.g_det.net)
    h_W, h_B = extract_mlp(model.h.net)

    # h is 4 layers in our def: in→32, 32→16, 16→1 — that's 3 Linears.
    head_h0 = h_W[0].shape[0]   # 32
    head_h1 = h_W[1].shape[0]   # 16

    no_skip = 1 if args.no_skip else 0

    obs_mean = np.array(ckpt["obs_mean"], dtype=np.float32)
    obs_std  = np.array(ckpt["obs_std"], dtype=np.float32)
    det_mean = np.array(ckpt["det_mean"], dtype=np.float32)
    det_std  = np.array(ckpt["det_std"], dtype=np.float32)
    pair_mean = np.array(ckpt["pair_mean"], dtype=np.float32)
    pair_std  = np.array(ckpt["pair_std"], dtype=np.float32)

    from bench._artefact_meta import make_pt_meta, bin_trailer, write_meta_sidecar
    pt_meta = ckpt.get("_meta")
    bin_meta = make_pt_meta(
        artefact_kind="match_cost_two_tower_bin",
        args=args,
        hparams={
            "obs_in": obs_in, "det_in": det_in, "pair_in": pair_in,
            "e_dim": e_dim, "tower_hidden": tower_hidden,
            "head_h0": head_h0, "head_h1": head_h1,
            "alpha": float(alpha), "lambda": float(lam), "no_skip": int(no_skip),
            "magic": "UP3P", "version": VERSION,
        },
        dataset_info={"source_pt": args.inp, "source_pt_meta": pt_meta},
    )

    with open(args.out, "wb") as out:
        out.write(struct.pack("<II", MAGIC, VERSION))
        out.write(struct.pack("<IIIIIII", obs_in, det_in, pair_in,
                              e_dim, tower_hidden, head_h0, head_h1))
        out.write(struct.pack("<ffI", float(alpha), float(lam), no_skip))
        write_mlp(out, fobs_W, fobs_B)
        write_mlp(out, gdet_W, gdet_B)
        write_mlp(out, h_W, h_B)
        out.write(struct.pack("<I", obs_in))
        out.write(obs_mean.tobytes()); out.write(obs_std.tobytes())
        out.write(struct.pack("<I", det_in))
        out.write(det_mean.tobytes()); out.write(det_std.tobytes())
        out.write(struct.pack("<I", pair_in))
        out.write(pair_mean.tobytes()); out.write(pair_std.tobytes())
        out.write(bin_trailer(bin_meta))
    write_meta_sidecar(bin_meta, args.out)

    print(f"wrote {args.out}")
    print(f"  obs_in={obs_in} det_in={det_in} pair_in={pair_in} "
          f"e_dim={e_dim} tower_hidden={tower_hidden} "
          f"head_h0={head_h0} head_h1={head_h1}")
    print(f"  alpha={alpha} lambda={lam} no_skip={no_skip}")
    n_w = sum(w.size for w in fobs_W + gdet_W + h_W)
    n_b = sum(b.size for b in fobs_B + gdet_B + h_B)
    print(f"  total weights={n_w}  biases={n_b}")


if __name__ == "__main__":
    main()
