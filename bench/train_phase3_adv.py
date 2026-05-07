"""
Phase 3 advanced aggregators — non-fixed-α variants of the per-track
accumulator, motivated by the observation that fixed-α EMA forces all
matched obs to contribute equally regardless of how informative they
are. Goal: let *salient* obs dominate the accumulator, suppressing
noisy/ambiguous obs.

Variants implemented (`--aggregator`):

  ema_fixed     — current production: e ← (1-α)·e + α·f(obs)
  ema_gated     — scalar gate per obs: gate_t = σ(MLP(f(obs)));
                  e ← (1-gate_t)·e + gate_t·f(obs)
  ema_perdim    — per-dim gate: gate_t ∈ R^{e_dim}, σ-activated;
                  e ← (1-gate_t) ⊙ e + gate_t ⊙ f(obs)
  gru           — GRU cell over f(obs) input, hidden state = e_track
  maxpool       — running elementwise max over a saliency-weighted
                  f(obs) ; e ← max(decay·e, salience·f(obs))

All are drop-in replacements for the fixed-α EMA. Inference cost in C
is small for ema_gated/ema_perdim (one extra small MLP per matched
obs); larger for gru (full update gate); maxpool is cheapest.

Output checkpoint format extends train_phase3.py's:
  - adds 'aggregator' name and aggregator state_dict keys
  - same obs/det/pair feature names so corpus generators stay compatible
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from bench.explore_score import auc
from bench.train_phase3 import (
    OBS_FEATURE_NAMES, DET_FEATURE_NAMES, PAIR_FEATURE_NAMES,
    build_obs_matrix, build_det_matrix, build_pair_matrix,
    standardise, apply_norm, MLP, TwoTower,
)


# ------------- aggregators -------------

class FixedEMA(nn.Module):
    """e ← (1-α)·e + α·f(obs); first matched obs initialises e to f(obs)."""
    def __init__(self, e_dim: int, alpha: float):
        super().__init__()
        self.e_dim = e_dim
        self.alpha = alpha
        self.kind = "ema_fixed"

    def step(self, e: torch.Tensor, f_obs: torch.Tensor, seen: torch.Tensor) -> torch.Tensor:
        # seen: float tensor 0/1 indicating whether the track has had any prior obs.
        # When seen=0, we initialise e=f_obs (no prior history). When seen=1, EMA.
        a = self.alpha
        ema = (1.0 - a) * e + a * f_obs
        return torch.where(seen.unsqueeze(-1) > 0.5, ema, f_obs)


class GatedEMA(nn.Module):
    """Scalar gate per obs from an MLP on f_obs. Bounded in (0, 1) via sigmoid.

    Initialised at training start so the gate sits near the fixed-α reference
    point — this gives a like-for-like ablation against ema_fixed.
    """
    def __init__(self, e_dim: int, alpha_init: float = 0.2, hidden: int = 16):
        super().__init__()
        self.e_dim = e_dim
        self.kind = "ema_gated"
        self.gate = nn.Sequential(
            nn.Linear(e_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1)
        )
        # Initialise output bias so sigmoid(b) ≈ alpha_init
        b0 = math.log(alpha_init / max(1e-6, 1.0 - alpha_init))
        with torch.no_grad():
            self.gate[-1].weight.mul_(0.01)
            self.gate[-1].bias.fill_(b0)

    def step(self, e: torch.Tensor, f_obs: torch.Tensor, seen: torch.Tensor) -> torch.Tensor:
        a = torch.sigmoid(self.gate(f_obs))  # (B, 1)
        ema = (1.0 - a) * e + a * f_obs
        return torch.where(seen.unsqueeze(-1) > 0.5, ema, f_obs)


class PerDimEMA(nn.Module):
    """Per-dim gate: each e_dim coordinate has its own mix weight,
    function of f_obs. Lets some dims hold older info (small gate)
    while others adapt fast (large gate)."""
    def __init__(self, e_dim: int, alpha_init: float = 0.2, hidden: int = 24):
        super().__init__()
        self.e_dim = e_dim
        self.kind = "ema_perdim"
        self.gate = nn.Sequential(
            nn.Linear(e_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, e_dim)
        )
        b0 = math.log(alpha_init / max(1e-6, 1.0 - alpha_init))
        with torch.no_grad():
            self.gate[-1].weight.mul_(0.01)
            self.gate[-1].bias.fill_(b0)

    def step(self, e: torch.Tensor, f_obs: torch.Tensor, seen: torch.Tensor) -> torch.Tensor:
        a = torch.sigmoid(self.gate(f_obs))  # (B, e_dim)
        ema = (1.0 - a) * e + a * f_obs
        return torch.where(seen.unsqueeze(-1) > 0.5, ema, f_obs)


class GRUAccumulator(nn.Module):
    """Standard GRU cell over f_obs. Hidden state = e_track."""
    def __init__(self, e_dim: int, alpha_init: float = 0.2):
        super().__init__()
        self.e_dim = e_dim
        self.kind = "gru"
        self.cell = nn.GRUCell(e_dim, e_dim)
        # Bias the update gate so it leaks similar to alpha_init (rough init)
        # GRUCell weights live in (W_ih, W_hh) of shape (3*e_dim, e_dim).
        # The first e_dim block is reset gate, second is update gate, third is new gate.
        with torch.no_grad():
            b_ih = self.cell.bias_ih
            b_hh = self.cell.bias_hh
            # Update gate bias — sigmoid(b) ≈ alpha_init
            target = math.log(alpha_init / max(1e-6, 1.0 - alpha_init))
            b_ih[self.e_dim:2*self.e_dim].fill_(target)
            b_hh[self.e_dim:2*self.e_dim].fill_(0.0)

    def step(self, e: torch.Tensor, f_obs: torch.Tensor, seen: torch.Tensor) -> torch.Tensor:
        new = self.cell(f_obs, e)
        return torch.where(seen.unsqueeze(-1) > 0.5, new, f_obs)


class MaxPoolDecay(nn.Module):
    """Salient max-pool: e_d ← max(decay · e_d, salience · f_obs_d).

    Lets a single high-confidence obs lock a feature in; quiet obs decay.
    """
    def __init__(self, e_dim: int, alpha_init: float = 0.2):
        super().__init__()
        self.e_dim = e_dim
        self.kind = "maxpool"
        self.decay = nn.Parameter(torch.tensor(0.95))   # learned scalar in (0,1) via sigmoid
        # Salience per obs is a scalar gate >= 0 from f_obs.
        self.sal = nn.Sequential(
            nn.Linear(e_dim, 16), nn.ReLU(inplace=True), nn.Linear(16, 1)
        )
        b0 = math.log(0.5 / 0.5)  # sigmoid(0) = 0.5 init
        with torch.no_grad():
            self.sal[-1].weight.mul_(0.01)
            self.sal[-1].bias.fill_(b0)

    def step(self, e: torch.Tensor, f_obs: torch.Tensor, seen: torch.Tensor) -> torch.Tensor:
        d = torch.sigmoid(self.decay)         # (,)
        s = torch.sigmoid(self.sal(f_obs))     # (B, 1)
        # Element-wise max of decayed history and salience-weighted current obs
        new = torch.maximum(d * e, s * f_obs)
        return torch.where(seen.unsqueeze(-1) > 0.5, new, f_obs)


def make_aggregator(name: str, e_dim: int, alpha_init: float) -> nn.Module:
    name = name.lower()
    if name == "ema_fixed":
        return FixedEMA(e_dim, alpha_init)
    if name == "ema_gated":
        return GatedEMA(e_dim, alpha_init)
    if name == "ema_perdim":
        return PerDimEMA(e_dim, alpha_init)
    if name == "gru":
        return GRUAccumulator(e_dim, alpha_init)
    if name == "maxpool":
        return MaxPoolDecay(e_dim, alpha_init)
    raise ValueError(f"unknown aggregator: {name}")


# ------------- model -------------

class TwoTowerAdv(nn.Module):
    """Same towers as TwoTower, plus a swappable aggregator."""
    def __init__(self, obs_in: int, det_in: int, pair_in: int,
                 e_dim: int, hidden: int, aggregator: str,
                 alpha_init: float):
        super().__init__()
        self.f_obs = MLP([obs_in, hidden, e_dim])
        self.g_det = MLP([det_in, hidden, e_dim])
        self.h = MLP([2 * e_dim + pair_in, 32, 16, 1], last_zero=True)
        self.aggregator = make_aggregator(aggregator, e_dim, alpha_init)
        self.e_dim = e_dim

    def encode_obs(self, x: torch.Tensor) -> torch.Tensor:
        return self.f_obs(x)

    def encode_det(self, x: torch.Tensor) -> torch.Tensor:
        return self.g_det(x)

    def head(self, e_track: torch.Tensor, e_det: torch.Tensor,
             pair_feats: torch.Tensor) -> torch.Tensor:
        return self.h(torch.cat([e_track, e_det, pair_feats], dim=-1)).squeeze(-1)


# ------------- e_track replay (advanced) -------------

@torch.no_grad()
def _precompute_aggregator_inputs(
    model: TwoTowerAdv, obs_x: np.ndarray, device: str, batch_size: int = 8192,
):
    """Run f_obs (and any gate that depends only on f_obs) on all pairs in
    batched torch, return numpy arrays. The per-pair recurrence loop then
    runs in pure numpy with no GPU sync per pair.

    Returns dict with keys depending on aggregator kind:
      kind='ema_fixed':   {f_all, alpha (scalar)}
      kind='ema_gated':   {f_all, gate (n,1)}
      kind='ema_perdim':  {f_all, gate (n,e_dim)}
      kind='maxpool':     {f_all, sal (n,1), decay (scalar)}
      kind='gru':         {f_all, gru_weights} — handled in numpy step path
    """
    e_dim = model.e_dim
    n = len(obs_x)
    f_all = np.zeros((n, e_dim), dtype=np.float32)
    obs_t = torch.from_numpy(obs_x.astype(np.float32))
    for i in range(0, n, batch_size):
        b = obs_t[i:i+batch_size].to(device)
        f_all[i:i+batch_size] = model.f_obs(b).cpu().numpy()

    agg = model.aggregator
    kind = agg.kind
    out = {"kind": kind, "f_all": f_all}

    if kind == "ema_fixed":
        out["alpha"] = float(agg.alpha)
    elif kind in ("ema_gated", "ema_perdim"):
        gate_chunks = []
        f_t = torch.from_numpy(f_all)
        for i in range(0, n, batch_size):
            b = f_t[i:i+batch_size].to(device)
            g = torch.sigmoid(agg.gate(b)).cpu().numpy()
            gate_chunks.append(g)
        out["gate"] = np.concatenate(gate_chunks, axis=0).astype(np.float32)
    elif kind == "maxpool":
        sal_chunks = []
        f_t = torch.from_numpy(f_all)
        for i in range(0, n, batch_size):
            b = f_t[i:i+batch_size].to(device)
            s = torch.sigmoid(agg.sal(b)).cpu().numpy()
            sal_chunks.append(s)
        out["sal"] = np.concatenate(sal_chunks, axis=0).astype(np.float32)
        out["decay"] = float(torch.sigmoid(agg.decay).item())
    elif kind == "gru":
        # Extract GRU weights to numpy. nn.GRUCell stores:
        #   weight_ih: (3*H, I), weight_hh: (3*H, H)
        #   bias_ih: (3*H,), bias_hh: (3*H,)
        # Order within each block: (reset, update, new).
        cell = agg.cell
        out["w_ih"] = cell.weight_ih.detach().cpu().numpy()
        out["w_hh"] = cell.weight_hh.detach().cpu().numpy()
        out["b_ih"] = cell.bias_ih.detach().cpu().numpy()
        out["b_hh"] = cell.bias_hh.detach().cpu().numpy()
        out["e_dim"] = e_dim
    else:
        raise ValueError(f"unknown aggregator kind {kind}")
    return out


def _gru_step_np(e_prev: np.ndarray, f_obs: np.ndarray, ws: dict) -> np.ndarray:
    """Single GRU step in numpy. e_prev (e_dim,), f_obs (e_dim,)."""
    H = ws["e_dim"]
    w_ih = ws["w_ih"]; w_hh = ws["w_hh"]
    b_ih = ws["b_ih"]; b_hh = ws["b_hh"]
    gi = w_ih @ f_obs + b_ih
    gh = w_hh @ e_prev + b_hh
    r = 1.0 / (1.0 + np.exp(-(gi[:H] + gh[:H])))
    z = 1.0 / (1.0 + np.exp(-(gi[H:2*H] + gh[H:2*H])))
    n = np.tanh(gi[2*H:] + r * gh[2*H:])
    return (1.0 - z) * n + z * e_prev


def compute_etrack_for_pairs_adv(
    model: TwoTowerAdv, obs_x: np.ndarray,
    scene_ids: np.ndarray, track_ids: np.ndarray,
    frame_times: np.ndarray, det_indices: np.ndarray,
    was_matched: np.ndarray,
    device: str, batch_size: int = 8192,
) -> np.ndarray:
    """Replay aggregator over each track's matched obs sequence.

    Strategy: precompute f_obs and any gate that depends only on f_obs in
    batched torch, then walk pairs in numpy. No GPU sync per pair.

    For each (scene, track), record e_track BEFORE the current pair
    contributes (NNUE timing). Update aggregator state on matched obs
    only.
    """
    e_dim = model.e_dim
    ws = _precompute_aggregator_inputs(model, obs_x, device, batch_size)
    f_all = ws["f_all"]
    kind = ws["kind"]
    n = len(obs_x)

    order = np.lexsort((det_indices, frame_times, track_ids, scene_ids))
    e_track_per_pair = np.zeros((n, e_dim), dtype=np.float32)

    cur_scene = -1
    cur_track = -1
    cur_e = np.zeros(e_dim, dtype=np.float32)
    cur_seen = False

    if kind == "ema_fixed":
        alpha = ws["alpha"]
        for k in order:
            sid = scene_ids[k]; tid = track_ids[k]
            if sid != cur_scene or tid != cur_track:
                cur_scene = sid; cur_track = tid
                cur_e = np.zeros(e_dim, dtype=np.float32); cur_seen = False
            if cur_seen:
                e_track_per_pair[k] = cur_e
            if was_matched[k]:
                f = f_all[k]
                cur_e = (1.0 - alpha) * cur_e + alpha * f if cur_seen else f
                cur_seen = True
    elif kind in ("ema_gated", "ema_perdim"):
        gate = ws["gate"]
        for k in order:
            sid = scene_ids[k]; tid = track_ids[k]
            if sid != cur_scene or tid != cur_track:
                cur_scene = sid; cur_track = tid
                cur_e = np.zeros(e_dim, dtype=np.float32); cur_seen = False
            if cur_seen:
                e_track_per_pair[k] = cur_e
            if was_matched[k]:
                f = f_all[k]; a = gate[k]
                cur_e = (1.0 - a) * cur_e + a * f if cur_seen else f
                cur_seen = True
    elif kind == "maxpool":
        sal = ws["sal"]; d = ws["decay"]
        for k in order:
            sid = scene_ids[k]; tid = track_ids[k]
            if sid != cur_scene or tid != cur_track:
                cur_scene = sid; cur_track = tid
                cur_e = np.zeros(e_dim, dtype=np.float32); cur_seen = False
            if cur_seen:
                e_track_per_pair[k] = cur_e
            if was_matched[k]:
                f = f_all[k]; s = sal[k]
                cur_e = np.maximum(d * cur_e, s * f) if cur_seen else (s * f)
                cur_seen = True
    elif kind == "gru":
        for k in order:
            sid = scene_ids[k]; tid = track_ids[k]
            if sid != cur_scene or tid != cur_track:
                cur_scene = sid; cur_track = tid
                cur_e = np.zeros(e_dim, dtype=np.float32); cur_seen = False
            if cur_seen:
                e_track_per_pair[k] = cur_e
            if was_matched[k]:
                f = f_all[k]
                cur_e = _gru_step_np(cur_e, f, ws) if cur_seen else f
                cur_seen = True
    else:
        raise ValueError(f"unknown aggregator kind {kind}")
    return e_track_per_pair


# ------------- training -------------

def train(args):
    print(f"loading data from {args.data_dir}...")
    tr = np.load(f"{args.data_dir}/pairs_train.npz", allow_pickle=True)
    va = np.load(f"{args.data_dir}/pairs_val.npz", allow_pickle=True)

    r_tr = tr["records"]; y_tr = tr["labels"].astype(np.float32)
    r_va = va["records"]; y_va = va["labels"].astype(np.float32)

    s_tr = tr["scene_ids"]; s_va = va["scene_ids"]
    n_tr = dict(tr["scene_id_to_name"])
    n_va = dict(va["scene_id_to_name"])

    def fam_of(name: str) -> str:
        for prefix in ("MOT17", "MOT20", "UKof", "INof"):
            if name.startswith(prefix): return prefix
        return "other"

    fam_tr = np.array([fam_of(n_tr.get(int(s), "?")) for s in s_tr])
    fam_target = {"MOT17": args.mot17_boost, "MOT20": args.mot20_weight,
                  "UKof": 1.0, "INof": 1.0, "other": 1.0}
    w = np.ones(len(s_tr), dtype=np.float32)
    fam_counts = {f: int((fam_tr == f).sum()) for f in np.unique(fam_tr)}
    for f, n in fam_counts.items():
        w[fam_tr == f] = fam_target.get(f, 1.0) / max(1, n)
    w *= len(s_tr) / w.sum()

    print("building features...")
    obs_tr = build_obs_matrix(r_tr); obs_va = build_obs_matrix(r_va)
    det_tr = build_det_matrix(r_tr); det_va = build_det_matrix(r_va)
    pair_tr = build_pair_matrix(r_tr); pair_va = build_pair_matrix(r_va)

    obs_tr_n, obs_mean, obs_std = standardise(obs_tr)
    det_tr_n, det_mean, det_std = standardise(det_tr)
    pair_tr_n, pair_mean, pair_std = standardise(pair_tr)
    obs_va_n = apply_norm(obs_va, obs_mean, obs_std)
    det_va_n = apply_norm(det_va, det_mean, det_std)
    pair_va_n = apply_norm(pair_va, pair_mean, pair_std)

    pre_tr = r_tr["pre_thr_score"].astype(np.float32)
    pre_va = r_va["pre_thr_score"].astype(np.float32)
    base_va = auc(pre_va, y_va.astype(bool))
    print(f"  baseline val AUC (no NN): {base_va:.5f}")

    sid_tr = s_tr.astype(np.int64); tid_tr = r_tr["track_id"].astype(np.int64)
    sid_va = s_va.astype(np.int64); tid_va = r_va["track_id"].astype(np.int64)
    ft_tr = r_tr["frame_time"].astype(np.float32)
    ft_va = r_va["frame_time"].astype(np.float32)
    di_tr = r_tr["det_index"].astype(np.int32)
    di_va = r_va["det_index"].astype(np.int32)
    wm_tr = r_tr["was_matched"].astype(np.int32)
    wm_va = r_va["was_matched"].astype(np.int32)

    rev_va = {v: int(k) for k, v in n_va.items()}
    sid_mot17_02 = rev_va.get("MOT17-02")
    sid_oh008 = rev_va.get("UKof_LD_Indoor_Light_OHcam_008")
    m_mot17_02 = (s_va == sid_mot17_02) if sid_mot17_02 is not None else np.zeros(len(s_va), bool)
    m_oh008 = (s_va == sid_oh008) if sid_oh008 is not None else np.zeros(len(s_va), bool)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TwoTowerAdv(obs_in=obs_tr_n.shape[1], det_in=det_tr_n.shape[1],
                        pair_in=pair_tr_n.shape[1], e_dim=args.e_dim,
                        hidden=args.tower_hidden,
                        aggregator=args.aggregator, alpha_init=args.alpha).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  TwoTowerAdv params: {n_params}  agg={args.aggregator}  e_dim={args.e_dim}  device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    obs_tr_g = torch.from_numpy(obs_tr_n).to(device)
    det_tr_g = torch.from_numpy(det_tr_n).to(device)
    pair_tr_g = torch.from_numpy(pair_tr_n).to(device)
    pre_tr_g = torch.from_numpy(pre_tr).to(device)
    y_tr_g = torch.from_numpy(y_tr).to(device)
    w_tr_g = torch.from_numpy(w).to(device)

    obs_va_g = torch.from_numpy(obs_va_n).to(device)
    det_va_g = torch.from_numpy(det_va_n).to(device)
    pair_va_g = torch.from_numpy(pair_va_n).to(device)
    pre_va_g = torch.from_numpy(pre_va).to(device)

    bs = args.batch_size
    lam = args.lam
    n = obs_tr_g.shape[0]
    print(f"\nepochs={args.epochs} bs={bs} lam={lam}")
    print(f"{'epoch':>5s}  {'tr_loss':>8s}  {'va_AUC':>8s}  "
          f"{'mot17-02':>9s}  {'OHcam_008':>10s}  {'Δ vs base':>10s}")

    best_va = base_va
    best_state = None

    for ep in range(args.epochs):
        # Stage 1: replay e_track with current f_obs (frozen wrt grad).
        model.eval()
        e_track_tr = compute_etrack_for_pairs_adv(
            model, obs_tr_n, sid_tr, tid_tr, ft_tr, di_tr, wm_tr, device)
        e_track_va = compute_etrack_for_pairs_adv(
            model, obs_va_n, sid_va, tid_va, ft_va, di_va, wm_va, device)
        e_track_tr_g = torch.from_numpy(e_track_tr).to(device)
        e_track_va_g = torch.from_numpy(e_track_va).to(device)

        # Stage 2: SGD with skip path so f_obs and aggregator both train.
        model.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            ob = obs_tr_g[idx]; dt = det_tr_g[idx]; pf = pair_tr_g[idx]
            pb = pre_tr_g[idx]; yb = y_tr_g[idx]; wb = w_tr_g[idx]
            et = e_track_tr_g[idx]

            e_det = model.encode_det(dt)
            e_obs_now = model.encode_obs(ob)
            if args.no_skip:
                e_combined = et
            else:
                # Skip: blend historical e_track with one aggregator step on
                # *current* obs. This trains both f_obs and the aggregator.
                seen = torch.ones(et.shape[0], device=device)  # treat history as present
                e_combined = model.aggregator.step(et, e_obs_now, seen)
            logit_res = model.head(e_combined, e_det, pf)
            logit = pb + lam * logit_res
            losses = bce(logit, yb)
            loss = (losses * wb).sum() / wb.sum().clamp(min=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss) * ob.shape[0]

        # Eval
        model.eval()
        with torch.no_grad():
            e_det_va = model.encode_det(det_va_g)
            if args.no_skip:
                e_combined_va = e_track_va_g
            else:
                e_obs_va = model.encode_obs(obs_va_g)
                seen = torch.ones(e_track_va_g.shape[0], device=device)
                e_combined_va = model.aggregator.step(e_track_va_g, e_obs_va, seen)
            logit_res_va = model.head(e_combined_va, e_det_va, pair_va_g)
            logit_va = (pre_va_g + lam * logit_res_va).cpu().numpy()
        va_auc = auc(logit_va, y_va.astype(bool))
        a_mot17_02 = auc(logit_va[m_mot17_02], y_va[m_mot17_02].astype(bool)) if m_mot17_02.sum() > 50 else 0.0
        a_oh008 = auc(logit_va[m_oh008], y_va[m_oh008].astype(bool)) if m_oh008.sum() > 50 else 0.0

        marker = "  ★" if va_auc > best_va else ""
        if va_auc > best_va:
            best_va = va_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"{ep:5d}  {total_loss/n:8.5f}  {va_auc:8.5f}  "
              f"{a_mot17_02:9.5f}  {a_oh008:10.5f}  "
              f"{va_auc-base_va:+10.5f}{marker}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final: replay with the best model and report per-scene
    model.eval()
    with torch.no_grad():
        e_track_va = compute_etrack_for_pairs_adv(
            model, obs_va_n, sid_va, tid_va, ft_va, di_va, wm_va, device)
        e_track_va_g = torch.from_numpy(e_track_va).to(device)
        e_det_va = model.encode_det(det_va_g)
        if args.no_skip:
            e_combined_va = e_track_va_g
        else:
            e_obs_va = model.encode_obs(obs_va_g)
            seen = torch.ones(e_track_va_g.shape[0], device=device)
            e_combined_va = model.aggregator.step(e_track_va_g, e_obs_va, seen)
        logit_res_va = model.head(e_combined_va, e_det_va, pair_va_g)
        logit_va = (pre_va_g + lam * logit_res_va).cpu().numpy()

    print(f"\n=== Final (best epoch) val AUC = {best_va:.5f}  Δ={best_va-base_va:+.5f} ===")
    print(f"  Per-scene AUC:")
    for sid in np.unique(s_va):
        m = s_va == sid
        n_s = int(m.sum())
        if n_s < 200: continue
        a_b = auc(pre_va[m], y_va[m].astype(bool))
        a_n = auc(logit_va[m], y_va[m].astype(bool))
        nm = str(n_va.get(int(sid), "?"))[:38]
        print(f"    {nm:40s}  n={n_s:6d}  base={a_b:.5f}  +nn={a_n:.5f}  Δ={a_n-a_b:+.5f}")

    if args.save:
        torch.save({
            "state_dict": model.state_dict(),
            "obs_mean": obs_mean.tolist(), "obs_std": obs_std.tolist(),
            "det_mean": det_mean.tolist(), "det_std": det_std.tolist(),
            "pair_mean": pair_mean.tolist(), "pair_std": pair_std.tolist(),
            "obs_in": obs_tr_n.shape[1], "det_in": det_tr_n.shape[1],
            "pair_in": pair_tr_n.shape[1], "e_dim": args.e_dim,
            "tower_hidden": args.tower_hidden,
            "alpha": args.alpha, "lambda": lam,
            "aggregator": args.aggregator,
            "obs_feature_names": OBS_FEATURE_NAMES,
            "det_feature_names": DET_FEATURE_NAMES,
            "pair_feature_names": PAIR_FEATURE_NAMES,
            "best_va_auc": float(best_va),
        }, args.save)
        print(f"\nsaved → {args.save}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--mot17_boost", type=float, default=1.5)
    p.add_argument("--mot20_weight", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.2,
                   help="initial mixing weight (also fixed alpha for ema_fixed)")
    p.add_argument("--e_dim", type=int, default=16)
    p.add_argument("--tower_hidden", type=int, default=24)
    p.add_argument("--aggregator", default="ema_gated",
                   choices=["ema_fixed", "ema_gated", "ema_perdim", "gru", "maxpool"])
    p.add_argument("--no_skip", action="store_true")
    p.add_argument("--save", default=None)
    p.add_argument("--data_dir", default="bench/data")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
