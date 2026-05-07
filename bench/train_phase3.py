"""
Phase 3 / Proposal B (UTRACK_NN.md): two-tower model with per-track
accumulator. **Offline, frozen-accumulator** training — no BPTT, no C-side
state. Goal is to *gate* whether the architecture is worth implementing in
the C runtime.

Setup (all in PyTorch, val-AUC compared against the Phase 2 residual MLP):

  obs_features (per matched (track, det)):
    [reid_cos_raw, reid_z, kf_d2, of_score, kf_score, sim_term,
     ocm_cos, track_speed, log(observations+1), log(num_missed+1),
     pose_kp_visible, det_conf, prev_det_conf]   → 13 dims

  det_features (per detection):
    [det_conf, det_w, det_h, det_aspect, pose_kp_visible]
    → 5 dims

  pair_features (per (track, det) pair, fed into head alongside the
  embeddings):
    [iou, kf_d2, ocm_cos, conf_delta, size_ratio, h_ratio, a_ratio,
     reid_z·valid, pre_thr_score, of_score, kf_score, sim_term,
     pass_0, pass_1, pass_2]   → 15 dims

  Architecture:
    f_obs : 13 → 24 → 16        (per matched obs; emits update vector)
    g_det : 5 → 16 → 16         (per detection; emits e_det)
    h     : 16+16+15 = 47 → 32 → 16 → 1   (per pair; emits logit residual)

  Per-track accumulator: e_track[16], EMA over f_obs(matched obs).
    e_track_t = (1-α) e_track_{t-1} + α · f_obs(obs_t)        for matched obs
    e_track_0 = f_obs(first matched obs)                      (no prior)

  At training time, e_track is computed one epoch *behind* (computed at
  start of each epoch using the f_obs from the end of the prior epoch,
  detached) — this is the "frozen-accumulator pretrain" stage. No BPTT.

Output is logit *residual* added to pre_thr_score with a learned λ:
    score = pre_thr_score + λ · h(...)

This matches the Phase 2 framing so the two are comparable on val AUC.
"""
from __future__ import annotations

import argparse
import json
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from bench.explore_score import auc


OBS_FEATURE_NAMES = [
    "reid_cos_raw", "reid_z", "kf_d2", "of_score", "kf_score", "sim_term",
    "ocm_cos", "track_speed", "log_observations", "log_num_missed",
    "pose_kp_visible", "det_conf", "prev_det_conf",
]

# v2 schema (face/subbox) features. The model gets to see them only
# when the corpus carries them — older v7/v8 corpora will fail the
# schema check on these fields, which is the desired behaviour.
OBS_FEATURE_NAMES_V2 = OBS_FEATURE_NAMES + [
    "det_subbox_conf", "track_subbox_conf", "det_fiqa_score",
]

DET_FEATURE_NAMES = [
    "det_conf", "det_w", "det_h", "det_aspect", "pose_kp_visible",
]

PAIR_FEATURE_NAMES = [
    "iou", "kf_d2", "ocm_cos", "conf_delta", "size_ratio", "h_ratio",
    "a_ratio", "reid_z_masked", "pre_thr_score", "of_score", "kf_score",
    "sim_term", "pass_0", "pass_1", "pass_2",
]


def compute_iou(records: np.ndarray) -> np.ndarray:
    """Compute IoU between track box (x0,y0,x1,y1) and det box."""
    tx0 = records["track_x0"]; ty0 = records["track_y0"]
    tx1 = records["track_x1"]; ty1 = records["track_y1"]
    dx0 = records["det_x0"]; dy0 = records["det_y0"]
    dx1 = records["det_x1"]; dy1 = records["det_y1"]
    ix0 = np.maximum(tx0, dx0); iy0 = np.maximum(ty0, dy0)
    ix1 = np.minimum(tx1, dx1); iy1 = np.minimum(ty1, dy1)
    iw = np.maximum(0, ix1 - ix0); ih = np.maximum(0, iy1 - iy0)
    inter = iw * ih
    a_t = np.maximum(0, tx1 - tx0) * np.maximum(0, ty1 - ty0)
    a_d = np.maximum(0, dx1 - dx0) * np.maximum(0, dy1 - dy0)
    union = a_t + a_d - inter
    iou = np.where(union > 1e-9, inter / np.maximum(union, 1e-9), 0.0)
    return iou.astype(np.float32)


def build_obs_matrix(r: np.ndarray, *, with_face: bool = False) -> np.ndarray:
    """f_obs input matrix (raw, not z-scored — model has BatchNorm).

    `with_face=True` appends the v2 face/subbox features
    (det_subbox_conf, track_subbox_conf, det_fiqa_score). The base
    13-feature variant is preserved for old corpora.
    """
    cols = [
        np.nan_to_num(r["reid_cos_raw"]).astype(np.float32),
        np.nan_to_num(r["reid_z"]).astype(np.float32),
        r["kf_d2"].astype(np.float32),
        r["of_score"].astype(np.float32),
        r["kf_score"].astype(np.float32),
        r["sim_term"].astype(np.float32),
        r["ocm_cos"].astype(np.float32),
        r["track_speed"].astype(np.float32),
        np.log1p(r["observations"].astype(np.float32)),
        np.log1p(r["num_missed"].astype(np.float32)),
        r["pose_kp_visible"].astype(np.float32),
        r["det_conf"].astype(np.float32),
        r["prev_det_conf"].astype(np.float32),
    ]
    if with_face:
        cols += [
            r["det_subbox_conf"].astype(np.float32),
            r["track_subbox_conf"].astype(np.float32),
            r["det_fiqa_score"].astype(np.float32),
        ]
    x = np.stack(cols, axis=1)
    x = np.where(np.isnan(x), 0.0, x)
    return x


def build_det_matrix(r: np.ndarray) -> np.ndarray:
    cols = [
        r["det_conf"].astype(np.float32),
        (r["det_x1"] - r["det_x0"]).astype(np.float32),
        (r["det_y1"] - r["det_y0"]).astype(np.float32),
        ((r["det_x1"] - r["det_x0"]) / np.maximum(1e-3, r["det_y1"] - r["det_y0"])).astype(np.float32),
        r["pose_kp_visible"].astype(np.float32),
    ]
    x = np.stack(cols, axis=1)
    x = np.where(np.isnan(x), 0.0, x)
    return x


def build_pair_matrix(r: np.ndarray, *, with_face: bool = False) -> np.ndarray:
    """Pair feature matrix. With `with_face=True`, append v2 features:
    subbox_iou, det_subbox_conf, track_subbox_conf, det_fiqa_score —
    same numbers also fed to f_obs but it can also serve as a direct
    input to the head (skip path).
    """
    iou = compute_iou(r)
    reid_z = np.nan_to_num(r["reid_z"]).astype(np.float32)
    valid = r["reid_stats_valid"].astype(np.float32)
    pid = r["pass_id"].astype(np.int64)
    pid_oh = np.zeros((len(pid), 3), dtype=np.float32)
    pid_oh[np.arange(len(pid)), np.clip(pid, 0, 2)] = 1.0
    cols = [
        iou,
        r["kf_d2"].astype(np.float32),
        r["ocm_cos"].astype(np.float32),
        r["conf_delta"].astype(np.float32),
        r["size_ratio"].astype(np.float32),
        r["h_ratio"].astype(np.float32),
        r["a_ratio"].astype(np.float32),
        reid_z * valid,
        r["pre_thr_score"].astype(np.float32),
        r["of_score"].astype(np.float32),
        r["kf_score"].astype(np.float32),
        r["sim_term"].astype(np.float32),
        pid_oh[:, 0], pid_oh[:, 1], pid_oh[:, 2],
    ]
    if with_face:
        cols += [
            r["subbox_iou"].astype(np.float32),
            r["det_subbox_conf"].astype(np.float32),
            r["track_subbox_conf"].astype(np.float32),
            r["det_fiqa_score"].astype(np.float32),
        ]
    x = np.stack(cols, axis=1)
    x = np.where(np.isnan(x), 0.0, x)
    return x


def standardise(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    safe_std = np.where(std < 1e-6, 1.0, std)
    return (x - mean) / safe_std, mean.astype(np.float32), safe_std.astype(np.float32)


def apply_norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


# ------------- two-tower model -------------

class MLP(nn.Module):
    def __init__(self, dims: List[int], last_zero: bool = False):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i + 1 < len(dims) - 1:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
        if last_zero:
            with torch.no_grad():
                self.net[-1].weight.mul_(0.01)
                self.net[-1].bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoTower(nn.Module):
    def __init__(self, obs_in: int, det_in: int, pair_in: int,
                 e_dim: int = 16, hidden: int = 24):
        super().__init__()
        self.f_obs = MLP([obs_in, hidden, e_dim])      # per matched obs
        self.g_det = MLP([det_in, hidden, e_dim])      # per detection
        self.h = MLP([2 * e_dim + pair_in, 32, 16, 1], last_zero=True)
        self.e_dim = e_dim

    def encode_obs(self, x: torch.Tensor) -> torch.Tensor:
        return self.f_obs(x)

    def encode_det(self, x: torch.Tensor) -> torch.Tensor:
        return self.g_det(x)

    def head(self, e_track: torch.Tensor, e_det: torch.Tensor,
             pair_feats: torch.Tensor) -> torch.Tensor:
        return self.h(torch.cat([e_track, e_det, pair_feats], dim=-1)).squeeze(-1)


# ------------- e_track (frozen accumulator) -------------

@torch.no_grad()
def compute_etrack_for_pairs(model: TwoTower, obs_x: np.ndarray,
                             scene_ids: np.ndarray, track_ids: np.ndarray,
                             frame_times: np.ndarray, det_indices: np.ndarray,
                             was_matched: np.ndarray,
                             alpha: float, device: str,
                             batch_size: int = 8192) -> np.ndarray:
    """For each pair, return the e_track value as of *just before* this pair
    is scored (i.e. EMA over all prior matched pairs in the same track).

    e_track for the very first observation in a track = zeros (we haven't
    seen anything yet). After the first matched obs, it becomes a running
    EMA. Note: a pair scored BEFORE its track has any prior matched obs
    sees e_track=0; after the first match, subsequent scoring sees the
    running value.
    """
    n = len(obs_x)
    # Precompute f_obs only for the matched pairs (others contribute nothing
    # to e_track; we still need their e_track value for scoring though).
    e_dim = model.e_dim
    # Process in batches to avoid GPU OOM
    f_all = np.zeros((n, e_dim), dtype=np.float32)
    obs_t = torch.from_numpy(obs_x.astype(np.float32))
    for i in range(0, n, batch_size):
        b = obs_t[i:i+batch_size].to(device)
        f_all[i:i+batch_size] = model.f_obs(b).cpu().numpy()

    # Walk records in (scene, track, frame, det) order. Build a stable sort
    # key. det_index is given for tie-break within frame.
    order = np.lexsort((det_indices, frame_times, track_ids, scene_ids))

    e_track_per_pair = np.zeros((n, e_dim), dtype=np.float32)

    cur_scene = -1
    cur_track = -1
    cur_e = np.zeros(e_dim, dtype=np.float32)
    cur_seen = False  # have we seen a matched obs for this track yet?

    for k in order:
        sid = scene_ids[k]; tid = track_ids[k]
        if sid != cur_scene or tid != cur_track:
            cur_scene = sid; cur_track = tid
            cur_e = np.zeros(e_dim, dtype=np.float32)
            cur_seen = False
        # Record e_track BEFORE this pair contributes (matches NNUE: score at t,
        # then update for matched obs at t).
        e_track_per_pair[k] = cur_e if cur_seen else np.zeros(e_dim, dtype=np.float32)
        # If this pair was matched, it updates the accumulator for *future*
        # pairs in this track.
        if was_matched[k]:
            if not cur_seen:
                cur_e = f_all[k]   # init at first matched obs
                cur_seen = True
            else:
                cur_e = (1.0 - alpha) * cur_e + alpha * f_all[k]
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

    # Family weighting (mirror v3 / prod recipe)
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

    # Build the three feature matrices
    print("building features...")
    obs_tr = build_obs_matrix(r_tr); obs_va = build_obs_matrix(r_va)
    det_tr = build_det_matrix(r_tr); det_va = build_det_matrix(r_va)
    pair_tr = build_pair_matrix(r_tr); pair_va = build_pair_matrix(r_va)

    # Z-score using train stats
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

    # Per-track sequence keys
    sid_tr = s_tr.astype(np.int64); tid_tr = r_tr["track_id"].astype(np.int64)
    sid_va = s_va.astype(np.int64); tid_va = r_va["track_id"].astype(np.int64)
    ft_tr = r_tr["frame_time"].astype(np.float32)
    ft_va = r_va["frame_time"].astype(np.float32)
    di_tr = r_tr["det_index"].astype(np.int32)
    di_va = r_va["det_index"].astype(np.int32)
    wm_tr = r_tr["was_matched"].astype(np.int32)
    wm_va = r_va["was_matched"].astype(np.int32)

    # Cache val masks for regression scenes
    rev_va = {v: int(k) for k, v in n_va.items()}
    sid_mot17_02 = rev_va.get("MOT17-02")
    sid_oh008 = rev_va.get("UKof_LD_Indoor_Light_OHcam_008")
    m_mot17_02 = (s_va == sid_mot17_02) if sid_mot17_02 is not None else np.zeros(len(s_va), bool)
    m_oh008 = (s_va == sid_oh008) if sid_oh008 is not None else np.zeros(len(s_va), bool)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TwoTower(obs_in=obs_tr_n.shape[1], det_in=det_tr_n.shape[1],
                     pair_in=pair_tr_n.shape[1], e_dim=args.e_dim,
                     hidden=args.tower_hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  TwoTower params: {n_params}  e_dim={args.e_dim}  device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    # Move data to GPU
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
    print(f"\nepochs={args.epochs} bs={bs} lam={lam} alpha={args.alpha}")
    print(f"{'epoch':>5s}  {'tr_loss':>8s}  {'va_AUC':>8s}  "
          f"{'mot17-02':>9s}  {'OHcam_008':>10s}  {'Δ vs base':>10s}")

    best_va = base_va
    best_state = None

    for ep in range(args.epochs):
        # Stage 1: compute e_track for all train and val pairs using current f_obs
        # (frozen, no grad). This is the "frozen accumulator" pretrain.
        model.eval()
        e_track_tr = compute_etrack_for_pairs(
            model, obs_tr_n, sid_tr, tid_tr, ft_tr, di_tr, wm_tr,
            args.alpha, device)
        e_track_va = compute_etrack_for_pairs(
            model, obs_va_n, sid_va, tid_va, ft_va, di_va, wm_va,
            args.alpha, device)
        e_track_tr_g = torch.from_numpy(e_track_tr).to(device)
        e_track_va_g = torch.from_numpy(e_track_va).to(device)

        # Stage 2: SGD over the head + g_det (and indirectly f_obs, since
        # next epoch's e_track uses the updated f_obs).
        model.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            ob = obs_tr_g[idx]; dt = det_tr_g[idx]; pf = pair_tr_g[idx]
            pb = pre_tr_g[idx]; yb = y_tr_g[idx]; wb = w_tr_g[idx]
            et = e_track_tr_g[idx]

            # f_obs trains via the implicit "self obs" — at this pair, the
            # current obs feature is fed through f_obs and used to produce
            # the e_obs for THIS step (separately from e_track which is
            # historical). We use this as an additional skip path so f_obs
            # gets gradient even with frozen accumulator.
            e_det = model.encode_det(dt)
            if args.no_skip:
                e_combined = et
            else:
                # Mix e_track (historical) and e_obs_now (current) — this lets
                # f_obs train without true BPTT, and matches the cheap inference
                # variant that re-uses obs features the C runtime already has.
                e_obs_now = model.encode_obs(ob)
                e_combined = (1.0 - args.alpha) * et + args.alpha * e_obs_now
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
                e_combined_va = (1.0 - args.alpha) * e_track_va_g + args.alpha * e_obs_va
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

    # Final per-scene AUC
    model.eval()
    with torch.no_grad():
        e_track_va = compute_etrack_for_pairs(
            model, obs_va_n, sid_va, tid_va, ft_va, di_va, wm_va,
            args.alpha, device)
        e_track_va_g = torch.from_numpy(e_track_va).to(device)
        e_det_va = model.encode_det(det_va_g)
        if args.no_skip:
            e_combined_va = e_track_va_g
        else:
            e_obs_va = model.encode_obs(obs_va_g)
            e_combined_va = (1.0 - args.alpha) * e_track_va_g + args.alpha * e_obs_va
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
            "obs_feature_names": OBS_FEATURE_NAMES,
            "det_feature_names": DET_FEATURE_NAMES,
            "pair_feature_names": PAIR_FEATURE_NAMES,
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
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--e_dim", type=int, default=16)
    p.add_argument("--tower_hidden", type=int, default=24)
    p.add_argument("--no_skip", action="store_true",
                   help="Disable the train-time skip path; head sees only "
                        "historical e_track (closer to NNUE/spec).")
    p.add_argument("--save", default=None)
    p.add_argument("--data_dir", default="bench/data",
                   help="dir containing pairs_train.npz, pairs_val.npz, feature_norm.json")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
