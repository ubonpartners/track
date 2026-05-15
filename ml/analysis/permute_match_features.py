"""Permutation-importance audit for the match-cost NN.

For each feature column in {obs, det, pair}, shuffle that column across
all val rows and recompute the combined score (pre_thr_score + λ·residual).
The drop in AUC vs the un-shuffled baseline = that feature's importance.

Run against the shipped v10 head + the corpus it was trained on:

    python -m ml.analysis.permute_match_features \
        --ckpt ml/data/phase3_v10_face.pt \
        --val  ml/data/pairs_val.npz

The script reports two numbers per feature:
  - Δ-AUC under training-λ (= ckpt['lambda'], usually 1.0): how much the
    head's *own* signal depends on this feature.
  - Δ-AUC under deployment-λ (= 0.05 in uc_v11.yaml): how much the
    combined score that the C runtime actually computes depends on it.

The obs-feature path is special: shuffling obs perturbs the input to f_obs,
which means the per-track e_track accumulator changes too. We recompute
e_track from scratch for every shuffled obs feature.

Repeats over multiple permutation seeds and averages so the rank is robust
to permutation noise.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from ml.train.train_phase3 import (
    TwoTower,
    auc,
    build_obs_matrix,
    build_det_matrix,
    build_pair_matrix,
    apply_norm,
    compute_etrack_for_pairs,
    OBS_FEATURE_NAMES_V2,
    OBS_FEATURE_NAMES_V3,
    DET_FEATURE_NAMES,
    PAIR_FEATURE_NAMES_V2,
    PAIR_FEATURE_NAMES_V3,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ml/data/phase3_v10_face.pt")
    p.add_argument("--val",  default="ml/data/pairs_val.npz")
    p.add_argument("--lam-deploy", type=float, default=0.05,
                   help="runtime nn_lambda used in uc_v11.yaml")
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = args.device
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    obs_in  = ck["obs_in"]
    det_in  = ck["det_in"]
    pair_in = ck["pair_in"]
    e_dim   = ck["e_dim"]
    tower_h = ck["tower_hidden"]
    alpha   = ck["alpha"]
    lam_train = float(ck["lambda"])
    obs_names  = ck.get("obs_feature_names",  OBS_FEATURE_NAMES_V2[:obs_in])
    det_names  = ck.get("det_feature_names",  DET_FEATURE_NAMES[:det_in])
    pair_names = ck.get("pair_feature_names", PAIR_FEATURE_NAMES_V2[:pair_in])

    model = TwoTower(obs_in=obs_in, det_in=det_in, pair_in=pair_in,
                     e_dim=e_dim, hidden=tower_h).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    obs_mean = np.array(ck["obs_mean"],  dtype=np.float32)
    obs_std  = np.array(ck["obs_std"],   dtype=np.float32)
    det_mean = np.array(ck["det_mean"],  dtype=np.float32)
    det_std  = np.array(ck["det_std"],   dtype=np.float32)
    pair_mean = np.array(ck["pair_mean"], dtype=np.float32)
    pair_std  = np.array(ck["pair_std"],  dtype=np.float32)

    va = np.load(args.val, allow_pickle=True)
    r  = va["records"]
    y  = va["labels"].astype(bool)
    s  = va["scene_ids"].astype(np.int64)
    tid = r["track_id"].astype(np.int64)
    ft  = r["frame_time"].astype(np.float32)
    di  = r["det_index"].astype(np.int32)
    wm  = r["was_matched"].astype(np.int32)
    pre = r["pre_thr_score"].astype(np.float32)

    has_face = obs_in >= 16
    has_time = obs_in == 17
    obs_raw  = build_obs_matrix(r,  with_face=has_face, with_time=has_time)
    det_raw  = build_det_matrix(r)
    pair_raw = build_pair_matrix(r, with_face=has_face, with_time=has_time)
    assert obs_raw.shape[1]  == obs_in
    assert det_raw.shape[1]  == det_in
    assert pair_raw.shape[1] == pair_in

    print(f"val pairs: {len(y):,}  positives: {y.sum():,}  "
          f"obs_in={obs_in} det_in={det_in} pair_in={pair_in}")
    print(f"λ_train={lam_train:.3f}  λ_deploy={args.lam_deploy:.3f}")
    print()

    @torch.no_grad()
    def score(obs_z, det_z, pair_z, etrack_already=None):
        """Return raw residual logit (per pair). etrack_already lets the
        caller skip the per-feature e_track recomputation when shuffling
        only det/pair columns."""
        obs_t  = torch.from_numpy(obs_z).to(device)
        det_t  = torch.from_numpy(det_z).to(device)
        pair_t = torch.from_numpy(pair_z).to(device)
        if etrack_already is None:
            et = compute_etrack_for_pairs(model, obs_z, s, tid, ft, di, wm,
                                          alpha, device)
        else:
            et = etrack_already
        e_track_g = torch.from_numpy(et).to(device)
        e_det_g   = model.encode_det(det_t)
        e_obs_now = model.encode_obs(obs_t)
        e_comb    = (1.0 - alpha) * e_track_g + alpha * e_obs_now
        return model.head(e_comb, e_det_g, pair_t).cpu().numpy()

    # Baseline (un-shuffled) scoring.
    obs_z  = apply_norm(obs_raw,  obs_mean,  obs_std)
    det_z  = apply_norm(det_raw,  det_mean,  det_std)
    pair_z = apply_norm(pair_raw, pair_mean, pair_std)
    et_base = compute_etrack_for_pairs(model, obs_z, s, tid, ft, di, wm,
                                       alpha, device)
    res_base = score(obs_z, det_z, pair_z, etrack_already=et_base)
    base_auc       = auc(pre, y)
    full_auc_train = auc(pre + lam_train * res_base, y)
    full_auc_dep   = auc(pre + args.lam_deploy * res_base, y)
    res_only_auc   = auc(res_base, y)
    print(f"baseline AUC (pre_thr only) : {base_auc:.5f}")
    print(f"residual alone AUC          : {res_only_auc:.5f}")
    print(f"pre + λ_train * residual    : {full_auc_train:.5f}   "
          f"(Δ {full_auc_train - base_auc:+.5f})")
    print(f"pre + λ_deploy * residual   : {full_auc_dep:.5f}   "
          f"(Δ {full_auc_dep - base_auc:+.5f})")
    print()

    rng = np.random.default_rng(args.seed)

    def perm_importance(view, col, repeats):
        """Return mean Δ-AUC across `repeats` shuffles for the
        (view, col) feature. Δ-AUC = baseline_full_dep - shuffled_full_dep
        (deployment-λ combined score). Also returns the same under
        training-λ for the secondary view."""
        deltas_train = []
        deltas_dep   = []
        n = len(y)
        for k in range(repeats):
            order = rng.permutation(n)
            if view == "obs":
                obs_z_p = obs_z.copy()
                obs_z_p[:, col] = obs_z[order, col]
                res = score(obs_z_p, det_z, pair_z)
            elif view == "det":
                det_z_p = det_z.copy()
                det_z_p[:, col] = det_z[order, col]
                res = score(obs_z, det_z_p, pair_z, etrack_already=et_base)
            elif view == "pair":
                pair_z_p = pair_z.copy()
                pair_z_p[:, col] = pair_z[order, col]
                res = score(obs_z, det_z, pair_z_p, etrack_already=et_base)
            else:
                raise ValueError(view)
            ad = auc(pre + args.lam_deploy * res, y)
            at = auc(pre + lam_train      * res, y)
            deltas_dep.append(full_auc_dep   - ad)
            deltas_train.append(full_auc_train - at)
        return (float(np.mean(deltas_train)),
                float(np.mean(deltas_dep)),
                float(np.std(deltas_dep)))

    rows = []
    for view, names in (("obs",  obs_names),
                        ("det",  det_names),
                        ("pair", pair_names)):
        for col, name in enumerate(names):
            dt, dd, sd = perm_importance(view, col, args.repeats)
            rows.append((view, col, name, dt, dd, sd))

    print(f"Permutation importance (avg over {args.repeats} shuffles)")
    print(f"{'view':5s} {'col':>3s} {'feature':28s} "
          f"{'Δ_train':>10s} {'Δ_deploy':>10s} {'σ_deploy':>10s}")
    print("-" * 75)
    # Sort by Δ_deploy descending (most important first).
    for view, col, name, dt, dd, sd in sorted(rows, key=lambda r: -r[4]):
        print(f"{view:5s} {col:>3d} {name:28s} "
              f"{dt:+10.5f} {dd:+10.5f} {sd:10.5f}")

    # Aggregate per underlying-feature-name (some features appear in
    # multiple views).
    agg = {}
    for view, col, name, dt, dd, sd in rows:
        agg.setdefault(name, []).append((view, dt, dd))
    print()
    print("Aggregate by feature name (sum across views):")
    print(f"{'feature':28s} {'views':12s} {'Σ Δ_train':>10s} {'Σ Δ_deploy':>11s}")
    print("-" * 65)
    agg_sorted = sorted(agg.items(), key=lambda x: -sum(v[2] for v in x[1]))
    for name, lst in agg_sorted:
        st = sum(v[1] for v in lst)
        sd = sum(v[2] for v in lst)
        views = "+".join(v[0] for v in lst)
        print(f"{name:28s} {views:12s} {st:+10.5f} {sd:+11.5f}")


if __name__ == "__main__":
    sys.exit(main() or 0)
