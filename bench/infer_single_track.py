"""Single-track diagnostic: walk one (sequence, track_id) through the
decoupled head and print the per-frame (p_TP, μ_TP, μ_FP) trajectory.

The head must be testable in isolation — given a track's frame
sequence, return three numbers per frame, with no dependency on the
runtime state machine. This script is the proof.

Usage:
    # First track in val corpus
    python -m bench.infer_single_track \\
        --head bench/data/state_head_dc_v1.pt \\
        --val  bench/data/state_corpus_v13_val.npz

    # Specific track
    python -m bench.infer_single_track \\
        --head bench/data/state_head_dc_v1.pt \\
        --val  bench/data/state_corpus_v13_val.npz \\
        --sequence MOT20-05 --track-id 3203334145
"""
from __future__ import annotations
import argparse
import numpy as np
import torch

from bench.train_state_head_gru   import build_input_matrix_no_state, group_rows_by_track
from bench.train_state_head_decoupled import DecoupledGRUHead
from bench.train_bayesian_head    import compute_track_labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", required=True)
    p.add_argument("--val",  required=True)
    p.add_argument("--sequence", default=None)
    p.add_argument("--track-id", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=80)
    p.add_argument("--n-tracks", type=int, default=3,
                   help="when --sequence/--track-id not given, walk this "
                        "many tracks (mix of TP and FP)")
    # Phase 29 — sensitivity test: override the 6 scene-stat columns with
    # a fixed value across all rows so we can compare head outputs under
    # different "scene priors". Only meaningful for in_dim==25 heads.
    p.add_argument("--scene-override", default=None,
                   metavar="K=V,K=V,...",
                   help="override scene-stat columns to a fixed value. "
                        "Keys: promote_rate, mean_det_TRK, mean_det_unm, "
                        "track_density, alive_age. The derived feature "
                        "det_conf_minus_scene_TP_avg is recomputed. "
                        "Example: --scene-override "
                        "promote_rate=0.9,mean_det_TRK=0.85")
    args = p.parse_args()

    print(f"loading head from {args.head}")
    ckpt = torch.load(args.head, map_location="cpu", weights_only=False)
    in_dim = int(ckpt["in_dim"]); hidden = int(ckpt["hidden"])
    model = DecoupledGRUHead(in_dim=in_dim, hidden=hidden)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    with_scene = (in_dim == 25)
    print(f"  in_dim={in_dim} hidden={hidden} with_scene={with_scene}")

    print(f"loading {args.val}")
    rec = np.load(args.val, allow_pickle=False)["records"]
    X = build_input_matrix_no_state(rec, with_scene=with_scene)

    # Apply scene overrides — replaces the relevant cols of X with the
    # fixed values so the per-frame scene priors become constants. Lets
    # us A/B the same track under different "scene means" and see how
    # the head's outputs respond.
    if args.scene_override:
        if not with_scene:
            print("  (warning: --scene-override given but head in_dim != 25; ignored)")
        else:
            ovr = {}
            for kv in args.scene_override.split(","):
                k, v = kv.split("=")
                ovr[k.strip()] = float(v)
            # Column indices in the 25-dim layout: scene_promote_rate=19,
            # scene_mean_det_conf_TRACKED=20, scene_mean_det_conf_unmatched=21,
            # log1p(scene_track_density_smooth)=22, log1p(scene_mean_alive_track_age)=23,
            # det_conf_minus_scene_TP_avg=24. det_conf is at column 5.
            if "promote_rate" in ovr:    X[:, 19] = ovr["promote_rate"]
            if "mean_det_TRK" in ovr:    X[:, 20] = ovr["mean_det_TRK"]
            if "mean_det_unm" in ovr:    X[:, 21] = ovr["mean_det_unm"]
            if "track_density" in ovr:   X[:, 22] = float(np.log1p(ovr["track_density"]))
            if "alive_age" in ovr:       X[:, 23] = float(np.log1p(ovr["alive_age"]))
            # det_conf_minus_scene_TP_avg = det_conf (col 5) - scene_mean_det_conf_TRACKED.
            X[:, 24] = X[:, 5] - X[:, 20]
            print(f"  applied scene-override: {ovr}")
    is_TP, rem = compute_track_labels(rec)
    seq = rec["sequence"].astype(str)
    tid = rec["track_id"]
    fi  = rec["frame_idx"]
    matched = rec["matched"].astype(bool)
    obs = rec["observations"]

    groups = group_rows_by_track(rec)

    # Pick which tracks to walk
    picks = []
    if args.sequence is not None and args.track_id is not None:
        for gi, rows in enumerate(groups):
            if seq[rows[0]] == args.sequence and int(tid[rows[0]]) == args.track_id:
                picks.append(gi)
                break
        if not picks:
            print(f"no track matching ({args.sequence}, {args.track_id})")
            return
    else:
        # Mix of TP and FP, picked at random
        rng = np.random.default_rng(0)
        tp_gi = [gi for gi, rows in enumerate(groups)
                 if is_TP[rows[0]] > 0.5 and len(rows) >= 5]
        fp_gi = [gi for gi, rows in enumerate(groups)
                 if is_TP[rows[0]] <= 0.5 and len(rows) >= 3]
        rng.shuffle(tp_gi); rng.shuffle(fp_gi)
        for gi in tp_gi[:max(1, args.n_tracks - 1)]:
            picks.append(gi)
        for gi in fp_gi[:1]:
            picks.append(gi)

    for gi in picks:
        rows = groups[gi]
        L = min(len(rows), args.max_frames)
        feats = X[rows[:L]]
        # Single-track inference: feed feats as (1, T, in_dim).
        x = torch.from_numpy(feats).unsqueeze(0)
        with torch.no_grad():
            llr_l, mtp_l, mfp_l, _ = model(x)
        p_TP  = torch.sigmoid(llr_l[0]).numpy()
        mu_TP = np.expm1(np.clip(mtp_l[0].numpy(), 0, 10))
        mu_FP = np.expm1(np.clip(mfp_l[0].numpy(), 0, 10))

        truth = "TP" if is_TP[rows[0]] > 0.5 else "FP"
        print(f"\n=== Track ({seq[rows[0]]}, {int(tid[rows[0]])}) — "
              f"truth={truth}, total_len={len(rows)}, walking first {L} frames ===")
        print(f"  remaining_lifetime at row[0]: {rem[rows[0]]:.2f}s")
        print(f"  {'k':>3} {'frame':>5} {'matched':>7} {'obs':>4} "
              f"{'p_TP':>6} {'μ_TP(s)':>8} {'μ_FP(s)':>8}")
        for k in range(L):
            r = rows[k]
            print(f"  {k:>3} {fi[r]:>5} {int(matched[r]):>7} {obs[r]:>4} "
                  f"{p_TP[k]:>6.3f} {mu_TP[k]:>8.2f} {mu_FP[k]:>8.2f}")


if __name__ == "__main__":
    main()
