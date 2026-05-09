"""Compare offline state-machine sim with-demote vs no-demote.

Per the user: do we even NEED demote? With the output-fix in place, only
matched-this-frame TRACKED tracks emit. Demote (TRACKED → LOST) does
five things in the C runtime (Pass-3 cross-match, eviction priority,
hard timeout, pose-tracker exclusion, dedup-on-overlap) — but none of
those affect the offline cost-rule fitness directly.

This script measures the cost of removing demote in pure cost-rule
terms, on the val corpus. Real C-runtime impact will differ (the drop
rule still fires; we don't lose FP tracks forever) — but if no-demote
is bad even on this upper-bound benchmark, we know not to bother
implementing it in C.

Run:
    python -m bench.test_no_demote \\
        --head bench/data/state_head_dc_v8.pt \\
        --val  bench/data/state_corpus_v9_val.npz
"""
from __future__ import annotations
import argparse
import numpy as np
import torch

from bench.train_state_head_gru        import build_input_matrix_no_state, group_rows_by_track
from bench.train_state_head_decoupled  import DecoupledGRUHead
from bench.eval_decoupled_offline      import cost_decision, UNCONFIRMED, TRACKED, LOST


def cost_decision_no_demote(p_TP, mu_TP, mu_FP, state, matched,
                            c_MOTA, c_FP_track, c_FP_frame, match_rate_TP):
    """Same as cost_decision but TRACKED never transitions to LOST."""
    p_FP = 1.0 - p_TP
    delta_promote = (p_TP * c_MOTA * mu_TP * match_rate_TP
                     - p_FP * (c_FP_track + c_FP_frame * mu_FP))
    if state == UNCONFIRMED:
        if matched and delta_promote > 0:
            return TRACKED
        return UNCONFIRMED
    # TRACKED stays TRACKED forever (no demote, no drop here either).
    return TRACKED if state == TRACKED else state


def simulate(p_TP, mu_TP, mu_FP, matched, gt, groups, *,
             demote: bool, c_MOTA, c_FP_track, c_FP_frame, match_rate_TP=0.95):
    n_rows = len(p_TP)
    state_arr = np.full(n_rows, -1, dtype=np.int8)
    is_TP_track = np.zeros(len(groups), dtype=bool)
    promoted = np.zeros(len(groups), dtype=bool)
    n_demotes = 0

    decide = cost_decision if demote else cost_decision_no_demote
    for gi, rows in enumerate(groups):
        is_TP_track[gi] = bool(np.any(gt[rows] != -1))
        s = UNCONFIRMED
        prev = s
        for r in rows:
            s = decide(float(p_TP[r]), float(mu_TP[r]), float(mu_FP[r]),
                       s, bool(matched[r]),
                       c_MOTA=c_MOTA, c_FP_track=c_FP_track,
                       c_FP_frame=c_FP_frame, match_rate_TP=match_rate_TP)
            state_arr[r] = s
            if prev == TRACKED and s == LOST:
                n_demotes += 1
            if s == TRACKED:
                promoted[gi] = True
            prev = s

    aligned = (gt != -1)
    is_TRACKED = state_arr == TRACKED
    n_tp_tracked = int((is_TRACKED & aligned).sum())
    n_fp_tracked = int((is_TRACKED & ~aligned).sum())
    n_aligned = int(aligned.sum())
    n_tp_rows = sum(len(groups[g]) for g in range(len(groups)) if is_TP_track[g])
    n_fp_rows = sum(len(groups[g]) for g in range(len(groups)) if not is_TP_track[g])
    fp_wrong_promote = int(((~is_TP_track) & promoted).sum())
    tp_correct_promote = int((is_TP_track & promoted).sum())

    mota_proxy = n_tp_tracked / max(1, n_aligned)
    fp_per_frame = n_fp_tracked / max(1, n_rows)
    # Same fitness shape as eval_decoupled_offline.
    fitness = mota_proxy - 0.0005 * fp_wrong_promote - 0.002 * fp_per_frame
    return {
        "demote_active": demote,
        "n_demotes": n_demotes,
        "promoted_TP": tp_correct_promote,
        "promoted_FP": fp_wrong_promote,
        "tp_coverage_proxy": n_tp_tracked / max(1, n_tp_rows),
        "fp_exposure_proxy": n_fp_tracked / max(1, n_fp_rows),
        "mota_proxy": mota_proxy,
        "fp_per_frame_proxy": fp_per_frame,
        "fitness_proxy": fitness,
        "tp_frames_tracked": n_tp_tracked,
        "fp_frames_tracked": n_fp_tracked,
    }


def head_forward_all(model, X, groups, device, in_dim, batch=64):
    p_raw = np.zeros(len(X), dtype=np.float32)
    mu_TP = np.zeros(len(X), dtype=np.float32)
    mu_FP = np.zeros(len(X), dtype=np.float32)
    order = sorted(range(len(groups)), key=lambda i: len(groups[i]))
    with torch.no_grad():
        for s in range(0, len(groups), batch):
            picks = order[s:s+batch]
            T_max = max(len(groups[g]) for g in picks)
            B = len(picks)
            X_b = np.zeros((B, T_max, in_dim), dtype=np.float32)
            for bi, g in enumerate(picks):
                rows = groups[g]
                X_b[bi, :len(rows)] = X[rows]
            Xt = torch.from_numpy(X_b).to(device)
            llr_l, mtp_l, mfp_l, _ = model(Xt)
            p_b  = torch.sigmoid(llr_l).cpu().numpy()
            mtp  = np.expm1(np.clip(mtp_l.cpu().numpy(), 0, 10))
            mfp  = np.expm1(np.clip(mfp_l.cpu().numpy(), 0, 10))
            for bi, g in enumerate(picks):
                rows = groups[g]
                L = len(rows)
                p_raw[rows] = p_b[bi, :L]
                mu_TP[rows] = np.maximum(0.0, mtp[bi, :L])
                mu_FP[rows] = np.maximum(0.0, mfp[bi, :L])
    return p_raw, mu_TP, mu_FP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--val",  required=True)
    ap.add_argument("--c-mota",     type=float, default=1e-3)
    ap.add_argument("--c-fp-track", type=float, default=2.5e-2)
    ap.add_argument("--c-fp-frame", type=float, default=2e-3)
    args = ap.parse_args()

    print(f"loading head {args.head}", flush=True)
    ckpt = torch.load(args.head, map_location="cpu", weights_only=False)
    in_dim = int(ckpt["in_dim"]); hidden = int(ckpt["hidden"])
    model = DecoupledGRUHead(in_dim=in_dim, hidden=hidden)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"loading {args.val}", flush=True)
    rec = np.load(args.val, allow_pickle=False)["records"]
    X = build_input_matrix_no_state(rec)
    groups = group_rows_by_track(rec)
    matched = rec["matched"].astype(bool)
    gt = rec["gt_id_now"]
    print(f"  {len(rec)} rows, {len(groups)} tracks", flush=True)

    print("forward pass over corpus ...", flush=True)
    p_TP, mu_TP, mu_FP = head_forward_all(model, X, groups, device, in_dim)

    print(f"\n=== Cost rule sim (c_FP_track={args.c_fp_track}) ===")
    out_with    = simulate(p_TP, mu_TP, mu_FP, matched, gt, groups,
                           demote=True,
                           c_MOTA=args.c_mota,
                           c_FP_track=args.c_fp_track,
                           c_FP_frame=args.c_fp_frame)
    out_without = simulate(p_TP, mu_TP, mu_FP, matched, gt, groups,
                           demote=False,
                           c_MOTA=args.c_mota,
                           c_FP_track=args.c_fp_track,
                           c_FP_frame=args.c_fp_frame)

    width = 22
    print(f"\n  {'metric':<{width}}  {'with demote':>12}  {'no demote':>12}  {'delta':>10}")
    for k in ["promoted_TP", "promoted_FP",
              "tp_coverage_proxy", "fp_exposure_proxy",
              "mota_proxy", "fp_per_frame_proxy",
              "tp_frames_tracked", "fp_frames_tracked",
              "fitness_proxy", "n_demotes"]:
        a = out_with[k]; b = out_without[k]
        if isinstance(a, float):
            print(f"  {k:<{width}}  {a:12.4f}  {b:12.4f}  {b-a:+10.4f}")
        else:
            print(f"  {k:<{width}}  {a:12d}  {b:12d}  {b-a:+10d}")

    print("\nNotes:")
    print("  - This offline sim does NOT model cost-rule drop. In a real")
    print("    C-runtime no-demote variant, FP tracks would eventually drop,")
    print("    so fp_exposure here is an upper bound on the runtime cost.")
    print("  - tp_coverage gain (delta) is the real benefit: TP tracks that")
    print("    don't get demoted during occlusion stay TRACKED.")


if __name__ == "__main__":
    main()
