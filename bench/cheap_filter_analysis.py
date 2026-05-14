"""Phase 1 analysis for the matching-cheap-filter experiment.

For each detection event in the pair-log, compute:
  - n_candidates (number of tracks scored against this detection)
  - top-1 / top-2 pre_thr_score and the delta between them
  - "NN-flipped": whether the NN-augmented score ranks a different
    track first than the pre_thr score does
  - True-positive winner: the (track, det) pair the runtime ultimately
    assigned (was_matched=1) and its rank in both score orderings

Aggregate over the whole pair-log to produce:
  - n_candidates histogram → fraction of events with 1, 2, 3+ candidates
  - delta distribution
  - Pareto curve: for each candidate filter threshold (top-K, or
    delta) how many NN evaluations would be skipped, and what
    fraction of NN-induced decision flips would be lost?

Usage:
  python3 bench/cheap_filter_analysis.py \
      --pair-log /mldata/track_analysis_runs/pair_log_iter2_iter1NN/pair_log \
      --out /tmp/cheap_filter_analysis.json
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np


def analyse_clip(npz_path):
    """Return per-event records as parallel numpy arrays.

    Each detection event is summarised by:
      - n_candidates
      - pre_top, pre_2nd, pre_winner_idx (within-event)
      - mc_top, mc_2nd, mc_winner_idx
      - per-pair pre_thr scores (variable length, stored in `event_scores`)
      - flip = pre_winner_idx != mc_winner_idx

    Returns
    -------
    per_event : dict of arrays (one row per event)
    event_scores : list of np.ndarray (raw pre_thr_score per pair, one
        array per event, length == n_candidates) — needed for the
        Pareto sweep.
    """
    d = np.load(npz_path, allow_pickle=True)
    r = d['records']
    if len(r) == 0:
        return None, []

    order = np.argsort(r['frame_time'] * 100000.0 + r['det_index'].astype(np.float64))
    r = r[order]
    ft = r['frame_time']
    di = r['det_index']

    same = (ft[1:] == ft[:-1]) & (di[1:] == di[:-1])
    boundaries = np.concatenate(([0], np.where(~same)[0] + 1, [len(r)]))

    n_arr, pre_top, pre_2nd, mc_top, mc_2nd, flip = [], [], [], [], [], []
    # mc_winner_rank_in_pre[i] = rank (1-indexed) of the mc_winner
    # within the event's pre_thr_score descending ordering. 1 means the
    # mc_winner is also the pre_winner (no flip); higher means the NN
    # promoted a lower-pre-score candidate to the top. Bounds the
    # top-K and delta filter analysis exactly.
    mc_winner_rank_in_pre = []
    # pre_to_mc_winner_delta[i] = pre_top - pre_score[mc_winner]:
    # the gap that delta-filter must cross to keep the mc_winner.
    pre_to_mc_winner_delta = []
    event_scores = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i+1]
        if e <= s:
            continue
        rows = r[s:e]
        if rows[0]['frame_time'] == 0.0 and np.all(rows['pre_thr_score'] == 0.0):
            continue

        pre = rows['pre_thr_score'].astype(np.float64)
        mc  = rows['match_cost_score'].astype(np.float64)

        pre_order = np.argsort(-pre)
        mc_order  = np.argsort(-mc)
        n = len(rows)
        n_arr.append(n)
        pre_top.append(pre[pre_order[0]])
        pre_2nd.append(pre[pre_order[1]] if n > 1 else np.nan)
        mc_top.append(mc[mc_order[0]])
        mc_2nd.append(mc[mc_order[1]] if n > 1 else np.nan)
        flip.append(pre_order[0] != mc_order[0])
        event_scores.append(pre)

        mc_winner_idx = int(mc_order[0])
        # rank of mc_winner in pre ordering (1-indexed)
        rank = int(np.where(pre_order == mc_winner_idx)[0][0]) + 1
        mc_winner_rank_in_pre.append(rank)
        pre_to_mc_winner_delta.append(float(pre[pre_order[0]] - pre[mc_winner_idx]))

    return {
        'n':       np.asarray(n_arr,    dtype=np.int32),
        'pre_top': np.asarray(pre_top,  dtype=np.float64),
        'pre_2nd': np.asarray(pre_2nd,  dtype=np.float64),
        'mc_top':  np.asarray(mc_top,   dtype=np.float64),
        'mc_2nd':  np.asarray(mc_2nd,   dtype=np.float64),
        'flip':    np.asarray(flip,     dtype=bool),
        'mc_winner_rank_in_pre': np.asarray(mc_winner_rank_in_pre, dtype=np.int32),
        'pre_to_mc_winner_delta': np.asarray(pre_to_mc_winner_delta, dtype=np.float64),
    }, event_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair-log', required=True,
                    help='pair-log directory containing .npz files')
    ap.add_argument('--out', default='/tmp/cheap_filter_analysis.json')
    args = ap.parse_args()

    npz_paths = sorted(glob.glob(os.path.join(args.pair_log, '*.npz')))
    if not npz_paths:
        sys.exit(f'No .npz under {args.pair_log}')
    print(f'analysing {len(npz_paths)} pair-log files ...', flush=True)

    parts = []
    all_event_scores = []
    for p in npz_paths:
        per_event, ev_scores = analyse_clip(p)
        if per_event is None:
            continue
        parts.append(per_event)
        all_event_scores.extend(ev_scores)

    # Concatenate per-event arrays across all clips.
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    n_arr   = merged['n']
    pre_top = merged['pre_top']
    pre_2nd = merged['pre_2nd']
    flip    = merged['flip']
    mc_winner_rank = merged['mc_winner_rank_in_pre']
    pre_to_mc_delta = merged['pre_to_mc_winner_delta']

    print(f'{len(n_arr):,} matching events from {int(n_arr.sum()):,} pair-log rows', flush=True)

    # delta = pre_top - pre_2nd, defined only when n>=2
    has_2 = n_arr >= 2
    pre_delta = (pre_top - pre_2nd)[has_2]
    flipped   = flip[has_2]
    flipped_pre_delta = pre_delta[flipped]

    # ===== Aggregate stats =====
    print()
    print('=== n_candidates histogram ===')
    cnts = Counter(n_arr.tolist())
    total = len(n_arr)
    for k in sorted(cnts):
        if k <= 10 or cnts[k] >= total // 200:
            print(f'  n={k:3d}: {cnts[k]:>8,d}  ({100*cnts[k]/total:5.2f}%)')

    n1_frac = float((n_arr == 1).mean())
    n2_frac = float((n_arr == 2).mean())
    n_ge3   = float((n_arr >= 3).mean())
    print()
    print(f'  n=1   : {100*n1_frac:5.2f}%  (NN cannot help — only 1 candidate)')
    print(f'  n=2   : {100*n2_frac:5.2f}%')
    print(f'  n≥3   : {100*n_ge3:5.2f}%')

    print()
    print('=== pre_thr_score delta (top1 − top2) distribution, events with n≥2 ===')
    for q in (10, 25, 50, 75, 90, 95, 99):
        print(f'  p{q}: {np.percentile(pre_delta, q):8.4f}')
    print(f'  mean: {pre_delta.mean():8.4f}')

    print()
    print(f'=== NN-flip rate (pre_winner != mc_winner) among n≥2: '
          f'{100*flipped.mean():.2f}% ({flipped.sum():,} of {len(flipped):,}) ===')
    if len(flipped_pre_delta):
        print('pre_thr_delta on events the NN FLIPPED:')
        for q in (10, 25, 50, 75, 90):
            print(f'  p{q}: {np.percentile(flipped_pre_delta, q):8.4f}')

    # ===== Pareto: delta-filter aggressiveness vs flips preserved =====
    # For each event, count pairs with pre_thr_score >= top1 - delta.
    # NN must run on every surviving pair. NN-flip is preserved iff the
    # mc_winner (which is the row matching mc_top) survives the filter
    # — i.e. its pre_thr_score is within delta of pre_top.
    #
    # We don't have mc_winner's pre_thr score directly from `merged`, so
    # we use a proxy: assume the mc_winner is among the top-K by pre_thr.
    # For the K=1 / delta=0 case this is exact (only the pre_winner
    # survives → flips lost = all). For larger deltas we use the raw
    # event_scores to test: pre_score_of_mc_winner >= pre_top - delta?
    # We have the pre_thr scores per event but we don't track which
    # index is mc_winner. So we conservatively bound: the flip is
    # preserved IFF the second-best pre_thr is within delta of the top
    # (since the actual mc_winner has pre_thr at least as high as #2 by
    # construction of mc-flipping events — the mc_winner is by
    # definition NOT the pre_winner, hence its rank is >=2 in pre_thr
    # ordering, so pre_thr_of_mc_winner <= pre_2nd). Thus:
    #   "flip preserved at delta" >= "pre_2nd >= pre_top - delta"
    #                              = "pre_delta <= delta"
    print()
    print('=== Delta-filter Pareto ===')
    print('   delta_keep   pairs_kept   nn_evals_saved   flips_preserved (lower bd)')
    print('                (% of pairs) (% of pairs)     (% of NN-flips)')
    total_pairs = int(n_arr.sum())
    total_flips = int(flipped.sum())

    pareto_rows = []
    # Pre-extract the flipped-event mc_winner deltas (exact, not bounded)
    flipped_mc_winner_delta = pre_to_mc_delta[flip]
    for delta in (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 1e9):
        kept = 0
        for top, scores in zip(pre_top, all_event_scores):
            kept += int((scores >= top - delta).sum())
        # EXACT: flip preserved iff the mc_winner's own pre_thr score
        # is within delta of pre_top, i.e. pre_to_mc_winner_delta <= delta.
        flips_preserved = int((flipped_mc_winner_delta <= delta).sum())
        kept_pct  = 100 * kept / total_pairs
        saved_pct = 100 * (total_pairs - kept) / total_pairs
        flip_pct  = 100 * flips_preserved / max(total_flips, 1)
        label = f'{delta:>6.3f}' if delta < 1e8 else '   inf'
        print(f'   {label}    {kept_pct:>7.2f}      {saved_pct:>7.2f}         '
              f'{flip_pct:>7.2f}')
        pareto_rows.append({'delta': float(delta), 'pairs_kept': kept,
                            'pairs_kept_pct': kept_pct,
                            'pairs_saved_pct': saved_pct,
                            'flips_preserved': flips_preserved,
                            'flips_preserved_pct': flip_pct})

    # ===== Pareto: top-K filter =====
    print()
    print('=== Top-K filter Pareto ===')
    print('   K    pairs_kept   nn_evals_saved   flips_preserved')
    print('        (% of pairs) (% of pairs)     (% of NN-flips)')
    topk_rows = []
    flipped_mc_rank = mc_winner_rank[flip]
    for K in (1, 2, 3, 5, 8, 12, 1000):
        kept = int(np.minimum(n_arr, K).sum())
        kept_pct  = 100 * kept / total_pairs
        saved_pct = 100 * (total_pairs - kept) / total_pairs
        # EXACT: flip preserved iff mc_winner's pre_thr rank ≤ K.
        flips_preserved = int((flipped_mc_rank <= K).sum())
        flip_pct = 100 * flips_preserved / max(total_flips, 1)
        print(f'   K={K:<4d} {kept_pct:>7.2f}      {saved_pct:>7.2f}         '
              f'{flip_pct:>7.2f}')
        topk_rows.append({'K': K, 'pairs_kept': kept, 'pairs_kept_pct': kept_pct,
                          'pairs_saved_pct': saved_pct,
                          'flips_preserved': flips_preserved,
                          'flips_preserved_pct': flip_pct})

    # Direct picture of how often the NN really needs to "see deep" into
    # the pre_thr ordering. mc_winner_rank_in_pre histogram (n>=2 only).
    print()
    print('=== mc_winner rank in pre_thr ordering (events with n≥2) ===')
    has_2 = n_arr >= 2
    ranks_2plus = mc_winner_rank[has_2]
    flips_2plus = flip[has_2]
    for k in (1, 2, 3, 4, 5, 8, 12):
        c_all = int((ranks_2plus == k).sum())
        c_flip = int(((ranks_2plus == k) & flips_2plus).sum())
        print(f'  rank={k:2d}: all={c_all:>8,d} ({100*c_all/len(ranks_2plus):5.2f}%)   '
              f'flipped={c_flip:>6,d} ({100*c_flip/max(total_flips,1):5.2f}% of flips)')
    c_tail = int((ranks_2plus > 12).sum())
    if c_tail:
        c_tail_flip = int(((ranks_2plus > 12) & flips_2plus).sum())
        print(f'  rank>12: all={c_tail:>8,d} ({100*c_tail/len(ranks_2plus):5.2f}%)   '
              f'flipped={c_tail_flip:>6,d} ({100*c_tail_flip/max(total_flips,1):5.2f}% of flips)')

    out = {
        'total_events':       int(len(n_arr)),
        'total_pairs':        int(n_arr.sum()),
        'pareto_delta':       pareto_rows,
        'pareto_topk':        topk_rows,
        'n_candidates_histogram': {int(k): int(v) for k, v in cnts.items()},
        'n_eq_1_frac':        n1_frac,
        'n_eq_2_frac':        n2_frac,
        'n_ge_3_frac':        n_ge3,
        'pre_delta_pct': {f'p{q}': float(np.percentile(pre_delta, q))
                          for q in (10, 25, 50, 75, 90, 95, 99)},
        'pre_delta_mean':     float(pre_delta.mean()),
        'nn_flip_rate':       float(flipped.mean()),
        'nn_flip_count':      int(flipped.sum()),
        'n_ge_2_events':      int(len(flipped)),
        'flipped_pre_delta_pct': {f'p{q}': float(np.percentile(flipped_pre_delta, q))
                                  for q in (10, 25, 50, 75, 90)} if len(flipped_pre_delta) else None,
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
