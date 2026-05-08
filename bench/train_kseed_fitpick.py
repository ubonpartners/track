"""Phase 20.9 (1) — K-seed train + fitness-pick wrapper.

State-head training is bimodal across seeds (Phase 20.7): the same
config can produce a working head (deployment fitness ~0.6) or a
broken head (~0.02). Val combined AUC does NOT distinguish these
cases (Phase 20.8b — all 5 heads tested had identical val behaviour).

The only reliable signal is closed-loop fitness on the diverse 29-
clip subset (`bench/eval_head_fitness.py`). This wrapper:

1. Runs `train_state_head.py` K times with seeds 0..K-1.
2. Exports each .pt → .bin via `export_state_head.py`.
3. Runs the diverse 29-clip fitness eval on each .bin.
4. Saves the best-by-fitness .bin (and .pt) under the user-supplied
   output prefix; reports per-seed fitness as a table.

Usage:
    python -m bench.train_kseed_fitpick \\
        --train bench/data/state_corpus_v9_train.npz \\
        --val   bench/data/state_corpus_v9_val.npz \\
        --out-prefix bench/data/state_head_v23 \\
        --epochs 30 --no-history --seeds 5

The output prefix produces:
    bench/data/state_head_v23.pt        # best-fitness checkpoint
    bench/data/state_head_v23.bin       # corresponding .bin
    bench/data/state_head_v23.kseed.json # per-seed table

Forward unrecognised flags to `train_state_head.py` so this wrapper
inherits its full CLI without re-declaration.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-prefix", required=True,
                   help="prefix for .pt/.bin output (e.g. bench/data/state_head_v23)")
    p.add_argument("--seeds", type=int, default=5,
                   help="number of seeds to train (default 5)")
    p.add_argument("--workers", type=int, default=4,
                   help="fitness-eval workers per head (default 4)")
    p.add_argument("--keep-all", action="store_true",
                   help="keep all per-seed .pt/.bin (default: only best)")
    args, passthrough = p.parse_known_args()

    prefix = args.out_prefix
    rows = []
    seed_paths = []
    t_start = time.time()

    for seed in range(args.seeds):
        pt = f"{prefix}.s{seed}.pt"
        bn = f"{prefix}.s{seed}.bin"
        seed_paths.append((seed, pt, bn))

        # 1. Train.
        t0 = time.time()
        cmd = [sys.executable, "-m", "bench.train_state_head",
               "--save", pt, "--seed", str(seed)] + passthrough
        print(f"\n[kseed {seed}/{args.seeds-1}] training → {pt}", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  TRAIN FAILED: {r.stderr.strip()[-400:]}")
            continue
        # Pull val AUC from the trainer's stdout.
        val_auc = None
        for ln in r.stdout.splitlines():
            if ln.startswith("best combined AUC"):
                try: val_auc = float(ln.split("=")[1].strip())
                except Exception: pass
        train_dt = time.time() - t0

        # 2. Export.
        t0 = time.time()
        r = subprocess.run([sys.executable, "-m", "bench.export_state_head",
                            "--in", pt, "--out", bn],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  EXPORT FAILED: {r.stderr.strip()[-400:]}")
            continue
        export_dt = time.time() - t0

        # 3. Fitness on diverse subset.
        t0 = time.time()
        from bench.eval_head_fitness import eval_config, load_subset, make_yaml_with_state_bin
        clips = load_subset("diverse")
        yaml_path = make_yaml_with_state_bin(bn)
        try:
            res = eval_config(yaml_path, clips=clips, workers=args.workers)
        finally:
            try: os.unlink(yaml_path)
            except OSError: pass
        fit_dt = time.time() - t0
        ov = res["overall"]

        rows.append({
            "seed": seed, "pt": pt, "bin": bn,
            "val_auc": val_auc,
            "fitness": ov["fitness"], "mota": ov["mota"],
            "fp_tracks": ov["fp_tracks"], "fp_per_frame": ov["fp_per_frame"],
            "train_sec": train_dt, "export_sec": export_dt, "fit_sec": fit_dt,
        })
        print(f"  [s{seed}] AUC={val_auc!s} fit={ov['fitness']:.4f} "
              f"mota={ov['mota']:.4f} FPTr={ov['fp_tracks']} "
              f"({train_dt:.0f}s train, {fit_dt:.0f}s fit)", flush=True)

    if not rows:
        sys.exit("no successful seeds")

    rows.sort(key=lambda r: -r["fitness"])
    best = rows[0]

    # Promote the best to the canonical prefix.
    import shutil
    shutil.copyfile(best["pt"],  f"{prefix}.pt")
    shutil.copyfile(best["bin"], f"{prefix}.bin")
    print(f"\n[kseed] best seed = {best['seed']} (fitness {best['fitness']:.4f})", flush=True)
    print(f"[kseed] saved → {prefix}.pt / {prefix}.bin", flush=True)

    # Per-seed table.
    print(f"\n{'seed':>4s}  {'val_AUC':>8s}  {'fitness':>8s}  {'mota':>7s}  {'FPTr':>5s}")
    for r in sorted(rows, key=lambda x: x["seed"]):
        auc_s = f"{r['val_auc']:.4f}" if r["val_auc"] is not None else "  n/a "
        marker = "  ←best" if r["seed"] == best["seed"] else ""
        print(f"  {r['seed']:>2d}  {auc_s:>8s}  {r['fitness']:8.4f}  "
              f"{r['mota']:7.4f}  {r['fp_tracks']:5d}{marker}")

    summary = {
        "prefix": prefix,
        "total_sec": time.time() - t_start,
        "best_seed": best["seed"],
        "best_fitness": best["fitness"],
        "rows": rows,
    }
    Path(f"{prefix}.kseed.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[kseed] summary → {prefix}.kseed.json")

    if not args.keep_all:
        for seed, pt, bn in seed_paths:
            if seed == best["seed"]: continue
            for f in (pt, bn):
                try: os.unlink(f)
                except OSError: pass
        print("[kseed] cleaned up non-best per-seed artefacts")


if __name__ == "__main__":
    main()
