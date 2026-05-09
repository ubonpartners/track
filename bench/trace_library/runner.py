"""Run the trace regression library against a state head.

Loads every `.npz` under `bench/trace_library/traces/`, runs the head
forward over each trace's input rows, walks the cost rule on the
predicted (p_TP, μ_TP, μ_FP) + observed `matched`, and checks the trace's
declared assertions. Exits 0 on all-pass, 1 on any failure.

Run:
    python -m bench.trace_library.runner --head bench/data/state_head_dc_v9.pt
    python -m bench.trace_library.runner --head bench/data/state_head_dc_v9.pt \\
        --trace mu_tp_collapse_occlusion       # single trace
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from typing import Tuple

import numpy as np
import torch

from bench.train_state_head_decoupled import DecoupledGRUHead
from bench.eval_decoupled_offline import cost_decision, UNCONFIRMED, TRACKED, LOST


STATE_NAMES = {UNCONFIRMED: "UNCONFIRMED", TRACKED: "TRACKED", LOST: "LOST"}
STATE_BY_NAME = {v: k for k, v in STATE_NAMES.items()}
OPS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    ">":  lambda a, b: a >  b,
    "<":  lambda a, b: a <  b,
}


def parse_rows(spec, n_rows):
    """Spec is int, "lo-hi" string, or "all". Returns sorted list[int]."""
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, str):
        if spec == "all":
            return list(range(n_rows))
        if "-" in spec:
            lo, hi = spec.split("-", 1)
            return list(range(int(lo), int(hi) + 1))
        return [int(spec)]
    if isinstance(spec, list):
        out = []
        for s in spec:
            out.extend(parse_rows(s, n_rows))
        return sorted(set(out))
    raise ValueError(f"unparseable rows spec: {spec!r}")


def run_head(model: torch.nn.Module,
             inputs: np.ndarray,
             device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward one trace (T, in_dim) through the head; return per-row
    (p_TP, μ_TP, μ_FP)."""
    with torch.no_grad():
        x = torch.from_numpy(inputs.astype(np.float32))[None, ...].to(device)
        llr_l, mtp_l, mfp_l, _ = model(x)
        p_TP  = torch.sigmoid(llr_l).cpu().numpy()[0]
        mu_TP = np.expm1(np.clip(mtp_l.cpu().numpy()[0], 0, 10))
        mu_FP = np.expm1(np.clip(mfp_l.cpu().numpy()[0], 0, 10))
    return p_TP, np.maximum(0.0, mu_TP), np.maximum(0.0, mu_FP)


def walk_cost_rule(p_TP, mu_TP, mu_FP, matched,
                   c_MOTA=1e-3, c_FP_track=5e-4, c_FP_frame=2e-3,
                   match_rate_TP=0.95, disable_demote=False) -> np.ndarray:
    """Per-row state from the offline cost-rule sim. Mirrors the C runtime.

    `disable_demote` mirrors the `bayes_disable_demote` YAML knob in the C
    runtime: TRACKED tracks never transition to LOST; the bench's intent is
    that TRACKED-unmatched-too-long → drop replaces the LOST middle state.
    The trace library tests the rule the head will actually run under, so
    when the deployment mode uses no-demote, the runner should too."""
    state = UNCONFIRMED
    out = np.zeros(len(p_TP), dtype=np.int8)
    for k in range(len(p_TP)):
        next_state = cost_decision(
            float(p_TP[k]), float(mu_TP[k]), float(mu_FP[k]),
            state, bool(matched[k]),
            c_MOTA=c_MOTA, c_FP_track=c_FP_track,
            c_FP_frame=c_FP_frame, match_rate_TP=match_rate_TP)
        if disable_demote and state == TRACKED and next_state == LOST:
            next_state = TRACKED
        state = next_state
        out[k] = state
    return out


def check_trace(trace_path: str, model, device,
                c_MOTA: float, c_FP_track: float, c_FP_frame: float,
                disable_demote: bool = False):
    """Returns (n_pass, n_fail, [failure_messages])."""
    blob = np.load(trace_path, allow_pickle=False)
    inputs   = blob["inputs"]
    matched  = blob["matched"].astype(bool)
    meta     = json.loads(str(blob["meta_json"]))
    trace_id = meta.get("trace_id", os.path.basename(trace_path))

    p_TP, mu_TP, mu_FP = run_head(model, inputs, device)
    states = walk_cost_rule(p_TP, mu_TP, mu_FP, matched,
                            c_MOTA=c_MOTA,
                            c_FP_track=c_FP_track,
                            c_FP_frame=c_FP_frame,
                            disable_demote=disable_demote)

    n_rows = len(p_TP)
    failures = []
    n_checks = 0
    for assertion in meta.get("asserts", []):
        rows  = parse_rows(assertion["rows"], n_rows)
        field = assertion["field"]
        op    = assertion["op"]
        val   = assertion["value"]
        tag   = assertion.get("tag", "")
        n_checks += 1

        if field == "p_TP":
            actual = p_TP[rows]
        elif field == "mu_TP":
            actual = mu_TP[rows]
        elif field == "mu_FP":
            actual = mu_FP[rows]
        elif field == "cost_rule_state":
            actual = states[rows]
            if isinstance(val, str):
                val = STATE_BY_NAME[val]
        else:
            failures.append(f"  [{trace_id}] unknown field {field!r}")
            continue

        ok = bool(np.all(OPS[op](actual, val)))
        if not ok:
            if field == "cost_rule_state":
                got = [STATE_NAMES.get(int(s), str(s)) for s in actual]
            else:
                got = [round(float(x), 3) for x in actual]
            failures.append(
                f"  [{trace_id}] {tag or field}: rows={rows} "
                f"expected {field} {op} {val}; got {got}"
            )

    return n_checks - len(failures), len(failures), failures, trace_id, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True,
                    help="path to a state_head_dc_v*.pt checkpoint")
    ap.add_argument("--trace-dir", default="bench/trace_library/traces",
                    help="directory of .npz traces to run")
    ap.add_argument("--trace", default=None,
                    help="single trace name (without .npz) to run")
    # Defaults match the C runtime's defaults in utrack.c around line 226-230:
    #   bayes_c_MOTA=1e-3, bayes_c_FP_track=5e-4, bayes_c_FP_frame=2e-3.
    # Earlier this defaulted to c_FP_track=0.025 which over-suppressed
    # traces relative to runtime; the trace lib's pass/fail must match
    # the cost rule the head actually runs under.
    ap.add_argument("--c-mota",     type=float, default=1e-3)
    ap.add_argument("--c-fp-track", type=float, default=5e-4)
    ap.add_argument("--c-fp-frame", type=float, default=2e-3)
    ap.add_argument("--disable-demote", action="store_true",
                    help="mirror runtime's bayes_disable_demote knob: "
                         "TRACKED never drops to LOST")
    args = ap.parse_args()

    ckpt = torch.load(args.head, map_location="cpu", weights_only=False)
    in_dim = int(ckpt["in_dim"]); hidden = int(ckpt["hidden"])
    model = DecoupledGRUHead(in_dim=in_dim, hidden=hidden)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.trace:
        paths = [os.path.join(args.trace_dir, f"{args.trace}.npz")]
        if not os.path.exists(paths[0]):
            sys.exit(f"trace not found: {paths[0]}")
    else:
        paths = sorted(glob.glob(os.path.join(args.trace_dir, "*.npz")))

    if not paths:
        sys.exit(f"no traces found under {args.trace_dir}")

    print(f"head: {args.head}  in_dim={in_dim} hidden={hidden}")
    print(f"running {len(paths)} trace(s) "
          f"(c_FP_track={args.c_fp_track}, c_MOTA={args.c_mota}, "
          f"disable_demote={args.disable_demote})\n")

    total_pass = 0
    total_fail = 0
    all_failures = []
    for p in paths:
        n_pass, n_fail, fails, tid, meta = check_trace(
            p, model, device,
            c_MOTA=args.c_mota,
            c_FP_track=args.c_fp_track,
            c_FP_frame=args.c_fp_frame,
            disable_demote=args.disable_demote)
        total_pass += n_pass
        total_fail += n_fail
        all_failures.extend(fails)
        status = "PASS" if n_fail == 0 else f"FAIL({n_fail})"
        issue = meta.get("issue", "?")
        print(f"  [{status:8}]  {tid:48}  {issue}")

    print(f"\n{total_pass} pass, {total_fail} fail "
          f"(across {len(paths)} traces)\n")
    for f in all_failures:
        print(f)
    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
