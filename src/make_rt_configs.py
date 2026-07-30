"""Regenerate the rt_benchmark tracker configs used by the PM/capacity campaign.

The campaign's configs lived in /tmp on the Jetson and would not survive a reboot,
which would have made every measurement in docs/research/detection-free-frames.md
unreproducible. This regenerates them from the base tracker yaml.

Each config is the base tracker yaml plus a small, named delta — that is the whole
point: one variable at a time, so a result can be attributed. Run on the machine
under test (the base yaml path must resolve there):

    python -m src.make_rt_configs --out ~/rtcfg_pm_opt
    ./build/rt_benchmark --tracker-yaml ~/rtcfg_pm_opt/L_mixed.yaml --min 36 --max 36 ...

Knobs that rt_benchmark OVERRIDES from its own CLI defaults, so setting them in the
yaml is silently ineffective (learned the hard way — E38, E43):
    performance.pm_log          -> use -L
    performance.skip_target_sec -> use -S <sec>
    the OFA engine pool         -> use --of-pool-size N
"""
import argparse
import os
import re

BASE = "/mldata/config/track/trackers/uc_v11.yaml"

# name -> (description, transform). Transforms take the base yaml text.
def _perf(block):
    """Insert a performance: block (base has none by default)."""
    def f(t):
        return t.rstrip("\n") + "\nperformance:\n" + block
    return f

def _top(line):
    def f(t):
        return t.rstrip("\n") + "\n" + line + "\n"
    return f

def _compose(*fns):
    def f(t):
        for fn in fns:
            t = fn(t)
        return t
    return f

def _motiontrack(extra):
    """Add keys inside the existing motiontrack: block."""
    def f(t):
        assert "motiontrack:" in t, "base yaml has no motiontrack: block"
        return t.replace("motiontrack:", "motiontrack:\n" + extra, 1)
    return f

def _replace_top(key, value):
    def f(t):
        pat = re.compile(rf"^{re.escape(key)}:.*$", re.M)
        assert pat.search(t), f"{key} not found at top level"
        return pat.sub(f"{key}: {value}", t, count=1)
    return f


CONFIGS = {
    # --- the two ladders, and a TRUE baseline -------------------------------
    "B_baseline": (
        "no performance block at all — the genuine pre-change default. NOTE: "
        "setting degrade_policy is itself what switches a stream onto its own "
        "ladder, so L_res_first is NOT a clean baseline (E31).",
        lambda t: t),
    "L_res_first": (
        "resolution-first ladder (the historical mapping), explicitly selected",
        _perf("  degrade_policy: resolution_first\n")),
    "L_mixed": (
        "mixed ladder — the banked win, +0.022 to +0.050 over baseline at 24-48 "
        "streams (E23, E31, E34)",
        _perf("  degrade_policy: mixed\n")),

    # --- MOTION / NVOF carry ------------------------------------------------
    "M_mixed_carry": (
        "mixed + PM shed delivered as MOTION carry frames. A wash on the curve "
        "with the OFA pool raised (E32, E36)",
        _compose(_perf("  degrade_policy: mixed\n  skip_mode: motion\n"),
                 _top("min_time_delta_motion: 0.09"))),
    "X_nodrop_noflow": (
        "shed frames DELIVERED but with no flow hop (skip_mode: motion without "
        "min_time_delta_motion). The isolation that showed delivering frames is "
        "free and the flow hop is the entire cost (E37)",
        _perf("  degrade_policy: mixed\n  skip_mode: motion\n")),

    # --- flow working size (separate from the motion image) -----------------
    "L_mixed_of320": (
        "mixed + flow at 320 instead of 512. Cuts a flow call 10-16%, moves the "
        "operating point by nothing (E34)",
        _compose(_perf("  degrade_policy: mixed\n"),
                 _motiontrack("  of_max_width: 320\n  of_max_height: 320\n"))),
    "M_carry_of320": (
        "carry + flow at 320",
        _compose(_perf("  degrade_policy: mixed\n  skip_mode: motion\n"),
                 _top("min_time_delta_motion: 0.09"),
                 _motiontrack("  of_max_width: 320\n  of_max_height: 320\n"))),

    # --- per-content ladders, selected by stream hint ------------------------
    "P_percontent": (
        "per-stream ladders keyed on stream_hint; pair with "
        "--stream-hints cctv,egomo. Worth ~nothing measured (E28, E30)",
        _perf("  degrade_policy(hint:cctv): mixed\n"
              "  degrade_policy(hint:egomo): resolution_first\n")),

    # --- worker threads (measured null, E35) --------------------------------
    "L_mixed_wt10": ("mixed with 10 worker threads (auto gives 5 on a 6-core Orin)",
                     _compose(_perf("  degrade_policy: mixed\n"),
                              _replace_top("num_worker_threads", "10"))),
    "L_mixed_wt16": ("mixed with 16 worker threads",
                     _compose(_perf("  degrade_policy: mixed\n"),
                              _replace_top("num_worker_threads", "16"))),
    "M_mixed_carry_wt16": ("carry with 16 worker threads",
                           _compose(_perf("  degrade_policy: mixed\n  skip_mode: motion\n"),
                                    _top("min_time_delta_motion: 0.09"),
                                    _replace_top("num_worker_threads", "16"))),

    # --- RT batch linger (measured ~2%, E27) --------------------------------
    "B_ling25": ("mixed + dispatch-relative RT batch linger, 25 ms",
                 _perf("  degrade_policy: mixed\n  rt_linger_ms: 25\n"
                       "  rt_batch_target: 8\n  rt_max_age_ms: 400\n")),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default=os.path.expanduser("~/rtcfg_pm_opt"))
    a = ap.parse_args()

    base = open(a.base).read()
    os.makedirs(a.out, exist_ok=True)
    for name, (desc, fn) in sorted(CONFIGS.items()):
        text = fn(base)
        path = os.path.join(a.out, name + ".yaml")
        with open(path, "w") as f:
            f.write(f"# {name}: {desc}\n"
                    f"# Regenerate with: python -m src.make_rt_configs --out {a.out}\n")
            f.write(text)
        print(f"{name:22s} {desc[:78]}")
    print(f"\n{len(CONFIGS)} configs written to {a.out}")


if __name__ == "__main__":
    main()
