"""Attribute GPU time to pipeline stages, from nsys kernel summaries.

Built because four successive explanations for where a MOTION/NVOF carry frame's
cost goes were wrong (docs/research/detection-free-frames.md E33/E35/E36), and each
time the reason was the same: every counter available on the contended path mixes
WORK with WAITING. Host wall-clock around a CUDA call measures submit plus queue
plus execution; the pipeline's own timers cannot separate them. A profiler can —
`nsys` reports per-kernel GPU execution time directly.

So: profile two configurations, bucket the kernels by what they belong to, and
compare GPU seconds per wall-clock second. That answers "does this feature add GPU
work, and where" without another plausible story.

Usage:
    nsys profile -o prof_X -t cuda ./rt_benchmark ...
    nsys stats --report cuda_gpu_kern_sum --format csv -o kern_X prof_X.nsys-rep
    python -m src.gpu_attrib --window 20 \
        --csv baseline=kern_A_cuda_gpu_kern_sum.csv \
        --csv carry=kern_B_cuda_gpu_kern_sum.csv

--window is the profiled wall-clock duration in seconds (runtime + warmup), used
to normalise; without it, totals are reported raw.
"""
import argparse
import csv
import re
import sys
from collections import defaultdict

# Kernel name -> bucket. Ordered: first match wins, so put specific before broad.
# Grounded in the actual kernel names in src/cuda/*.cu and the TensorRT/NPP
# families, not guessed from the profile output.
BUCKETS = (
    # --- the detector engine (TensorRT) ---
    ("detector",      (r"_trt\b", r"xmma", r"^__myl_", r"cuInt8::", r"CUTENSOR",
                       r"generatedNativePointwise", r"nvinfer", r"cudnn",
                       r"cask_", r"trt_")),
    # --- our own scale/convert/copy image ops (src/cuda/cuda_kernels_*.cu) ---
    ("image_scale",   (r"^Scale\b", r"^Scale_uv\b", r"^Resize\b",
                       r"downsample_2x2_kernel")),
    ("image_convert", (r"deinterleave_uv_kernel", r"interleave_uv_kernel",
                       r"rgb24_to_yuv420_kernel", r"rgb24_to_planar_fp_kernel",
                       r"fp16_planar_to_RGB24_kernel", r"fp32_planar_to_RGB24_kernel",
                       r"float_to_half_kernel", r"half_to_float_kernel",
                       r"warpYUVToPlanarRGBKernel", r"rotate_plane_kernel")),
    ("image_crop",    (r"cropYPlaneKernel", r"cropUVPlaneKernel")),
    # --- motion detection (the mask, not the flow: OFA is not a CUDA kernel) ---
    ("motion_detect", (r"kernel_block_mad_4x4", r"compute_motion_bytemask",
                       r"row_hash_kernel")),
    # --- detection post-processing on GPU ---
    ("nms_postproc",  (r"doSuppressionKernel", r"filterAndCollectKernel",
                       r"gatherFeaturesKernel", r"centerToCornerKernel",
                       r"buildPairwiseMaskKernel")),
    # --- NPP (used by the YUV420 scaler) ---
    ("npp",           (r"^nppi", r"Npp", r"^npp")),
)


def bucket_of(name):
    for bucket, pats in BUCKETS:
        for p in pats:
            if re.search(p, name):
                return bucket
    return "other"


def load(path):
    """nsys cuda_gpu_kern_sum csv -> {bucket: (total_ns, instances)}."""
    tot = defaultdict(lambda: [0, 0])
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("Name") or ""
            try:
                ns = float(row["Total Time (ns)"])
                inst = int(float(row["Instances"]))
            except (KeyError, ValueError):
                continue
            b = bucket_of(name)
            tot[b][0] += ns
            tot[b][1] += inst
    return {k: tuple(v) for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--window", type=float, default=0.0,
                    help="profiled wall-clock seconds, for normalisation")
    ap.add_argument("--top-other", type=int, default=6,
                    help="list this many largest unclassified kernels")
    a = ap.parse_args()

    runs = {}
    for spec in a.csv:
        name, _, path = spec.partition("=")
        runs[name] = load(path)

    buckets = sorted({b for r in runs.values() for b in r},
                     key=lambda b: -max(runs[n].get(b, (0, 0))[0] for n in runs))
    unit = "GPU s/s" if a.window else "GPU s"
    div = a.window if a.window else 1.0

    print(f"GPU time by bucket ({unit}); instances in brackets")
    hdr = f"{'bucket':16s}"
    for n in runs:
        hdr += f"{n:>22s}"
    print(hdr)
    for b in buckets:
        line = f"{b:16s}"
        for n in runs:
            ns, inst = runs[n].get(b, (0, 0))
            line += f"{ns/1e9/div:12.3f} [{inst:6d}]"
        print(line)
    line = f"{'TOTAL':16s}"
    for n in runs:
        ns = sum(v[0] for v in runs[n].values())
        inst = sum(v[1] for v in runs[n].values())
        line += f"{ns/1e9/div:12.3f} [{inst:6d}]"
    print(line)

    if len(runs) == 2:
        (n1, r1), (n2, r2) = list(runs.items())
        print(f"\ndelta ({n2} - {n1}), {unit}:")
        for b in buckets:
            d = (r2.get(b, (0, 0))[0] - r1.get(b, (0, 0))[0]) / 1e9 / div
            if abs(d) > 1e-4:
                print(f"  {b:16s} {d:+8.3f}")
        d = (sum(v[0] for v in r2.values())
             - sum(v[0] for v in r1.values())) / 1e9 / div
        print(f"  {'TOTAL':16s} {d:+8.3f}")

    # Unclassified kernels are a correctness risk for the whole table, so surface
    # the biggest ones rather than letting them hide in "other".
    if a.top_other:
        print(f"\nlargest unclassified kernels (bucket=other):")
        for name, path in ((n, p.partition("=")[2]) for n, p in
                           ((s.partition("=")[0], s) for s in a.csv)):
            rows = []
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    nm = row.get("Name") or ""
                    if bucket_of(nm) != "other":
                        continue
                    try:
                        rows.append((float(row["Total Time (ns)"]), nm))
                    except (KeyError, ValueError):
                        pass
            rows.sort(reverse=True)
            print(f"  [{name}]")
            for ns, nm in rows[:a.top_other]:
                print(f"    {ns/1e9/div:8.3f}  {nm[:96]}")


if __name__ == "__main__":
    main()
