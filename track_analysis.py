import argparse
import os
import pprint
import sys

import stuff

from src.pair_log import run_track_analysis


def main() -> int:
    parser = argparse.ArgumentParser(prog="track_analysis.py")
    parser.add_argument("--logging", type=str, default="info", help="Logging config: level[:console|file]")
    parser.add_argument(
        "--config",
        type=str,
        default="/mldata/config/track/analysis/track_analysis_v10.yaml",
        help="analysis config yaml",
    )
    parser.add_argument("--workers", type=int, default=None, help="override worker count")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "all"],
        default="train",
        help="which dataset split to run: train, val, or all",
    )
    parser.add_argument(
        "--force-regen",
        action="store_true",
        help="regenerate UBTRK2 files even if already present",
    )
    parser.add_argument(
        "--show-per-sequence",
        action="store_true",
        help="print per-sequence pair-logger metrics",
    )
    args = parser.parse_args()

    stuff.makedir(os.path.join(os.getcwd(), "tmp", "log"))
    stuff.configure_root_logger(args.logging, log_dir=os.path.join(os.getcwd(), "tmp", "log"))

    if not os.path.isfile(args.config):
        print(f"Config not found: {args.config}")
        return 2

    config = stuff.load_dictionary(args.config)
    if not isinstance(config, dict):
        print(f"Failed to load config as dictionary: {args.config}")
        return 2

    if args.workers is not None:
        config["workers"] = int(args.workers)
    config["dataset_split"] = args.split
    if args.force_regen:
        config["force_regen"] = True

    result = run_track_analysis(config)

    print("Track analysis complete")
    print(f"Output root: {result['output_root']}")
    print(f"Summary JSON: {result['summary_json']}")
    print(f"Summary MD: {result['summary_md']}")

    print("\nGeneration:")
    pprint.pprint(result["summary"]["generation"])
    print("\nAggregate metrics:")
    pprint.pprint(result["summary"]["aggregates"])

    if args.show_per_sequence:
        print("\nPer-sequence metrics:")
        per_seq = result["summary"].get("per_sequence_metrics", {})
        for seq in sorted(per_seq.keys()):
            print(f"\n[{seq}]")
            for k in sorted(per_seq[seq].keys()):
                v = per_seq[seq][k]
                if isinstance(v, float):
                    print(f"  {k}: {v:.6f}")
                else:
                    print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
