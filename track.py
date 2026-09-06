"""Compatibility entry point (repo_cleanup.md stage 6).

The command line is `python -m src.cli <verb> ...` (src/cli.py). This
file keeps every old `python track.py --flag` form working: it parses the
old flags, prints the new spelling once, and runs the same code.
"""
import argparse
import sys

import src.paths as paths


def old_parser():
    parser = argparse.ArgumentParser(prog="track.py")
    parser.add_argument('--logging', type=str, default='info')
    parser.add_argument('--trackset', type=str, default=paths.tier2('mot', 'annotation', 'MOT20-01.json'))
    parser.add_argument('--view', action='store_true')
    for name in ("mot", "personpath22", "jaad", "otw", "meva", "cevo"):
        parser.add_argument(f'--{name}', action='store_true')
    parser.add_argument('--personpath22-amodal', action='store_true')
    parser.add_argument('--test', type=str, default=None)
    parser.add_argument('--search', type=str, default=None)
    parser.add_argument('--eval', nargs='?', type=str, default=None, const='')
    parser.add_argument('--results-location', type=str, default=None)
    parser.add_argument('--tracker-config', type=str, default=None)
    parser.add_argument('--eval-split', type=str, default='both', choices=['train', 'val', 'both'])
    parser.add_argument('--eval-permissive', type=str, default='auto', choices=['auto', 'on', 'off'])
    parser.add_argument('--pm', type=int, default=None)
    parser.add_argument('--track', action='store_true')
    parser.add_argument('--compare', type=str, default=None)
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--paths', action='store_true')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--save-trackset', type=str, default=None)
    parser.add_argument('--proxy', type=str, default=None)
    return parser


def translate(argv):
    """Old track.py flags -> `src.cli` argv (None when no action was given).
    Dispatch order is the old one: paths, importers, track, compare,
    search, eval, test, view."""
    o = old_parser().parse_args(argv)
    common = ["--logging", o.logging] + (["--pm", str(o.pm)] if o.pm is not None else [])
    if o.paths:
        return ["paths"]
    for name in ("mot", "personpath22", "jaad", "otw", "meva", "cevo"):
        if getattr(o, name):
            extra = ["--amodal"] if name == "personpath22" and o.personpath22_amodal else []
            return common + ["import", name] + extra
    if o.track:
        args = ["track", o.trackset]
        if o.config: args += ["--config", o.config]
        if o.display: args.append("--display")
        if o.output: args += ["--output", o.output]
        if o.save_trackset: args += ["--save-trackset", o.save_trackset]
        if o.proxy: args += ["--proxy", o.proxy]
        return common + args
    if o.compare is not None:
        return common + ["compare", o.compare]
    if o.search is not None:
        return common + ["search", o.search]
    if o.eval is not None:
        args = ["eval"] + ([o.eval] if o.eval else [])
        args += ["--split", o.eval_split, "--permissive", o.eval_permissive]
        if o.results_location: args += ["--results-location", o.results_location]
        if o.tracker_config: args += ["--tracker-config", o.tracker_config]
        return common + args
    if o.test is not None:
        return common + ["test", o.test]
    if o.view and o.trackset is not None:
        return common + ["view", o.trackset]
    return None


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    new = translate(argv)
    if new is None:
        print("No option specified")
        return 0
    print(f"note: `python track.py {' '.join(argv)}` is now `python -m src.cli {' '.join(new)}`",
          file=sys.stderr)
    import src.cli as cli
    return cli.main(new)


if __name__ == '__main__':
    sys.exit(main())
