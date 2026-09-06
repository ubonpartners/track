"""`python -m src.cli <verb> ...` — the command line (repo_cleanup.md stage 6).

Verbs: view, track, compare, eval, search, test, import, corpus, paths.
Each verb's function is the existing entry point in the package; no
logic lives here except argument handling and the two single-sequence
drivers (compare_track, test_track) that used to sit in track.py.
`python track.py --old-flags` still works: track.py translates to these
verbs and prints the new spelling on every run.
"""
import argparse
import os
import sys
import time

import stuff

import src.paths as paths

IMPORTERS = {
    # verb `import <name>` -> callable(**kw); kwargs from the parser
    "mot": lambda **kw: _importers().convert_mot(),
    "personpath22": lambda amodal=False, **kw: _importers().convert_personpath22(
        anno_variant="amodal" if amodal else "visible"),
    "jaad": lambda **kw: _importers().convert_jaad(),
    "otw": lambda **kw: _importers().convert_otw(),
    "meva": lambda **kw: _importers().convert_meva(),
    "cevo": lambda **kw: _importers().convert_cevo(),
    "chirla": lambda **kw: _importers().convert_chirla(),
    "roundabouthd": lambda **kw: _importers().convert_roundabouthd(),
    "uvg_vcm": lambda **kw: _importers().convert_uvg_vcm(),
    "bdd100k": lambda **kw: _importers().convert_bdd100k_kaggle(),
    "raw_movies": lambda **kw: _importers().convert_raw_movies(),
    "bwc_videotext": lambda **kw: _importers().convert_bwc_videotext(),
    "antare": lambda **kw: __import__("src.import_antare", fromlist=["main"]).main([]),   # its own argparse: never sys.argv
}


def _importers():
    import src.corpus.importers as importers
    return importers


# ---------------------------------------------------------------- drivers

def compare_track(t, compare_config=None, display=True):
    # verbatim from the old track.py (repo_cleanup.md stage 6); the module
    # imports moved inside so importing src.cli stays cheap
    import src.core.trackset as ts
    import src.core.display as core_display
    import src.eval.metrics as eval_metrics
    import src.eval.report as eval_report
    import src.tracker.run as tracker_run
    try:
        import ubon_pycstuff.ubon_pycstuff as upyc
        upyc.enable_file_trace("uc_compare.log")
    except:
        print("Could not enable file tracing for upyc")

    config=stuff.load_dictionary(compare_config)

    assert "gt_trackset" in config
    assert "configs_to_compare" in config

    trackset_gt=ts.TrackSet(config["gt_trackset"])
    configs_to_compare=config["configs_to_compare"]
    import_start_time=0
    import_end_time=60.0
    print("Trimming")
    trackset_gt.trim(import_start_time, import_end_time)

    trackset_compare=[]
    metrics_compare=[]
    names_compare=[]
    frame_events_list=[]
    for i,c in enumerate(configs_to_compare):
        this_config=configs_to_compare[c]

        params=None
        if "params" in this_config:
            params=this_config["params"]
        track_min_interval=this_config.get("track_min_interval", 0.199)
        #max_duration=this_config.get("max_duration", 1000.0)
        if params is None:
            params={}
        params["simple"]=False #True

        trackset=ts.TrackSet()
        start_time=time.time()
        print(f"Import/create {c}....")
        tracker_run.import_create(trackset, trackset_gt,
                               config_file=this_config["config"],
                               params=params,
                               track_min_interval=track_min_interval,
                               debug=False,
                               debug_enable=True,
                               start_time=import_start_time,
                               end_time=import_end_time)

        trackset.name=c
        import_time=time.time()
        print("Computing metrics....")
        metrics, frame_events=eval_metrics.compute_metrics(trackset_gt, trackset,
                                                         frame_metrics=True,
                                                         eval_rate_divisor=1)
        metrics_time=time.time()
        elapsed_import=import_time-start_time
        elapsed_metrics=metrics_time-import_time
        #print(frame_events)
        print(metrics)
        print("--Summary--")
        print(eval_report.summary_string(metrics)+f"  Import: {elapsed_import:.2f}s Metrics: {elapsed_metrics:.2f}s")
        trackset_compare.append(trackset)
        metrics_compare.append(metrics)
        frame_events_list.append(frame_events)
        names_compare.append(c)

    print("\nPer-frame MOTA")
    nfr=len(frame_events_list[0])
    for i in range(nfr):
        s=f"{i:4d} {frame_events_list[0][i]['frame_time']:6.3f}"
        for j,fe in enumerate(frame_events_list):
            s+=f" {fe[i]['stats']['mota']:0.6f} "
            if j!=0:
                delta=fe[i]['stats']['mota']-frame_events_list[0][i]['stats']['mota']
                if (abs(delta)>0.001):
                    s+=f" E {delta:0.6f} "
        print(s)

    print("\nMetrics:")

    keys = list(metrics_compare[0].keys())

    # Prepare table headers: "Metric" followed by "Run 1", "Run 2", ...
    headers = ["Metric"] + names_compare

    # Build table rows
    rows = []
    for key in keys:
        row = [key]
        for run in metrics_compare:
            val = run.get(key, "")
            if isinstance(val, float):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))
        rows.append(row)

    # Print table
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*([headers] + rows))]
    row_format = "  ".join(f"{{:<{w}}}" for w in col_widths)

    print(row_format.format(*headers))
    print("-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    for row in rows:
        print(row_format.format(*row))

    print("\nComparison:")
    for i, x in enumerate(trackset_compare):
        print(f"{i} {names_compare[i]:20s}) {eval_report.summary_string(metrics_compare[i])}")
    if display:
        core_display.display_trackset(trackset_list=trackset_compare, trackset_gt=trackset_gt, frame_events_list=frame_events_list, output=None)


def test_track(t, config_file, display=False, output=None, proxy=None, save_trackset=None):
    # verbatim from the old track.py (repo_cleanup.md stage 6); the module
    # imports moved inside so importing src.cli stays cheap
    import src.core.trackset as ts
    import src.core.display as core_display
    import src.eval.metrics as eval_metrics
    import src.eval.report as eval_report
    import src.tracker.run as tracker_run
    trackset_gt=ts.TrackSet(t)
    trackset=ts.TrackSet()
    start_time=time.time()
    params=None
    if proxy is not None:
        params={"proxy":proxy}
    tracker_run.import_create(trackset, trackset_gt,
                           track_min_interval=0.199,
                           debug=False,
                           config_file=config_file,
                           debug_enable=True,
                           params=params)

    import_time=time.time()
    metrics, frame_events=eval_metrics.compute_metrics(trackset_gt,
                                                     trackset,
                                                     frame_metrics=True,
                                                     eval_rate_divisor=1,
                                                     show_pbar=True,
                                                     eval_min_framerate=5)
    metrics_time=time.time()
    elapsed_import=import_time-start_time
    elapsed_metrics=metrics_time-import_time
    print(metrics)
    print("--Summary--")
    print(eval_report.summary_string(metrics)+f"  Import: {elapsed_import:.2f}s Metrics: {elapsed_metrics:.2f}s")
    if save_trackset is not None:
        trackset.export_track_file(save_trackset)
        print(f"Saved tracked run to {save_trackset}")
    if display:
        core_display.display_trackset(trackset_list=[trackset], trackset_gt=trackset_gt, frame_events_list=[frame_events], output=output)


# ---------------------------------------------------------------- parser

def build_parser():
    p = argparse.ArgumentParser(prog="python -m src.cli",
                                description="track: tracker evaluation, parameter search and dataset import")
    p.add_argument("--logging", default="info", help="Logging config: level[:console|file]")
    p.add_argument("--pm", type=int, default=None, metavar="N",
                   help="detector performance-mode tier for eval/search/test streams: "
                        "0=full res, higher=cheaper (0..3 today = 640/512/416/320). Global "
                        "override — beats a per-test 'pm:' key in the yaml.")
    sub = p.add_subparsers(dest="verb", required=True, metavar="<verb>")

    v = sub.add_parser("view", help="replay a GT annotation or a tracked run (UBTRK2) with boxes")
    v.add_argument("trackset", help="annotation json/yaml or .ubtrk2 run")

    t = sub.add_parser("track", help="track one GT clip with one config and print metrics")
    t.add_argument("trackset", help="GT annotation json")
    t.add_argument("--config", default=None, help="tracker yaml (default: the production config)")
    t.add_argument("--display", action="store_true", help="visualise the run")
    t.add_argument("--output", default=None, help="output mp4 name (with --display)")
    t.add_argument("--save-trackset", default=None, help="save the tracked run as a UBTRK2 file")
    t.add_argument("--proxy", default=None, help="proxy addr:port of a remote jetson, e.g. 192.168.1.35:18861")

    c = sub.add_parser("compare", help="run several tracker configs over one GT clip and compare")
    c.add_argument("config", help="compare yaml: gt_trackset + configs_to_compare")
    c.add_argument("--no-display", action="store_true", help="skip the viewer at the end")

    e = sub.add_parser("eval", help="THE measurement path for tracker A/Bs")
    e.add_argument("yaml", nargs="?", default=None,
                   help="omit to run the one objective config that `search` optimises; a path is "
                        "allowed for one-off probes and prints a loud warning that the result is "
                        "not the objective")
    e.add_argument("--split", default="both", choices=["train", "val", "both"], help="dataset split")
    e.add_argument("--permissive", default="auto", choices=["auto", "on", "off"],
                   help="convention-permissive matching override")
    e.add_argument("--results-location", default=None,
                   help="output dir for the reports; overrides results_location in the yaml")
    e.add_argument("--tracker-config", default=None,
                   help="tracker yaml to evaluate; overrides tests.*.config (the way to A/B a tracker)")

    s = sub.add_parser("search", help="coordinate-descent parameter search")
    s.add_argument("yaml", help="search config yaml (the objective)")

    x = sub.add_parser("test", help="benchmark tracker configs over datasets (test yaml)")
    x.add_argument("yaml", help="test yaml: tests + datasets (+ optional results_location)")

    i = sub.add_parser("import", help="tier 0 -> tier 1 import of a labelled dataset")
    i.add_argument("corpus", choices=sorted(IMPORTERS))
    i.add_argument("--amodal", action="store_true", help="personpath22: amodal (occluded) boxes instead of visible")

    k = sub.add_parser("corpus", help="tier-1 manifest and tier-2 derivation")
    k.add_argument("action", choices=["build", "verify", "derive", "check"])
    k.add_argument("corpus", nargs="+")
    k.add_argument("--hint", choices=["static", "bodycam"], default=None, help="derive: camera class")
    k.add_argument("--max-seconds", type=float, default=None, help="derive: duration cap")
    k.add_argument("--divisor", type=int, default=None, help="derive: force the framerate divisor")
    k.add_argument("--purge-legacy", action="store_true", help="check: delete legacy artefacts")

    sub.add_parser("paths", help="print every filesystem root the package resolves")
    return p


LOGGED_VERBS = {"view", "track", "compare", "eval", "search", "test"}


def dispatch(ns):
    if ns.verb == "paths":
        for k, v in paths.describe().items():
            print(f"{k:16s} {v}")
        return 0
    if ns.verb in LOGGED_VERBS:
        # per-run log dir, as track.py always did for these actions, before
        # any logging happens (import/corpus never had it: their old CLIs
        # did not wipe ./tmp, so they still do not)
        stuff.rmdir(os.path.join(os.getcwd(), "tmp"))
        log_dir = os.path.join(os.getcwd(), "tmp/log")
        stuff.makedir(log_dir)
        stuff.configure_root_logger(ns.logging, log_dir=log_dir)
    if ns.pm is not None:
        # read only by the eval runner's _resolve_pm (eval/search/test);
        # harmless for the other verbs, so set unconditionally
        import src.eval.runner as eval_runner
        eval_runner.PM_OVERRIDE = ns.pm
    if ns.verb == "import" and ns.amodal and ns.corpus != "personpath22":
        raise SystemExit("--amodal is a personpath22 option")
    if ns.verb == "view":
        import src.core.display as core_display
        core_display.display_trackset(trackset_gt=ns.trackset)
    elif ns.verb == "track":
        test_track(ns.trackset, ns.config or paths.tracker_config(), display=ns.display,
                   output=ns.output, proxy=ns.proxy, save_trackset=ns.save_trackset)
    elif ns.verb == "compare":
        compare_track(None, compare_config=ns.config, display=not ns.no_display)
    elif ns.verb == "eval":
        import src.track_search as track_search
        perm = {"auto": None, "on": True, "off": False}[ns.permissive]
        track_search.eval_track(ns.yaml, split=ns.split, convention_permissive=perm,
                                results_location=ns.results_location,
                                tracker_config=ns.tracker_config)
    elif ns.verb == "search":
        import src.track_search as track_search
        track_search.search_track(ns.yaml)
    elif ns.verb == "test":
        import src.eval.runner as eval_runner
        eval_runner.track_test(ns.yaml)
    elif ns.verb == "import":
        IMPORTERS[ns.corpus](amodal=ns.amodal)
    elif ns.verb == "corpus":
        import src.corpus.manifest as manifest
        import src.corpus.derive as derive
        ok = True
        for corpus in ns.corpus:
            if ns.action == "derive":
                r = derive.derive_tracking(corpus, hint=ns.hint, max_seconds=ns.max_seconds,
                                           divisor=ns.divisor)
            elif ns.action == "check":
                r = derive.check_tracking(corpus, purge_legacy=ns.purge_legacy)
            else:
                r = {"build": manifest.build, "verify": manifest.verify}[ns.action](corpus)
            ok = ok and (r is not False)
        return 0 if ok else 1
    return 0


def main(argv=None):
    ns = build_parser().parse_args(argv)
    return dispatch(ns)


if __name__ == "__main__":
    sys.exit(main())
