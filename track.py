import src.trackset as ts
import src.corpus.importers as importers
import src.eval.metrics as eval_metrics
import src.eval.report as eval_report
import src.eval.runner as eval_runner
import src.paths as paths
import src.track_search as track_search
import os
import stuff
import argparse
import time

def compare_track(t, compare_config=None, display=True):
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
        trackset.import_create(trackset_gt,
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
        ts.display_trackset(trackset_list=trackset_compare, trackset_gt=trackset_gt, frame_events_list=frame_events_list, output=None)

def test_track(t, config_file, display=False, output=None, proxy=None, save_trackset=None):
    trackset_gt=ts.TrackSet(t)
    trackset=ts.TrackSet()
    start_time=time.time()
    params=None
    if proxy is not None:
        params={"proxy":proxy}
    trackset.import_create(trackset_gt,
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
        ts.display_trackset(trackset_list=[trackset], trackset_gt=trackset_gt, frame_events_list=[frame_events], output=output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='view.py')
    parser.add_argument('--logging', type=str, default='info', help="Logging config: level[:console|file]")
    parser.add_argument('--trackset', type=str, default=paths.tier2('mot', 'annotation', 'MOT20-01.json'))
    parser.add_argument('--view', action='store_true', help='view a trackset')
    parser.add_argument('--mot', action='store_true', help='make MOT sequences')
    parser.add_argument('--personpath22', action='store_true', help='make PersonPath22 sequences')
    parser.add_argument('--personpath22-amodal', action='store_true', help='use amodal (occluded) boxes for PersonPath22 instead of visible')
    parser.add_argument('--jaad', action='store_true', help='make JAAD sequences')
    parser.add_argument('--otw', action='store_true', help='make Out the Window (OTW) sequences')
    parser.add_argument('--meva', action='store_true', help='make MEVA sequences')
    parser.add_argument('--cevo', action='store_true', help='make new CEVO videos')
    parser.add_argument('--test', type=str, default=None, help='test yaml file')
    parser.add_argument('--search', type=str, default=None, help='search config yaml file')
    parser.add_argument('--eval', nargs='?', type=str, default=None, const='',
                        help='THE measurement path for tracker A/Bs. Pass NO path: it runs the one '
                             'objective config (track_search_v11_mc.yaml) that --search optimises, so eval and '
                             'search cannot describe different datasets. Set the output dir with '
                             '--results-location (never by copying the yaml -- that is how a second, '
                             'differently-weighted "canonical" config appeared and invalidated results three '
                             'times). Use --eval-split val for search-comparable scores and compare runs with '
                             'python -m src.eval_compare. Passing a path is allowed for one-off probes and '
                             'prints a loud warning that the result is not the objective.')
    parser.add_argument('--results-location', type=str, default=None,
                        help='output dir for --eval reports; overrides results_location in the yaml')
    parser.add_argument('--tracker-config', type=str, default=None,
                        help='tracker yaml to evaluate; overrides tests.*.config. Use this for '
                             'tracker A/Bs instead of copying the objective config -- varying the '
                             'thing under test must not mean forking the measurement.')
    parser.add_argument('--eval-split', type=str, default='both', choices=['train', 'val', 'both'], help='dataset split for --eval')
    parser.add_argument('--eval-permissive', type=str, default='auto', choices=['auto', 'on', 'off'], help='convention-permissive matching override for --eval')
    parser.add_argument('--pm', type=int, default=None, metavar='N',
                        help='detector performance-mode tier for --eval/--search streams: '
                             '0=full res, higher=cheaper (0..3 today = 640/512/416/320). '
                             'Eval streams are non-realtime, so this sets nrt_pm and was '
                             'previously fixed at 0. Global override — beats a per-test "pm:" '
                             'key in the yaml. Omit to use whatever the yaml specifies.')
    parser.add_argument('--track', action='store_true', help='test tracker on a single sequence')
    parser.add_argument('--compare', type=str, default=None, help='compare multiple sets of tracking results')
    parser.add_argument('--display', action='store_true', help='visualise results')
    parser.add_argument('--config', type=str, default=paths.tracker_config(), help="config")
    parser.add_argument('--paths', action='store_true', help='print every filesystem root the package resolves (src/paths.py) and exit')
    parser.add_argument('--output', type=str, default=None, help='output mp4 name')
    parser.add_argument('--save-trackset', type=str, default=None, help='save tracked run as UBTRK2 trackset file')
    parser.add_argument('--proxy', type=str, default=None, help='proxy addr:port remote jetson e.g. 192.168.1.35:18861')
    opt = parser.parse_args()
    if opt.paths:
        for k, v in paths.describe().items():
            print(f"{k:16s} {v}")
        exit()
    stuff.rmdir(os.path.join(os.getcwd(), "tmp"))
    log_dir = os.path.join(os.getcwd(), "tmp/log")
    stuff.makedir(log_dir)
    stuff.configure_root_logger(opt.logging, log_dir=log_dir)
    if opt.mot:
        importers.convert_mot()
        exit()
    if opt.personpath22:
        variant = "amodal" if opt.personpath22_amodal else "visible"
        importers.convert_personpath22(anno_variant=variant)
        exit()
    if opt.jaad:
        importers.convert_jaad()
        exit()
    if opt.otw:
        importers.convert_otw()
        exit()
    if opt.meva:
        importers.convert_meva()
        exit()
    if opt.cevo:
        importers.convert_cevo()
        exit()
    if opt.track:
        test_track(opt.trackset, opt.config, display=opt.display, output=opt.output, proxy=opt.proxy, save_trackset=opt.save_trackset)
        exit()
    if opt.compare is not None:
        compare_track(opt.trackset, compare_config=opt.compare)
        exit()
    if opt.pm is not None:
        eval_runner.PM_OVERRIDE = opt.pm
    if opt.search is not None:
        track_search.search_track(opt.search)
        exit()
    if opt.eval is not None:
        perm = {"auto": None, "on": True, "off": False}[opt.eval_permissive]
        track_search.eval_track(opt.eval or None, split=opt.eval_split,
                                convention_permissive=perm,
                                results_location=opt.results_location,
                                tracker_config=opt.tracker_config)
        exit()
    if opt.test is not None:
        eval_runner.track_test(opt.test)
        exit()
    if opt.view and opt.trackset is not None:
        ts.display_trackset(trackset_gt=opt.trackset)
        exit()

    print("No option specified")
