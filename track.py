import src.trackset as ts
import src.track_test as track_test
import src.track_search as track_search
import src.utrack.motion_track as mt
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
        metrics, frame_events=track_test.compute_metrics(trackset_gt, trackset,
                                                         frame_metrics=True,
                                                         eval_rate_divisor=1)
        metrics_time=time.time()
        elapsed_import=import_time-start_time
        elapsed_metrics=metrics_time-import_time
        #print(frame_events)
        print(metrics)
        print("--Summary--")
        print(track_test.summary_string(metrics)+f"  Import: {elapsed_import:.2f}s Metrics: {elapsed_metrics:.2f}s")
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
        print(f"{i} {names_compare[i]:20s}) {track_test.summary_string(metrics_compare[i])}")
    if display:
        ts.display_trackset(trackset_list=trackset_compare, trackset_gt=trackset_gt, frame_events_list=frame_events_list, output=None)

def test_track(t, config_file, display=False, output=None, proxy=None, metrics="python"):
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
    metrics, frame_events=track_test.compute_metrics(trackset_gt,
                                                     trackset,
                                                     frame_metrics=True,
                                                     eval_rate_divisor=1,
                                                     show_pbar=True,
                                                     eval_min_framerate=5,
                                                     metrics=metrics)
    metrics_time=time.time()
    elapsed_import=import_time-start_time
    elapsed_metrics=metrics_time-import_time
    print(metrics)
    print("--Summary--")
    print(track_test.summary_string(metrics)+f"  Import: {elapsed_import:.2f}s Metrics: {elapsed_metrics:.2f}s")
    if display:
        ts.display_trackset(trackset_list=[trackset], trackset_gt=trackset_gt, frame_events_list=[frame_events], output=output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='view.py')
    parser.add_argument('--logging', type=str, default='info', help="Logging config: level[:console|file]")
    parser.add_argument('--trackset', type=str, default='/mldata/tracking/mot/annotation/MOT20-01.json')
    parser.add_argument('--view', action='store_true', help='view a trackset')
    parser.add_argument('--caltech', action='store_true', help='make caltech pedestrian sequences')
    parser.add_argument('--mot', action='store_true', help='make MOT sequences')
    parser.add_argument('--cevo', action='store_true', help='make new CEVO videos')
    parser.add_argument('--test', type=str, default=None, help='test yaml file')
    parser.add_argument('--search', type=str, default=None, help='search config yaml file')
    parser.add_argument('--track', action='store_true', help='test tracker on a single sequence')
    parser.add_argument('--compare', type=str, default=None, help='compare multiple sets of tracking results')
    parser.add_argument('--display', action='store_true', help='visualise results')
    parser.add_argument('--config', type=str, default="/mldata/config/track/trackers/uc_v10.yaml", help="config")
    parser.add_argument('--output', type=str, default=None, help='output mp4 name')
    parser.add_argument('--metrics', type=str, default="python", help='metric computation: python or c')
    parser.add_argument('--proxy', type=str, default=None, help='proxy addr:port remote jetson e.g. 192.168.1.35:18861')
    parser.add_argument('--motiontracker-test',action='store_true', help='run motiontracker test')
    opt = parser.parse_args()
    stuff.rmdir(os.path.join(os.getcwd(), "tmp"))
    log_dir = os.path.join(os.getcwd(), "tmp/log")
    stuff.makedir(log_dir)
    stuff.configure_root_logger(opt.logging, log_dir=log_dir)
    if opt.motiontracker_test:
        mt.motiontracker_test()
        exit()
    if opt.caltech:
        ts.convert_caltech_pedestrian()
        exit()
    if opt.mot:
        ts.convert_mot()
        exit()
    if opt.cevo:
        ts.convert_cevo()
        exit()
    if opt.track:
        test_track(opt.trackset, opt.config, display=opt.display, output=opt.output, proxy=opt.proxy, metrics=opt.metrics)
        exit()
    if opt.compare is not None:
        compare_track(opt.trackset, compare_config=opt.compare)
        exit()
    if opt.search is not None:
        track_search.search_track(opt.search)
        exit()
    if opt.test is not None:
        track_test.track_test(opt.test)
        exit()
    if opt.view and opt.trackset is not None:
        ts.display_trackset(trackset_gt=opt.trackset)
        exit()

    print("No option specified")
