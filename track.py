import src.trackset as ts
import src.track_test as track_test
import src.track_search as track_search
import src.utrack.motion_track as mt
import stuff
import argparse
import time

def compare_track(t, compare_config=None, display=True):
    config=stuff.load_dictionary(compare_config)

    assert "gt_trackset" in config
    assert "configs_to_compare" in config

    trackset_gt=ts.TrackSet(config["gt_trackset"])
    configs_to_compare=config["configs_to_compare"]

    trackset_compare=[]
    metrics_compare=[]
    names_compare=[]
    for i,c in enumerate(configs_to_compare):
        this_config=configs_to_compare[c]

        params=None
        if "params" in this_config:
            params=this_config["params"]
        track_min_interval=0.159
        if "track_min_interval" in this_config:
            track_min_interval=this_config["track_min_interval"]

        trackset=ts.TrackSet()
        start_time=time.time()
        trackset.import_create(trackset_gt,
                               config_file=this_config["config"],
                               params=params,
                               track_min_interval=track_min_interval, #-1,
                               debug=False,
                               debug_enable=True)
        trackset.name=c
        import_time=time.time()
        print("Computing metrics....")
        metrics, frame_events=track_test.compute_metrics(trackset_gt, trackset, frame_metrics=True, eval_rate_divisor=1)
        metrics_time=time.time()
        elapsed_import=import_time-start_time
        elapsed_metrics=metrics_time-import_time
        print(metrics)
        print("--Summary--")
        print(track_test.summary_string(metrics)+f"  Import: {elapsed_import:.2f}s Metrics: {elapsed_metrics:.2f}s")
        trackset_compare.append(trackset)
        metrics_compare.append(metrics)
        names_compare.append(c)

    print("\nComparison:")
    for i, x in enumerate(trackset_compare):
        print(f"{i} {names_compare[i]:20s}) {track_test.summary_string(metrics_compare[i])}")
    if display:
        ts.display_trackset(trackset_list=trackset_compare, trackset_gt=trackset_gt, frame_events=frame_events, output=None)

def test_track(t, config_file, display=False, output=None):
    print(ts)
    trackset_gt=ts.TrackSet(t)
    trackset=ts.TrackSet()
    start_time=time.time()
    trackset.import_create(trackset_gt,
                           track_min_interval=0.159,
                           debug=False,
                           config_file=config_file,
                           debug_enable=True)
    import_time=time.time()
    metrics, frame_events=track_test.compute_metrics(trackset_gt, trackset, frame_metrics=True, eval_rate_divisor=1)
    metrics_time=time.time()
    elapsed_import=import_time-start_time
    elapsed_metrics=metrics_time-import_time
    print(metrics)
    print("--Summary--")
    print(track_test.summary_string(metrics)+f"  Import: {elapsed_import:.2f}s Metrics: {elapsed_metrics:.2f}s")
    if display:
        ts.display_trackset(trackset_list=[trackset, trackset], trackset_gt=trackset_gt, frame_events=frame_events, output=output)

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
    parser.add_argument('--config', type=str, default="/mldata/config/track/bytetrack_nofuse.yaml", help="config")
    parser.add_argument('--output', type=str, default=None, help='output mp4 name')
    parser.add_argument('--motiontracker-test',action='store_true', help='run motiontracker test')
    opt = parser.parse_args()
    stuff.configure_root_logger(opt.logging)
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
        test_track(opt.trackset, opt.config, display=opt.display, output=opt.output)
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