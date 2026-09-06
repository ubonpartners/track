"""Eval orchestration: the multiprocess work queue path, the shared-stream
tracker runner (run_single_shared), packed-result parsing, track_test().

Moved verbatim from src/track_test.py (repo_cleanup.md stage 4a).
"""
import copy
import datetime
import json
import logging
import os
import pickle
import time
import stuff
import src.core.trackset as ts
import src.tracker.run as tracker_run

from src.eval.metrics import score_tracksets
from src.eval.report import _write_eval_summary_json, display_results



def track_test_work_fn(params, mpwq_context, mpwq_progress_fn):
    logging.debug("Running here")
    trackset=ts.TrackSet()
    trackset_gt=ts.TrackSet(params["ds_path"])
    logging.debug(f"import create")
    tracker_run.import_create(trackset, trackset_gt,
                           track_min_interval=params["min_interval"],
                           display=params["display"],
                           config_file=params["config"],
                           params=params,
                           mpwq_context=mpwq_context,
                           mpwq_progress_fn=mpwq_progress_fn,
                           # max_duration is a REAL compute cap since
                           # 2026-07-23: upyc_tracker truncates the h264 at
                           # extraction (duration-suffixed cache entry, so
                           # capped and full evals never share files) — the
                           # C pipeline only ever sees the window. Scoring
                           # is capped by the same value in compute_metrics.
                           end_time=(100000 if params.get("max_duration_is_default")
                                     else params.get("max_duration", 100000)))
    result=score_tracksets(trackset_gt, trackset, params)

    del trackset
    del trackset_gt
    logging.debug(f"set entry")
    entry={"params":params,
           "result":result,
           "time":datetime.datetime.now()}

    logging.debug(f"done")
    return entry



def _clip_meta(json_path):
    """{original_video, frame_rate} for a clip, via a tiny sidecar cache —
    scheduling must not json.load a 200MB MOT20 annotation in the main
    process. Built once per annotation (invalidated by annotation mtime)."""
    meta_path = json_path + ".meta.json"
    try:
        m = json.load(open(meta_path))
        if m.get("_mtime") == os.path.getmtime(json_path):
            return m
    except (OSError, ValueError):
        pass
    md = json.load(open(json_path)).get("metadata") or {}
    m = {"original_video": md.get("original_video"),
         "frame_rate": md.get("frame_rate"),
         # lite provenance == guaranteed I+P encoding (dataset_lite -bf 0),
         # the precondition for the B-frame-skipping mp4-direct ingest
         "lite": bool(md.get("lite")),
         "_mtime": os.path.getmtime(json_path)}
    with open(meta_path, "w") as f:
        json.dump(m, f)
    return m



def _parse_packed_results(blob, class_names, skip_framerate_rt):
    """Parse a get_results_packed() blob (magic UPK1; see the binding) into
    trackset frames — numpy-fast, no per-detection Python churn on the pump
    thread. Frames with result_type == skip_framerate are dropped (parity
    with the dict path's frame_times filter); a 0xFFFFFFFF det count is a
    NULL list -> objects None (parity with skip frames)."""
    import struct
    off = 0
    (magic, nf) = struct.unpack_from("<II", blob, off); off += 8
    assert magic == 0x314B5055, f"bad packed magic {magic:#x}"
    frames = []
    times = []
    DET = struct.Struct("<QIfffffffffff")
    FR = struct.Struct("<difI")
    for _ in range(nf):
        stamp, rt, motion, nd = FR.unpack_from(blob, off); off += FR.size
        objects = None
        if nd != 0xFFFFFFFF:
            objects = {}
            for _d in range(nd):
                (tid, cl, conf, x0, y0, x1, y1,
                 sx0, sy0, sx1, sy1, sconf, fiqa) = DET.unpack_from(blob, off)
                off += DET.size
                objects[tid] = {"box": [x0, y0, x1, y1], "class": cl,
                                "confidence": conf,
                                "subbox": [sx0, sy0, sx1, sy1],
                                "subbox_conf": sconf, "fiqa_score": fiqa}
        if rt == skip_framerate_rt:
            continue
        frames.append({"frame_time": stamp, "result_type": rt,
                       "motion_score": motion, "motion_roi": None,
                       "inference_roi": None, "inference_dets": None,
                       "clip_embedding": None, "objects": objects,
                       "image_path": None, "debug": None})
        times.append(stamp)
    return frames, times



def _single_metrics_worker_packed(args):
    """CPU pool worker (packed variant): parse blob + GT load + scoring."""
    (params, blob, class_names, skip_rt) = args
    frames, times = _parse_packed_results(blob, class_names, skip_rt)
    trackset_gt = ts.TrackSet(params["ds_path"])
    trackset = ts.TrackSet()
    trackset.frames = frames
    trackset.frame_times = times
    trackset.metadata = {
        "frame_rate": trackset_gt.metadata["frame_rate"],
        "width": trackset_gt.metadata["width"],
        "height": trackset_gt.metadata["height"],
        # objects carry RAW model class ints -> metadata must name them;
        # scoring remaps by NAME exactly as everywhere else.
        "classes": list(class_names),
    }
    result = score_tracksets(trackset_gt, trackset, params)
    return {"params": params, "result": result,
            "time": datetime.datetime.now()}



def _single_metrics_worker(args):
    """CPU pool worker for the single-shared-state path: GT load + results
    conversion + scoring. No GPU, no CUDA context."""
    (params, track_results, class_names, attr_names) = args
    from src.tracker.upyc import upyc_results_view
    target_classes = params.get("target_classes", ["person", "face"])
    view = upyc_results_view(track_results, class_names, attr_names, target_classes)
    trackset_gt = ts.TrackSet(params["ds_path"])
    trackset = ts.TrackSet()
    tracker_run.import_create(trackset, trackset_gt,
                           track_min_interval=params["min_interval"],
                           display=params["display"],
                           config_file=None,
                           params={"target_classes": target_classes},
                           tracker=view,
                           mpwq_context=None,
                           mpwq_progress_fn=lambda *a, **k: None,
                           end_time=(100000 if params.get("max_duration_is_default")
                                     else params.get("max_duration", 100000)))
    result = score_tracksets(trackset_gt, trackset, params)
    return {"params": params, "result": result,
            "time": datetime.datetime.now()}



# Global detector performance-mode override (track.py --pm). None = leave whatever
# the yaml says. Eval/search streams are created non-realtime (the C binding
# defaults realtime=false), so they take the BATCH pm path — historically pinned
# to PM0/full resolution — and `nrt_pm` is the knob that moves them. Per-test
# `pm:` in a test entry does the same for one test; this overrides all of them.
# See ubon_cstuff/docs/design/nrt_pm_and_detection_lanes.md.
PM_OVERRIDE = None



def _resolve_pm(per_test_pm):
    """CLI --pm beats a per-test `pm:`, which beats whatever the yaml already had."""
    pm = PM_OVERRIDE if PM_OVERRIDE is not None else per_test_pm
    return None if pm is None else int(pm)



def run_single_shared(config, tests_to_run, desc, max_streams):
    """The single-shared-state eval path (MB design 2026-07-23): ONE
    c_track_shared_state per test (one engine set, one CUDA context),
    every clip submitted as its own c_track_stream — at most max_streams
    in flight, harvesting the OLDEST before submitting the next. Per-clip
    config (stream_hint) rides the per-stream yaml, exactly the production
    merge path. Scoring runs in a CPU pool as streams complete, so the GPU
    pump never waits on metrics. Replaces 8 processes x 8 engine copies
    with 1 x 1 and much higher stream concurrency."""
    import ubon_pycstuff.ubon_pycstuff as upyc
    from src.tracker.upyc import trim_aux_outputs, h264_for_video
    from multiprocessing import Pool
    from collections import deque
    import src.core.trackset as _  # noqa: ensure module import before workers fork

    by_test = {}
    for p in tests_to_run:
        by_test.setdefault(p["test_key"], []).append(p)

    entries = []
    n_cpu = max(4, (os.cpu_count() or 8) // 4)
    with Pool(n_cpu) as pool:
        pending = []
        for test_key, items in by_test.items():
            # shared config = the tracker yaml + overrides, munged EXACTLY
            # like the per-clip path (import_create's merge + aux trim).
            base = items[0]
            param_dict = {}
            cfg = base["config"]
            cfg = stuff.load_dictionary(cfg) if isinstance(cfg, str) else copy.deepcopy(cfg)
            param_dict.update(copy.deepcopy(cfg))
            override = base.get("main_config_override")
            if override:
                def _deep_merge(dst, src):
                    for k, v in src.items():
                        if isinstance(v, dict) and isinstance(dst.get(k), dict):
                            _deep_merge(dst[k], v)
                        else:
                            dst[k] = v
                _deep_merge(param_dict, copy.deepcopy(override))
            param_dict.pop("target_classes", None)
            # Detector PM for these (non-realtime) eval streams — see PM_OVERRIDE.
            _pm = _resolve_pm(base.get("nrt_pm"))
            if _pm is not None:
                param_dict["nrt_pm"] = _pm
            trim_aux_outputs(param_dict)
            import yaml as _yaml
            shared = upyc.c_track_shared_state(_yaml.dump(param_dict))
            md = shared.get_model_description()
            class_names, attr_names = md["class_names"], md["person_attribute_names"]

            # Harvest-ANY-completed, NO polling (MB 2026-07-23): each
            # in-flight stream gets a thread that BLOCKS in the C wait
            # (get_results_packed releases the GIL for wait+pack), so
            # completions are event-driven and the executor refills the
            # window the moment any stream finishes. The old harvest-oldest
            # loop was head-of-line blocking: NVDEC burst to 84% then sat
            # near-idle while younger finished streams waited on the head.
            # Stream create/run/destroy hold the GIL (pybind default) so
            # those calls stay serialized — no concurrent-create question.
            packed = (hasattr(upyc.c_track_stream, "get_results_packed")
                      and not globals().get("_FORCE_DICT", False))
            skip_rt = int(upyc.TRACK_FRAME_SKIP_FRAMERATE)
            from concurrent.futures import ThreadPoolExecutor

            def clip_task(item):
                meta = _clip_meta(item["ds_path"])
                cap = None
                if not item.get("max_duration_is_default"):
                    cap = item.get("max_duration", 100000)
                    cap = cap if cap < 9000 else None
                video = meta["original_video"]
                # mp4-direct only for lite-provenance clips: dataset_lite
                # encodes I+P-only, so the ingest's B-frame skip is a no-op.
                # Non-lite mp4s (mot/personpath22 full-rate) are B-framed
                # and keep the h264 path until they get a lite pass.
                mp4_direct = (cap is None and video.endswith(".mp4")
                              and meta.get("lite"))
                if not mp4_direct:
                    # duration-capped runs still need the trimmed h264
                    # (no time-range support in run_on_mp4_file yet)
                    h264 = h264_for_video(video, max_seconds=cap)
                per_stream = {}
                if item.get("stream_hint"):
                    per_stream["stream_hint"] = item["stream_hint"]
                # cadence rows (multi_class_and_hints.md §5 extras): the
                # debug mask is a per-STREAM key, so in the single-shared
                # path it must ride the per-stream config like stream_hint
                if item.get("debug_analytics_mask"):
                    per_stream["debug_analytics_mask"] = item["debug_analytics_mask"]
                if per_stream:
                    st = upyc.c_track_stream(shared, _yaml.dump(per_stream))
                else:
                    st = upyc.c_track_stream(shared)
                st.set_name(item["ds_key"])
                st.set_frame_intervals(item["min_interval"], -1.0)
                if mp4_direct:
                    # container pts drive frame timing (no synthetic-fps
                    # clock to get wrong — the stale-h264 lesson);
                    # base_time=0 keeps result times on the GT's 0-based
                    # clock. NB analytics skips B-frames on this path:
                    # fine for lite files (encoded I+P), wrong for
                    # B-framed full-rate sources — those keep the h264
                    # path via max_duration or a non-mp4 extension.
                    st.run_on_mp4_file(video, 0.0)
                else:
                    st.run_on_video_file(h264, upyc.SIMPLE_DECODER_CODEC_H264,
                                         meta["frame_rate"], False)
                if packed:
                    blob = st.get_results_packed(3600.0)
                    del st
                    return pool.apply_async(
                        _single_metrics_worker_packed,
                        ((item, blob, class_names, skip_rt),))
                results = st.get_results(3600.0)
                del st
                return pool.apply_async(
                    _single_metrics_worker,
                    ((item, results, class_names, attr_names),))

            with ThreadPoolExecutor(max_workers=max_streams) as ex:
                for fut in [ex.submit(clip_task, item) for item in items]:
                    pending.append(fut.result())
            del shared
        entries = [p.get() for p in pending]
    return entries



def on_result_callback(mpwq_context, result):
    cache=True
    ds_key=result["params"]["ds_key"]
    if "no_cache" in mpwq_context["config"]["datasets"][ds_key]:
        if mpwq_context["config"]["datasets"][ds_key]["no_cache"]==True:
            cache=False
    if cache is True and mpwq_context["resultfile"] is not None:
        mpwq_context["cached_results"].append(result)
        stuff.save_atomic_pickle(mpwq_context["cached_results"], mpwq_context["resultfile"])
        #logging.info(f"Saved {len(mpwq_context["cached_results"])} cached results")

def track_test(config, split=None, desc="track test"):
    start_time=time.time()
    if isinstance(config, str):
        config=stuff.load_dictionary(config)

    if "framerates" in config:
        expanded_tests={}
        for t in config["tests"]:
            c=config["tests"][t]
            if "min_interval" in c:
                expanded_tests[t]=c
                continue
            for f in config["framerates"]:
                t_fr=copy.deepcopy(c)
                if f<0:
                    t_fr["min_interval"]=f
                else:
                    t_fr["min_interval"]=1/(f+0.01)
                expanded_tests[t+f", {f}fps"]=t_fr
        config["tests"]=expanded_tests

    resultfile=None
    if "results_cache_file" in config:
        resultfile=config["results_cache_file"]
    num_workers=stuff.resolve_num_workers(config["num_workers"])
    cached_results=[]
    if resultfile is not None and os.path.isfile(resultfile):
        with open(resultfile, 'rb') as handle:
            cached_results = pickle.load(handle)

    datasets=config["datasets"]
    tests=config["tests"]
    columns=config["columns"]
    output_results=[]

    # Optional family allow-list. Absent/empty => use all families.
    include_families=config.get("include_families")
    if isinstance(include_families,str):
        include_families=[f.strip() for f in include_families.split(",") if f.strip()]
    if include_families:
        include_families=set(include_families)

    tests_to_run=[]

    for _,ds_key in enumerate(datasets):
        dataset=datasets[ds_key]
        if include_families and dataset.get("family") not in include_families:
            continue
        if split is not None:
            if "split" in dataset:
                if dataset["split"]!=split:
                    continue
        for test_key in tests:
            result=None
            for r in cached_results:
                if r["params"]["test_key"]==test_key and r["params"]["ds_key"]==ds_key:
                    if "regenerate" in datasets[ds_key] and datasets[ds_key]["regenerate"]==True:
                        r["params"]["need_regenerate"]=True
                        continue
                    if "regenerate" in tests[test_key] and tests[test_key]["regenerate"]==True:
                        r["params"]["need_regenerate"]=True
                        continue
                    result=r
            if result is None:

                test=tests[test_key]
                params={}
                for p in test:
                    params[p]=test[p]
                # Per-test `pm:` (and the --pm global override) select the
                # detector tier for this test's streams. Normalised to `nrt_pm`
                # here so BOTH the single-shared path and the per-clip path see
                # one key — the latter passes params through to the tracker
                # config yaml verbatim, exactly as stream_hint rides across.
                _pm = _resolve_pm(test.get("pm"))
                if _pm is not None:
                    params["nrt_pm"] = _pm
                params.pop("pm", None)
                if not "max_duration" in params:
                    params["max_duration"]=1000
                    # metrics-window default only — NOT a media cap. The
                    # truncated-h264 path must never build _t1000 variants
                    # of every clip off this default (found 2026-07-23:
                    # 306 pointless serial demuxes).
                    params["max_duration_is_default"]=True
                # copy some parameters from top level to each test config
                params_to_copy=["eval_rate_divisor", "eval_min_framerate", "min_person_height",
                                "classes", "classes_for_det_map", "fitness_weights"]
                for p in params_to_copy:
                    if p in config:
                        params[p]=config[p]
                # Dataset extras (multi_class_and_hints.md §5): every
                # per-dataset key beyond the harness-reserved set rides
                # into params → import_create's param_dict → the tracker
                # config yaml verbatim. This is how `stream_hint: bodycam`
                # reaches track_stream_create (the C side resolves the
                # (hint:x) variant axis from it).
                reserved={"path","split","group","family","regenerate","no_cache"}
                for k,v in dataset.items():
                    if k not in reserved:
                        params[k]=v
                params["ds_path"]=dataset["path"]
                params["display"]=f"{len(tests_to_run):02d}: "+ds_key+"/"+test_key
                params["ds_key"]=ds_key
                params["test_key"]=test_key

                tests_to_run.append(params)
            else:
                output_results.append(result)


    # LARGEST-FIRST dispatch: tried 2026-07-23 and REVERTED on measurement —
    # 793s vs 182s unsorted (4.3x WORSE). Sorting by annotation size
    # co-schedules the giant-GT clips (MOT20/PP22, 100-200MB JSONs ->
    # gigabytes parsed EACH) into one concurrency window; the resulting
    # memory pressure dwarfs the ~40s queue-tail it was meant to save.
    # If tail-packing is ever revisited it must interleave memory monsters,
    # not cluster them.

    # single_shared_streams: N = the MB single-shared-state path (one engine
    # set, N concurrent streams, CPU-pool scoring). Absent = the mp path.
    if config.get("single_shared_streams"):
        results = run_single_shared(config, tests_to_run, desc,
                                    int(config["single_shared_streams"]))
        for entry in results:
            output_results.append(entry)
        for o in output_results:
            if "time" in o:
                o["result"]["time"]=(datetime.datetime.now()-o["time"]).total_seconds()
            if "group" in config["datasets"][o["params"]["ds_key"]]:
                o["group"]=config["datasets"][o["params"]["ds_key"]]["group"]
        results2=display_results(config, output_results, columns, config["sort_key"])
        elapsed=time.time()-start_time
        _write_eval_summary_json(config, output_results, results2, elapsed)
        print(f"All done (single-shared): Evaluated {len(tests_to_run)} tests in {stuff.timestr(elapsed)}")
        return results2

    cached_results_new=[r for r in cached_results if "need_regenerate" not in r["params"]]
    logging.info(f"cached results {len(cached_results)}; deleting {len(cached_results)-len(cached_results_new)} need to run {len(tests_to_run)} tests")
    cached_results=cached_results_new

    on_result_context={"cached_results": cached_results,
                       "config":config,
                       "resultfile": resultfile}

    results = stuff.mp_workqueue_run(tests_to_run,
                                     track_test_work_fn,
                                     num_workers=num_workers,
                                     desc=desc,
                                     result_callback_context=on_result_context,
                                     result_callback=on_result_callback)

    for entry in results:
        output_results.append(entry)

    for o in output_results:
        if "time" in o:
            o["result"]["time"]=(datetime.datetime.now()-o["time"]).total_seconds()
        if "group" in config["datasets"][o["params"]["ds_key"]]:
            o["group"]=config["datasets"][o["params"]["ds_key"]]["group"]

    results2=display_results(config, output_results, columns, config["sort_key"])
    elapsed=time.time()-start_time
    _write_eval_summary_json(config, output_results, results2, elapsed)
    print(f"All done: Evaluated {len(tests_to_run)} tests in {stuff.timestr(elapsed)}")
    return results2
