"""Run a tracker over a video (or a GT TrackSet's frames) and fill a
TrackSet with its results: the body of the old TrackSet.import_create,
moved verbatim with `self` renamed `ts` (repo_cleanup.md stage 4c). It
lives in the tracker layer because it creates the tracker; core must not
import tracker.
"""
from tqdm.auto import tqdm
import cv2
import stuff

from src.core.trackset import TrackSet
import src.tracker.factory as factory


def import_create(ts,
                  video,
                  track_min_interval=0.05,
                  display="Tracking...",
                  pbar=None,
                  config_file=None,
                  params=None,
                  debug=False,
                  debug_enable=False,
                  mpwq_context=None,
                  mpwq_progress_fn=None,
                  start_time=0,
                  end_time=100000,
                  tracker=None):

    assert len(ts.frame_times)==0

    param_dict={}
    if config_file is not None:
        if isinstance(config_file, str):
            ts.name=f"Import-create {stuff.name_from_file(config_file)}"
            config=stuff.load_dictionary(config_file)
            for c in config:
                param_dict[c]=config[c]
        else:
            ts.name=f"Import-create noname"
            for c in config_file:
                param_dict[c]=config_file[c]
    if params is not None:
        for p in params:
            param_dict[p]=params[p]
    # main_config_override: deep-merge overrides into the loaded tracker
    # config. Lets a search/eval yaml disable aux stages (faces/clip/
    # audio jpegs+embeddings) for offline runs without editing the
    # tracker config file itself. Nested dicts merge key-by-key; any
    # other value replaces.
    override=param_dict.pop("main_config_override", None)
    if override:
        def _deep_merge(dst, src):
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    _deep_merge(dst[k], v)
                else:
                    dst[k]=v
        _deep_merge(param_dict, override)
    # target_classes: which tracker output classes this import KEEPS
    # (multi_class_and_hints.md §1). utrack emits vehicle/animal tracks
    # now; the old hardcoded ["person","face"] dropped them on the
    # floor. A test entry (or main_config_override) requests e.g.
    # ["person","face","vehicle","animal"]; default unchanged. Popped:
    # it is an import policy, not a tracker config key.
    target_classes=param_dict.pop("target_classes", ["person", "face"])
    # tracker injection (single-shared-state eval path): the GPU work
    # already happened on a stream elsewhere; the caller hands us a
    # results VIEW exposing the same interface, and this import is pure
    # dict->frame conversion.
    if tracker is None:
        param_dict["original_trackset"]=video
        tracker=factory.create_tracker(param_dict,
                                        track_min_interval=track_min_interval,
                                        debug_enable=debug_enable,
                                        start_time=start_time,
                                        end_time=end_time,
                                        classes=target_classes)

    frame_times=None
    if hasattr(tracker, 'get_frame_times'):
        frame_times=tracker.get_frame_times()

    needs_frames=True
    if hasattr(tracker, 'needs_frames'):
        needs_frames=tracker.needs_frames()

    cap=None

    if isinstance(video, TrackSet):
        if video.source_name is not None:
            ts.name+=f":{stuff.name_from_file(video.source_name)}"
        else:
            ts.name+=f" none {video.name}"
        fps=video.metadata["frame_rate"]
        duration=video.duration_seconds()
        width=video.metadata["width"]
        height=video.metadata["height"]
    else:
        ts.name+=f" Video={stuff.name_from_file(video)}"
        ts.source_name=video
        cap = cv2.VideoCapture(video)
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration=fps*frame_count
        if needs_frames is False:
            del cap

    t=0

    ts.metadata={
            "frame_rate": fps,
            "width": width,
            "height": height,
            "classes": target_classes,
        }

    if isinstance(video, TrackSet):
        if "original_video" in video.metadata:
            ts.metadata["original_video"]=video.metadata["original_video"]

    if frame_times is None:
        frame_times=[]
        t=0
        while t<=duration and t<end_time:
            if (t>=start_time):
                frame_times.append(t)
            t+=(1.0/fps)

    frame_times=[t for t in frame_times if t>=start_time and t<=end_time]

    if pbar is None and mpwq_progress_fn is None:
        if stuff.platform_stuff.is_jetson():
            tqdm.monitor_interval = 0 # seems to crash here otherwise,
        pbar=tqdm(total=len(frame_times),
                  desc=f"{display:35s}",
                  colour="#ffcc00",
                  leave=False)
    elif mpwq_progress_fn is not None:
        mpwq_progress_fn(mpwq_context, desc=f"{display:35s}", total=len(frame_times))

    if debug:
        display=stuff.Display(width=1280, height=720)

    fn=0
    for t in frame_times:
        frame=None
        if needs_frames:
            if cap is not None:
                success, frame = cap.read()
                if success is False:
                    break
            else:
                frame=video.img_at_time(t)
            if frame is None:
                break

        frame_result=tracker.track_frame(frame, t, debug_enable=debug_enable)
        objects=frame_result.get("objects")

        if debug:
            display.clear()
            if objects is not None:
                for o in objects:
                    o.draw(display, clr=(128,255,255,255), thickness=1)
            display.show(frame, title=f"time={t:5.2f}")
            events=display.get_events(0)

        if frame_result is not None:
            # VFR/jittery sources (OTW doorbell cams) can step a frame
            # timestamp BACKWARD (the decoder logs "unexpected delta ...;
            # resetting" and carries on). A non-monotonic result used to
            # trip add_frame's assert and kill the whole eval worker at
            # 90% of a clip — drop the blip instead and say so once.
            ft=frame_result["frame_time"]
            if len(ts.frame_times)>0 and ft<=ts.frame_times[-1]:
                nonmono_dropped=getattr(ts, "_nonmono_dropped", 0)+1
                ts._nonmono_dropped=nonmono_dropped
                if nonmono_dropped==1:
                    import logging
                    logging.warning(
                        f"{ts.name}: non-monotonic frame time {ft:.4f} after "
                        f"{ts.frame_times[-1]:.4f} — dropping (VFR source jitter; "
                        f"further drops counted silently)")
            else:
                img_path=video.img_path_at_time(t) if cap is None else None
                ts.add_frame_result(frame_result, img_path=img_path)
        fn+=1
        if pbar is not None:
            pbar.update(1)
        elif mpwq_progress_fn is not None:
            mpwq_progress_fn(mpwq_context, update=1)

    if getattr(ts, "_nonmono_dropped", 0) > 0:
        import logging
        logging.warning(f"{ts.name}: dropped {ts._nonmono_dropped} "
                        f"non-monotonic frame result(s) (VFR source jitter)")
    if cap is not None:
        cap.release()
