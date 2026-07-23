
import tempfile
import stuff
import src.track_util as tu
import os
import yaml
import logging
from pathlib import Path
import ubon_pycstuff.ubon_pycstuff as upyc


RESULT_TYPE_NAMES = {
    upyc.TRACK_FRAME_SKIP_FRAMERATE: "skip_framerate",
    upyc.TRACK_FRAME_SKIP_NO_MOTION: "skip_no_motion",
    upyc.TRACK_FRAME_SKIP_NO_IMG: "skip_no_img",
    upyc.TRACK_FRAME_TRACKED_ROI: "tracked_roi",
    upyc.TRACK_FRAME_TRACKED_FULL_REFRESH: "tracked_full_refresh",
}

class upyc_tracker:
    def __init__(self, params,
                 track_min_interval,
                 debug_enable=False,
                 cache_h264=True,
                 classes=["person","face"],
                 max_duration=10000.0,
                 start_time=0,
                 end_time=10000.0):

        self.params=params
        self.classes=classes

        trackset=self.params["original_trackset"]
        del self.params["original_trackset"]

        video=trackset.metadata["original_video"]
        fps=trackset.metadata["frame_rate"]

        name=os.path.basename(video)
        logging.debug(f"upyc tracker init {name}")

        # convert mp4 file into h264 using ffmpeg
        # by default we will put the converted file in a "generated" subfolder
        # of where the mp4 is so we can reuse it next time - if you don't want
        # to do this use cache_h264=False which will use a temp file instead

        # end_time < the sentinel default = a REAL time cap: truncate the
        # elementary stream so the C pipeline decodes/tracks only the window
        # (run_on_video_file has no stop-time of its own). The cap is baked
        # into the CACHE NAME — an uncapped eval must never pick up a
        # truncated file, and capped search iterations reuse their cut.
        cap = end_time if (end_time and end_time < 9000.0) else None
        suffix = f"_t{int(cap)}" if cap else ""
        h264_file_temp=None
        h264_file=None
        if cache_h264 and video.endswith(".mp4"):
            p = Path(video)
            h264_file=str(p.with_name("generated_h264") / (p.stem + suffix + ".h264"))
            gen_dir = p.with_name("generated_h264")
            gen_dir.mkdir(parents=True, exist_ok=True)
            if not os.path.isfile(h264_file):
                stuff.mp4_to_h264(video, h264_file, max_seconds=cap)
        else:
            h264_file=tempfile.NamedTemporaryFile(delete=False, suffix=".h264").name
            stuff.rm(h264_file)
            stuff.mp4_to_h264(video, h264_file, max_seconds=cap)
            h264_file_temp=h264_file

        assert os.path.isfile(h264_file), "Failed to create h264 file"

        # By default we trim auxiliary outputs for faster metric runs.
        # If debug is enabled, preserve the full configured detector/tracker output.
        if debug_enable is False:
            # whole-frame preview stream (renamed main_jpeg -> thumbnail_stream)
            for _k in ("thumbnail_stream", "main_jpeg"):
                if _k in params:
                    params[_k]["enabled"]=False
            if "faces" in params:
                params["faces"]["embeddings_enabled"]=False
                params["faces"]["jpegs_enabled"]=False
            if "clip" in params:
                params["clip"]["frame_embeddings_enabled"]=False
                params["clip"]["object_embeddings_enabled"]=False
                params["clip"]["jpegs_enabled"]=False
            if "fiqa" in params:
                params["fiqa"]["enabled"]=False

        yaml_string=yaml.dump(params)

        if "proxy" in params:
            port=18861
            proxy=params['proxy']
            if ":" in proxy:
                ip=proxy.split(":")[0]
                port=int(proxy.split(":")[1])
            else:
                ip=proxy
            import ubon_cproxy
            # remote_cli=ubon_cproxy.upyc_proxy() # to run on this PC
            remote_cli =ubon_cproxy.upyc_proxy(ip, port)
            track_shared=remote_cli.c_track_shared_state(yaml_string)
            self.md=track_shared.get_model_description()
            track_stream=remote_cli.c_track_stream(track_shared)
            track_stream.set_frame_intervals(track_min_interval, -1.0)  # -1 = leave at yaml default
            track_stream.run_on_video_file(h264_file, remote_cli.SIMPLE_DECODER_CODEC_H264, fps, False)
        else:
            logging.debug(f"upyc create")
            track_shared=upyc.c_track_shared_state(yaml_string)
            self.md=track_shared.get_model_description()
            track_stream=upyc.c_track_stream(track_shared)
            track_stream.set_name(name)
            track_stream.set_frame_intervals(track_min_interval, -1.0)  # -1 = leave at yaml default
            logging.debug(f"upyc run")
            track_stream.run_on_video_file(h264_file, upyc.SIMPLE_DECODER_CODEC_H264, fps, False)

        logging.debug(f"get results")
        self.track_results=track_stream.get_results(120.0, include_full_debug=debug_enable)
        del track_stream
        del track_shared
        self.frame_times=[]
        self.frame_indexes=[]
        for i,r in enumerate(self.track_results):
            if r['result_type']!=upyc.TRACK_FRAME_SKIP_FRAMERATE:
                # upyc renamed the frame's session media stamp from "time"
                # to "media_stamp" in the epoch-clock rework
                self.frame_times.append(r.get("media_stamp", r.get("time")))
                self.frame_indexes.append(i)

        if h264_file_temp is not None:
            stuff.rm(h264_file_temp)

        self.class_remap=stuff.make_class_remap_table(self.md['class_names'], classes)

        self.fn=0
        logging.debug(f"upyc done")

    # This function checks the file_path and if it is not present
    # checks a given default_path with respect to the PWD.  When
    # both are not present, raise and exception
    def check_and_get_filepath_or_default(self, file_path, default_path):
        if os.path.isfile(file_path):
            return os.path.realpath(file_path)

        default_path = os.path.join(os.getcwd(), default_path)
        if os.path.isfile(default_path):
            return os.path.realpath(default_path)

        raise FileNotFoundError(f"File not found {file_path} or {default_path}")

    def get_frame_times(self):
        return self.frame_times

    def needs_frames(self):
        return False

    def track_frame(self, frame, time, debug_enable=False):
        assert time in self.frame_times
        idx=self.frame_indexes[self.frame_times.index(time)]
        r=self.track_results[idx]

        objects=None
        if 'track_dets' in r and r['track_dets'] is not None:
            objects=[]
            for d in r['track_dets']:
                o=tu.Object(detection=d, time=time)
                o.track_id=d['track_id']
                o.cl=self.class_remap[o.cl]
                if o.cl is not None:
                    objects.append(o)

        debug=dict(r.get("debug") or {})
        inference_dets=r.get("inference_dets")
        if inference_dets is not None and "detector_output" not in debug:
            debug["detector_output"] = {
                "type": "detections",
                "data": {
                    "detections": inference_dets,
                    "class_names": self.md["class_names"],
                    "attribute_names": self.md["person_attribute_names"],
                },
            }
        if inference_dets is not None:
            # Convenience mapped overlay for current target classes, while retaining
            # the full detector output above for analysis.
            out_dets=[]
            for d in inference_dets:
                cl=self.class_remap[d["class"]]
                if cl is not None:
                    mapped=dict(d)
                    mapped["class"]=cl
                    out_dets.append(mapped)
            debug["detector_output_mapped"] = {
                "type": "detections",
                "data": {
                    "detections": out_dets,
                    "class_names": self.classes,
                    "attribute_names": self.md["person_attribute_names"],
                },
            }

        return {
            "frame_time": time,
            "result_type": RESULT_TYPE_NAMES.get(r["result_type"], str(r["result_type"])),
            "motion_score": r.get("motion_score", 0.0),
            "motion_roi": r.get("motion_roi"),
            "inference_roi": r.get("inference_roi"),
            "inference_dets": inference_dets,
            "clip_embedding": r.get("clip_embedding"),
            "objects": objects,
            "debug": debug if len(debug) > 0 else None,
        }
