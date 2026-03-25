
import tempfile
import stuff
import src.track_util as tu
import os
import yaml
import logging
from pathlib import Path
import ubon_pycstuff.ubon_pycstuff as upyc

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

        h264_file_temp=None
        h264_file=None
        if cache_h264 and video.endswith(".mp4"):
            p = Path(video)
            h264_file=str(p.with_name("generated_h264") / p.with_suffix(".h264").name)
            gen_dir = p.with_name("generated_h264")
            gen_dir.mkdir(parents=True, exist_ok=True)
            if not os.path.isfile(h264_file):
                stuff.mp4_to_h264(video, h264_file)
        else:
            h264_file=tempfile.NamedTemporaryFile(delete=False, suffix=".h264").name
            stuff.rm(h264_file)
            stuff.mp4_to_h264(video, h264_file)
            h264_file_temp=h264_file

        assert os.path.isfile(h264_file), "Failed to create h264 file"

        # disable a bunch of stuff not needed to measure MOTA stats
        # this makes it quite a bit faster to run

        params["main_jpeg"]["enabled"]=False
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
            track_stream.set_frame_intervals(track_min_interval, 120.0)
            track_stream.run_on_video_file(h264_file, remote_cli.SIMPLE_DECODER_CODEC_H264, fps, False)
        else:
            logging.debug(f"upyc create")
            track_shared=upyc.c_track_shared_state(yaml_string)
            self.md=track_shared.get_model_description()
            track_stream=upyc.c_track_stream(track_shared)
            track_stream.set_name(name)
            track_stream.set_frame_intervals(track_min_interval, 120.0)
            logging.debug(f"upyc run")
            track_stream.run_on_video_file(h264_file, upyc.SIMPLE_DECODER_CODEC_H264, fps, False)

        logging.debug(f"get results")
        self.track_results=track_stream.get_results(120.0)
        del track_stream
        del track_shared
        self.frame_times=[]
        self.frame_indexes=[]
        for i,r in enumerate(self.track_results):
            if r['result_type']!=upyc.TRACK_FRAME_SKIP_FRAMERATE:
                self.frame_times.append(r["time"])
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

        debug={}
        debug|={"detection_roi": {"type": "roi", "data": {"roi":r['inference_roi']}}}
        debug|={"motion_roi": {"type": "roi", "data": {"roi":r['motion_roi']}}}

        objects=None
        if 'track_dets' in r and r['track_dets'] is not None:
            objects=[]
            for d in r['track_dets']:
                o=tu.Object(detection=d, time=time)
                o.track_id=d['track_id']
                o.cl=self.class_remap[o.cl]
                if o.cl is not None:
                    objects.append(o)

        if 'inference_dets' in r and r['inference_dets'] is not None:
            # remap classes to target class set
            out_dets=[]
            for d in r['inference_dets']:
                cl=self.class_remap[d["class"]]
                if cl is not None:
                    d["class"]=cl
                    out_dets.append(d)
            debug|={"detections": {"type": "yolo_detections",
                                   "data":{"detections":out_dets,
                                           "class_names":self.classes,
                                           "attributes":self.md['person_attribute_names']}}}

        return objects, debug
