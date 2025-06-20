
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
                 classes=["person","face"]):
        self.params=params
        self.classes=classes
        trackset=self.params["original_trackset"]
        del self.params["original_trackset"]

        # write config to yaml file so we can pass it as parameter to cevo binary

        fd, self.tmp_config_file=tempfile.mkstemp(dir="/tmp", prefix="cevo_config_", suffix=".yaml")
        os.close(fd)
        with open(self.tmp_config_file, 'w') as outfile:
            yaml.dump(self.params, outfile, default_flow_style=False)
        video=trackset.metadata["original_video"]
        fps=trackset.metadata["frame_rate"]
        divisor=1

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
                stuff.mp4_to_h264(video, h264_file, debug=exe_debug)
        else:
            h264_file=tempfile.NamedTemporaryFile(delete=False, suffix=".h264").name
            stuff.rm(h264_file)
            stuff.mp4_to_h264(video, h264_file, debug=exe_debug)
            h264_file_temp=h264_file

        assert os.path.isfile(h264_file), "Failed to create h264 file"

        track_shared=upyc.c_track_shared_state("/mldata/config/track/trackers/uc_test.yaml")
        self.md=track_shared.get_model_description()
        track_stream=upyc.c_track_stream(track_shared)
        track_stream.set_frame_intervals(track_min_interval, 120.0)
        track_stream.run_on_video_file(h264_file, upyc.SIMPLE_DECODER_CODEC_H264, fps)
        self.track_results=track_stream.get_results()
        self.frame_times=[]
        self.frame_indexes=[]
        for i,r in enumerate(self.track_results):
            if r['result_type']!=upyc.TRACK_FRAME_SKIP_FRAMERATE:
                self.frame_times.append(r["time"])
                self.frame_indexes.append(i)

        if h264_file_temp is not None:
            stuff.rm(h264_file_temp)

        self.fn=0

    def __del__(self):
        try:
            os.remove(self.tmp_config_file)
        except Exception as e:
            print("ERROR :: ", "cevo_tracker", e)

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
                objects.append(o)

        if 'inference_dets' in r and r['inference_dets'] is not None:
            debug|={"detections": {"type": "yolo_detections",
                                   "data":{"detections":r['inference_dets'],
                                           "class_names":self.md['class_names'],
                                           "attributes":self.md['person_attribute_names']}}}

        return objects, debug
