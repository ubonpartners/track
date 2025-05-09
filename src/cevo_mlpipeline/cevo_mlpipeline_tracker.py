
import tempfile
import stuff
import src.track_util as tu
import os
import yaml
from pathlib import Path

class cevo_mlpipe_tracker:
    def __init__(self, params, track_min_interval, debug_enable=False, cache_h264=True, classes=["person","face"]):
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
        fps=int(trackset.metadata["frame_rate"]+0.5)
        divisor=1
        while(divisor/fps<track_min_interval):
            divisor+=1

        exe_debug=debug_enable

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

        cevo_out_folder=tempfile.NamedTemporaryFile(delete=False, suffix=".out").name
        stuff.rm(cevo_out_folder)

        # run Cevo video_dec_trt; runs tracker and outputs JSON annotations

        cmd=[params["exe"],
            "-c", self.tmp_config_file,
            "-n", "1", "-p", "1", "-d", cevo_out_folder,
            "--pix_fmt", "H264",
            "-f",f"{fps}/{divisor}",
            "-v", h264_file,
            "--trt-enginefile", params["trt"],
            "--display", "0x02", "--dbg-level", "0"]

        stuff.run_cmd(cmd, debug=exe_debug)

        assert os.path.isdir(cevo_out_folder), "Failed to create output cevo data"

        frames=os.listdir(cevo_out_folder+"/"+os.path.basename(h264_file))
        frames=[f for f in frames if "_det" not in f]
        frames.sort()
        self.frames={}
        for f in frames:
            fname=cevo_out_folder+"/"+os.path.basename(h264_file)+"/"+f
            d=stuff.load_dictionary(fname)
            fname_det=fname.replace(".json", "_det.json")
            if os.path.exists(fname_det):
                d["det_debug"]=stuff.load_dictionary(fname_det)
            fn=d["frame_num"]-1
            d["test_time"]=fn/trackset.metadata["frame_rate"]
            self.frames[fn]=d

        if h264_file_temp is not None:
            stuff.rm(h264_file_temp)
        stuff.rmdir(cevo_out_folder)

        self.fn=0

    def __del__(self):
        try:
            os.remove(self.tmp_config_file)
        except Exception as e:
            print("ERROR :: ", "cevo_tracker", e)

    def track_frame(self, frame, time, debug_enable=False):
        if not self.fn in self.frames:
            self.fn+=1
            return None, None
        h,w,_=frame.shape
        det_frame=self.frames[self.fn]
        objects=[]
        #print(det_frame)
        if det_frame["bbox"] is None:
            self.fn+=1
            return [], None

        assert abs(time-det_frame["test_time"])<0.1

        for det in det_frame["bbox"]:
            b=det["bbox"]
            conf=det["confidence"]
            track_id=det["track_id"]
            bx=b["x"]
            by=b["y"]
            bw=b["w"]
            bh=b["h"]
            box=[stuff.clip01((bx)/w),
                 stuff.clip01((by)/h),
                 stuff.clip01((bx+bw)/w),
                 stuff.clip01((by+bh)/h)]
            # currently JSON only contains person tracks and no class field
            d={"box":box, "class":self.classes.index("person"), "confidence":conf}
            o=tu.Object(detection=d, time=time)
            o.track_id=track_id
            objects.append(o)
        debug=None

        if "bbox_organised_dets" in det_frame:
            inter_w,inter_h=det_frame["inter_width"], det_frame["inter_height"]
            infer_w,infer_h=det_frame["infer_width"], det_frame["infer_height"]
            roi_x=det_frame["roi"]["x"]/inter_w
            roi_y=det_frame["roi"]["y"]/inter_h
            roi_w=det_frame["roi"]["w"]/inter_w
            roi_h=det_frame["roi"]["h"]/inter_h
            dets=[]
            for d in det_frame["bbox_organised_dets"]:
                if d["class_id"]=='Person' or d["class_id"]=='Face':
                    x0=d["bbox"]["cx"]
                    y0=d["bbox"]["cy"]
                    w=d["bbox"]["w"]
                    h=d["bbox"]["h"]
                    x1=x0+w/2
                    y1=y0+h/2
                    x0=x0-w/2
                    y0=y0-h/2
                    conf=d["confidence"]
                    x0=x0/infer_w
                    y0=y0/infer_h
                    x1=x1/infer_w
                    y1=y1/infer_h
                    box=[roi_x+roi_w*x0, roi_y+roi_h*y0, roi_x+roi_w*x1, roi_y+roi_h*y1]
                    det={"box":box, "class":(self.classes.index(d["class_id"].lower())), "confidence":conf}
                    dets.append(det)
            debug={"detections": {"type": "yolo_detections", "data":{"detections":dets, "class_names":["person"], "attributes":None}}}
            debug|={"detection_roi": {"type": "roi", "data": {"roi":[roi_x, roi_y, roi_x+roi_w, roi_y+roi_h]}}}

        #print(self.frames[self.fn])
        self.fn+=1
        return objects, debug