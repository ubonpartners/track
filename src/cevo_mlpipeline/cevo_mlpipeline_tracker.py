import subprocess
import tempfile
import stuff
import src.track_util as tu
import os

def run_cmd(cmd, debug=False):
    result=subprocess.run(cmd, stdout=subprocess.DEVNULL, capture_output=True, text=True)
    stdout_str = result.stdout
    stderr_str = result.stderr
    if result.returncode!=0:
        print(f" command {cmd} failed")
        print(f" STDOUT: {stdout_str}")
        print(f" STDERR: {stderr_str}")
        exit()

class cevo_mlpipe_tracker:
    def __init__(self, params, track_min_interval):
        #print(params)
        trackset=params["original_trackset"]
        video=trackset.metadata["original_video"]
        fps=int(trackset.metadata["frame_rate"]+0.5)
        divisor=1
        while(divisor/fps<track_min_interval):
            divisor+=1
        
        #print(fps, track_min_interval, divisor)
        #print(video)

        exe_debug=False

        h264_file=tempfile.NamedTemporaryFile(delete=False, suffix=".h264").name
        cevo_out_folder=tempfile.NamedTemporaryFile(delete=False, suffix=".out").name
        stuff.rm(h264_file)
        stuff.rm(cevo_out_folder)

        # use FFMPEG to convert original video file to h264 stream

        cmd=["ffmpeg",
            "-i", video,
            "-c:v", "copy",
            "-bsf:v", "h264_mp4toannexb",
            "-an",
            "-f", "h264",
            h264_file]

        run_cmd(cmd, debug=exe_debug)

        assert os.path.isfile(h264_file), "Failed to create h264 file"

        # run Cevo video_dec_trt; runs tracker and outputs JSON annotations
        
        cmd=[params["exe"],
            "-n", "1", "-p", "1", "-d", cevo_out_folder,
            "--pix_fmt", "H264",
            "-f",f"{fps}/{divisor}",
            "-v", h264_file,
            "--trt-enginefile", params["trt"],
            "--display", "0x00", "--dbg-level", "0"]
        
        run_cmd(cmd, debug=exe_debug)

        assert os.path.isdir(cevo_out_folder), "Failed to create output cevo data"

        frames=os.listdir(cevo_out_folder+"/"+os.path.basename(h264_file))
        frames.sort()
        self.frames={}
        for f in frames:
            d=stuff.load_dictionary(cevo_out_folder+"/"+os.path.basename(h264_file)+"/"+f)
            fn=d["frame_num"]-1
            d["test_time"]=fn/trackset.metadata["frame_rate"]
            self.frames[fn]=d

        stuff.rm(h264_file)
        stuff.rmdir(cevo_out_folder)

        self.fn=0

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
            d={"box":box, "class":0, "confidence":conf}
            o=tu.Object(detection=d, time=time)
            o.track_id=track_id
            objects.append(o)
        #print(self.frames[self.fn])
        self.fn+=1
        return objects, None