import ultralytics
import src.track_util as tu
import os
import tempfile
import yaml
import stuff
import src.utrack.utracker as ut
import src.upyc_tracker.upyc_tracker as upyct

class ultralytics_tracker:

    def __init__(self, params, track_min_interval, debug_enable=False):
        self.tmp_file=None
        if "tracker_model" in params:
            self.yolo = ultralytics.YOLO(params["tracker_model"])
        else:
            self.yolo = ultralytics.YOLO(params["model"])
        self.track_min_interval=track_min_interval
        self.class_names=[self.yolo.names[i] for i in range(len(self.yolo.names))]
        self.last_track_time=-1000
        self.frames_skipped_count=0
        self.params=params
        self.person_class_index=self.class_names.index("person")
        fd, self.tmp_config_file=tempfile.mkstemp(dir="/tmp", prefix="yolo_config", suffix=".yaml")
        os.close(fd)
        if "original_trackset" in self.params:
            del self.params["original_trackset"]
        with open(self.tmp_config_file, 'w') as outfile:
            yaml.dump(self.params, outfile, default_flow_style=False)

    def __del__(self):
        try:
            os.remove(self.tmp_config_file)
        except Exception as e:
            print("ERROR :: ", "ultralytics_tracker", e)

    def track_frame(self, img, t, debug_enable=False):

        if self.track_min_interval>=0:
            do_track=t-self.last_track_time>=self.track_min_interval
        else:
            do_track=self.frames_skipped_count+1>=(-self.track_min_interval)
            if do_track:
                self.frames_skipped_count=0
            else:
                self.frames_skipped_count+=1

        det_w=stuff.get_dict_param(self.params, "det_w", 640)
        det_h=stuff.get_dict_param(self.params, "det_h", 640)

        objects=None
        if do_track:
            results = self.yolo.track(img,
                                      imgsz=[det_h, det_w],
                                      persist=True,
                                      classes=[self.person_class_index],
                                      verbose=False,
                                      rect=True,
                                      conf=max(0.01, min(0.95, self.params["track_low_thresh"])),
                                      iou=max(0.01, min(0.95, self.params["nms_iou"])),
                                      half=True,
                                      max_det=600,
                                      tracker=self.tmp_config_file)

            out_det=stuff.yolo_results_to_dets(results[0],
                                            det_thr=max(0.01, min(0.95, self.params["track_low_thresh"])),
                                            yolo_class_names=self.class_names,
                                            class_names=self.class_names,
                                            face_kp=True,
                                            pose_kp=True,
                                            params=self.params)

            person_dets=[d for d in out_det if d["class"]==self.person_class_index]
            objects=[]
            for d in person_dets:
                o=tu.Object(detection=d, time=t)
                if d["id"] is not None:

                    o.track_id=d["id"]
                    objects.append(o)

            self.last_track_time=t
        return objects, None


def create_tracker(param_dict, track_min_interval,
                   debug_enable=False,
                   start_time=0,
                   end_time=1000000.0,
                   classes=None):
    assert "tracker_type" in param_dict, "tracker type must be specified"

    if not "model" in param_dict:
        param_dict["model"]="/mldata/weights/good/yolo11l-dpa-131224.pt"
        #print(f"WARNING: Model not specified in config; using default model {param_dict['model']}")

    tracker_type=param_dict["tracker_type"]
    if tracker_type=="bytetrack" or tracker_type=="botsort":
        tracker=ultralytics_tracker(param_dict, track_min_interval=track_min_interval, debug_enable=debug_enable)
    elif tracker_type=="utrack":
        tracker=ut.utracker(param_dict, track_min_interval=track_min_interval, debug_enable=debug_enable, start_time=start_time, end_time=end_time)
    elif tracker_type.startswith("upyc"):
        tracker=upyct.upyc_tracker(param_dict, track_min_interval=track_min_interval,
                                   debug_enable=debug_enable,
                                   start_time=start_time,
                                   end_time=end_time,
                                   classes=classes)
    else:
        raise ValueError(f"Unknown tracker type {tracker_type}")
    return tracker
