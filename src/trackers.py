import ultralytics
import src.track_util as tu
import os
import tempfile
import yaml
import stuff
import copy
import src.bytetrack.byte_tracker as bt
import src.utrack.utracker as ut
import src.cevo_mlpipeline.cevo_mlpipeline_tracker as cm

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


class cevo_tracker:

    def __init__(self, params, track_min_interval, debug_enable=False):

        self.params=params
        self.yolo = ultralytics.YOLO(self.params["model"])
        self.track_min_interval=track_min_interval
        self.class_names=[self.yolo.names[i] for i in range(len(self.yolo.names))]
        self.byte_tracker=bt.BYTETracker(self.params)
        self.last_track_time=-1000
        self.attributes=[]
        self.person_class_index=self.class_names.index("person")
        self.frames_skipped_count=0
        for c in self.class_names:
            if c.startswith("person_"):
                self.attributes.append("person:"+c[len("person_"):])

    def track_frame(self, frame, time, debug_enable=False):
        self.debug_enable=debug_enable
        self.debug={}

        if self.track_min_interval>=0:
            do_track=time-self.last_track_time>=self.track_min_interval
        else:
            do_track=self.frames_skipped_count+1>=(-self.track_min_interval)
            if do_track:
                self.frames_skipped_count=0
            else:
                self.frames_skipped_count+=1
        if do_track==False:
            return None, None

        det_w=stuff.get_dict_param(self.params, "det_w", 640)
        det_h=stuff.get_dict_param(self.params, "det_h", 640)

        result=self.yolo(frame,
                         classes=[self.person_class_index],
                         imgsz=[det_h, det_w],
                         half=True,
                         conf=self.params["track_low_thresh"],
                         iou=self.params["nms_iou"],
                         max_det=600,
                         verbose=False,
                         rect=True)

        h=result[0].orig_shape[0]
        w=result[0].orig_shape[1]

        out_det=stuff.yolo_results_to_dets(result[0],
                                        det_thr=self.params["track_low_thresh"],
                                        yolo_class_names=self.class_names,
                                        class_names=self.class_names,
                                        attributes=self.attributes,
                                        face_kp=True,
                                        pose_kp=True,
                                        fold_attributes=True)

        if self.debug_enable:
            self.debug|={"detections": {"type": "yolo_detections", "data":{"detections":out_det, "class_names":self.class_names, "attributes":self.attributes}}}


        person_dets=[d for d in out_det if d["class"]==self.person_class_index]

        scores=[]
        bboxes=[]
        for d in person_dets:
            box=copy.copy(d["box"])
            box[0]*=w
            box[1]*=h
            box[2]*=w
            box[3]*=h
            bboxes.append(box)
            scores.append(d["confidence"])

        output_stracks=self.byte_tracker.update(bboxes, scores, time)

        objects=[]
        for s in output_stracks:
            b = s.mean[:4].copy()
            b[2] *= b[3]

            box=[stuff.clip01((b[0]-0.5*b[2])/w),
                 stuff.clip01((b[1]-0.5*b[3])/h),
                 stuff.clip01((b[0]+0.5*b[2])/w),
                 stuff.clip01((b[1]+0.5*b[3])/h)]
            d={"box":box, "class":self.person_class_index, "confidence":1.0}
            o=tu.Object(detection=d, time=time)
            o.track_id=s.track_id
            objects.append(o)

        self.last_track_time=time
        return objects, self.debug

def create_tracker(param_dict, track_min_interval, debug_enable=False):
    assert "tracker_type" in param_dict, "tracker type must be specified"

    if not "model" in param_dict:
        param_dict["model"]="/mldata/weights/good/yolo11l-dpa-131224.pt"
        #print(f"WARNING: Model not specified in config; using default model {param_dict['model']}")

    tracker_type=param_dict["tracker_type"]
    if tracker_type=="bytetrack" or tracker_type=="botsort":
        tracker=ultralytics_tracker(param_dict, track_min_interval=track_min_interval, debug_enable=debug_enable)
    elif tracker_type=="utrack":
        tracker=ut.utracker(param_dict, track_min_interval=track_min_interval, debug_enable=debug_enable)
    elif tracker_type=="cevo":
        tracker=cevo_tracker(param_dict, track_min_interval=track_min_interval, debug_enable=debug_enable)
    elif tracker_type=="cevo_mlpipe":
        tracker=cm.cevo_mlpipe_tracker(param_dict, track_min_interval=track_min_interval, debug_enable=debug_enable)
    elif tracker_type=="cevo_analyser":
        tracker=cm.cevo_analyser(param_dict, track_min_interval=track_min_interval, debug_enable=debug_enable)
    else:
        print(f"Unkown tracker type {tracker_type}")
        exit()
    return tracker
