import ultralytics
import src.track_util as tu
import os
import tempfile
import yaml
import stuff

class ultralytics_tracker:

    def __init__(self, model, track_min_interval, config_file=None, params=None):
        self.tmp_file=None
        self.yolo = ultralytics.YOLO(model)
        self.track_min_interval=track_min_interval
        self.class_names=[self.yolo.names[i] for i in range(len(self.yolo.names))]
        self.last_track_time=-1000
        self.nms_iou=0.5
        self.config={}
        self.config_file=config_file
        self.params=params

        #print("model",model)
        #print("track_min_interval",track_min_interval)
        #print("config",config_file)
        #print("Params",params)

        if os.path.exists(config_file) and params is not None:
            self.config=stuff.load_dictionary(config_file)
            fd, self.tmp_file=tempfile.mkstemp(dir="/tmp", prefix="yolo_config", suffix=".yaml")
            os.close(fd)
            for p in params:
                self.config[p]=params[p]
            with open(self.tmp_file, 'w') as outfile:
                yaml.dump(self.config, outfile, default_flow_style=False)
            self.config_file=self.tmp_file
        elif params is not None:
            print("Params only works if a config file specified")
            exit()

        if "nms_iou" in self.config:
            self.nms_iou=self.config["nms_iou"]

    def __del__(self):
        if hasattr(self, 'tmp_file'):
            if self.tmp_file is not None:
                os.remove(self.tmp_file)

    def track_frame(self, img, t):
        do_track=t-self.last_track_time>=self.track_min_interval
        objects=None
        if do_track:
            results = self.yolo.track(img,
                                      imgsz=640,
                                      persist=True,
                                      classes=[0],
                                      verbose=False,
                                      rect=True,
                                      conf=0.05,
                                      iou=self.nms_iou,
                                      half=True,
                                      max_det=600,
                                      tracker=self.config_file)

            out_det=stuff.yolo_results_to_dets(results[0],
                                            det_thr=0.05,
                                            yolo_class_names=self.class_names,
                                            class_names=self.class_names,
                                            face_kp=True,
                                            pose_kp=True,
                                            params=self.params)
            objects=[]
            for d in out_det:
                if d["class"]==0:
                    o=tu.Object(detection=d, time=t)
                    if d["id"] is not None:
                        o.track_id=d["id"]
                        objects.append(o)

            self.last_track_time=t
        return objects
    
class nvof_tracker:
    
    def __init__(self, model, track_min_interval, config_file=None, params=None):
        self.yolo = ultralytics.YOLO(model)
        self.track_min_interval=track_min_interval
        self.class_names=[self.yolo.names[i] for i in range(len(self.yolo.names))]
        self.last_track_time=-1000
        
        if config_file is not None:
            self.config=stuff.load_dictionary(config_file)
            self.params={}
            for c in self.config:
                self.params[c]=self.config[c]
        if params is not None:
            for p in params:
                self.params[p]=params[p]
        self.motiontracker=tu.MotionTracker(params=self.params)
        self.objecttracker=tu.ObjectTracker(params=self.params)

        self.attributes=[]
        for c in self.class_names:
            if c.startswith("person_"):
                self.attributes.append("person:"+c[len("person_"):])

    def track_frame(self, frame, time):
        do_track=time-self.last_track_time>=self.track_min_interval
        
        objects=None
        result=None
        detection_roi=None
        self.motiontracker.add_frame(frame, time)
        roi=[0,0,1.0,1.0] #motiontracker.get_roi(100)
        if stuff.coord.box_a(roi)>0.005 and do_track:
            roi=[0,0,1.0,1.0]#motiontracker.get_roi(80)
            detection_roi=roi
            h,w,_=frame.shape
            roi_l=int(roi[0]*w)
            roi_r=int(roi[2]*w)
            roi_t=int(roi[1]*h)
            roi_b=int(roi[3]*h)
            self.motiontracker.set_roi_detected(roi)
            #print(roi_l,roi_t,roi_r,roi_b)
            img_roi=frame[roi_t:roi_b, roi_l:roi_r]
            result=self.yolo(img_roi,
                             half=True,
                             conf=0.05,
                             iou=self.params["nms_iou"],
                             max_det=600,
                             verbose=False,
                             rect=True)

            out_det=stuff.yolo_results_to_dets(result[0],
                                            det_thr=0.1,
                                            yolo_class_names=self.class_names,
                                            class_names=self.class_names,
                                            attributes=self.attributes,
                                            face_kp=True,
                                            pose_kp=True,
                                            fold_attributes=True)
            
            for d in out_det:
                if d["class"]==0:
                    o=tu.Object(detection=d, time=time)
                    self.objecttracker.add_object(roi, o)
            self.last_track_time=time

            ret=self.objecttracker.update_predict(self.motiontracker, detection_roi, time)
            return ret
        return None

