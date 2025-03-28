import ultralytics
import src.track_util as tu
import os
import tempfile
import yaml
import stuff
import src.bytetrack.byte_tracker as bt
import  src.bytetrack.basetrack as basetrack

class ultralytics_tracker:

    def __init__(self, params, track_min_interval):
        self.tmp_file=None
        self.yolo = ultralytics.YOLO(params["model"])
        self.track_min_interval=track_min_interval
        self.class_names=[self.yolo.names[i] for i in range(len(self.yolo.names))]
        self.last_track_time=-1000
        self.nms_iou=0.5
        self.params=params

        fd, self.tmp_config_file=tempfile.mkstemp(dir="/tmp", prefix="yolo_config", suffix=".yaml")
        os.close(fd)
        with open(self.tmp_config_file, 'w') as outfile:
            yaml.dump(self.params, outfile, default_flow_style=False)

    def __del__(self):
        os.remove(self.tmp_config_file)

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
                                      conf=self.params["track_low_thresh"],
                                      iou=self.params["nms_iou"],
                                      half=True,
                                      max_det=600,
                                      tracker=self.tmp_config_file)
            
            out_det=stuff.yolo_results_to_dets(results[0],
                                            det_thr=self.params["track_low_thresh"],
                                            yolo_class_names=self.class_names,
                                            class_names=self.class_names,
                                            face_kp=True,
                                            pose_kp=True,
                                            params=self.params)
            
            person_dets=[d for d in out_det if self.class_names[d["class"]]=="person"]
            objects=[]
            for d in person_dets:
                o=tu.Object(detection=d, time=t)
                if d["id"] is not None:

                    o.track_id=d["id"]
                    objects.append(o)

            self.last_track_time=t
        return objects
    
class nvof_tracker:
    
    def __init__(self, params, track_min_interval):
        self.yolo = ultralytics.YOLO(params["model"])
        self.track_min_interval=track_min_interval
        self.class_names=[self.yolo.names[i] for i in range(len(self.yolo.names))]
        self.last_track_time=-1000
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
    
class cevo_tracker:
    
    def __init__(self, params, track_min_interval):
        
        self.params=params
        self.yolo = ultralytics.YOLO(self.params["model"])
        self.track_min_interval=track_min_interval
        self.class_names=[self.yolo.names[i] for i in range(len(self.yolo.names))]
        self.byte_tracker=bt.BYTETracker(self.params)
        self.last_track_time=-1000
        self.attributes=[]
        for c in self.class_names:
            if c.startswith("person_"):
                self.attributes.append("person:"+c[len("person_"):])

    def track_frame(self, frame, time):
        do_track=time-self.last_track_time>=self.track_min_interval
        if do_track==False:
            return None

        result=self.yolo(frame,
                         classes=[0],
                         imgsz=640,
                         half=True,
                         conf=self.params["track_low_thresh"],
                         iou=self.params["nms_iou"],
                         max_det=600,
                         verbose=False,
                         rect=True)

        out_det=stuff.yolo_results_to_dets(result[0],
                                        det_thr=self.params["track_low_thresh"],
                                        yolo_class_names=self.class_names,
                                        class_names=self.class_names,
                                        attributes=self.attributes,
                                        face_kp=True,
                                        pose_kp=True,
                                        fold_attributes=True)
    
        person_dets=[d for d in out_det if self.class_names[d["class"]]=="person"]
        output_stracks=self.byte_tracker.update(person_dets, time)
        
        objects=[]
        for s in output_stracks:
            #b=s.tlbr
            b = s.mean[:4].copy()
            b[2] *= b[3]

            box=[stuff.clip01(b[0]-0.5*b[2]), 
                 stuff.clip01(b[1]-0.5*b[3]), 
                 stuff.clip01(b[0]+0.5*b[2]), 
                 stuff.clip01(b[1]+0.5*b[3])]
            d={"box":box, "class":0, "confidence":1.0}
            o=tu.Object(detection=d, time=time)
            o.track_id=s.track_id
            objects.append(o)
       
        self.last_track_time=time    
        return objects




