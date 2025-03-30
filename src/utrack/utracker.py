
import ultralytics
import src.utrack.kalman as kalman
import src.track_util as tu
import src.utrack.motion_track as motion_track
import stuff

from scipy.io import loadmat

class UTracker_ObjectTracker:
    def __init__(self, params=None):
        self.params=params
        self.current_time=0
        self.objects=[]
        self.detected_objects=[]
        self.next_track_id=1
        self.print_log=False

    def log(self, txt):
        if self.print_log:
            print(txt)

    def set_time(self, current_time):
        self.current_time=current_time

    def reset(self):
        self.objects=[]
        self.detected_objects=[]

    def add_object(self, roi, o):
        o.time=self.current_time
        stuff.coord.unmap_roi_box(roi, o.box)
        for pt in o.pose_pos:
            stuff.coord.unmap_roi_point(roi, pt)
        self.detected_objects.append(o)

    def update_predict(self, motiontracker, roi, time):

        self.log(f"Update-predict {len(self.objects)} old objects {len(self.detected_objects)} new objects")
        
        for o in self.objects:
            o.update_predict(motiontracker)

        for o in self.objects:
            o.predicted_box=o.kf.predict(time)

        for o in self.objects:
            o.time=time

        pose_conf=self.params["pose_conf"]
        for o in self.detected_objects:
            o.adjusted_confidence=o.confidence+pose_conf*sum(o.pose_conf)

        output_objects=[]
        max_miss_time=self.params["track_buffer"]/30.0

        for match_pass in [0,1]:
            min_conf=self.params["track_high_thresh"] if match_pass==0 else self.params["track_low_thresh"]
            match_thr=self.params["match_thresh_high"] if match_pass==0 else self.params["match_thresh_low"]
            mfn_context={"min_conf":min_conf,
                         "kf_weight":self.params["kf_weight"], 
                         "kp_weight":self.params["kp_weight"],
                         "fuse_scores":self.params["fuse_scores"],
                         "match_thr":match_thr}
            new_ind, old_ind, scores=stuff.match_lsa(self.detected_objects, self.objects, mfn_context=mfn_context)
            
            num_matches=len(new_ind)
            self.log(f"...Time {time:8.3f} : ROI {roi} {num_matches} matches")

            for i in range(num_matches):
                if scores[i]>0:
                    new_obj=self.detected_objects[new_ind[i]]
                    old_obj=self.objects[old_ind[i]]
                    self.log(f" ({match_pass}) Match to old obj {old_obj.track_id} score {scores[i]:0.3f} conf {new_obj.confidence:0.2f}")
                    new_obj.num_detections=old_obj.num_detections+1
                    new_obj.attr=[max(x,y)*0.9+min(x,y)*0.1 for x,y in zip(new_obj.attr, old_obj.attr)]
                    new_obj.track_id=old_obj.track_id
                    new_obj.kf=old_obj.kf
                    new_obj.kf.update(new_obj.box, time)
                    old_obj.kf=None
                    new_obj.num_missed=0
                    new_obj.last_detect_time=time
                    new_obj.confirmed=True if (match_pass==0) else old_obj.confirmed
                    new_obj.predicted_box=old_obj.predicted_box
                    output_objects.append(new_obj)
                    self.objects[old_ind[i]]=None
                    self.detected_objects[new_ind[i]]=None

        for i,obj in enumerate(self.detected_objects):
            if obj is None:
                continue
            if obj.adjusted_confidence>self.params["new_track_thresh"]:
                obj.track_id=self.next_track_id
                self.log(f"Unmatched new obj {obj.track_id} conf {obj.confidence:0.3f}")
                obj.kf=kalman.KalmanBoxTracker(obj.box, time)
                obj.last_detect_time=time
                obj.confirmed=obj.adjusted_confidence>self.params["immediate_confirm_thresh"]
                self.next_track_id+=1
                output_objects.append(obj)
                self.detected_objects[i]=None

        for i,obj in enumerate(self.objects):
            if obj is None:
                continue
            if roi is not None:
                obj.num_missed+=1
            time_since_detection=time-obj.last_detect_time
            
            keep=time_since_detection<max_miss_time
            keep=keep and (obj.num_missed==0 or obj.confirmed)
            
            if keep:
                output_objects.append(obj)
            else:
                self.log(f"Deleting object {obj.track_id}")

        self.detected_objects=[]
        self.objects=output_objects
        
        if roi is None:
            return None
        
        ret_objects=[]
        for o in output_objects:
            if o.num_missed<2 and o.confirmed:
                ret_objects.append(o)

        for o in output_objects:
            self.log(f"... obj {o.track_id} last_det {o.last_detect_time:5.3f} confirmed {o.confirmed} output {o in ret_objects}")

        return ret_objects

    def draw(self, img):
        for o in self.objects:
            o.draw(img)

class utracker:
    def __init__(self, params, track_min_interval):
        self.params=params
        self.yolo = ultralytics.YOLO(params["model"])
        self.track_min_interval=track_min_interval
        self.class_names=[self.yolo.names[i] for i in range(len(self.yolo.names))]
        self.last_track_time=-1000
        self.motiontracker=motion_track.MotionTracker(params=self.params)
        self.objecttracker=UTracker_ObjectTracker(params=self.params)
        self.attributes=[]
        self.person_class_index=self.class_names.index("person")
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
                if d["class"]==self.person_class_index:
                    o=tu.Object(detection=d, time=time)
                    self.objecttracker.add_object(roi, o)
            self.last_track_time=time

            ret=self.objecttracker.update_predict(self.motiontracker, detection_roi, time)
            return ret
        return None