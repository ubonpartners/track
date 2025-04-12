import copy
import ultralytics
import src.utrack.kalman as kalman
import src.track_util as tu
import src.utrack.motion_track as motion_track
import stuff
import math

class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

def lost_object_match_score(self, other, context):
    iou=context["iou"]
    if self is None:
        return 0
    if other is None:
        return 0
    if self.track_state!=TrackState.Tracked:
        return 0
    if other.track_state!=TrackState.Lost:
        return 0
    box_score=stuff.coord.box_iou(self.box, other.box)
    if box_score<iou:
        return 0
    return box_score

def object_match_score(new_obj, tracked_obj, context):
    kf_weight=context["kf_weight"]
    kp_weight=context["kp_weight"]
    of_weight=1.0

    of_score=stuff.coord.box_iou(new_obj.box, tracked_obj.of_predicted_box)
    kf_score=stuff.coord.box_iou(new_obj.box, tracked_obj.kf_predicted_box)
    if of_score+kf_score==0:
        return 0

    if tracked_obj.observations<2:
        kf_weight=0
    else: #tracked_obj.observations<3:
        f=min(0.1, 1/tracked_obj.observations)
        kf_weight*=(1-f)
        of_weight*=f
    #print(tracked_obj.observations)

    kp_score=None
    if True and len(new_obj.pose_pos)==17 and len(tracked_obj.pose_pos)==17 and (of_score+kf_score)!=0:
        scales=[0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089]
        scales=[x*2 for x in scales] # scale=2*sigma
        ss=0.5*(stuff.coord.box_a(new_obj.box)+stuff.coord.box_a(tracked_obj.of_predicted_box))*0.53 # approximation of shape area from box area
        ss*=(context["kp_distance_scale"]*context["kp_distance_scale"])
        num=0
        denom=0
        for i,_ in enumerate(new_obj.pose_pos):
            if new_obj.pose_conf[i]>0.0 and tracked_obj.pose_conf[i]>0.0: # is point labelled
                dx=new_obj.pose_pos[i][0]-tracked_obj.of_predicted_pose_pos[i][0]
                dy=new_obj.pose_pos[i][1]-tracked_obj.of_predicted_pose_pos[i][1]
                num+=math.exp(-(dx*dx+dy*dy)/(2.0*ss*scales[i]*scales[i]+1e-7))
                denom+=1.0
        if denom>4:
            kp_score=num/(denom+1e-7)

    #if kp_score is not None:
    #    print(f"{new_obj.track_id} {tracked_obj.track_id}  box:{box_score:0.2f} kp:{kp_score:0.3f},{denom} kf:{kf_score:0.2f}")
    
    if kp_score is None:
        score=(of_score*of_weight+kf_score*kf_weight)/(of_weight+kf_weight)
    else:
        score=(of_score*of_weight+kf_score*kf_weight+kp_score*kp_weight)/(of_weight+kf_weight+kp_weight)
    if score<context["match_thr"]:
        return 0
    
    score=score*math.pow(new_obj.confidence, context["fuse_scores"])
    return score

class utracker:
    def __init__(self, params, track_min_interval, debug_enable=False):
        self.params=params
        self.yolo = ultralytics.YOLO(params["model"])
        self.track_min_interval=track_min_interval
        self.class_names=[self.yolo.names[i] for i in range(len(self.yolo.names))]
        self.last_track_time=-1000
        self.motiontracker=motion_track.MotionTracker(params=self.params)
        self.attributes=[]
        self.person_class_index=self.class_names.index("person")
        for c in self.class_names:
            if c.startswith("person_"):
                self.attributes.append("person:"+c[len("person_"):])
        self.tracked_objects=[]
        self.next_track_id=1
        self.print_log=False
        self.debug_enable=False
    
    def log(self, txt):
        if self.print_log:
            print(txt)

    def reset(self):
        self.tracked_objects=[]

    def predict_tracked_object_positions(self, motiontracker, time):
        for o in self.tracked_objects:
            o.of_predicted_box=motiontracker.predict_box(o.box, time)
            o.of_predicted_pose_pos=[0]*o.num_pose
            for i in range(o.num_pose):
                o.of_predicted_pose_pos[i]=motiontracker.predict_point(o.pose_pos[i], time)

        for o in self.tracked_objects:
            o.kf_predicted_box=o.kf.predict(time)

        debug_kf_prediction={}
        debug_of_prediction={}
        if self.debug_enable:
            for o in self.tracked_objects:
                debug_kf_prediction[o.track_id]={"from":copy.copy(o.box), "to":copy.copy(o.kf_predicted_box)}
                debug_of_prediction[o.track_id]={"from":copy.copy(o.box), 
                                                 "to":copy.copy(o.of_predicted_box),
                                                 "pose_from":copy.copy(o.pose_pos),
                                                 "pose_to":copy.copy(o.of_predicted_pose_pos),
                                                 "pose_conf":copy.copy(o.pose_conf)}
            self.debug|={"kf_predictions":{"type":"box_prediction", "data":debug_kf_prediction}}
            self.debug|={"of_prediction":{"type":"box_prediction", "data":debug_of_prediction}}

    def update_predict(self, detected_objects, motiontracker, roi, time):
        self.log(f"Update-predict {len(self.tracked_objects)} old objects {len(detected_objects)} new objects")

        for o in self.tracked_objects:
            o.matched=False
            o.time=time

        pose_conf=self.params["pose_conf"]
        for o in detected_objects:
            o.matched=False
            o.adjusted_confidence=o.confidence+pose_conf*sum(o.pose_conf)
            o.track_state=TrackState.New
            o.observations=1

        max_miss_time=self.params["track_buffer_seconds"]

        # match new objects to existing objects
        output_objects=[]
        for match_pass in [0,1,2]:
            mfn_context={"kf_weight":self.params["kf_weight"], 
                         "kp_weight":self.params["kp_weight"],
                         "kp_distance_scale":self.params["kp_distance_scale"],
                         "fuse_scores":self.params["fuse_scores"]}
            
            if match_pass==0:
                mfn_context["det_select"]=lambda o : o.adjusted_confidence>self.params["track_initial_thresh"]
                mfn_context["tracked_select"]=lambda o : o.track_state!=TrackState.Lost
                mfn_context["match_thr"]=self.params["match_thresh_initial"]
            elif match_pass==1:
                mfn_context["det_select"]=lambda o : o.adjusted_confidence>self.params["track_high_thresh"]
                mfn_context["tracked_select"]=lambda o : True
                mfn_context["match_thr"]=self.params["match_thresh_high"]
            elif match_pass==2:
                mfn_context["det_select"]=lambda o : o.adjusted_confidence>self.params["track_low_thresh"]
                mfn_context["tracked_select"]=lambda o : True
                mfn_context["match_thr"]=self.params["match_thresh_low"]

            det_filtered=[o for o in detected_objects if o.matched==False and mfn_context["det_select"](o)]
            tracked_filtered=[o for o in self.tracked_objects if o.matched==False and mfn_context["tracked_select"](o)]

            new_ind, old_ind, scores=stuff.match_lsa(det_filtered, tracked_filtered, mfn=object_match_score, mfn_context=mfn_context)
            #print(f"{match_pass} {len(det_filtered)} {len(tracked_filtered)}")
            num_matches=len(new_ind)
            self.log(f"...Time {time:8.3f} : ROI {roi} {num_matches} matches")

            for i in range(num_matches):
                if scores[i]>0:
                    new_obj=det_filtered[new_ind[i]]
                    old_obj=tracked_filtered[old_ind[i]]
                    self.log(f" ({match_pass}) Match to old obj {old_obj.track_id} score {scores[i]:0.3f} conf {new_obj.confidence:0.2f}")
                    new_obj.num_detections=old_obj.num_detections+1
                    new_obj.attr=[max(x,y)*0.9+min(x,y)*0.1 for x,y in zip(new_obj.attr, old_obj.attr)]
                    new_obj.track_id=old_obj.track_id
                    new_obj.kf=old_obj.kf
                    new_obj.kf.update(new_obj.box, time)
                    new_obj.observations=old_obj.observations+1
                    old_obj.kf=None
                    new_obj.num_missed=0
                    new_obj.last_detect_time=time
                    if (match_pass==0) or (match_pass==1):
                        new_obj.track_state=TrackState.Tracked
                    elif old_obj.track_state==TrackState.Lost:
                        new_obj.track_state=TrackState.Tracked
                    else:
                        new_obj.track_state=old_obj.track_state
                    new_obj.kf_predicted_box=old_obj.kf_predicted_box
                    new_obj.deleted=False
                    output_objects.append(new_obj)
                    tracked_filtered[old_ind[i]].matched=True
                    det_filtered[new_ind[i]].matched=True

        # deal with new objects that don't match any existing objects

        for i,obj in enumerate(detected_objects):
            if obj.matched==True:
                continue
            if obj.adjusted_confidence>self.params["new_track_thresh"]:
                obj.track_id=self.next_track_id
                self.log(f"Unmatched new obj {obj.track_id} conf {obj.confidence:0.3f}")
                obj.kf=kalman.KalmanBoxTracker(obj.box, time)
                obj.last_detect_time=time
                if obj.adjusted_confidence>self.params["immediate_confirm_thresh"]:
                    obj.track_state=TrackState.Tracked
                obj.deleted=False
                self.next_track_id+=1
                output_objects.append(obj)

        # determine which objects to delete
        for i,obj in enumerate(self.tracked_objects):
            if obj.matched==True:
                continue
            if roi is not None:
                obj.num_missed+=1
            time_since_detection=time-obj.last_detect_time
            
            keep=time_since_detection<max_miss_time
            keep=keep and (obj.num_missed<10 or obj.track_state==TrackState.Tracked)

            if keep:
                output_objects.append(obj)
            else:
                obj.track_state=TrackState.Removed
                self.log(f"Deleting object {obj.track_id}")

        detected_objects=[]
        self.tracked_objects=output_objects
        
        if roi is None:
            return None, None
        
        # determine "lost" objects"
        for o in self.tracked_objects:
            if o.track_state==TrackState.Tracked:
                if o.num_missed>=2:
                    o.track_state=TrackState.Lost

        # remove duplicated objects

        lost_object_context={"iou":self.params["delete_dup_iou"]}
        new_ind, old_ind, scores=stuff.match_lsa(self.tracked_objects, self.tracked_objects, mfn=lost_object_match_score, mfn_context=lost_object_context)
        for i,s in enumerate(scores):
            if s>0:
                self.tracked_objects[old_ind[i]].track_state=TrackState.Removed
        
        self.tracked_objects=[o for o in self.tracked_objects if o.track_state!=TrackState.Removed]

        # determine objects to return as visible

        ret_objects=[o for o in self.tracked_objects if o.track_state==TrackState.Tracked]
        for o in ret_objects:
            self.log(f"... obj {o.track_id} last_det {o.last_detect_time:5.3f} stats {o.track_state} output {o in ret_objects}")

        return ret_objects

    def track_frame(self, frame, time, debug_enable=False):
        self.debug_enable=debug_enable
        self.debug={}

        # skip running if below minimnum frame interval

        do_track=time-self.last_track_time>=self.track_min_interval
        if do_track==False:
            return None, None
        
        # check if skip running due to not enough motion

        #result=None
        #detection_roi=None
        self.motiontracker.add_frame(frame, time)
        motion_roi=self.motiontracker.get_roi(10)
        #if stuff.coord.box_a(motion_roi)<0.005:
        #    return None, None

        roi=[0,0,1.0,1.0] #motiontracker.get_roi(100)

        if debug_enable:
            self.debug|=self.motiontracker.get_debug()

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
                                        det_thr=0.05,
                                        yolo_class_names=self.class_names,
                                        class_names=self.class_names,
                                        attributes=self.attributes,
                                        face_kp=True,
                                        pose_kp=True,
                                        fold_attributes=True)
        
        detected_objects=[]
        for d in out_det:
            if d["class"]==self.person_class_index:
                o=tu.Object(detection=d, time=time, expand_by_pose=True)
                o.time=time
                stuff.coord.unmap_roi_box(roi, o.box)
                for pt in o.pose_pos:
                    stuff.coord.unmap_roi_point(roi, pt)
                detected_objects.append(o)

        self.last_track_time=time
        if self.debug_enable:
            self.debug|={"detections": {"type": "yolo_detections", "data":{"detections":out_det, "class_names":self.class_names, "attributes":self.attributes}}}
            self.debug|={"test_roi": {"type": "roi", "data": {"roi":copy.copy(motion_roi)}}}

        self.predict_tracked_object_positions(self.motiontracker, time)

        ret=self.update_predict(detected_objects, self.motiontracker, detection_roi, time)
        return ret, self.debug