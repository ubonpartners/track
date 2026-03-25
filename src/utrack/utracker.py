import copy
import stuff.inference_wrapper
import src.utrack.kalman as kalman
import src.track_util as tu
import src.utrack.motion_track as motion_track
import stuff
import math
import logging
import numpy as np
import cv2
import os
import tempfile
from pathlib import Path

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

def partition_fn(obj, context):
    return obj.partition_mask

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def center_and_normalize(vec, mean):
    centered = vec - mean
    norm = np.linalg.norm(centered)
    return centered / norm if norm > 0 else centered

def object_match_score(new_obj, tracked_obj, context):
    kf_weight=context["kf_weight"]
    kp_weight=context["kp_weight"]
    of_weight=1.0

    of_score=stuff.coord.box_iou(new_obj.box, tracked_obj.of_predicted_box)
    kf_score=stuff.coord.box_iou(new_obj.box, tracked_obj.kf_predicted_box)

    #if context["simple"]:
     #   return (of_score+kf_score)/2
    # better with below removed
    #if of_score+kf_score==0:
    #    return 0

    sim=0.0
    if new_obj.reid_vector is not None and tracked_obj.reid_vector is not None:
        sim=np.dot(new_obj.reid_vector_norm, tracked_obj.reid_vector_norm) # =cosine dist, as already normalized
        sim=sim*context["sim_weight"]

    of_score_face=0
    if tracked_obj.subbox is not None and new_obj.subbox is not None:
        of_score_face=stuff.coord.box_iou(new_obj.subbox, tracked_obj.of_predicted_subbox)
    of_score+=of_score_face*context["face_weight"]

    if tracked_obj.observations<2:
        kf_weight=0
    else: #tracked_obj.observations<3:
        f=math.pow(max(0.1, 1/tracked_obj.observations),context["kf_warmup"])
        kf_weight*=(1-f)

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
    score+=sim
    if score<context["match_thr"]:
        return 0

    score=score*math.pow(new_obj.confidence, context["fuse_scores"])
    return score

class utracker:
    def __init__(self, params, track_min_interval, cache_h264=True, debug_enable=False, start_time=0, end_time=100000):
        try:
            import ubon_pycstuff.ubon_pycstuff as upyc
        except:
            assert False, "Motion_tracker equires ubon_pycstuff"
        self.upyc=upyc
        self.logger=logging.getLogger('utracker')
        self.params=params

        trackset=self.params["original_trackset"]
        del self.params["original_trackset"]

        video=trackset.metadata["original_video"]
        self.video_fps=trackset.metadata["frame_rate"]

        self.simple=params.get("simple", False)

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
        with open(h264_file, "rb") as f:
            self.video_bitstream = f.read()
        self.video_frames=[]
        self.video_decoder=self.upyc.c_decoder(self.upyc.SIMPLE_DECODER_CODEC_H264)
        self.video_decoder.set_framerate(self.video_fps)

        if h264_file_temp is not None:
            stuff.rm(h264_file_temp)

        self.class_names=["person", "face"]
        self.infer=stuff.inference_wrapper(params["model"],
                                          half=True,
                                          rect=True,
                                          thr=0.05,
                                          nms_thr=self.params["nms_thr"],
                                          max_det=1000,
                                          class_names=self.class_names,
                                          face_kp=True,
                                          pose_kp=True)
        self.track_min_interval=track_min_interval
        self.last_track_time=-1000
        self.motiontracker=motion_track.MotionTracker(params=self.params)
        self.attributes=[]
        self.person_class_index=self.class_names.index("person")
        for c in self.class_names:
            if c.startswith("person_"):
                self.attributes.append("person:"+c[len("person_"):])
        self.tracked_objects=[]
        self.next_track_id=0xbeef0000
        self.print_log=False
        self.debug_enable=False

    def file_trace(self, txt):
        self.upyc.file_trace("[PYTHON] "+txt)

    def reset(self):
        self.tracked_objects=[]

    def predict_tracked_object_positions(self, motiontracker, time):
        for o in self.tracked_objects:
            o.of_predicted_box=motiontracker.predict_box(o.box, time)
            o.of_predicted_subbox=motiontracker.predict_box(o.subbox, time)
            o.of_predicted_pose_pos=[0]*o.num_pose
            for i in range(o.num_pose):
                o.of_predicted_pose_pos[i]=motiontracker.predict_point(o.pose_pos[i], time)
                if o.pose_conf[i]>0.05:
                    o.of_predicted_box[0]=min(o.of_predicted_box[0], o.of_predicted_pose_pos[i][0])
                    o.of_predicted_box[1]=min(o.of_predicted_box[1], o.of_predicted_pose_pos[i][1])
                    o.of_predicted_box[2]=max(o.of_predicted_box[2], o.of_predicted_pose_pos[i][0])
                    o.of_predicted_box[3]=max(o.of_predicted_box[3], o.of_predicted_pose_pos[i][1])

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
        self.logger.debug(f"Update-predict {len(self.tracked_objects)} old objects {len(detected_objects)} new objects")

        use_new_match=True
        partition_size=(8,8)

        for o in self.tracked_objects:
            o.matched=False
            o.time=time
            if use_new_match:
                vbox=stuff.box_union(stuff.box_union(o.kf_predicted_box, o.of_predicted_box), o.box)
                vbox=stuff.box_expand(vbox, 1.1)
                o.vbox=vbox
                o.partition_mask=stuff.uniform_grid_partition(vbox, context=partition_size)
                self.file_trace(f"MASKGEN {vbox[0]:0.4f} {vbox[1]:0.4f} {vbox[2]:0.4f} {vbox[3]:0.4f} {o.partition_mask:016x} ")

        all_embeddings=[]
        for o in detected_objects:
            if o.reid_vector is not None:
                o.reid_vector=np.array(o.reid_vector)
                all_embeddings.append(o.reid_vector)
        for o in self.tracked_objects:
            if o.reid_vector is not None:
                all_embeddings.append(o.reid_vector)
        if len(all_embeddings)>0:
            mean_vector = np.mean(all_embeddings, axis=0)
            for o in detected_objects:
                if o.reid_vector is not None:
                    o.reid_vector_norm=center_and_normalize(o.reid_vector, mean_vector)
            for o in self.tracked_objects:
                if o.reid_vector is not None:
                    o.reid_vector_norm=center_and_normalize(o.reid_vector, mean_vector)

        pose_conf=self.params["pose_conf"]
        for i,o in enumerate(detected_objects):
            o.matched=False
            o.adjusted_confidence=o.confidence+pose_conf*sum(o.pose_conf)
            o.track_state=TrackState.New
            o.observations=1
            if use_new_match:
                o.partition_mask=stuff.uniform_grid_partition(o.box, context=partition_size)

        detected_objects.sort(key=lambda o: o.adjusted_confidence, reverse=True)

        for i,o in enumerate(detected_objects):
            self.file_trace(f"Initial det {i:2d} msk {o.partition_mask:016x} conf {o.adjusted_confidence:.4f} box {[f'{x:.4f}' for x in o.box]} area {stuff.box_a(o.box):.3f}")

        for i,o in enumerate(self.tracked_objects):
            s=f"Initial tracked {i} {o.track_id} {o.partition_mask:016x} "
            s=s+f"[{' '.join(f'{x:.4f}' for x in o.box)}]"
            s+=f" OF [{' '.join(f'{x:.4f}' for x in o.of_predicted_box)}]"
            s+=f" KF [{' '.join(f'{x:.4f}' for x in o.kf_predicted_box)}]"
            self.file_trace(s)

        max_miss_time=self.params["track_buffer_seconds"]

        # match new objects to existing objects
        output_objects=[]
        for match_pass in [0,1,2]:
            mfn_context={"kf_weight":self.params["kf_weight"],
                         "kp_weight":self.params["kp_weight"],
                         "kp_distance_scale":self.params["kp_distance_scale"],
                         "fuse_scores":self.params["fuse_scores"],
                         "kf_warmup":self.params["kf_warmup"],
                         "face_weight":self.params["face_weight"],
                         "sim_weight":self.params["sim_weight"],
                         "simple":self.params.get("simple",False)}

            if match_pass==0:
                mfn_context["det_select"]=lambda o : o.adjusted_confidence>self.params["track_initial_thr"]
                mfn_context["tracked_select"]=lambda o : o.track_state!=TrackState.Lost
                mfn_context["match_thr"]=self.params["match_thr_initial"]
            elif match_pass==1:
                mfn_context["det_select"]=lambda o : o.adjusted_confidence>self.params["track_high_thr"]
                mfn_context["tracked_select"]=lambda o : True
                mfn_context["match_thr"]=self.params["match_thr_high"]
            elif match_pass==2:
                mfn_context["det_select"]=lambda o : o.adjusted_confidence>self.params["track_low_thr"]
                mfn_context["tracked_select"]=lambda o : True
                mfn_context["match_thr"]=self.params["match_thr_low"]

            det_filtered=[o for o in detected_objects if o.matched==False and mfn_context["det_select"](o)]
            tracked_filtered=[o for o in self.tracked_objects if o.matched==False and mfn_context["tracked_select"](o)]

            if use_new_match:
                new_ind, old_ind, scores=stuff.match_lsa2(det_filtered,
                                                          tracked_filtered,
                                                          mfn=object_match_score,
                                                          mfn_context=mfn_context,
                                                          partition_fn=partition_fn,
                                                          max_partitions=64)
            else:
                new_ind, old_ind, scores=stuff.match_lsa(det_filtered,
                                                          tracked_filtered,
                                                          mfn=object_match_score,
                                                          mfn_context=mfn_context)

            # sort - just for easier debug
            combined = list(zip(scores, new_ind, old_ind))
            # Sort by scores in descending order
            combined_sorted = sorted(combined, key=lambda x: x[0], reverse=True)
            # Unzip the sorted result
            if combined_sorted:
                scores, new_ind, old_ind= zip(*combined_sorted)

            #print(f"{match_pass} {len(det_filtered)} {len(tracked_filtered)}")
            num_matches=len(new_ind)
            self.logger.debug(f"...Time {time:8.3f} : ROI {roi} {num_matches} matches")

            for i in range(num_matches):
                if scores[i]>0:
                    new_obj=det_filtered[new_ind[i]]
                    old_obj=tracked_filtered[old_ind[i]]

                    self.file_trace(f"match pass {match_pass}:{i}/{num_matches} Match {new_ind[i]}<->{old_ind[i]} score {scores[i]}")
                    self.logger.debug(f" ({match_pass}) Match to old obj {old_obj.track_id} score {scores[i]:0.3f} conf {new_obj.confidence:0.2f}")
                    new_obj.attr=[max(x,y)*0.9+min(x,y)*0.1 for x,y in zip(new_obj.attr, old_obj.attr)]
                    ema=0.5
                    if new_obj.reid_vector is not None and old_obj.reid_vector is not None:
                        new_obj.reid_vector=ema*new_obj.reid_vector+(1.0-ema)*old_obj.reid_vector

                    #elif old_obj.reid_vector is not None:
                    #    new_obj.reid_vector=old_obj.reid_vector
                    new_obj.track_id=old_obj.track_id
                    new_obj.kf=old_obj.kf
                    new_obj.kf.update(new_obj.box, time)
                    old_obj.kf=None
                    new_obj.observations=old_obj.observations+1
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
            if obj.adjusted_confidence>self.params["new_track_thr"]:
                obj.track_id=self.next_track_id
                self.logger.debug(f"Unmatched new obj {obj.track_id} conf {obj.confidence:0.3f}")
                if False: # use python kalman filter
                    obj.kf=kalman.KalmanBoxTracker(obj.box, time)
                else:
                    obj.kf=self.upyc.c_kalmanboxtracker(obj.box, time)
                obj.last_detect_time=time
                if obj.adjusted_confidence>self.params["immediate_confirm_thr"]:
                    obj.track_state=TrackState.Tracked
                    #print(f"Here {time} {obj.confidence} {obj.adjusted_confidence} {self.params["immediate_confirm_thr"]}")
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

            keep=time_since_detection<max_miss_time and obj.num_missed<10
            keep=keep or obj.track_state==TrackState.Tracked

            if keep:
                output_objects.append(obj)
            else:
                obj.track_state=TrackState.Removed
                self.logger.debug(f"Deleting object {obj.track_id}")

        # update tracked object list
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
        if use_new_match:
            new_ind, old_ind, scores=stuff.match_lsa2(self.tracked_objects,
                                                  self.tracked_objects,
                                                  mfn=lost_object_match_score,
                                                  mfn_context=lost_object_context,
                                                  partition_fn=partition_fn,
                                                  max_partitions=64)
        else:
            new_ind, old_ind, scores=stuff.match_lsa(self.tracked_objects,
                                                  self.tracked_objects,
                                                  mfn=lost_object_match_score,
                                                  mfn_context=lost_object_context)
        for i,s in enumerate(scores):
            if s>0:
                self.tracked_objects[old_ind[i]].track_state=TrackState.Removed

        self.tracked_objects=[o for o in self.tracked_objects if o.track_state!=TrackState.Removed]

        # determine objects to return as visible

        ret_objects=[o for o in self.tracked_objects if o.track_state==TrackState.Tracked]
        for o in ret_objects:
            self.logger.debug(f"... obj {o.track_id} last_det {o.last_detect_time:5.3f} stats {o.track_state} output {o in ret_objects}")
        for i,o in enumerate(ret_objects):
            self.file_trace(f"- return det {i} conf {o.confidence:.3f} box {[f'{x:.4f}' for x in o.box]} area {stuff.box_a(o.box):.3f}")

        for i,o in enumerate(self.tracked_objects):
            s=f"FINAL tracked {i} {o.track_id:x}  "
            s=s+f"[{' '.join(f'{x:.4f}' for x in o.box)}]"
            s+=f" MSS {o.num_missed} OBS {o.observations} TSS {time-o.last_detect_time} TSS {o.track_state}"
            self.file_trace(s)

        return ret_objects

    #def get_frame_times(self):
    #    return self.frame_times

    def track_frame(self, frame, time, debug_enable=False):
        #print(time, self.video_fps)
        self.debug_enable=debug_enable
        self.debug={}

        # decode video
        while True:
            while len(self.video_frames)==0 and len(self.video_bitstream)>0:
                chunk=512*1024
                frames=self.video_decoder.decode(self.video_bitstream[0:chunk])
                self.video_bitstream=self.video_bitstream[chunk:]
                self.video_frames+=frames

            if len(self.video_frames)==0:
                return None, None

            cframe=self.video_frames[0]
            self.video_frames=self.video_frames[1:]
            if cframe.time+0.0001>time:
                break

        # skip running if below minimum frame interval

        do_track=time-self.last_track_time>=self.track_min_interval
        if do_track==False:
            return None, None

        # convert image into C domain, and do an aspect-ratio correct scaling to fit
        # in max_widthxmax_height (typically 1280x1280)

        #h,w,_=frame.shape
        h,w=cframe.size
        frame_c=cframe
        #frame_c=self.upyc.c_image.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_c=frame_c.convert(self.upyc.YUV420_DEVICE)

        w,h=stuff.determine_scale_size(w, h, self.params["max_width"], self.params["max_height"])
        frame_c=frame_c.scale(w, h)

        # do motiontrack
        # check if skip running due to not enough motion

        self.motiontracker.add_frame(frame_c, time)
        motion_roi=self.motiontracker.get_roi()

        if debug_enable:
            self.debug|=self.motiontracker.get_debug()

        motion_area=stuff.coord.box_a(motion_roi)
        if motion_area<self.params["completely_skip_frame_area"]:
            self.logger.debug(f"Completely skipped frame (Area {motion_area:0.4f})")
            self.last_track_time=time
            return None, self.debug

        self.predict_tracked_object_positions(self.motiontracker, time)

        # expand ROI to include the predicted position of all tracked objects
        expanded_roi=motion_roi.copy()

        self.file_trace(f"Time {time}")
        self.file_trace(f"Motion ROI [{' '.join(f'{x:.4f}' for x in expanded_roi)}]")
        for o in self.tracked_objects:
            s=f"det bx [{' '.join(f'{x:.4f}' for x in o.box)}]"
            s+=f" of [{' '.join(f'{x:.4f}' for x in o.of_predicted_box)}]"
            s+=f" kf [{' '.join(f'{x:.4f}' for x in o.kf_predicted_box)}]"
            self.file_trace(s)
            expanded_roi[0]=min(o.kf_predicted_box[0], min(o.of_predicted_box[0], expanded_roi[0]))
            expanded_roi[1]=min(o.kf_predicted_box[1], min(o.of_predicted_box[1], expanded_roi[1]))
            expanded_roi[2]=max(o.kf_predicted_box[2], max(o.of_predicted_box[2], expanded_roi[2]))
            expanded_roi[3]=max(o.kf_predicted_box[3], max(o.of_predicted_box[3], expanded_roi[3]))

        self.file_trace(f"Predicted object ROI pre-expand [{' '.join(f'{x:.4f}' for x in expanded_roi)}]")

        e_w=max(0.05, expanded_roi[2]-expanded_roi[0])
        e_h=max(0.05, expanded_roi[3]-expanded_roi[1])
        expanded_roi[0]=stuff.clip01(expanded_roi[0]-self.params["roi_expand_ratio"]*0.5*e_w)
        expanded_roi[1]=stuff.clip01(expanded_roi[1]-self.params["roi_expand_ratio"]*0.5*e_h)
        expanded_roi[2]=stuff.clip01(expanded_roi[2]+self.params["roi_expand_ratio"]*0.5*e_w)
        expanded_roi[3]=stuff.clip01(expanded_roi[3]+self.params["roi_expand_ratio"]*0.5*e_h)

        self.file_trace(f"Predicted object {time} ROI post-expand [{' '.join(f'{x:.4f}' for x in expanded_roi)}]")

        detection_roi=expanded_roi
        if self.simple:
            detection_roi=[0,0,1,1]

        #print("B", detection_roi[0], detection_roi[2]);
        #print(detection_roi[2]-detection_roi[0],e_w,(detection_roi[2]-detection_roi[0])/ e_w, self.params["roi_expand_ratio"])

        self.logger.debug(f"Detection ROI post-expand [{' '.join(f'{x:.4f}' for x in expanded_roi)}]")

        if False:
            roi_l=int(detection_roi[0]*w)
            roi_r=int(detection_roi[2]*w)
            roi_t=int(detection_roi[1]*h)
            roi_b=int(detection_roi[3]*h)
            roi_l=(roi_l)&(~1)
            roi_t=(roi_t)&(~1)
            roi_r=(roi_r+1)&(~1)
            roi_b=(roi_b+1)&(~1)
            detection_roi=[roi_l/w, roi_t/h, roi_r/w, roi_b/h]

            self.motiontracker.set_roi_detected(detection_roi)

            img_roi_c=frame_c.crop(roi_l, roi_t, roi_r-roi_l, roi_b-roi_t)

        img_roi_c, detection_roi=frame_c.crop_roi(detection_roi)
        self.motiontracker.set_roi_detected(detection_roi)
        out_det=self.infer.infer([img_roi_c])[0]

        self.file_trace("==================================")
        self.file_trace(f"utracker run!: time {time}; {len(out_det)} detections")

        detected_objects=[]
        for d in out_det:
            if d["class"]==self.person_class_index:

                #stuff.check_pose_points(d)
                o=tu.Object(detection=d, time=time, expand_by_pose=True)
                o.time=time
                stuff.coord.unmap_roi_box_inplace(detection_roi, o.box)
                if o.subbox is not None:
                    stuff.coord.unmap_roi_box_inplace(detection_roi, o.subbox)
                for pt in o.pose_pos:
                    stuff.coord.unmap_roi_point_inplace(detection_roi, pt)
                for pt in o.face_pos:
                    stuff.coord.unmap_roi_point_inplace(detection_roi, pt)

                detected_objects.append(o)

            stuff.coord.unmap_roi_box_inplace(detection_roi, d["box"])
            if "pose_points" in d:
                for i in range(len(d["pose_points"])//3):
                    d["pose_points"][i*3:i*3+2]=stuff.coord.unmap_roi_point(detection_roi, d["pose_points"][i*3:i*3+2])
            if "face_points" in d:
                for i in range(len(d["face_points"])//3):
                    d["face_points"][i*3:i*3+2]=stuff.coord.unmap_roi_point(detection_roi, d["face_points"][i*3:i*3+2])

        self.last_track_time=time
        #if self.debug_enable:
        self.debug|={"detections": {"type": "yolo_detections", "data":{"detections":out_det, "class_names":self.class_names, "attributes":self.attributes}}}
        self.debug|={"detection_roi": {"type": "roi", "data": {"roi":copy.copy(detection_roi)}}}
        self.debug|={"motion_roi": {"type": "roi", "data": {"roi":copy.copy(motion_roi)}}}

        ret=self.update_predict(detected_objects, self.motiontracker, detection_roi, time)
        return ret, self.debug