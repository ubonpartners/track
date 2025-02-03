import cv2
import numpy as np
import math
import copy
import src.kalman as kalman
import stuff
import yaml
import os
from scipy.io import loadmat

def object_interpolate(a, b, frac):
    a2=copy.deepcopy(a)
    stuff.interpolate(a2.box, b.box, frac)
    a2.conf=stuff.interpolate(a2.confidence, b.confidence, frac)
    assert(a.num_pose==b.num_pose)
    for i in range(a.num_pose):
        if a2.pose_conf[i]>0 and b.pose_conf[i]>0:
            stuff.interpolate(a2.pose_pos[i], b.pose_pos[i], frac)
            a2.pose_conf[i]=stuff.interpolate(a2.pose_conf[i], b.pose_conf[i], frac)
        else:
            a2.pose_conf[i]=0
    return a2

class Object:
    def __init__(self, box=None, cl=None, conf=None, 
                 pose=None, pose_conf=None, pose_pos=None,
                 attr=None, time=None, detection=None):
        if detection is not None:
            box=detection["box"]
            cl=detection["class"]
            conf=detection["confidence"]
            if "pose_points" in detection:
                pose=detection["pose_points"]
            if "attrs" in detection:
                attr=detection["attrs"]

        self.box=box
        self.num_detections=0
        self.num_missed=0
        self.confidence=conf
        if pose is not None:
            self.num_pose=len(pose)//3
            self.pose_pos=[[0,0] for n in range(self.num_pose)]
            self.pose_conf=[0]*self.num_pose
            for i in range(self.num_pose):
                self.pose_pos[i][0]=pose[i*3+0]
                self.pose_pos[i][1]=pose[i*3+1]
                self.pose_conf[i]=pose[i*3+2]
                if self.pose_conf[i]>0.05:
                    self.box[0]=min(self.box[0], self.pose_pos[i][0])
                    self.box[1]=min(self.box[1], self.pose_pos[i][1])
                    self.box[2]=max(self.box[2], self.pose_pos[i][0])
                    self.box[3]=max(self.box[3], self.pose_pos[i][1])
        elif pose_conf is not None:
            self.num_pose=len(pose_conf)
            self.pose_pos=pose_pos
            self.pose_conf=pose_conf
        else:
            self.num_pose=0
            self.pose_pos=[]
            self.pose_conf=[]
        self.cl=cl
        if attr is None:
            attr=[0]*35
        self.attr=attr
        self.time=time
        self.track_id=None

    def draw(self, display, clr=(255,255,255,255), thickness=1):
        display.draw_box(self.box, clr=clr, thickness=thickness, select_context=self.track_id)
        if hasattr(self, "predicted_box"):
            #print(self.predicted_box)
            display.draw_box(self.predicted_box, clr=(180,255,100,255), thickness=thickness, select_context=self.track_id)
        if self.num_pose>0:
            lines=[[0,1],[0,2],[0, 5, 6],[1, 3],[2, 4],[5, 6],
               [5, 11],[6,12],[11,12],[5,7],[7,9],[6,8],
               [8,10],[11,13],[13,15],[12,14],[14,16]]
            for l in lines:
                ok=True
                for pt in l:
                    if self.pose_conf[pt]<0.01:
                        ok=False
                if ok:
                    if len(l)==2:
                        display.draw_line(self.pose_pos[l[0]], self.pose_pos[l[1]], clr=clr)
                    else:
                        p1=self.pose_pos[l[1]]
                        p2=self.pose_pos[l[2]]
                        mid=[0.5*(p1[0]+p2[0]), 0.5*(p1[1]+p2[1])]
                        display.draw_line(self.pose_pos[l[0]], mid, clr=clr)
        display.draw_text(f"ID {int(self.track_id)} {self.cl} {self.confidence:0.2f}", 
                          self.box[0],
                          self.box[1],
                          fontScale=0.5,
                          fontColor=(200,255,255,255),
                          bgColor=clr)

    def update_predict(self, motiontracker):
        motiontracker.predict_box(self.box)
        for i in range(self.num_pose):
            motiontracker.predict_point(self.pose_pos[i])

    def match_score(self, other, context):
        min_conf=context["min_conf"]
        kf_weight=context["kf_weight"]
        kp_weight=context["kp_weight"]

        if self is None:
            return 0
        if other is None:
            return 0
    
        if self.adjusted_confidence<min_conf:
            return 0
        
        box_score=stuff.coord.box_iou(self.box, other.box)
        kf_score=stuff.coord.box_iou(self.box, other.predicted_box)

        kp_score=None
        if True and len(self.pose_pos)==17 and len(other.pose_pos)==17 and (box_score+kf_score)!=0:
            scales=[0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089]
            scales=[x*2 for x in scales] # scale=2*sigma
            ss=0.5*(stuff.coord.box_a(self.box)+stuff.coord.box_a(other.box))*0.53 # approximation of shape area from box area
            ss*=10 # fudge
            num=0
            denom=0
            for i,_ in enumerate(self.pose_pos):
                if self.pose_conf[i]>0.0 and other.pose_conf[i]>0.0: # is point labelled
                    dx=self.pose_pos[i][0]-other.pose_pos[i][0]
                    dy=self.pose_pos[i][1]-other.pose_pos[i][1]
                    num+=math.exp(-(dx*dx+dy*dy)/(2.0*ss*scales[i]*scales[i]+1e-7))
                denom+=1.0
            kp_score=num/(denom+1e-7)

        attr_score=0
        if False and self.attr!=None and other.attr!=None and box_score!=0:
            a=np.array(self.attr)
            b=np.array(other.attr)
            attr_score=np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)+1e-7)
            a0 = " ".join([f"{num:.2f}" for num in self.attr])
            a1 = " ".join([f"{num:.2f}" for num in other.attr])
            #print(a0)
            #print(a1)
            #print(attr_score)

        #if box_score+kp_score>0:
        #    print(self.track_id, other.track_id, box_score, kp_score, attr_score)
        #if other.track_id==60:
        #    if box_score+kf_score+kp_score!=0:
        #        print(box_score,kf_score,kp_score)

        if kp_score is None:
            score=(box_score+kf_score*kf_weight)/(1.0+kf_weight)
        else:
            score=(box_score+kf_score*kf_weight+kp_score*kp_weight)/(1.0+kf_weight+kp_weight)
        if score<context["match_thr"]:
            return 0
        
        score=score*math.pow(self.confidence, context["fuse_scores"])
        return score
    
class ObjectTracker:
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

class MotionTracker:
    def __init__(self, params=None):
        self.params=params
        self.width=0
        self.height=0
        self.last_frame=None
        self.last_frame_time=None
        self.num_frames=0
        self.grid_size=0
        self.flow=None
        self.motion_array=None
        self.nvof_w=0
        self.nvof_h=0
        self.roi=[0,0,0,0]
        self.accumulated_delta=None

    def predict_delta(self, pt):
        if self.motion_array is None:
            return 0,0
        x=pt[0]
        y=pt[1]
        grid_h=self.motion_array.shape[0]
        grid_w=self.motion_array.shape[1]
        ix=max(0, min(grid_w-1, int(x*grid_w)))
        iy=max(0, min(grid_h-1, int(y*grid_h)))
        dx=self.motion_array[iy][ix][0]
        dy=self.motion_array[iy][ix][1]
        return dx,dy

    def predict_point(self, pt):
        dx,dy=self.predict_delta(pt)
        pt[0]=stuff.coord.clip01(pt[0]-dx)
        pt[1]=stuff.coord.clip01(pt[1]-dy)

    def predict_box(self, box):
        dxs=0
        dys=0
        for i,k in enumerate([[0.5, 0.5], [0.25, 0.5], [0.75,0.5], [0.25, 0.5], [0.75, 0.5]]):
            xf=k[0]
            yf=k[0]
            pt=[box[0]*xf+box[2]*(1-xf), box[1]*yf+box[3]*(1-yf)]
            dx,dy=self.predict_delta(pt)
            s=4 if k==0 else 1
            dxs+=dx*s
            dys+=dy*s
        dxs/=8
        dys/=8
        box[0]=stuff.coord.clip01(box[0]-dxs)
        box[1]=stuff.coord.clip01(box[1]-dys)
        box[2]=stuff.coord.clip01(box[2]-dxs)
        box[3]=stuff.coord.clip01(box[3]-dys)

    def add_frame(self, img, time):
        img_height, img_width, _ = img.shape
        scalef=min(640/img_height, 640/img_width)
        scale_w=int(img_width*scalef)
        scale_h=int(img_height*scalef)
        scale_w=4*((scale_w+3)//4)
        scale_h=4*((scale_h+3)//4)

        if scale_w!=self.width or scale_h!=self.height:
            #print(f"MotionTracker : resize to {scale_w} x {scale_h}")
            if True:
                self.nvof = cv2.cuda_NvidiaOpticalFlow_2_0.create(imageSize=[scale_w, scale_h],
                                                              enableCostBuffer=True,
                                                              enableTemporalHints=True,
                                                              perfPreset=cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_PERF_LEVEL_MEDIUM,
                                                              outputGridSize=cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_OUTPUT_VECTOR_GRID_SIZE_4)
            else:
                self.nvof = cv2.cuda_NvidiaOpticalFlow_1_0.create(imageSize=[scale_w, scale_h],
                                                              enableCostBuffer=True,
                                                              enableTemporalHints=True,
                                                              perfPreset=cv2.cuda.NvidiaOpticalFlow_1_0_NV_OF_PERF_LEVEL_FAST)
                    
            self.grid_size=self.nvof.getGridSize()
            self.last_frame=None
            self.num_frames=0
            self.width=scale_w
            self.height=scale_h
            self.flow=None
            self.nvof_w=int(scale_w/self.grid_size)
            self.nvof_h=int(scale_h/self.grid_size)
            self.accumulated_delta=np.ones((self.nvof_h, self.nvof_w), dtype=int)*1000

        img_scaled=cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (self.width, self.height))

        if self.num_frames!=0:
            self.flow = self.nvof.calc(img_scaled, self.last_frame, None)
            delta=self.flow[1]
            delta=np.maximum(0, (delta//20)-5)
            self.accumulated_delta+=delta
            #print(delta)
            #print(self.accumulated_delta)
            self.motion_array = self.flow[0].astype(np.float32)
            self.motion_array[..., 0]*=(1.0/(32*self.width))
            self.motion_array[..., 1]*=(1.0/(32*self.height))

        self.last_frame=img_scaled
        self.num_frames+=1
    
    def get_roi(self, thr):
        roi=[0,0,0,0]
        rows, cols = np.where(self.accumulated_delta > thr)
        if rows.size > 0 and cols.size > 0:
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            roi= [min_col/self.nvof_w, min_row/self.nvof_h, (max_col+1)/self.nvof_w, (max_row+1)/self.nvof_h]
        return roi
    
    def set_roi_detected(self, roi):
        l=int(self.nvof_w*roi[0]+0.99)
        t=int(self.nvof_h*roi[1]+0.99)
        r=int(self.nvof_w*roi[2]+0.99)
        b=int(self.nvof_h*roi[3]+0.99)
        #print("CLEAR ",roi, l,r,t,b)
        self.accumulated_delta[t:b, l:r]=0

    def draw(self, img):
        if self.flow is None:
            return img
        scale=8
        
        grid_h=self.flow[0].shape[0]
        grid_w=self.flow[0].shape[1]
        overlay=np.zeros((grid_h*scale, grid_w*scale, 3), dtype=np.uint8)
        m=scale

        #print(f"grid {grid_w}x{grid_h} overlay {overlay.shape}")
        for y in range(grid_h):
            for x in range(grid_w):
                v=self.flow[1][y][x]
                vx=int(self.motion_array[y][x][0]*grid_w*scale)
                vy=int(self.motion_array[y][x][1]*grid_h*scale)
                inside=False
                #if self.bounding_box!=None:
                #    min_row, max_row, min_col, max_col = self.bounding_box
                #    inside=y>=min_row and y<max_row and x>=min_col and x<max_col
                #if inside:
                #    c=200
                #else:
                #    c=0

                clr=(0,max(0,min(255,int(self.accumulated_delta[y][x]))),0)
                cv2.rectangle(overlay,
                              (x*m, y*m), 
                              ((x+1)*m, (y+1)*m), clr, cv2.FILLED)
                if abs(vx)>8 or abs(vy)>8:
                    cv2.line(overlay, (x*m+m//2, y*m+m//2), (x*m+m//2+(vx//8), y*m+m//2+(vy//8)), (0, 255, 0), thickness=1)

        alpha=0.4
        img_h, img_w, _=img.shape
        overlay=cv2.resize(overlay, (img_w, img_h))
        img2=cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        return img2
