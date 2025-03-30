
import numpy as np
import math
import copy
import stuff

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
            box=copy.copy(detection["box"])
            cl=detection["class"]
            conf=detection["confidence"]
            if "pose_points" in detection:
                pose=copy.copy(detection["pose_points"])
            if "attrs" in detection:
                attr=copy.copy(detection["attrs"])

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
                if False and self.pose_conf[i]>0.05:
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
    