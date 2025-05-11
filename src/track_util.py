
import numpy as np
import math
import copy
import stuff

def object_interpolate(a, b, frac):
    if True:
        box=stuff.interpolate2(a.box, b.box, frac)
        conf=stuff.interpolate2(a.confidence, b.confidence, frac)
        pose_pos=[stuff.interpolate2(a.pose_pos[i], b.pose_pos[i], frac) for i in range(len(a.pose_pos))]
        pose_conf=[a.pose_conf[i]*b.pose_conf[i] for i in range(len(a.pose_conf))]
        return Object(box=box, cl=a.cl, conf=conf, pose_conf=pose_conf, pose_pos=pose_pos)
    else:
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

def object_class_remap(objects, initial_classes, target_classes):
    if target_classes is None:
        return objects
    remap=[None]*len(initial_classes)
    for i,cl in enumerate(target_classes):
        if cl in initial_classes:
            remap[initial_classes.index(cl)]=i
    ret=copy.deepcopy(objects)
    for r in ret:
        if r.cl in remap:
            r.cl=remap[r.cl]
        else:
            r.cl=None
    return [o for o in ret if o.cl is not None]

class Object:
    def __init__(self, box=None, cl=None, conf=None,
                 pose=None, pose_conf=None, pose_pos=None,
                 attr=None, time=None, detection=None, expand_by_pose=False):
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
                if expand_by_pose and self.pose_conf[i]>0.05:
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

    def draw(self, display, clr=(255,255,255,255), thickness=1, label_prefix=None):
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
        label=f"ID {int(self.track_id)} {self.cl} {self.confidence:0.2f}"
        if label_prefix is not None:
            label=label_prefix+label
        display.draw_text(label,
                          self.box[0],
                          self.box[1],
                          fontScale=0.5,
                          fontColor=(200,255,255,255),
                          bgColor=clr)
