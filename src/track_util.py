
import numpy as np
import math
import copy
import stuff

def object_interpolate(a, b, frac):
    box=stuff.interpolate2(a.box, b.box, frac)

    if a.subbox is not None and b.subbox is not None:
        subbox=stuff.interpolate2(a.subbox, b.subbox, frac)
        subbox_conf=stuff.interpolate2(a.subbox_conf, b.subbox_conf, frac)
    else:
        source = a if frac<0.5 else b
        subbox=copy.deepcopy(source.subbox)
        subbox_conf=source.subbox_conf

    conf=stuff.interpolate2(a.confidence, b.confidence, frac)

    if len(a.pose_pos)==len(b.pose_pos) and len(a.pose_pos)>0:
        pose_pos=[stuff.interpolate2(a.pose_pos[i], b.pose_pos[i], frac) for i in range(len(a.pose_pos))]
        pose_conf=[a.pose_conf[i]*b.pose_conf[i] for i in range(len(a.pose_conf))]
    else:
        if len(a.pose_pos)>=len(b.pose_pos):
            chosen=a if len(a.pose_pos)>0 else b
        else:
            chosen=b
        pose_pos=copy.deepcopy(chosen.pose_pos)
        pose_conf=copy.deepcopy(chosen.pose_conf)

    face_pos=None
    face_conf=None
    if a.num_face_points>0 and b.num_face_points>0 and a.num_face_points==b.num_face_points:
        face_pos=[stuff.interpolate2(a.face_pos[i], b.face_pos[i], frac) for i in range(len(a.face_pos))]
        face_conf=[a.face_conf[i]*b.face_conf[i] for i in range(len(a.face_conf))]
    else:
        if a.num_face_points>0 or b.num_face_points>0:
            chosen=a if frac<0.5 else b
            if chosen.num_face_points==0:
                chosen=a if a.num_face_points>0 else b
            if chosen.num_face_points>0:
                face_pos=copy.deepcopy(chosen.face_pos)
                face_conf=copy.deepcopy(chosen.face_conf)

    obj=Object(box=box, cl=a.cl, conf=conf,
               pose_conf=pose_conf,
               pose_pos=pose_pos,
               face_pos=face_pos,
               face_conf=face_conf,
               subbox=subbox,
               subbox_conf=subbox_conf)
    if frac<0.5:
        obj.face_jpeg=a.face_jpeg
        obj.clip_jpeg=a.clip_jpeg
        obj.face_embedding=a.face_embedding
        obj.clip_embedding=a.clip_embedding
    else:
        obj.face_jpeg=b.face_jpeg
        obj.clip_jpeg=b.clip_jpeg
        obj.face_embedding=b.face_embedding
        obj.clip_embedding=b.clip_embedding
    return obj

def object_class_remap(objects, initial_classes, target_classes):
    if initial_classes==target_classes:
        return objects
    if target_classes is None:
        return objects
    remap=stuff.make_class_remap_table(initial_classes, target_classes)
    ret=copy.deepcopy(objects)
    for r in ret:
        if r.cl in remap:
            r.cl=remap[r.cl]
        else:
            r.cl=None
    return [o for o in ret if o.cl is not None]

class Object:
    def to_det(self):
        det={}
        det["class"]=self.cl
        det["box"]=self.box
        det["confidence"]=self.confidence
        det["track_id"]=int(self.track_id)
        assert str(det["track_id"])==str(self.track_id)
        return det

    def __init__(self, box=None, cl=None, conf=None,
                 pose=None, pose_conf=None, pose_pos=None,
                 face_points=None, face_conf=None, face_pos=None,
                 attr=None, subbox=None, subbox_conf=0.0, time=None, detection=None, expand_by_pose=False):

        self.reid_vector=None
        self.face_jpeg=None
        self.clip_jpeg=None
        self.face_embedding=None
        self.clip_embedding=None
        if detection is not None:
            box=copy.copy(detection["box"])
            cl=detection["class"]
            conf=detection["confidence"]
            if "pose_points" in detection:
                pose=copy.copy(detection["pose_points"])
            if "face_points" in detection:
                face_points=copy.copy(detection["face_points"])
            if "attrs" in detection:
                attr=copy.copy(detection["attrs"])
            if "reid_vector" in detection:
                self.reid_vector=detection["reid_vector"]
            if "subbox" in detection:
                subbox=detection["subbox"]
            if "subbox_conf" in detection:
                subbox_conf=detection["subbox_conf"]
            if "face_jpeg" in detection:
                self.face_jpeg=detection["face_jpeg"]
            if "clip_jpeg" in detection:
                self.clip_jpeg=detection["clip_jpeg"]
            if "face_embedding" in detection:
                self.face_embedding=detection["face_embedding"]
            if "clip_embedding" in detection:
                self.clip_embedding=detection["clip_embedding"]

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

        if face_points is not None:
            self.num_face_points=len(face_points)//3
            self.face_pos=[[0,0] for n in range(self.num_face_points)]
            self.face_conf=[0]*self.num_face_points
            for i in range(self.num_face_points):
                self.face_pos[i][0]=face_points[i*3+0]
                self.face_pos[i][1]=face_points[i*3+1]
                self.face_conf[i]=face_points[i*3+2]
        elif face_conf is not None:
            self.num_face_points=len(face_conf)
            self.face_conf=face_conf
            self.face_pos=face_pos
        else:
            self.num_face_points=0
            self.face_pos=[]
            self.face_conf=[]

        self.cl=cl
        if attr is None:
            attr=[0]*35
        self.subbox=subbox
        self.subbox_conf=subbox_conf
        self.attr=attr
        self.time=time
        self.track_id=None

    def draw(self, display, clr=(255,255,255,255), thickness=1, label_prefix=None):
        display.draw_box(self.box, clr=clr, thickness=thickness, select_context=self.track_id)
        if self.face_jpeg is not None:
            display.draw_jpeg(self.face_jpeg, select_context=self.track_id)
        if hasattr(self, "predicted_box"):
            #print(self.predicted_box)
            display.draw_box(self.predicted_box, clr=(180,255,100,255), thickness=thickness, select_context=self.track_id)
        if hasattr(self, "subbox") and self.subbox is not None:
            display.draw_box(self.subbox, clr=(255,100,100,255), thickness=thickness, select_context=self.track_id)
        if self.num_face_points>0:
            assert self.num_face_points==5
            for i in range(5):
                if self.face_conf[i]!=0:
                    clr="half_red"
                    if i==0 or i==3: # RIGHT points
                        clr="half_yellow"
                    display.draw_circle(self.face_pos[i], radius=0.002, clr=clr)
        if self.num_pose>0:
            clr="half_yellow"
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
        label=f"ID {int(self.track_id):x} {self.cl} {self.confidence:0.2f}"
        if label_prefix is not None:
            label=label_prefix+label
        display.draw_text(label,
                          self.box[0],
                          self.box[1],
                          fontScale=0.5,
                          fontColor=(200,255,255,255),
                          bgColor=clr)
