import cv2
import stuff
import copy
import yaml

class MotionTracker:
    def __init__(self, params=None):
        try:
            import ubon_pycstuff.ubon_pycstuff as upyc
        except:
            assert False, "Motion_tracker equires ubon_pycstuff"

        self.upyc=upyc
        self.params=params
        self.last_frame_time=None
        self.num_frames=0
        self.motion_array=None
        self.simple=params.get("simple", False)
        self.motiontracker=self.upyc.c_motion_tracker(yaml.dump(params))

    def predict_delta(self, pt):
        if self.motion_array is None:
            return 0,0
        x=pt[0]
        y=pt[1]
        grid_h=self.motion_array.shape[0]
        grid_w=self.motion_array.shape[1]
        ix=max(0, min(grid_w-1, int(x*grid_w+0.5)))
        iy=max(0, min(grid_h-1, int(y*grid_h+0.5)))
        dx=self.motion_array[iy][ix][0]
        dy=self.motion_array[iy][ix][1]
        return dx,dy

    def predict_point(self, pt, t):
        assert t==self.last_frame_time
        dx,dy=self.predict_delta(pt)
        return [stuff.coord.clip01(pt[0]-dx), stuff.coord.clip01(pt[1]-dy)]

    def predict_box(self, box, t):
        if box is None:
            return None
        assert t==self.last_frame_time
        dxs=0
        dys=0

        for i,k in enumerate([[0.5, 0.5, 0.5], [0.35, 0.5, 0.125], [0.65,0.5, 0.125], [0.5, 0.35, 0.125], [0.5, 0.65, 0.125]]):
            xf=k[0]
            yf=k[1]
            s=k[2]
            pt=[box[0]*xf+box[2]*(1-xf), box[1]*yf+box[3]*(1-yf)]
            dx,dy=self.predict_delta(pt)
            dxs+=dx*s
            dys+=dy*s
        return [stuff.coord.clip01(box[0]-dxs),
                stuff.coord.clip01(box[1]-dys),
                stuff.coord.clip01(box[2]-dxs),
                stuff.coord.clip01(box[3]-dys)]

    def add_frame(self, img, time):
        self.motiontracker.add_frame(img)
        self.motion_array=copy.deepcopy(self.motiontracker.get_of_results())
        self.nvof_h, self.nvof_w, _=self.motion_array.shape
        self.last_frame_time=time
        self.num_frames+=1

    def get_roi(self):
        return self.motiontracker.get_roi()

    def set_roi_detected(self, roi):
        self.motiontracker.set_roi(roi)

    def get_debug(self):
        debug={"motion_track": {"type": "motion_track", "data":{"motion_array":copy.deepcopy(self.motion_array)}}}
        return debug

def motiontracker_test():
    img=cv2.imread("/mldata/image/arrest2.jpg")
    mt=MotionTracker()
    images=[]
    w=256
    h=256
    for x in [0, -4, -8]:
        images.append(img[100:(100+h), (100+x):(100+x+w)])
    mt.add_frame(images[0], 0)
    mt.add_frame(images[2], 1)
    box=[0.4,0.4,0.6,0.6]
    out_box=mt.predict_box(box, 1)

    print(out_box)
    print((out_box[0]-box[0])*w)
    print((out_box[1]-box[1])*h)
    print((out_box[2]-box[2])*w)
    print((out_box[3]-box[3])*h)