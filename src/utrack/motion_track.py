import cv2
import numpy as np
import src.utrack.kalman as kalman
import ubon_pycstuff.ubon_pycstuff as upyc
import stuff
import copy

class MotionTracker:
    def __init__(self, params=None):
        self.params=params
        self.last_frame_time=None
        self.num_frames=0
        self.flow=None
        self.motion_array=None
        self.roi=[0,0,0,0]
        self.accumulated_delta=None
        self.flow_engine=None
        self.old_motion_surf=None

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

    def predict_point(self, pt, t):
        assert t==self.last_frame_time
        dx,dy=self.predict_delta(pt)
        return [stuff.coord.clip01(pt[0]-dx), stuff.coord.clip01(pt[1]-dy)]

    def predict_box(self, box, t):
        assert t==self.last_frame_time
        dxs=0
        dys=0
        for i,k in enumerate([[0.5, 0.5], [0.35, 0.5], [0.65,0.5], [0.35, 0.5], [0.65, 0.5]]):
            xf=k[0]
            yf=k[1]
            pt=[box[0]*xf+box[2]*(1-xf), box[1]*yf+box[3]*(1-yf)]
            dx,dy=self.predict_delta(pt)
            s=4 if i==0 else 1
            dxs+=dx*s
            dys+=dy*s
        dxs/=8
        dys/=8
        return [stuff.coord.clip01(box[0]-dxs),
                stuff.coord.clip01(box[1]-dys),
                stuff.coord.clip01(box[2]-dxs),
                stuff.coord.clip01(box[3]-dys)]

    def add_frame(self, img, time):
        cimg=upyc.c_image.from_numpy(img).convert(upyc.MONO_DEVICE)
        cimg=cimg.scale(320,180)
        self.new_motion_surf=cimg.blur()
        if self.old_motion_surf!=None:
            mad_surf=self.new_motion_surf.mad_4x4(self.old_motion_surf)
            self.mad_cost=mad_surf.to_numpy()
        else:
            self.mad_cost=None

        if self.flow_engine is None:
            self.flow_engine=upyc.c_nvof(320, 320)
        costs, self.motion_array = copy.deepcopy(self.flow_engine.run(upyc.c_image.from_numpy(img)))
        self.nvof_h, self.nvof_w=costs.shape
        #print(self.nvof_w, self.nvof_h)
        if self.num_frames==0:
            self.accumulated_delta=np.ones((self.nvof_h, self.nvof_w), dtype=int)*1
        self.last_frame_time=time
        #print("motion", self.motion_array.shape)
        self.accumulated_delta+=(2*costs)
        self.accumulated_delta+= (1 * np.abs(self.motion_array[..., 0])).astype(self.accumulated_delta.dtype)
        self.accumulated_delta+= (1 * np.abs(self.motion_array[..., 1])).astype(self.accumulated_delta.dtype)
            
        self.num_frames+=1
    
    def get_roi(self, thr):
        roi=[0,0,0,0]
        #rows, cols = np.where(self.accumulated_delta > thr)
        #if rows.size > 0 and cols.size > 0:
        #    min_row, max_row = rows.min(), rows.max()
        #    min_col, max_col = cols.min(), cols.max()
        #    roi= [min_col/self.nvof_w, min_row/self.nvof_h, (max_col+1)/self.nvof_w, (max_row+1)/self.nvof_h]
        if self.mad_cost is not None:
            rows, cols = np.where(self.mad_cost>self.params["motiontracker_mad_delta"])
            if len(rows)==0:
                return [0,0,0,0]
        else:
            return [0,0,1,1]

        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        h,w=self.mad_cost.shape
        roi= [min_col/w, min_row/h, (max_col+1)/w, (max_row+1)/h]
        return roi
    
    def set_roi_detected(self, roi):
        l=int(self.nvof_w*roi[0]+0.99)
        t=int(self.nvof_h*roi[1]+0.99)
        r=int(self.nvof_w*roi[2]+0.99)
        b=int(self.nvof_h*roi[3]+0.99)
        #print("CLEAR ",roi, l,r,t,b)
        self.accumulated_delta[t:b, l:r]=0

        if stuff.box_a(roi)>0.9 or self.old_motion_surf is None:
            self.old_motion_surf=self.new_motion_surf
        else:
            h,w=180,320
            l=int(w*roi[0])
            t=int(h*roi[1])
            r=int(w*roi[2])
            b=int(h*roi[3])
            l=l&(~3)
            t=t&(~3)
            r=(r+3)&(~3)
            b=(b+3)&(~3)
            #print(roi,l,r,t,b)
            self.old_motion_surf=self.old_motion_surf.blend(self.new_motion_surf, l,t,r-l,b-t,l,t)
            #self.old_motion_surf.display("old_motion_surf")
    def get_debug(self):
        return {"motion_track": {"type": "motion_track", "data":{"motion_array":copy.deepcopy(self.motion_array), "delta_array":copy.copy(self.accumulated_delta)}}}
            

    def draw(self, img):
        if self.flow is None:
            return img
        scale=8
        
        grid_h=self.motion_array.shape[0]
        grid_w=self.motion_array.shape[1]
        overlay=np.zeros((grid_h*scale, grid_w*scale, 3), dtype=np.uint8)
        m=scale
        #print(f"grid {grid_w}x{grid_h} overlay {overlay.shape}")
        for y in range(grid_h):
            for x in range(grid_w):
                v=self.costs[y][x]
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
    
def motiontracker_test():
    img=cv2.imread("/mldata/image/arrest2.jpg")
    img_h, img_w, _=img.shape
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