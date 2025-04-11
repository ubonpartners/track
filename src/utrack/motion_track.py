import cv2
import numpy as np
import src.utrack.kalman as kalman
import stuff
import copy

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
        img_height, img_width, _ = img.shape
        self.last_frame_time=time
        scalef=min(320/img_height, 320/img_width)
        scale_w=int(img_width*scalef)
        scale_h=int(img_height*scalef)
        scale_w=4*((scale_w+3)//4)
        scale_h=4*((scale_h+3)//4)

        if scale_w!=self.width or scale_h!=self.height:
            #print(f"MotionTracker : resize to {scale_w} x {scale_h}")
            self.nvof = cv2.cuda_NvidiaOpticalFlow_2_0.create(imageSize=[scale_w, scale_h],
                                                              enableCostBuffer=True,
                                                              enableTemporalHints=False,
                                                              perfPreset=cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_PERF_LEVEL_MEDIUM,
                                                              outputGridSize=cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_OUTPUT_VECTOR_GRID_SIZE_4,
                                                              hintGridSize=cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_HINT_VECTOR_GRID_SIZE_4)
                    
            self.grid_size=self.nvof.getGridSize()
            self.last_frame=None
            self.num_frames=0
            self.width=scale_w
            self.height=scale_h
            self.flow=None
            self.nvof_w=int(scale_w/self.grid_size)
            self.nvof_h=int(scale_h/self.grid_size)
            self.hint_buffer=np.zeros((self.nvof_h, self.nvof_w, 2), dtype=np.int16)
            self.accumulated_delta=np.ones((self.nvof_h, self.nvof_w), dtype=int)*1#1000

        img_scaled=cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (self.width, self.height))

        if self.num_frames!=0:
            self.flow = self.nvof.calc(img_scaled, self.last_frame, None, hint=self.hint_buffer)
            delta=self.flow[1]
            delta=np.maximum(0, delta-5)
            self.accumulated_delta+=delta
            self.motion_array = self.flow[0].astype(np.float32)
            self.motion_array[..., 0]*=(1.0/(32*self.width))
            self.motion_array[..., 1]*=(1.0/(32*self.height))
            self.accumulated_delta+= (100 * np.abs(self.motion_array[..., 0])).astype(self.accumulated_delta.dtype)
            self.accumulated_delta+= (100 * np.abs(self.motion_array[..., 1])).astype(self.accumulated_delta.dtype)

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

    def get_debug(self):
        return {"motion_track": {"type": "motion_track", "data":{"motion_array":copy.copy(self.motion_array), "delta_array":copy.copy(self.accumulated_delta)}}}
            

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
    
def motiontracker_test():
    img=cv2.imread("/mldata/image/arrest2.jpg")
    img_h, img_w, _=img.shape
    mt=MotionTracker()
    images=[]
    w=256
    h=256
    for x in [0, 4, 8]:
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