import src.trackset as ts
import src.trackers as trackers
import argparse
import math
import stuff

def do_track(video, model):
    #tracker=trackers.nvof_tracker(model, 0.05, None)
    tracker=trackers.ultralytics_tracker(model, 0.05, "botsort.yaml")
    trackset_gt=ts.TrackSet(video)
    fps=trackset_gt.metadata["frame_rate"]
    paused=True
    time=0
    display=stuff.Display(width=1280, height=720)

    while True:
        if time>trackset_gt.duration_seconds():
            break

        frame=trackset_gt.img_at_time(time)
        objs=tracker.track_frame(frame, time)
        if objs is not None:
            for o in objs:
                o.draw(display)

        #stuff.draw.draw_text(display, f"Time {time:5.3f} Objects: {len(objs)}", 0.05, 0.05)
        display.show(frame, title=f"time={time:5.2f}")
        #events=display.get_events(10)
        time+=(1.0/fps)
    
def test_track(t, config_file):
    trackset_gt=ts.TrackSet(t)
    trackset=ts.TrackSet()

    trackset.import_create(trackset_gt,
                           track_min_interval=0.159,
                           debug=False,
                           config_file=config_file)
        
    metrics, frame_events=ts.compute_metrics(trackset_gt, trackset, frame_metrics=True)
    print(metrics)
    show_trackset(trackset=trackset, trackset_gt=trackset_gt, frame_events=frame_events)

def show_trackset(trackset=None, trackset_gt=None, frame_events=None):
    if isinstance(trackset, str):
        trackset=ts.TrackSet(trackset)
    if isinstance(trackset_gt, str):
        trackset_gt=ts.TrackSet(trackset_gt)

    trackset_base=trackset_gt if trackset_gt is not None else trackset
    duration=trackset_base.duration_seconds()
    t=0
    paused=True
    show_gts=True
    show_det=True

    display=stuff.Display(width=1280, height=720)
    selected_ids=[]

    while(t<duration):
        display.clear()

        img=trackset_base.img_at_time(t)

        events={}
        if frame_events:
            best_diff=100000
            best_index=0
            for i, e in enumerate(frame_events):
                diff=abs(e["frame_time"]-t)
                if diff<best_diff:
                    best_diff=diff
                    best_index=i
            events=frame_events[best_index]["events"]

        if trackset_gt and show_gts:
            objs_gt=trackset_gt.objects_at_time(t)
            for o in objs_gt:
                a=255 if o.track_id in selected_ids else 48
                clr=(a,0,0,0)
                thickness=2
                for e in events:
                    if math.isnan(events[e]["OId"]):
                        continue
                    if int(events[e]["OId"])==o.track_id:
                        if events[e]["Type"]=="SWITCH":
                            clr=(a,0,128,128)
                        elif events[e]["Type"]=="MATCH":
                            clr=(a,0,128,0)
                        elif events[e]["Type"]=="MISS":
                            clr=(a,0,0,128)
                            thickness=4
                o.draw(display, clr=clr, thickness=thickness)

        if trackset and show_det:
            objs=trackset.objects_at_time(t)
            for o in objs:
                a=255 if o.track_id in selected_ids else 48
                clr=(a,255,255,255)
                thickness=2
                for e in events:
                    if math.isnan(events[e]["HId"]):
                        continue
                    if int(events[e]["HId"])==o.track_id:
                        if events[e]["Type"]=="SWITCH":
                            clr=(a,0,255,255)
                        elif events[e]["Type"]=="MATCH":
                            clr=(a,0,255,0)
                        elif events[e]["Type"]=="FP":
                            clr=(a,0,0,255)
                            thickness=4
                o.draw(display, clr=clr, thickness=thickness)

        display.show(img, title=f"time={t:5.2f}")
        events=display.get_events(10)
        for e in events:
            if 'selected' in e:
                selected_ids=[]
                for box in e['selected']:
                    selected_ids.append(box['context'])
            if e['key']=='g':
                show_gts=not show_gts
            if e['key']=='d':
                show_det=not show_det
            if e['key']==' ':
                paused=not paused
            if e['key']=='.':
                t+=0.033
            if e['key']==',':
                t-=0.033
        if paused is False:
            t+=0.033

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='view.py')
    parser.add_argument('--model', type=str, default='/mldata/weights/yolo11l-dpa-131224.pt', help='model to use')
    parser.add_argument('--video', type=str, default='/mldata/video/mall_escalators.264')
    parser.add_argument('--trackset', type=str, default='/mldata/downloaded_datasets/other/MOT20/train/MOT20-01/seqinfo.ini')
    parser.add_argument('--draw', action='store_true', help='overlay graphics')
    parser.add_argument('--view', action='store_true', help='view a trackset')
    parser.add_argument('--caltech', action='store_true', help='make caltech pedestrian sequences')
    parser.add_argument('--mot', action='store_true', help='make MOT sequences')
    parser.add_argument('--test', type=str, default=None, help='test yaml file')
    parser.add_argument('--search', type=str, default=None, help='search config yaml file')
    parser.add_argument('--track', action='store_true', help='test tracker')
    parser.add_argument('--tracker', type=str, default="ultralytics", help='tracker to use, ultralytics, nvof or cevo')
    parser.add_argument('--config', type=str, default="/mldata/config/track/bytetrack_nofuse.yaml", help="config")
    opt = parser.parse_args()
    if opt.caltech:
        ts.convert_caltech_pedestrian()
        exit()
    if opt.mot:
        ts.convert_mot()
        exit()
    if opt.track:
        test_track(opt.trackset, opt.config)
        exit()
    if opt.search is not None:
        ts.search_track(opt.search)
        exit()
    if opt.test is not None:
        ts.test_track(opt.test)
        exit()
    if opt.view and opt.trackset is not None:
        show_trackset(trackset_gt=opt.trackset)
        exit()

    do_track(opt.trackset, opt.model)