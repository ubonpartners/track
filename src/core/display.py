"""The OpenCV replay viewer for GT and tracked runs.

Moved verbatim from src/trackset.py (repo_cleanup.md stage 4c).
"""
import math

import stuff

from src.core.trackset import TrackSet


def onoff(x):
    if x:
        return "[ON]"
    else:
        return "[OFF]"


def display_trackset(trackset_list=None, trackset_gt=None, frame_events_list=None, cl=["person"], output=None, max_duration=10000):

    if trackset_list is None:
        if trackset_gt is None:
            raise ValueError("display_trackset requires at least one trackset.")
        trackset_list = [trackset_gt]

    tss=[]

    for i, ts in enumerate(trackset_list):
        name=f"Trackset {i}"
        if isinstance(ts, str):
            name+=":"+ts
            ts=TrackSet(ts)
        name+="["+ts.name+"]"
        frame_events=None
        if frame_events_list is not None:
            frame_events=frame_events_list[i]
        tss.append({"name": name,
                    "display":stuff.Display(width=1280, height=720, output=output, name=name),
                    "selected_ids":[],
                    "show":False,
                    "trackset":ts,
                    "frame_events":frame_events})

    if isinstance(trackset_gt, str):
        trackset_gt=TrackSet(trackset_gt)

    trackset_base=trackset_gt if trackset_gt is not None else trackset_list[0]
    # exact frame interval: t+=0.033 drifted 1% vs the frame grid,
    # forcing a duplicated frame at fixed times (~every 1.7s at 30fps,
    # e.g. T=3.35) — seen as deterministic playback hitches
    frame_dt=1.0/float(trackset_base.metadata.get("frame_rate", 30.0))
    duration=min(max_duration, trackset_base.duration_seconds())
    t=0
    paused=True
    show_gts=True
    show_det=True
    show_help=True
    show_stats=True

    debug_overlays_enabled={}
    while(t<duration):
        for ts in tss:
            trackset=ts["trackset"]
            display=ts["display"]
            selected_ids=ts["selected_ids"]
            frame_events=ts["frame_events"]
            display.clear()

            img=trackset_base.img_at_time(t)

            events={}
            stats={}
            if frame_events:
                best_diff=100000
                best_index=0
                for i, e in enumerate(frame_events):
                    diff=abs(e["frame_time"]-t)
                    if diff<best_diff:
                        best_diff=diff
                        best_index=i
                events=frame_events[best_index]["events"]
                stats=frame_events[best_index]["stats"]

            if trackset_gt and show_gts:
                objs_gt=trackset_gt.objects_at_time(t)
                for o in objs_gt:
                    obj_cl=trackset_gt.metadata["classes"][o.cl]
                    # Render "other" GT (crowd / ignore regions) distinctly: faint
                    # gray, no event matching, no track-id selection. Detections
                    # inside these are also ignored by compute_metrics.
                    if obj_cl == "other":
                        o.draw(display, clr=(80, 96, 96, 96), thickness=1,
                               label_prefix="[ign]")
                        continue
                    if not obj_cl in cl:
                        continue
                    a=200 if o.track_id in selected_ids else 48
                    clr=(a,0,0,0)
                    thickness=2
                    prefix="?"
                    #print(o.track_id)
                    for e in events:
                        if math.isnan(events[e]["OId"]):
                            continue
                       # print(f"OID is {events[e]["OId"]}")
                        if int(events[e]["OId"])==int(o.track_id):
                            if events[e]["Type"]=="SWITCH":
                                clr=(a,0,128,128)
                                prefix="[SW]"
                            elif events[e]["Type"]=="MATCH":
                                clr=(a,0,128,0)
                                if show_det:
                                    prefix=None # don't double-label OK if we show detections
                                else:
                                    prefix="[OK]"
                            elif events[e]["Type"]=="MISS":
                                clr=(a,128,0,0)
                                prefix="[MISS]"
                                thickness=4
                            elif events[e]["Type"] in ["TRANSFER","ASCEND","MIGRATE"]:
                                prefix="[TRANS]"
                            else:
                                prefix="[??]"
                                print(f"weird gt event type ", events[e]["Type"])
                    if prefix!=None:
                        o.draw(display, clr=clr, thickness=thickness, label_prefix=prefix)

            if trackset and show_det:
                objs=trackset.objects_at_time(t)
                for o in objs:
                    obj_cl=trackset.metadata["classes"][o.cl]
                    if not obj_cl in cl:
                        continue
                    a=200 if o.track_id in selected_ids else 48
                    clr=(a,255,255,255)
                    thickness=2
                    prefix="?"
                    for e in events:
                        if math.isnan(events[e]["HId"]):
                            continue
                        if int(events[e]["HId"])==o.track_id:
                            if events[e]["Type"]=="SWITCH" or events[e]["Type"]=="TRANSFER":
                                clr=(a,255,255,0)
                                prefix="[SW]"
                            elif events[e]["Type"]=="MATCH":
                                clr=(a,0,255,0)
                                prefix="[OK]"
                            elif events[e]["Type"]=="FP":
                                clr=(a,255,0,0)
                                thickness=4 #4
                                prefix="[FP]"
                            elif events[e]["Type"]=="MIGRATE":
                                clr=(a,255,255,0)
                                prefix="[MIG]"
                            elif events[e]["Type"]=="ASCEND":
                                clr=(a,255,255,0)
                                prefix="[ASC]"
                            else:
                                prefix="[??]"
                                print(f"weird det event type ", events[e]["Type"])
                    o.draw(display, clr=clr, thickness=thickness, label_prefix=prefix)

            debug, debug_time=trackset.debug_at_time(t, nearest=True)

            if trackset.skip_at_time(t, nearest=True):
                display.draw_text(f"Nearest processed frame at t={debug_time:5.2f} SKIPPED", 0.05,0.05)
            else:
                display.draw_text(f"Nearest processed frame at t={debug_time:5.2f} TRACKED", 0.05,0.05)

            if debug is not None:
                for i,d in enumerate(debug):
                    if not d in debug_overlays_enabled:
                        debug_overlays_enabled[d]=False
                    debug_entry=debug[d]
                    debug_entry_type=debug_entry["type"]
                    debug_entry_data=debug_entry["data"]
                    if not d in debug_overlays_enabled or debug_overlays_enabled[d]==False:
                        continue
                    if debug_entry_type=="detections":
                        stuff.draw_boxes(display,
                                        debug_entry_data["detections"],
                                        attributes=debug_entry_data.get("attribute_names"),
                                        highlight_index=None,
                                        class_names=debug_entry_data["class_names"])
                        if ts["show"]:
                            for i,d in enumerate(debug_entry_data["detections"]): #print(debug_entry_data["detections"])
                                print(f"{i} ", d["confidence"])
                            ts["show"]=False
                    if debug_entry_type=="motion_field":
                        flow=stuff.decode_payload(debug_entry_data["flow"]) if "flow" in debug_entry_data else debug_entry_data.get("motion_array")
                        if flow is not None:
                            grid_w=flow.shape[1]
                            grid_h=flow.shape[0]
                            for y in range(grid_h):
                                for x in range(grid_w):
                                    cx=(x+0.5)/grid_w
                                    cy=(y+0.5)/grid_h
                                    vx=flow[y][x][0]
                                    vy=flow[y][x][1]
                                    thr=0.001
                                    if abs(vx)>thr or abs(vy)>thr:
                                        display.draw_line([cx,cy],
                                                        [cx+vx, cy+vy],
                                                        clr=(128,255,255,0), thickness=1)
                            delta_array = None
                            if "delta" in debug_entry_data:
                                delta_array = stuff.decode_payload(debug_entry_data["delta"])
                            elif "delta_array" in debug_entry_data:
                                delta_array = debug_entry_data["delta_array"]
                            if delta_array is not None:
                                for y in range(grid_h):
                                    for x in range(grid_w):
                                        clr=max(0,min(255,int(delta_array[y][x])))
                                        box=[x/grid_w, y/grid_h, (x+1)/grid_w, (y+1)/grid_h]
                    if debug_entry_type=="cost_map":
                        cost_map=stuff.decode_payload(debug_entry_data["cost_map"])
                        scale=debug_entry_data["scale"]
                        if cost_map is not None:
                            grid_w=cost_map.shape[1]
                            grid_h=cost_map.shape[0]
                            for y in range(grid_h):
                                for x in range(grid_w):
                                    clr=max(0,min(255,int(scale*cost_map[y][x])))
                                    box=[x/grid_w, y/grid_h, (x+1)/grid_w, (y+1)/grid_h]
                                    display.draw_box(box, (clr,0,255,0), thickness=-1)
                    if debug_entry_type=="box_prediction":
                        for i in debug_entry_data:
                            display.draw_box(debug_entry_data[i]["from"], clr=(128,255,255,255), thickness=1)
                            display.draw_box(debug_entry_data[i]["to"], clr=(128,255,0,0), thickness=2)
                            if "pose_from" in debug_entry_data[i]:
                                stuff.draw_pose(display,
                                                pose_pos=debug_entry_data[i]["pose_from"],
                                                pose_conf=debug_entry_data[i]["pose_conf"],
                                                thickness=1, clr=(128,255,255,255))
                                stuff.draw_pose(display,
                                                pose_pos=debug_entry_data[i]["pose_to"],
                                                pose_conf=debug_entry_data[i]["pose_conf"],
                                                thickness=2, clr=(128,255,0,0))

                    if debug_entry_type=="roi":
                        box=debug_entry_data["roi"]
                        display.draw_box(box, clr=(16,255,255,0), thickness=-1)
                        display.draw_box(box, clr=(128,255,0,0), thickness=4)

            help="HELP\n"
            help+=f"h) toggle this help display {onoff(show_help)}\n"
            help+=f"s) toggle stats display {onoff(show_stats)}\n"
            help+=f"< > advance time, +SHIFT to skip to next tracked frame\n"
            help+=f"D, G toggle tracking Det {onoff(show_det)} GTs {onoff(show_gts)}\n"
            help+=f"<space> toggle continous playback {onoff(not paused)}\n"
            help+=f"Debug overlays-\n"
            for i,e in enumerate(debug_overlays_enabled):
                help+=(f"--- {i+1} Toggle : {e:20s} {onoff(debug_overlays_enabled[e])}\n")

            if show_help:
                display.draw_text(help, 0.05, 0.1)

            if show_stats and len(stats)>0:
                sstats="STATS\n"
                for s in stats:
                    sstats+=f"{s:20}: {stats[s]}\n"
                display.draw_text(sstats, 0.75, 0.1)

            title=ts["name"]+f"time={t:5.2f}"
            display.show(img, title=title)


        #end tss loop

        for i,ts in enumerate(tss):
            display=ts["display"]
            trackset=ts["trackset"]
            events=display.get_events(10)
            for e in events:
                if 'selected' in e:
                    selected_ids=[]
                    for box in e['selected']:
                        selected_ids.append(box['context'])
                    ts["selected_ids"]=selected_ids
                if e['key']=='g':
                    show_gts=not show_gts
                if e['key']=='d':
                    show_det=not show_det
                if e['key']==' ':
                    paused=not paused
                if e['key']=='>':
                    t=trackset.frame_time_after(t)
                if e['key']=='<':
                    t=trackset.frame_time_before(t)
                if e['key']=='.':
                    t+=frame_dt
                if e['key']==',':
                    t-=frame_dt
                if e['key']=='s':
                    show_stats=not show_stats
                if e['key']=='x':
                    show=True
                if e['key']=='h':
                    show_help=not show_help
                if e['key'] is not None and e['key']>='1' and e['key']<='9':
                    index=int(e['key'])-1
                    key = list(debug_overlays_enabled.keys())[index]
                    debug_overlays_enabled[key]=not debug_overlays_enabled[key]
        if paused is False:
            t+=frame_dt
    for ts in tss:
        ts["display"].close()
