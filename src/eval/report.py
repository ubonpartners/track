"""Result tables and rollups (__ovr<group>, _overall, _groupmean,
_arithmean), summary strings, the json + html eval writers.

Moved verbatim from src/track_test.py (repo_cleanup.md stage 4a).
"""
import copy
import datetime
import json
import os
import sys
import numpy as np
import stuff

from src.eval.metrics import fitness_multi_score, fitness_score


def summary_string(r):
    s=f" MOTA:{r['mota']:6.5f}"
    if 'idf1' in r:
        s+=f" IDF1:{r['idf1']:6.5f}"
    if 'fitness' in r:
        s+=f" Fit:{r['fitness']:6.5f}"
    if 'fp_per_frame' in r:
        s+=f" FPpf:{r['fp_per_frame']:5.2f}"
    if 'fp_tracks_frac' in r:
        s+=f" FPTf:{r['fp_tracks_frac']:5.3f}"
    if 'fn_per_obj' in r:
        s+=f" FNPo:{r['fn_per_obj']:5.3f}"
    if 'fp_tracks' in r:
        s+=f" FPTr:{r['fp_tracks']}"
    if 'fp_tracks_honest_v2' in r:
        s+=f" FPh2:{int(r['fp_tracks_honest_v2'])}"
    if 'switch_per_obj' in r:
        s+=f" SWPo:{r['switch_per_obj']:5.3f}"
    if 'frag_per_obj' in r:
        s+=f" FRPo:{r['frag_per_obj']:5.3f}"
    if 'tracked_frames_skipped_frac' in r:
        s+=f" Skip:{r['tracked_frames_skipped_frac']:0.2f}"
    if 'average_detection_roi_area' in r:
        s+=f" dROI:{r['average_detection_roi_area']:0.2f}"
    if 'det_ap_person' in r:
        s+=f" PmAP:{r['det_ap_person']:0.3f}"
    if 'det_ap_face' in r:
        s+=f" FmAP:{r['det_ap_face']:0.3f}"
    if 'det_ap_vehicle' in r:
        s+=f" VmAP:{r['det_ap_vehicle']:0.3f}"
    # vehicle block (unprefixed metrics above are the person pass)
    if 'mota_vehicle' in r:
        s+=f" vMOTA:{r['mota_vehicle']:6.5f}"
    if 'idf1_vehicle' in r:
        s+=f" vIDF1:{r['idf1_vehicle']:6.5f}"
    if 'fitness_vehicle' in r:
        s+=f" vFit:{r['fitness_vehicle']:6.5f}"
    if 'fp_per_frame_vehicle' in r:
        s+=f" vFPpf:{r['fp_per_frame_vehicle']:5.2f}"
    if 'fn_per_obj_vehicle' in r:
        s+=f" vFNPo:{r['fn_per_obj_vehicle']:5.3f}"
    if 'switch_per_obj_vehicle' in r:
        s+=f" vSWPo:{r['switch_per_obj_vehicle']:5.3f}"
    if 'fp_tracks_honest_v2_vehicle' in r:
        s+=f" vFPh2:{int(r['fp_tracks_honest_v2_vehicle'])}"
    if 'fitness_multi' in r:
        s+=f" FITm:{r['fitness_multi']:6.5f}"
    return s


def get_avg_scores(results, test, param, group=None):
    t=0.0
    n=0
    for r in results:
        if r["params"]["test_key"]==test:
            if group is None or ("group" in r and r["group"]==group):
                if param in r["result"]:
                    if isinstance(r["result"][param], int) or isinstance(r["result"][param], float):
                        t+=r["result"][param]
                        n=n+1
    if n>0:
        t=t/n
        return t
    else:
        return 0


def display_results(config, results, columns, sort_key):
    out_sort=[]
    out_txt=[]
    datasets=[result["params"]["ds_key"] for result in results]
    tests=[result["params"]["test_key"] for result in results]
    groups=[result["group"] if "group" in result else None for result in results]
    groups.append(None)
    datasets=list(set(datasets))
    tests=list(set(tests))
    groups=list(set(groups))
    paramset=set([])
    for r in results:
        paramset=paramset.union(set(r["result"].keys()))
    params=list(paramset)

    # Empty-GT exclusion: a clip with zero GT objects scores -inf fitness
    # and poisons every rollup containing it (jaad selected-subjects clips
    # with no pedestrian; meva clips emptied by the duration cap). Such
    # clips carry no signal for any objective — drop them from rollups,
    # loudly. `src.corpus.manifest check` flags them at the data layer too.
    _empty=[r["params"]["ds_key"] for r in results
            if r["result"].get("num_objects", 1) == 0]
    if _empty:
        print(f"EXCLUDING {len(_empty)} zero-GT clips from rollups: "
              f"{sorted(set(_empty))}", flush=True)
        results=[r for r in results
                 if r["result"].get("num_objects", 1) != 0]
        datasets=[d for d in datasets if d not in set(_empty)]
    results2=[]
    if len(datasets)>1:
        for g in groups:
            name="_overall" if g is None else f"__ovr{g}"
            for t in tests:
                filtered=[]
                for r in results:
                    if r["params"]["test_key"]==t:
                        if g is None or ("group" in r and r["group"]==g):
                            filtered.append(r)
                e={"result":{}, "params":{}}
                e["params"]["ds_key"]=name
                e["params"]["test_key"]=t
                er=e["result"]
                # Per-class namespacing (multi_class_and_hints.md §2): the
                # SAME derived-metric recompute runs once per class suffix
                # ("" = person/base, "_vehicle", ...) from that suffix's own
                # summed counts. Suffixes are detected from the idtp_* keys.
                nonsum_bases=["fitness","fp_per_frame","fn_per_obj","switch_per_obj","frag_per_obj"]
                suffixes=[""]+sorted({p[len("idtp"):] for p in params
                                      if p.startswith("idtp") and p!="idtp"})
                nonsum={b+sfx for b in nonsum_bases for sfx in suffixes} | {"fitness_multi"}
                # Per-clip volume cap (search_review.md §1.3): with
                # clip_weight_cap_pctl set, a clip whose GT volume
                # (num_objects, per class suffix) exceeds that percentile of
                # its peers contributes scaled-down COUNTS — so one crowd
                # monster (MOT20-05 = 75.6% of v11's val boxes) cannot own a
                # rollup. Rates/derived metrics recompute from the scaled
                # counts, so everything stays self-consistent.
                cap_pctl=config.get("clip_weight_cap_pctl")
                row_w={}   # id(row) -> {suffix: weight}
                if cap_pctl:
                    for sfx in suffixes:
                        vk="num_objects"+sfx
                        vols=[r["result"][vk] for r in filtered
                              if vk in r.get("result",{}) and r["result"][vk]>0]
                        if len(vols)>=2:
                            cap=float(np.percentile(vols, float(cap_pctl)))
                            for r in filtered:
                                v=r.get("result",{}).get(vk, 0)
                                if v>cap>0:
                                    row_w.setdefault(id(r),{})[sfx]=cap/v
                capped_bases={"num_frames","num_objects","num_false_positives",
                              "num_misses","num_switches","num_unique_objects",
                              "num_fragmentations","num_matches","idtp","idfp",
                              "idfn","fp_tracks","fp_tracks_honest_v2",
                              "fp_h2_nm","fp_h2_inrun","duration","missed",
                              "mostly_tracked","partially_tracked",
                              "mostly_lost","mostly_lost2"}
                def _split_suffix(key):
                    for sfx in sorted(suffixes, key=len, reverse=True):
                        if sfx and key.endswith(sfx):
                            return key[:-len(sfx)], sfx
                    return key, ""
                def _w(r, key):
                    base, sfx=_split_suffix(key)
                    if base not in capped_bases:
                        return 1.0
                    return row_w.get(id(r), {}).get(sfx, 1.0)
                for p in params:
                    if p not in nonsum:
                        vals=[_w(r, p)*r["result"][p] for r in filtered
                              if "result" in r and p in r["result"]]
                        # A key NO row in this group has stays absent — an
                        # empty sum would fabricate zero counts, and the
                        # derived recompute would then report e.g.
                        # mota_vehicle=1.0 for a group with no vehicle GT.
                        if vals:
                            er[p]=sum(vals)
                for sfx in suffixes:
                    if ("idtp"+sfx) not in er:
                        continue
                    weighted_motp_sum=0
                    for r in filtered:
                        rr=r.get("result", {})
                        if ("motp"+sfx) in rr and ("idtp"+sfx) in rr:
                            weighted_motp_sum += rr["motp"+sfx]*rr["idtp"+sfx]*_w(r, "idtp"+sfx)
                    er["idf1"+sfx]= (2 * er["idtp"+sfx]) / (2 * er["idtp"+sfx] + er["idfp"+sfx] + er["idfn"+sfx]+1e-7)
                    er["mota"+sfx]= 1 - (er["num_false_positives"+sfx] + er["num_misses"+sfx] + er["num_switches"+sfx]) / (er["num_objects"+sfx]+1e-7)
                    er["motp"+sfx]=weighted_motp_sum/(er["idtp"+sfx]+1e-7)
                    er["fp_per_frame"+sfx]=er["num_false_positives"+sfx]/(er["num_frames"+sfx]+1e-7) # false positive dets per frame
                    er["fn_per_obj"+sfx]=er["num_misses"+sfx]/(er["num_objects"+sfx]+1e-7) # num false negative dets per real object GT det
                    er["switch_per_obj"+sfx]=er["num_switches"+sfx]/(er["num_unique_objects"+sfx]+1e-7) # num switches per unique object
                    er["frag_per_obj"+sfx]=er["num_fragmentations"+sfx]/(er["num_unique_objects"+sfx]+1e-7)

                stats_to_avg=['mostly_tracked_frac','partially_tracked_frac','mostly_lost2_frac',
                              'missed_frac','fp_tracks_frac', 'time',
                              'tracked_frames','tracked_time','tracked_fps','tracked_frames_skipped_frac',
                              'average_detection_roi_area','det_ap_person', 'det_ap_face', 'det_ap_vehicle']

                for x in stats_to_avg:
                    if x in er:
                        er[x]=er[x]/len(filtered)
                # Suffixed fraction averages divide by the number of rows
                # that HAVE the class — clips without it must not dilute.
                for sfx in suffixes:
                    if sfx=="":
                        continue
                    for b in ['mostly_tracked_frac','partially_tracked_frac','mostly_lost2_frac',
                              'missed_frac','fp_tracks_frac','tracked_frames_skipped_frac']:
                        x=b+sfx
                        if x in er:
                            n=len([r for r in filtered if x in r.get("result",{})])
                            er[x]=er[x]/max(1,n)

                for sfx in suffixes:
                    if ("idtp"+sfx) in er:
                        view={b: er[b+sfx] for b in
                              ["mota","fp_per_frame","fp_tracks","fp_tracks_honest_v2","duration"]
                              if (b+sfx) in er}
                        er["fitness"+sfx]=fitness_score(view)
                fm=fitness_multi_score(er, config.get("fitness_weights"))
                if fm is not None:
                    er["fitness_multi"]=fm

                results2.append(e)
            datasets.append(name)

        # _groupmean (search_review.md §1.1): the BALANCED objective row —
        # weighted mean across the per-group __ovr rollups, so no group can
        # buy influence with GT density (micro-average within a group,
        # macro-average across groups). group_weights: {group: w} in the
        # yaml reweights; weights normalise over the groups PRESENT for
        # each key, so a metric one group lacks (e.g. vehicle keys in a
        # person-only group) averages over the groups that have it.
        real_groups=[g for g in groups if g is not None]
        if real_groups:
            gw_cfg=config.get("group_weights") or {}
            for t in tests:
                grows=[]
                for r2 in results2:
                    ds=r2["params"]["ds_key"]
                    if r2["params"]["test_key"]==t and ds.startswith("__ovr"):
                        grows.append((ds[len("__ovr"):], r2["result"]))
                if not grows:
                    continue
                er={}
                allkeys=set()
                for _, gr in grows:
                    allkeys |= set(gr.keys())
                for k in allkeys:
                    num=0.0; den=0.0
                    for gname, gr in grows:
                        if k in gr and isinstance(gr[k], (int, float)):
                            w=float(gw_cfg.get(gname, 1.0))
                            num+=w*gr[k]; den+=w
                    if den>0:
                        er[k]=num/den
                results2.append({"params":{"ds_key":"_groupmean","test_key":t},
                                 "result":er})
            datasets.append("_groupmean")

        for g in groups:
            if g is None:
                continue
            n=f"__mean({g})"
            for t in tests:
                e={"result":{}, "params":{}}
                e["params"]["ds_key"]=n
                e["params"]["test_key"]=t
                for p in params:
                    e["result"][p]=get_avg_scores(results, t, p, g)
                results2.append(e)
            datasets.append(n)
        for t in tests:
            e={"result":{}, "params":{}}
            e["params"]["ds_key"]="_arithmean"
            e["params"]["test_key"]=t
            for p in params:
                e["result"][p]=get_avg_scores(results, t, p)
            results2.append(e)
        datasets.append("_arithmean")

    datasets.sort()

    if True:
        all_results=results+results2

        column_text=[]
        column_keys=[]
        for c in columns:
            column_text.append(c.split(",")[1])
            column_keys.append(c.split(",")[0])

        result=[]
        for r in all_results:
            rc=copy.deepcopy(r["result"])
            ds=r["params"]["ds_key"]
            ds=ds[:2]+ds[2:].replace("_", "")
            rc["dataset"]=ds
            rc["test"]=r['params']['test_key']
            result.append(rc)

        unique_datasets = list(dict.fromkeys(r["dataset"] for r in result))
        unique_datasets.sort()

        for r in result:
            assert "dataset" in r

        def sort_fn(r):
            return unique_datasets.index(r["dataset"]) *1000 + r[sort_key]

        data_out = stuff.show_data(result, ["dataset","test"]+column_keys,
                        ["dataset","test"]+column_text, sort_fn)
        #result["params"]["ds_key"]
        cur_time=datetime.datetime.now().strftime('%Y%m%d-%H%M')
        if "results_location" in config:
            result_location=config["results_location"]
            stuff.makedir(result_location)
            out_file=result_location+"/results-"+ \
                cur_time+".txt"
            with open(out_file, "w") as f:
                f.write(data_out)
                f.write("\n")

    return results2


def _summary_metric_keys():
    """Float keys exposed in the per-test summary JSON sidecar. Per-class
    namespaced variants (multi_class_and_hints.md §2) ride along for every
    extra class, plus the combined objective."""
    base = [
        "fitness", "mota", "idf1", "fp_tracks",
        "fp_tracks_honest_v2", "fp_h2_nm", "fp_h2_inrun",
        "fp_per_frame", "fn_per_obj", "switch_per_obj", "frag_per_obj",
        "motp", "duration",
        "num_frames", "num_objects", "num_false_positives",
        "num_misses", "num_switches",
    ]
    out = list(base)
    for cls in ("vehicle", "animal"):
        out += [k + "_" + cls for k in base]
    out.append("fitness_multi")
    return out


def _result_subset(result, keys):
    out = {}
    for k in keys:
        if k in result:
            v = result[k]
            if isinstance(v, (np.floating, np.integer)):
                v = v.item()
            out[k] = v
    return out


def _write_eval_summary_json(config, output_results, rollups, elapsed):
    """Sidecar JSON next to the text results report. No-op without
    `results_location` in the config.

    Structure:
        {
          "elapsed_seconds": float,
          "num_clips": int,
          "tests": {
            "<test_key>": {
              "overall":   {fitness, mota, fp_tracks, ...},  # __ovr<group> if 1 group, else _arithmean
              "groups":    {"<group>": {...}, ...},          # one per __ovr<group>; "<group>_mean" for __mean(<group>)
              "arithmean": {...},                             # _arithmean rollup
              "clips":     {"<ds_key>": {...}, ...}           # raw per-clip metrics
            }
          }
        }
    """
    if "results_location" not in config:
        return
    location = config["results_location"]
    stuff.makedir(location)
    keys = _summary_metric_keys()

    tests_by_key = {}
    for entry in rollups:
        test_key = entry["params"]["test_key"]
        ds_key = entry["params"]["ds_key"]
        bucket = tests_by_key.setdefault(test_key, {
            "overall": None, "groups": {}, "arithmean": None, "clips": {},
        })
        metrics = _result_subset(entry["result"], keys)
        if ds_key.startswith("__ovr"):
            group = ds_key[len("__ovr"):]
            bucket["groups"][group] = metrics
        elif ds_key.startswith("__mean("):
            group = ds_key[len("__mean("):-1]
            bucket["groups"].setdefault(group, {})
            bucket["groups"][group + "_mean"] = metrics
        elif ds_key == "_arithmean":
            bucket["arithmean"] = metrics
        elif ds_key == "_groupmean":
            bucket["groupmean"] = metrics

    for entry in output_results:
        test_key = entry["params"]["test_key"]
        ds_key = entry["params"]["ds_key"]
        bucket = tests_by_key.setdefault(test_key, {
            "overall": None, "groups": {}, "arithmean": None, "clips": {},
        })
        bucket["clips"][ds_key] = _result_subset(entry["result"], keys)

    # "overall" is the single-group __ovr rollup when there is exactly one
    # group, else the arithmetic mean across all clips.
    for bucket in tests_by_key.values():
        groups = bucket["groups"]
        non_mean_groups = [k for k in groups if not k.endswith("_mean")]
        if len(non_mean_groups) == 1:
            bucket["overall"] = groups[non_mean_groups[0]]
        else:
            bucket["overall"] = bucket["arithmean"]

    summary = {
        "elapsed_seconds": elapsed,
        "num_clips": len({e["params"]["ds_key"] for e in output_results}),
        "tests": tests_by_key,
    }

    cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    out_path = os.path.join(location, f"results-{cur_time}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    try:
        _write_eval_summary_html(
            os.path.join(location, f"results-{cur_time}.html"), summary)
    except Exception:
        import traceback
        sys.stderr.write("eval html report failed: "
                         + traceback.format_exc() + "\n")


def _write_eval_summary_html(path, summary):
    """Self-contained sortable eval report (search_review.md §4.3):
    per-test rollup cards + a click-to-sort per-clip table, worst clips
    surfaced. Inline data, vanilla JS, no server, negatives visible."""
    payload = json.dumps(summary)
    html = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>eval results</title><style>
body{font-family:system-ui,sans-serif;margin:20px;background:#111;color:#ddd}
h1{font-size:18px} h2{font-size:14px;color:#aaa;margin-top:22px}
table{border-collapse:collapse;font-size:12px;margin-top:6px}
td,th{padding:2px 9px;border-bottom:1px solid #2a2a2a;text-align:right;white-space:nowrap}
th{color:#8ab;cursor:pointer;user-select:none} td:first-child,th:first-child{text-align:left}
.neg{color:#ff6b6b} .rollup td{color:#ffd43b}
</style></head><body><h1>eval results</h1><div id="root"></div>
<script>
const D = __PAYLOAD__;
const COLS = ["fitness","fitness_multi","mota","idf1","mota_vehicle","idf1_vehicle",
              "fp_per_frame","fp_tracks_honest_v2","num_objects","duration"];
function fmt(v){ if(v===undefined||v===null) return "";
  if(typeof v!=="number") return String(v);
  const s = Math.abs(v)>=1000 ? v.toFixed(0) : v.toFixed(3);
  return v<0 ? '<span class="neg">'+s+"</span>" : s; }
function table(rows, id){
  let sortk=COLS[0], asc=false;
  const el=document.createElement("table"); el.id=id;
  function render(){
    const sorted=[...rows].sort((a,b)=>{
      const x=a[1][sortk], y=b[1][sortk];
      return ((x===undefined)-(y===undefined)) || (asc?x-y:y-x) || 0; });
    el.innerHTML="<tr><th>clip</th>"+COLS.map(c=>
      '<th data-k="'+c+'">'+c+(c===sortk?(asc?" \u25b2":" \u25bc"):"")+"</th>").join("")+"</tr>"+
      sorted.map(([k,r])=>"<tr"+(k.startsWith("_")?' class="rollup"':"")+"><td>"+k+"</td>"+
        COLS.map(c=>"<td>"+fmt(r[c])+"</td>").join("")+"</tr>").join("");
    el.querySelectorAll("th[data-k]").forEach(th=>th.onclick=()=>{
      if(sortk===th.dataset.k) asc=!asc; else {sortk=th.dataset.k; asc=false;}
      render(); });
  }
  render(); return el;
}
const root=document.getElementById("root");
for(const [tk, t] of Object.entries(D.tests||{})){
  const h=document.createElement("h2"); h.textContent="test: "+tk; root.appendChild(h);
  const rows=[];
  if(t.overall) rows.push(["_overall", t.overall]);
  if(t.groupmean) rows.push(["_groupmean", t.groupmean]);
  for(const [g,m] of Object.entries(t.groups||{})) rows.push(["_"+g, m]);
  for(const [c,m] of Object.entries(t.clips||{})) rows.push([c, m]);
  root.appendChild(table(rows, "t_"+tk));
}
</script></body></html>"""
    with open(path, "w") as f:
        f.write(html.replace("__PAYLOAD__", payload))
