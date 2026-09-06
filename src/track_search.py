import copy
import datetime
import json
import os

import stuff

import src.paths as paths
import src.eval.report as eval_report
import src.eval.runner as eval_runner


def _is_int_param_value(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _normalise_param_value(value, is_int):
    numeric_value = float(value)
    if is_int:
        return int(round(numeric_value))
    return round(numeric_value, 3)


def _normalise_param_vec(param_vec, param_is_int):
    return [
        _normalise_param_value(value, is_int)
        for value, is_int in zip(param_vec, param_is_int)
    ]


def _is_variant_key(key):
    """True for keys addressing a variant on ANY config axis — `(hint:x)`
    (per-stream profile, section- or key-level) or `(class:x)` (per-class
    key-level, resolved inside utrack per tracked class; the two COMPOSE:
    a (class:) key inside a (hint:) block is the hint x class cell).
    Variant paths are CREATE-ON-WRITE: the C side deep-merges variants over
    their base, so the key legitimately does not exist until a search
    writes it (multi_class_and_hints.md §15.3)."""
    import re
    return re.search(r"\(\w+:", key) is not None


def _strip_variants(key):
    """The base path of a variant key: every `(...)` suffix removed from
    every segment — `utrack(hint:bodycam).kf_weight` → `utrack.kf_weight`.
    Initial values seed from here, so a split starts at the shared optimum."""
    import re
    return re.sub(r"\([^)]*\)", "", key)


def _find_by_bare_name(config_dict, key):
    """Match-anywhere lookup for a bare (dot-free) param name — the
    original behaviour, kept for every existing search yaml. Returns the
    single (parent, key) match; asserts on missing or ambiguous."""
    matches = []

    def walk(node, path):
        if isinstance(node, dict):
            for child_key, child_value in node.items():
                child_path = path + [child_key]
                if child_key == key:
                    matches.append((node, child_key, child_path))
                walk(child_value, child_path)

    walk(config_dict, [])
    if len(matches) == 0:
        raise AssertionError(f"Can't find param {key} in config")
    if len(matches) > 1:
        paths = ["/".join(path) for _, _, path in matches]
        raise AssertionError(f"Param {key} is ambiguous in config: {paths}")
    return matches[0][0], matches[0][1]


def _set_nested_param(config_dict, key, value):
    """Set one search parameter in the loaded tracker-config tree.

    Two addressing modes (multi_class_and_hints.md §15.2):
      - bare name ("kf_weight"): match anywhere, exactly one occurrence —
        unchanged legacy behaviour, still asserts on ambiguity;
      - dotted path ("motiontrack.mad_delta", "roi_scan.min_age_lo",
        "utrack(hint:bodycam).kf_weight"): explicit segments, so keys that
        are ambiguous as bare names (alpha, max_width, ...) are reachable.

    A missing PLAIN path asserts (typo protection). A missing VARIANT path
    is created (§15.3): the `(hint:x)` block is a legitimate new override.
    """
    if "." not in key:
        if _is_variant_key(key):
            # Flat-key variant ("kf_weight(hint:bodycam)"): create-on-write
            # BESIDE ITS BASE KEY — the C side resolves `key(hint:x)`
            # against siblings, so a variant of a section-scoped key must
            # live in that section (utrack: {kf_weight(hint:bodycam): x}),
            # not at top level. Base missing entirely -> top level (a
            # genuinely new flat key).
            base = _strip_variants(key)
            try:
                parent, _ = _find_by_bare_name(config_dict, base)
            except AssertionError:
                parent = config_dict
            parent[key] = value
            return
        parent, child_key = _find_by_bare_name(config_dict, key)
        parent[child_key] = value
        return
    segments = key.split(".")
    variant = _is_variant_key(key)
    node = config_dict
    for seg in segments[:-1]:
        if not isinstance(node.get(seg), dict):
            if seg in node and node[seg] is not None:
                raise AssertionError(
                    f"Param {key}: segment {seg} is not a section in config")
            if not variant:
                raise AssertionError(f"Can't find param {key} in config")
            node[seg] = {}
        node = node[seg]
    leaf = segments[-1]
    if leaf not in node and not variant:
        raise AssertionError(f"Can't find param {key} in config")
    node[leaf] = value


def _get_nested_param(config_dict, key):
    """Read a dotted-path param; (found, value)."""
    node = config_dict
    segments = key.split(".")
    for seg in segments[:-1]:
        if not isinstance(node, dict) or seg not in node:
            return False, None
        node = node[seg]
    if isinstance(node, dict) and segments[-1] in node:
        return True, node[segments[-1]]
    return False, None


def _update_initial_parameters(param_names, param_initial, source_dict, logfile, source_label):
    # Bare names: original match-anywhere walk. Dotted paths: explicit
    # lookup; a variant path absent from the config seeds from its BASE
    # path (multi_class_and_hints.md §15.3 — a split starts at the shared
    # optimum, so iteration 0 is behaviour-identical to the unsplit run).
    def note(key, value, how=""):
        if logfile:
            search_log(
                logfile,
                f"Setting parameter {key} initial value to {value} from {source_label}{how}",
            )

    bare = {n for n in param_names if "." not in n}

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in bare:
                    idx = param_names.index(key)
                    if param_initial[idx] is None:
                        param_initial[idx] = value
                        note(key, value)
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(source_dict)

    for idx, name in enumerate(param_names):
        if param_initial[idx] is not None:
            continue
        if "." not in name:
            if not _is_variant_key(name):
                continue        # plain bare name: the walk above was its one chance
            # Flat-key variant: seed from the base key WHEREVER it lives —
            # the same match-anywhere walk used to set it (a bare
            # vbox_expand sits under utrack:, not at top level).
            base = _strip_variants(name)
            try:
                parent, child = _find_by_bare_name(source_dict, base)
                found, value = True, parent[child]
            except AssertionError:
                found, value = False, None
            if found:
                note(name, value, " (base value; variant not yet in config)")
                param_initial[idx] = value
            continue
        found, value = _get_nested_param(source_dict, name)
        if not found and _is_variant_key(name):
            found, value = _get_nested_param(source_dict, _strip_variants(name))
            if found:
                note(name, value, " (base value; variant not yet in config)")
        elif found:
            note(name, value)
        if found:
            param_initial[idx] = value


def _expand_split_hints(search_params, logfile=None):
    """Directed split sugar — nothing splits by itself:

    `split_hints: [bodycam]`   -> plus one `(hint:x)` variant per hint.
       The suffix lands on the param's SECTION (first path segment) —
       hint variants are section-level blocks; a flat key takes it directly.
    `split_classes: [vehicle]` -> plus one `(class:x)` variant per class.
       The suffix lands on the KEY (last segment) — class variants are
       key-level, resolved per tracked class inside utrack.

    Each variant is an independently-steppable dimension seeded from its
    base value. NO automatic hint x class matrix: write the composed key
    (`utrack(hint:bodycam).kf_weight(class:vehicle)`) explicitly when a
    cell earns it (multi_class_and_hints.md §15.3/§15.4)."""
    out = {}
    for name, spec in search_params.items():
        spec = dict(spec or {})
        hints = spec.pop("split_hints", None)
        classes = spec.pop("split_classes", None)
        out[name] = spec
        for h in hints or []:
            if "." in name:
                first, rest = name.split(".", 1)
                vname = f"{first}(hint:{h}).{rest}"
            else:
                vname = f"{name}(hint:{h})"
            out[vname] = dict(spec)
            if logfile:
                search_log(logfile, f"split_hints: {name} -> {vname}")
        for c in classes or []:
            vname = f"{name}(class:{c})"
            out[vname] = dict(spec)
            if logfile:
                search_log(logfile, f"split_classes: {name} -> {vname}")
    return out


def _check_protect(config, results, test_key=None, logfile=None):
    """The §7 regression guard: `protect: [{group, param, floor}]` rejects
    any candidate whose `__ovr<group>` rollup drops below the floor —
    'don't break CCTV' as a hard constraint, not a weight-tuning hope.
    Returns None when every rule passes, else the violated rule."""
    rules = config.get("protect") or []
    if test_key is None:
        test_key = config["result_test_opt_key"]
    for rule in rules:
        group, param, floor = rule["group"], rule["param"], float(rule["floor"])
        row = None
        for r in results:
            if (r["params"]["test_key"] == test_key
                    and r["params"]["ds_key"] == f"__ovr{group}"):
                row = r["result"]
        assert row is not None, (
            f"protect: no __ovr{group} rollup in results — is any dataset "
            f"tagged group: {group}?")
        assert param in row, f"protect: {param} missing from __ovr{group}"
        if row[param] < floor:
            if logfile:
                search_log(logfile,
                           f"REJECT by protect: {group}.{param} "
                           f"{row[param]:0.4f} < floor {floor:0.4f}")
            return rule
    return None


def _round_numeric_metrics(metrics_dict, decimals=3):
    for key, value in list(metrics_dict.items()):
        if isinstance(value, (int, float)):
            metrics_dict[key] = round(value, decimals)


def search_log(logfile, x):
    logfile.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": ")
    logfile.write(x + "\n")
    logfile.flush()


def _candidate_key(split, param_vec):
    return (split or "train", tuple(param_vec))


def search_test_batch(
    config,
    params,
    cand_vecs,
    param_is_int,
    param_min,
    param_max,
    all_results,
    split="train",
    logfile=None,
    desc="search batch",
    journal_fn=None,
):
    """Evaluate N candidate vectors in ONE track_test pass (search_review.md
    §2.1): the up/down probes (or any batch) share one work-queue run, so the
    pool never idles between candidates and engines load once per batch.

    Per candidate: range check (reject −10000), memoisation by (split, vec) —
    validation runs are cached too, so an unchanged vec_best never re-runs
    the val set — a fresh eval otherwise. The base tracker config is loaded
    once and DEEP-COPIED per candidate, so nothing (§15 create-on-write
    variant blocks included) leaks between candidates or batches.

    Returns [(score, full_result_or_None, groups_or_None)] aligned with
    cand_vecs. groups = {group: opt_param value} from the __ovr rollups.
    """
    result_test_opt_key = config["result_test_opt_key"]
    result_dataset_opt_key = config["result_dataset_opt_key"]
    result_dataset_opt_param = config["result_dataset_opt_param"]

    out = [None] * len(cand_vecs)
    to_eval = []          # index -> fresh evaluation needed
    key_of = {}
    for i in range(len(cand_vecs)):
        vec = _normalise_param_vec(cand_vecs[i], param_is_int)
        cand_vecs[i] = vec
        if any(v < mn or v > mx for v, mn, mx in zip(vec, param_min, param_max)):
            # out-of-range: cached like every other reject so boundary
            # bounces never re-run anything
            all_results[_candidate_key(split, vec)] = {"score": -10000}
            out[i] = (-10000, None, None)
            continue
        ck = _candidate_key(split, vec)
        key_of[i] = ck
        if ck in all_results:
            c = all_results[ck]
            out[i] = (c["score"], c.get("full_result"), c.get("groups"))
            continue
        if any(key_of.get(j) == ck for j in to_eval):
            continue      # duplicate inside the batch: fill from cache after
        to_eval.append(i)

    if to_eval:
        base = config["tests"][result_test_opt_key]
        if isinstance(base.get("config"), str):
            base["config"] = stuff.load_dictionary(base["config"])
        run_cfg = {k: v for k, v in config.items() if k != "tests"}
        run_cfg["tests"] = {}
        tk_of = {}
        for i in to_eval:
            tk = f"cand{i:02d}"
            tk_of[i] = tk
            t = copy.deepcopy(base)
            for pi, pname in enumerate(params):
                _set_nested_param(t["config"], pname, cand_vecs[i][pi])
            run_cfg["tests"][tk] = t
        results = eval_runner.track_test(run_cfg, split=split, desc=desc)

        for i in to_eval:
            tk = tk_of[i]
            score = None
            full_result = None
            groups = {}
            for r in results:
                if r["params"]["test_key"] != tk:
                    continue
                ds = r["params"]["ds_key"]
                if ds == result_dataset_opt_key:
                    score = r["result"].get(result_dataset_opt_param)
                    full_result = r["result"]
                if ds.startswith("__ovr"):
                    g = ds[len("__ovr"):]
                    if result_dataset_opt_param in r["result"]:
                        groups[g] = r["result"][result_dataset_opt_param]
            if full_result is None:
                raise RuntimeError(
                    f"No result for candidate {tk} on dataset {result_dataset_opt_key}")
            # Regression guard (§7): reject + cache, same path as
            # out-of-range params.
            if _check_protect(config, results, test_key=tk, logfile=logfile) is not None:
                all_results[_candidate_key(split, cand_vecs[i])] = {
                    "score": -10000, "groups": groups}
                out[i] = (-10000, None, groups)
                continue
            _round_numeric_metrics(full_result)
            entry = {"score": score, "groups": groups, "full_result": full_result}
            all_results[_candidate_key(split, cand_vecs[i])] = entry
            out[i] = (score, full_result, groups)
            if journal_fn:
                stats = {k: full_result[k] for k in
                         ("fitness", "mota", "idf1", "fitness_vehicle",
                          "mota_vehicle", "idf1_vehicle", "fitness_multi")
                         if k in full_result}
                journal_fn(split, cand_vecs[i], score, groups, stats=stats)

    # fill any batch-internal duplicates / late cache hits
    for i in range(len(cand_vecs)):
        if out[i] is None:
            c = all_results[_candidate_key(split, cand_vecs[i])]
            out[i] = (c["score"], c.get("full_result"), c.get("groups"))
    return out


# THE objective config. There is exactly one, and search and eval BOTH read
# it, so they cannot describe different datasets. Do not add a second file and
# do not copy this one to change a field -- every knob an eval run needs is a
# CLI override. (ledger 2026-07-24 One objective config)


def eval_track(yaml_file=None, split=None, convention_permissive=None,
               results_location=None, tracker_config=None):
    """Single-pass parallel evaluation via the existing multi-process
    work queue. The yaml mirrors the search yaml minus search_params:

      tests:          one or more {test_key: {config: ..., min_interval: ...}}
      datasets:       {clip_name: {path: ..., split: ...}}   (split optional)
      num_workers:    int or "auto" (default: "auto" — 4 workers when ≤1 GPU
                      is visible, 2 × N workers when N > 1 GPUs are visible)
      columns:        list of "key,header,fmt" strings
      sort_key:       column to sort the report by (e.g. fitness or mota)
      results_location: optional dir for the persisted .txt + .json
                        reports. JSON sidecar shape:
                        `tests[<key>]` → {overall, groups, arithmean, clips}.
      include_families: optional list (or comma-string) — restrict the
                        run to datasets whose `family` is in the list.

    Properties:
      - shares loaded detector engines across the work queue
        (no per-clip process spinup);
      - evaluates multiple tracker variants in a single run (cartesian
        product of tests × datasets);
      - surfaces dead workers fast via mp_workqueue's liveness check
        instead of hanging on result_queue.get(timeout=300).

    Returns the aggregated results list (_overall + per-clip).
    """
    canonical = paths.search_yaml()
    if yaml_file is None:
        yaml_file = canonical
    elif os.path.realpath(yaml_file) != os.path.realpath(canonical):
        # Not fatal -- one-off probes are legitimate -- but it must be
        # impossible to do accidentally and then quote the number as if it
        # were the objective.
        print("!" * 100)
        print("!! NOT THE OBJECTIVE CONFIG. You passed:")
        print(f"!!     {yaml_file}")
        print(f"!! The objective is {canonical}")
        print("!! Numbers from this run are NOT comparable to search scores and must not be")
        print("!! quoted as an A/B result. Run `track.py --eval` with no path to use the objective.")
        print("!" * 100)
    config = stuff.load_dictionary(yaml_file)
    # a SEARCH yaml is a valid eval yaml once search_params is dropped —
    # its tests.search_config block carries the tracker config + eval
    # parameters. This gives a one-off eval of the exact search substrate
    # (ledger 2026-07-24 One objective config): src.cli eval <search yaml> --split ...
    config.pop("search_params", None)
    if "num_workers" not in config:
        config["num_workers"] = "auto"
    if "sort_key" not in config:
        config["sort_key"] = "fitness"
    if "columns" not in config:
        config["columns"] = [
            "num_frames,FR,{:5.0f}",
            "fp_tracks,FPTr,{:5.0f}",
            "fp_per_frame,FPpf,{:5.2f}",
            "mota,MOTA,{:6.3f}",
            "idf1,IDF1,{:6.3f}",
            "fitness,FIT,{:6.3f}",
        ]
    # Output dir is a CLI override, never a file edit: needing to change a field
    # in the yaml is what drove people to copy it in the first place.
    if results_location is not None:
        config["results_location"] = results_location
    # Tracker A/Bs point every test at a variant tracker yaml. Doing that by
    # copying the objective config is what produced a second, divergent
    # "canonical" eval; this keeps ONE objective config and varies only the
    # thing under test.
    if tracker_config is not None:
        for t in (config.get("tests") or {}).values():
            t["config"] = tracker_config
        print(f"tracker config overridden -> {tracker_config}")
    if convention_permissive is not None:
        for t in (config.get("tests") or {}).values():
            t["convention_permissive"] = convention_permissive
    if split in ("both", ""):
        split = None
    return eval_runner.track_test(config, split=split,
                                  desc=f"eval {yaml_file}"
                                       f" split={split or 'both'}")


def _journal_append(path, entry):
    entry = dict(entry)
    entry["ts"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _journal_load(path, param_names):
    """Preload a previous run's journal into the (split, vec) cache —
    the crash-resume path (search_review.md §2.5). Entries whose param-name
    set differs from the current search are skipped (a changed search space
    invalidates old scores)."""
    cache = {}
    entries = []
    if not path or not os.path.isfile(path):
        return cache, entries
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            vec_map = e.get("vec") or {}
            if set(vec_map.keys()) != set(param_names):
                continue
            vec = tuple(vec_map[n] for n in param_names)
            cache[(e.get("split") or "train", vec)] = {
                "score": e.get("score"), "groups": e.get("groups")}
            entries.append(e)
    return cache, entries


def _write_search_html(path, meta, entries):
    """Self-contained live search report (search_review.md §4.3): train
    score trace, validate markers, per-group traces at validates, and the
    best vector — inline data, vanilla JS, no server, regenerated at every
    validate so it is live during a run."""
    payload = json.dumps({"meta": meta, "entries": entries})
    html = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>track search</title><style>
body{font-family:system-ui,sans-serif;margin:20px;background:#111;color:#ddd}
h1{font-size:18px} h2{font-size:14px;color:#aaa;margin-top:24px}
canvas{background:#181818;border:1px solid #333;border-radius:6px}
table{border-collapse:collapse;font-size:12px;margin-top:8px}
td,th{padding:2px 10px;border-bottom:1px solid #2a2a2a;text-align:right}
th{color:#888} td:first-child,th:first-child{text-align:left}
.legend span{display:inline-block;margin-right:14px;font-size:12px}
</style></head><body>
<h1>track search <span id="title"></span></h1>
<h2>objective (train evals ordered; &#9650; = validate)</h2>
<canvas id="score" width="1100" height="300"></canvas>
<h2>per-group objective at validates</h2>
<canvas id="groups" width="1100" height="300"></canvas>
<div class="legend" id="glegend"></div>
<h2>per-class (train evals): person vs vehicle</h2>
<canvas id="classes" width="1100" height="300"></canvas>
<div class="legend" id="clegend"></div>
<h2>best vector</h2><table id="vec"></table>
<script>
const D = __PAYLOAD__;
document.getElementById("title").textContent =
  D.meta.name + " — " + D.entries.length + " journal rows";
const evals = D.entries.filter(e => (e.kind||"eval") === "eval" && e.split !== "val");
const vals  = D.entries.filter(e => e.split === "val");
function drawSeries(cv, series, markers) {
  const c = cv.getContext("2d");
  const all = series.flatMap(s => s.pts.map(p => p[1])).filter(v => v > -100);
  if (!all.length) return;
  const lo = Math.min(...all), hi = Math.max(...all), pad = (hi-lo || 1) * 0.08;
  const X = n => 40 + (cv.width-60) * n, Y = v => cv.height-24 - (cv.height-48)*(v-lo+pad)/(hi-lo+2*pad);
  c.clearRect(0,0,cv.width,cv.height);
  c.fillStyle="#666"; c.font="11px sans-serif";
  c.fillText(hi.toFixed(3), 2, Y(hi)+4); c.fillText(lo.toFixed(3), 2, Y(lo)+4);
  series.forEach(s => {
    c.strokeStyle = s.color; c.beginPath();
    s.pts.forEach((p,i) => { const x=X(p[0]), y=Y(Math.max(p[1],lo)); i?c.lineTo(x,y):c.moveTo(x,y); });
    c.stroke();
  });
  (markers||[]).forEach(m => {
    c.fillStyle = "#ffcc00";
    c.beginPath(); const x=X(m[0]), y=Y(Math.max(m[1],lo));
    c.moveTo(x,y-5); c.lineTo(x-4,y+3); c.lineTo(x+4,y+3); c.fill();
  });
}
const n = Math.max(evals.length-1, 1);
drawSeries(document.getElementById("score"),
  [{color:"#4da6ff", pts: evals.map((e,i)=>[i/n, e.score])}],
  vals.map(v => { const i = evals.findIndex(e => e.iter >= (v.iter||0));
                  return [(i<0?evals.length-1:i)/n, v.score]; }));
const gnames = [...new Set(vals.flatMap(v => Object.keys(v.groups||{})))].sort();
const colors = ["#4da6ff","#ff6b6b","#51cf66","#ffd43b","#c084fc","#ff922b","#38d9d9","#e599f7"];
const m = Math.max(vals.length-1, 1);
drawSeries(document.getElementById("groups"),
  gnames.map((g,gi) => ({color: colors[gi%colors.length],
    pts: vals.map((v,i)=>[i/m, (v.groups||{})[g]]).filter(p=>p[1]!==undefined)})));
const CSER = [["fitness","#4da6ff"],["fitness_vehicle","#ff6b6b"],
              ["mota","#51cf66"],["mota_vehicle","#ffd43b"]];
drawSeries(document.getElementById("classes"),
  CSER.map(([k,color]) => ({color,
    pts: evals.map((e,i)=>[i/n, (e.stats||{})[k]]).filter(p=>p[1]!==undefined)})));
document.getElementById("clegend").innerHTML =
  CSER.map(([k,c])=>'<span style="color:'+c+'">&#9632; '+k+"</span>").join("");
document.getElementById("glegend").innerHTML =
  gnames.map((g,gi)=>'<span style="color:'+colors[gi%colors.length]+'">&#9632; '+g+"</span>").join("");
const best = D.meta.best_vec || {};
document.getElementById("vec").innerHTML =
  "<tr><th>param</th><th>value</th></tr>" +
  Object.keys(best).map(k=>"<tr><td>"+k+"</td><td>"+best[k]+"</td></tr>").join("");
</script></body></html>"""
    html = html.replace("__PAYLOAD__", payload)
    with open(path, "w") as f:
        f.write(html)


def _registry_check(config, logfile):
    """Corpus-registry consultation (data_tiers spec section 4): every
    dataset row's corpus must approve tuning use. WARNS rather than
    fails while the `tune_tracker` vocabulary decision is pending —
    derived-GT corpora (meva/otw/bwc/movies) are currently registered
    train_detector-only yet sit in the search sets."""
    from src.corpus.manifest import load_capabilities
    unapproved = {}
    for name, row in (config.get("datasets") or {}).items():
        path = row.get("path", "")
        parts = path.split("/")
        corpus = parts[3] if len(parts) > 3 else None
        caps = load_capabilities(corpus) if corpus else None
        if caps is None:
            unapproved.setdefault("UNREGISTERED:" + str(corpus), []).append(name)
        elif not any(u in caps.get("approved_uses", [])
                     for u in ("screen", "val", "frozen_test", "tune_tracker")):
            unapproved.setdefault(corpus, []).append(name)
    for corpus, names in sorted(unapproved.items()):
        search_log(logfile,
                   f"REGISTRY WARNING: corpus '{corpus}' is not approved "
                   f"for eval/tuning use ({len(names)} clips, e.g. "
                   f"{names[0]}) — see capabilities.approved_uses")


def search_track(yaml_file):
    config = stuff.load_dictionary(yaml_file)
    result_log_file = config["result_log_file_path"]
    stuff.makedir(result_log_file)
    cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    logfile = open(result_log_file + "/search_log_" + cur_time + ".txt", "w")
    _registry_check(config, logfile)
    journal_path = result_log_file + f"/search_journal_{cur_time}.jsonl"
    html_path = result_log_file + f"/search_report_{cur_time}.html"
    param_names = []
    param_initial = []
    param_step = []
    param_min = []
    param_max = []
    param_is_int = []

    train_split = "train"
    if "do_train_split" in config:
        if config["do_train_split"] is False:
            train_split = None
    search_log(logfile, f"Setting train split to: {train_split}")

    step_multiplier = 4
    final_multiplier = 0.5
    if "initial_mult" in config:
        step_multiplier = config["initial_mult"]
    if "final_mult" in config:
        final_multiplier = config["final_mult"]
    search_log(logfile, f"Starting step multiplier set to {step_multiplier}")

    config["search_params"] = _expand_split_hints(config["search_params"], logfile=logfile)

    for p in config["search_params"]:
        param_names.append(p)
        param_initial.append(None)
        param_is_int.append(False)

    test_dict = stuff.load_dictionary(config["tests"]["search_config"]["config"])
    _update_initial_parameters(param_names, param_initial, test_dict, logfile, "base config")

    for i, p in enumerate(config["search_params"]):
        if param_initial[i] is not None:
            param_is_int[i] = _is_int_param_value(param_initial[i])
        if "initial" in config["search_params"][p]:
            raw_initial = config["search_params"][p]["initial"]
            param_initial[i] = _normalise_param_value(raw_initial, param_is_int[i])
            search_log(logfile, f"Setting parameter {p} initial value to {param_initial[i]} from search config")
        assert param_initial[i] is not None, f"Parameter {p} missing initial value"
        step = float(config["search_params"][p]["step"])
        # _normalise_param_value rounds floats to 3 dp — a finer step would
        # silently collapse (search_review.md §3).
        assert param_is_int[i] or step >= 0.001, \
            f"{p}: step {step} below the 0.001 float resolution"
        param_step.append(step)
        param_min.append(float(config["search_params"][p]["min"]))
        param_max.append(float(config["search_params"][p]["max"]))

    search_log(logfile, "Search params:" + str(param_names))

    for i, v in enumerate(param_initial):
        p = param_names[i]
        assert v >= param_min[i], f"{p} : Initial value {v} is less than min {param_min[i]}"
        assert v <= param_max[i], f"{p} : Initial value {v} is more than max {param_max[i]}"

    # (split, vec) memoisation, optionally preloaded from a previous run's
    # journal (config resume_from: <path to search_journal_*.jsonl>).
    results, journal_entries = _journal_load(config.get("resume_from"), param_names)
    if results:
        search_log(logfile, f"Resumed {len(results)} cached evals from {config['resume_from']}")

    iter_box = {"iter": 0}

    def journal_fn(split, vec, score, groups, kind="eval", stats=None):
        e = {"kind": kind, "iter": iter_box["iter"], "split": split or "train",
             "score": score, "groups": groups, "stats": stats or {},
             "vec": {n: v for n, v in zip(param_names, vec)}}
        journal_entries.append(e)
        _journal_append(journal_path, e)

    def write_html(best_vec):
        meta = {"name": os.path.basename(yaml_file),
                "best_vec": {n: v for n, v in zip(param_names, best_vec)}}
        try:
            _write_search_html(html_path, meta, journal_entries)
        except Exception as ex:      # the report must never kill a search
            search_log(logfile, f"html report failed: {ex}")

    def batch(vecs, split, desc):
        return search_test_batch(
            config, param_names, vecs, param_is_int, param_min, param_max,
            results, split=split, logfile=logfile, desc=desc,
            journal_fn=journal_fn)

    param_initial = _normalise_param_vec(param_initial, param_is_int)
    score_best, best_full_result, _g = batch(
        [param_initial], train_split, "search test: initial")[0]
    # best_full_result may be None on a journal-resumed run (the cache
    # carries score+groups, not the full row) — every consumer guards.
    vec_best = copy.copy(param_initial)

    iter_count = 0
    param_index = 0
    last_improvement_iter = 0
    improvements_since_validate = 0
    last_validate_iter = 0
    successive_improvements = 0
    best_val = {"score": None, "vec": None}
    last_val_groups = None
    search_log(logfile, f"Iter {iter_count:04d} initial score {score_best:0.4f}  [{eval_report.summary_string(best_full_result) if best_full_result else '(memoised)'} ]")
    search_log(logfile, f"  initial vector: {dict(zip(param_names, vec_best))}")

    total_improvement = [0.0] * len(param_names)

    def finish():
        search_log(logfile, "All done!")
        search_log(logfile, f"best by train: score {score_best:0.4f}  vector: {dict(zip(param_names, vec_best))}")
        if best_val["score"] is not None:
            search_log(logfile, f"best by val:   score {best_val['score']:0.4f}  vector: {dict(zip(param_names, best_val['vec']))}")
        write_html(vec_best)
        search_log(logfile, f"journal: {journal_path}")
        search_log(logfile, f"report:  {html_path}")
        logfile.close()

    while True:
        index = param_index % len(param_names)
        iter_box["iter"] = iter_count

        do_val = improvements_since_validate > 0 and iter_count >= last_validate_iter + 4
        if train_split is not None:
            if do_val or iter_count == 0:
                validate_score, full_result_val, val_groups = batch(
                    [vec_best], "val",
                    f"search test it:{iter_count} validate")[0]
                journal_fn("val", vec_best, validate_score, val_groups, kind="validate",
                           stats={k: full_result_val[k] for k in
                                  ("fitness", "mota", "idf1", "fitness_vehicle",
                                   "mota_vehicle", "idf1_vehicle", "fitness_multi")
                                  if full_result_val and k in full_result_val})
                search_log(logfile, "======================================================")
                vs = eval_report.summary_string(full_result_val) if full_result_val else "(memoised)"
                search_log(logfile, f"Iter {iter_count:04d}  **VALIDATE** score {validate_score:0.4f}  [{vs} ]")
                search_log(logfile, f"  vector: {dict(zip(param_names, vec_best))}")
                # Per-group deltas vs the previous validate — the "bwc got
                # better, cctv flat" line (search_review.md §4.2).
                if val_groups:
                    if last_val_groups:
                        deltas = ", ".join(
                            f"{g} {val_groups[g] - last_val_groups.get(g, 0):+0.4f}"
                            for g in sorted(val_groups))
                        search_log(logfile, f"  group deltas: {deltas}")
                    levels = ", ".join(f"{g} {val_groups[g]:0.4f}" for g in sorted(val_groups))
                    search_log(logfile, f"  group levels: {levels}")
                    last_val_groups = val_groups
                if best_val["score"] is None or validate_score > best_val["score"]:
                    best_val["score"] = validate_score
                    best_val["vec"] = copy.copy(vec_best)
                total = sum(total_improvement)
                if total > 0:
                    search_log(logfile, "  cumulative improvement by param:")
                    for i in range(len(param_names)):
                        if total_improvement[i] != 0:
                            search_log(logfile,
                                f"    {param_names[i]:25s} {total_improvement[i]:+8.5f} "
                                f"({100 * total_improvement[i] / total:4.1f}%)  value {vec_best[i]}")
                search_log(logfile, "======================================================")
                improvements_since_validate = 0
                last_validate_iter = iter_count
                write_html(vec_best)

        vec_up = copy.copy(vec_best)
        vec_down = copy.copy(vec_best)
        vec_up[index] += step_multiplier * param_step[index]
        vec_down[index] -= step_multiplier * param_step[index]
        # ONE eval pass for both probes (search_review.md §2.1) — cached /
        # out-of-range candidates cost nothing, fresh ones share the pool.
        probes = batch([vec_up, vec_down], train_split,
                       f"search test it:{iter_count} probe {param_names[index]}")
        (score_up, full_result_up, _gu) = probes[0]
        (score_down, full_result_down, _gd) = probes[1]
        vec_up, vec_down = _normalise_param_vec(vec_up, param_is_int), _normalise_param_vec(vec_down, param_is_int)
        if score_up > score_best:
            total_improvement[index] += score_up - score_best
            score_best = score_up
            vec_best = vec_up
            best_full_result = full_result_up
            last_improvement_iter = iter_count
        if score_down > score_best:
            total_improvement[index] += score_down - score_best
            score_best = score_down
            vec_best = vec_down
            best_full_result = full_result_down
            last_improvement_iter = iter_count
        if last_improvement_iter == iter_count:
            new_val = vec_best[index]
            fr = eval_report.summary_string(best_full_result) if best_full_result else "(memoised)"
            search_log(
                logfile,
                f"Iter {iter_count:04d} mult:{step_multiplier:>4} "
                f"{param_names[index]} → {new_val}  "
                f"score {score_best:0.4f}  [{fr} ]",
            )
            successive_improvements += 1
            improvements_since_validate += 1
            if successive_improvements >= 3:
                successive_improvements = 0
                param_index += 1
        else:
            search_log(
                logfile,
                f"Iter {iter_count:04d} mult:{step_multiplier:>4} "
                f"{param_names[index]} —  (best {score_best:0.4f}; up {score_up:0.4f}, dn {score_down:0.4f})",
            )
            successive_improvements = 0
            param_index += 1

        iter_count += 1
        if iter_count > last_improvement_iter + len(param_names) + 1:
            step_multiplier *= 0.5
            last_improvement_iter = iter_count
            search_log(logfile, f"Iter {iter_count:04d} ---- reducing multiplier to {step_multiplier}----")
            if step_multiplier < final_multiplier:
                finish()
                return
