import copy
import datetime

import stuff

import src.track_test as track_test


def _set_nested_param(config_dict, key, value):
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
    parent, child_key, _ = matches[0]
    parent[child_key] = value


def _update_initial_parameters(param_names, param_initial, source_dict, logfile, source_label):
    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in param_names:
                    idx = param_names.index(key)
                    if param_initial[idx] is None:
                        param_initial[idx] = value
                        if logfile:
                            search_log(
                                logfile,
                                f"Setting parameter {key} initial value to {value} from {source_label}",
                            )
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(source_dict)


def _round_numeric_metrics(metrics_dict, decimals=3):
    for key, value in list(metrics_dict.items()):
        if isinstance(value, (int, float)):
            metrics_dict[key] = round(value, decimals)


def search_log(logfile, x):
    logfile.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": ")
    logfile.write(x + "\n")
    logfile.flush()


def search_test(
    config,
    params,
    param_vec,
    param_min,
    param_max,
    all_results,
    split="train",
    logfile=None,
    desc="search result",
):
    # check if the parameters are within the min and max range
    if any(p < min_v or p > max_v for p, min_v, max_v in zip(param_vec, param_min, param_max)):
        return -10000, None

    is_train = split == "train" or split is None

    if is_train and tuple(param_vec) in all_results:
        return all_results[tuple(param_vec)]["score"], None

    result_test_opt_key = config["result_test_opt_key"]
    result_dataset_opt_key = config["result_dataset_opt_key"]
    result_dataset_opt_param = config["result_dataset_opt_param"]
    c = config["tests"][result_test_opt_key]
    if isinstance(c["config"], str):
        c["config"] = stuff.load_dictionary(c["config"])
    for i, p in enumerate(params):
        c = config["tests"][result_test_opt_key]
        _set_nested_param(c["config"], p, param_vec[i])
    results = track_test.track_test(config, split=split, desc=desc)

    val = None
    full_result = None
    for r in results:
        if r["params"]["test_key"] == result_test_opt_key and r["params"]["ds_key"] == result_dataset_opt_key:
            val = r["result"][result_dataset_opt_param]
            full_result = r["result"]
    if is_train:
        all_results[tuple(param_vec)] = {"score": val, "param_vec": param_vec}
    if full_result is None:
        raise RuntimeError(
            f"No result found for test {result_test_opt_key} on dataset {result_dataset_opt_key}"
        )
    _round_numeric_metrics(full_result)
    return val, full_result


def eval_track(yaml_file):
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
    config = stuff.load_dictionary(yaml_file)
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
    return track_test.track_test(config, split=None,
                                  desc=f"eval {yaml_file}")


def search_track(yaml_file):
    config = stuff.load_dictionary(yaml_file)
    result_log_file = config["result_log_file_path"]
    cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    logfile = open(result_log_file + "/search_log_" + cur_time + ".txt", "w")
    param_names = []
    param_initial = []
    param_step = []
    param_min = []
    param_max = []

    results = {}

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

    for p in config["search_params"]:
        param_names.append(p)
        param_initial.append(None)

    test_dict = stuff.load_dictionary(config["tests"]["search_config"]["config"])
    _update_initial_parameters(param_names, param_initial, test_dict, logfile, "base config")

    for i, p in enumerate(config["search_params"]):
        if "initial" in config["search_params"][p]:
            param_initial[i] = float(config["search_params"][p]["initial"])
            search_log(logfile, f"Setting parameter {p} initial value to {param_initial[i]} from search config")
        assert param_initial[i] is not None, f"Parameter {p} missing initial value"
        param_step.append(float(config["search_params"][p]["step"]))
        param_min.append(float(config["search_params"][p]["min"]))
        param_max.append(float(config["search_params"][p]["max"]))

    search_log(logfile, "Search params:" + str(param_names))

    for i, v in enumerate(param_initial):
        p = param_names[i]
        assert v >= param_min[i], f"{p} : Initial value {v} is less than min {param_min[i]}"
        assert v <= param_max[i], f"{p} : Initial value {v} is more than max {param_max[i]}"

    param_initial = [round(v, 3) for v in param_initial]
    score_best, best_full_result = search_test(
        config,
        param_names,
        param_initial,
        param_min,
        param_max,
        results,
        split=train_split,
        logfile=logfile,
        desc="search test: initial",
    )
    assert best_full_result is not None
    vec_best = copy.copy(param_initial)

    iter_count = 0
    param_index = 0
    last_improvement_iter = 0
    improvements_since_validate = 0
    last_validate_iter = 0
    successive_improvements = 0
    search_log(logfile, f"Iter {iter_count:04d} initial score {score_best:0.4f}  [{track_test.summary_string(best_full_result)} ]")
    search_log(logfile, f"  initial vector: {dict(zip(param_names, vec_best))}")

    total_improvement = [0.0] * len(param_names)

    while True:
        index = param_index % len(param_names)

        do_val = improvements_since_validate > 0 and iter_count >= last_validate_iter + 4
        if train_split is not None:
            if do_val or iter_count == 0:
                validate_score, full_result_val = search_test(
                    config,
                    param_names,
                    vec_best,
                    param_min,
                    param_max,
                    results,
                    split="val",
                    logfile=logfile,
                    desc=f"search test it:{iter_count} validate",
                )
                search_log(logfile, "======================================================")
                search_log(logfile, f"Iter {iter_count:04d}  **VALIDATE** score {validate_score:0.4f}  [{track_test.summary_string(full_result_val)} ]")
                search_log(logfile, f"  vector: {dict(zip(param_names, vec_best))}")
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

        vec_up = copy.copy(vec_best)
        vec_down = copy.copy(vec_best)
        vec_up[index] += step_multiplier * param_step[index]
        vec_up = [round(v, 3) for v in vec_up]
        vec_down[index] -= step_multiplier * param_step[index]
        vec_down = [round(v, 3) for v in vec_down]
        score_up, full_result_up = search_test(
            config,
            param_names,
            vec_up,
            param_min,
            param_max,
            results,
            split=train_split,
            logfile=logfile,
            desc=f"search test it:{iter_count} search param {param_names[index]} up",
        )
        score_down, full_result_down = search_test(
            config,
            param_names,
            vec_down,
            param_min,
            param_max,
            results,
            split=train_split,
            logfile=logfile,
            desc=f"search test it:{iter_count} search param {param_names[index]} down",
        )
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
            search_log(
                logfile,
                f"Iter {iter_count:04d} mult:{step_multiplier:>4} "
                f"{param_names[index]} → {new_val}  "
                f"score {score_best:0.4f}  [{track_test.summary_string(best_full_result)} ]",
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
                search_log(logfile, "All done!")
                logfile.close()
                return

import copy
import stuff
import src.track_test as track_test
import datetime
