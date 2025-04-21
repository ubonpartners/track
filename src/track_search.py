from datetime import datetime
import copy
import stuff
import src.track_test as track_test

def search_log(logfile, x):
    logfile.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+": ")
    logfile.write(x+"\n")
    logfile.flush()

def search_test(config, params, param_vec, param_min, param_max, all_results, split="train", logfile=None):
    
    param_vec_clipped = [max(min(v, max_v), min_v) for v, min_v, max_v in zip(param_vec, param_min, param_max)]
    
    is_train=split=="train" or split==None

    if is_train and tuple(param_vec_clipped) in all_results:
        return all_results[tuple(param_vec_clipped)]["score"], None
    
    result_test_opt_key=config["result_test_opt_key"]
    result_dataset_opt_key=config["result_dataset_opt_key"]
    result_dataset_opt_param=config["result_dataset_opt_param"]
  
    for i,p in enumerate(params):
        if param_vec_clipped[i]<0:
            print(f"WARNING: negative parameter {i} {param_vec_clipped[i]} {param_min[i]}-{param_max[i]}")
        config["tests"][result_test_opt_key][p]=param_vec_clipped[i]
    search_log(logfile, f"..... testing {param_vec_clipped}")
    results=track_test.track_test(config, split=split)
    
    val=None
    full_result=None
    for r in results:
        if r["params"]["test_key"]==result_test_opt_key and r["params"]["ds_key"]==result_dataset_opt_key:
            val=r["result"][result_dataset_opt_param]
            full_result=r["result"]
    if is_train:
        all_results[tuple(param_vec_clipped)]={"score":val, "param_vec":param_vec_clipped}
    for r in full_result:
        full_result[r]=round(full_result[r],3)
    return val,full_result

def search_track(yaml_file):
    config=stuff.load_dictionary(yaml_file)
    result_log_file=config["result_log_file"]
    logfile=open(result_log_file, "w")
    param_names=[]
    param_initial=[]
    param_step=[]
    param_min=[]
    param_max=[]

    results={}

    train_split="train"
    if "do_train_split" in config:
        if config["do_train_split"]==False:
            train_split=None
    search_log(logfile, f"Setting train split to: {train_split}")

    step_multiplier=4
    final_multiplier=0.5
    if "initial_mult" in config:
        step_multiplier=config["initial_mult"]
    if "final_mult" in config:
        final_multiplier=config["final_mult"]
    search_log(logfile, f"Starting step multiplier set to {step_multiplier}")

    for p in config["search_params"]:
        param_names.append(p)
        param_initial.append(None)
    
    test_dict=stuff.load_dictionary(config["tests"]["search_config"]["config"])
    for p in test_dict:
        if p in param_names:
            param_initial[param_names.index(p)]=test_dict[p]
            search_log(logfile, f"Setting parameter {p} initial value to {test_dict[p]} from base config")

    for i,p in enumerate(config["search_params"]):
        if "initial" in config["search_params"][p]:
            param_initial[i]=float(config["search_params"][p]["initial"])
            search_log(logfile, f"Setting parameter {p} initial value to {param_initial[i]} from search config")
        assert param_initial[i] is not None, f"Parameter {p} missing intial value"
        param_step.append(float(config["search_params"][p]["step"]))
        param_min.append(float(config["search_params"][p]["min"]))
        param_max.append(float(config["search_params"][p]["max"]))

    search_log(logfile, "Search params:"+str(param_names))

    score_best, best_full_result=search_test(config, param_names, param_initial, param_min, param_max, results, split=train_split, logfile=logfile)
    vec_best=copy.copy(param_initial)
    
    iter_count=0
    param_index=0
    last_improvement_iter=0
    improvements_since_validate=0
    last_validate_iter=0
    successive_improvements=0
    search_log(logfile, f"Iter {iter_count:04d} intial score {score_best:0.4f} at vector {vec_best}")
    search_log(logfile, f"---> Best score is {track_test.summary_string(best_full_result)}")

    total_improvement=[0.0]*len(param_names)

    while True:
        index = param_index % len(param_names)

        do_val=improvements_since_validate>0 and iter_count>=last_validate_iter+4
        if train_split is not None:
            if do_val or iter_count==0:
                validate_score, full_result_val=search_test(config, param_names, vec_best, param_min, 
                                                    param_max, results, split="val", logfile=logfile)
                search_log(logfile, "======================================================")
                search_log(logfile, f"Iter {iter_count:04d}  **VALIDATE** score {validate_score:0.4f} at vector {vec_best}")
                search_log(logfile, f"... full result {full_result_val}\n")
                for i,_ in enumerate(vec_best):
                    search_log(logfile, f"    {param_names[i]}: {vec_best[i]}")
                search_log(logfile, "======================================================")
                improvements_since_validate=0
                last_validate_iter=iter_count

        vec_up=copy.copy(vec_best)
        vec_down=copy.copy(vec_best)
        vec_up[index]+=step_multiplier*param_step[index]
        vec_up=[round(v,3) for v in vec_up]
        vec_down[index]-=step_multiplier*param_step[index]
        vec_down=[round(v,3) for v in vec_down]
        score_up,full_result_up=search_test(config, param_names, vec_up, param_min, param_max, results, split=train_split, logfile=logfile)
        score_down,full_result_down=search_test(config, param_names, vec_down, param_min, param_max, results, split=train_split, logfile=logfile)
        if score_up>score_best:
            total_improvement[index]+=(score_up-score_best)
            score_best=score_up
            vec_best=vec_up
            best_full_result=full_result_up
            last_improvement_iter=iter_count
        if score_down>score_best:
            total_improvement[index]+=(score_down-score_best)
            score_best=score_down
            vec_best=vec_down
            best_full_result=full_result_down
            last_improvement_iter=iter_count
        if last_improvement_iter==iter_count:
            search_log(logfile, f"Iter {iter_count:04d} mult: {step_multiplier} param {param_names[index]} 🎉🎉 new score_best 🎉🎉 : {score_best:0.4f} at vector {vec_best} total {len(results)} results")
            improvement=sum(total_improvement)
            if improvement!=0:
                for i in range(len(param_names)):
                    if total_improvement[i]!=0:
                        search_log(logfile, f"\t\t{param_names[i]:20s} improvement {total_improvement[i]:8.5f} {(100*total_improvement[i]/improvement):.1f}%%")
            successive_improvements+=1
            improvements_since_validate+=1
            if successive_improvements>=2:
                successive_improvements=0
                param_index+=1
        else:
            search_log(logfile, f"...param {param_names[index]} no improvement  score_best:{score_best:0.4f} score_u:{score_up:0.4f} score_d:{score_down:0.4f}")
            successive_improvements=0
            param_index+=1

        search_log(logfile, f"---> Best score is {track_test.summary_string(best_full_result)}")

        iter_count+=1
        if iter_count>last_improvement_iter+len(param_names)+1:
            step_multiplier*=0.5
            last_improvement_iter=iter_count
            search_log(logfile, f"Iter {iter_count:04d} ---- reducing multiplier to {step_multiplier}----")
            if step_multiplier<final_multiplier:
                print("All done!")
                exit()