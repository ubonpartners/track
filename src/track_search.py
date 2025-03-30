from datetime import datetime
import copy
import stuff
import src.track_test as track_test

def search_test(config, params, param_vec, param_min, param_max, all_results, split="train"):
    
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

def search_log(logfile, x):
    logfile.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+": ")
    logfile.write(x+"\n")
    logfile.flush()

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

    val, best_full_result=search_test(config, param_names, param_initial, param_min, param_max, results, split=train_split)
    vec_best=copy.copy(param_initial)
    val_best=val
    
    iter_count=0
    param_index=0
    last_improvement_iter=0
    improvements_since_validate=0
    last_validate_iter=0
    successive_improvements=0
    search_log(logfile, f"Iter {iter_count:04d} intial {val_best:0.4f} with vector {vec_best}")
    search_log(logfile, f"... best full result {best_full_result}\n")

    while True:
        index = param_index % len(param_names)

        do_val=improvements_since_validate>0 and iter_count>=last_validate_iter+4
        if train_split is not None:
            if do_val or iter_count==0:
                valval, full_result_val=search_test(config, param_names, vec_best, param_min, 
                                                    param_max, results, split="val")
                search_log(logfile, "======================================================")
                search_log(logfile, f"Iter {iter_count:04d}  **VALIDATE** {valval:0.4f} with vector {vec_best}")
                search_log(logfile, f"... best full result {full_result_val}\n")
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
        val_up,full_result_up=search_test(config, param_names, vec_up, param_min, param_max, results, split=train_split)
        val_down,full_result_down=search_test(config, param_names, vec_down, param_min, param_max, results, split=train_split)
        if val_up>val_best:
            val_best=val_up
            vec_best=vec_up
            best_full_result=full_result_up
            last_improvement_iter=iter_count
        if val_down>val_best:
            val_best=val_down
            vec_best=vec_down
            best_full_result=full_result_down
            last_improvement_iter=iter_count
        if last_improvement_iter==iter_count:
            search_log(logfile, f"Iter {iter_count:04d} mult: {step_multiplier} param {param_names[index]} new best {val_best:0.4f} with vector {vec_best} total {len(results)} results")
            search_log(logfile, f"... best full result {best_full_result}\n\n")
            successive_improvements+=1
            improvements_since_validate+=1
            if successive_improvements>=2:
                successive_improvements=0
                param_index+=1
        else:
            search_log(logfile, f"...param {param_names[index]} no improvement")
            successive_improvements=0
            param_index+=1

        iter_count+=1
        if iter_count>last_improvement_iter+len(param_names)+1:
            step_multiplier*=0.5
            last_improvement_iter=iter_count
            search_log(logfile, f"Iter {iter_count:04d} ---- reducing multiplier to {step_multiplier}----")
            if step_multiplier<final_multiplier:
                print("All done!")
                exit()