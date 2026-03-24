import torch
import numpy as np
import pandas as pd
import copy

def convert_mixed_array_to_2d(arr):
    """
    将 dtype=object 的 array (包含 tensor / numpy / nan) 转换为纯二维 numpy 数组。
    对于 nan 行，会用 np.nan 填充。
    """
    # 把每个元素都转成 numpy 或 nan
    processed = []
    for a in arr:
        if isinstance(a, torch.Tensor):
            processed.append(a.detach().cpu().numpy())
        elif isinstance(a, np.ndarray):
            processed.append(a)
        elif a is None or (isinstance(a, float) and np.isnan(a)):
            processed.append(np.nan)
        else:
            raise TypeError(f"Unsupported element type: {type(a)}")
    
    # 找出有效的非 nan 元素
    valid = [a for a in processed if isinstance(a, np.ndarray)]
    if not valid:
        raise ValueError("No valid tensor/array elements found.")
    
    # 检查维度一致性
    shapes = {a.shape for a in valid}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent tensor shapes found: {shapes}")
    
    d = valid[0].shape[0]  # 向量维度
    result = np.full((len(processed), d), np.nan, dtype=float)
    
    # 填充结果
    for i, a in enumerate(processed):
        if isinstance(a, np.ndarray):
            result[i] = a
    
    return result


    
import numpy as np
import pandas as pd
import copy

def prepare_covariates(
    adata_control, adata_train, adata_test, 
    cov_config, condition_keys, condition_combined_keys
):
    adatas = {"control": adata_control, "train": adata_train, "test": adata_test}
    

    # 在预处理阶段把cov_config整合成一个rulebook，写入adata.uns
    rulebook = {
        "categorical_mappings": {},
        "continuous_stats": {},
        "model_inputs": {"categorical": {}, "continuous": {}},
        "stratification": {"control_groups": [], "perturbed_groups": []},
        "condition_tuple_schema": [condition_keys] 
    }

    for cov_name, cfg in cov_config.items():
        if cfg.get("control_ot") == "groupwise":
            rulebook["stratification"]["control_groups"].append(cov_name)
        if cfg.get("perturbed_ot") == "groupwise":
            rulebook["stratification"]["perturbed_groups"].append(cov_name)
        
        if cfg.get("contain_in_condition"):
            rulebook["condition_tuple_schema"].append(cov_name)

        # 离散变量 
        if cfg["type"] == "categorical":
            all_series = pd.concat([a.obs[cov_name] for a in adatas.values() if cov_name in a.obs]).astype(str)
            cat2idx = {cat: idx for idx, cat in enumerate(sorted(all_series.unique()))}
            rulebook["categorical_mappings"][cov_name] = cat2idx #保存映射，inference时使用
            
            for adata in adatas.values():
                # 添加一个{cov_name}_idx的obs列，供模型输入
                adata.obs[f"{cov_name}_idx"] = adata.obs[cov_name].astype(str).map(cat2idx).fillna(-1).astype(np.int64)
                
            
            # 记录模型输入的数据来源
            if cfg.get("use_in_model"):
                rulebook["model_inputs"]["categorical"][cov_name] = {
                    "obs_col": f"{cov_name}_idx",
                    "model_source": cfg.get("model_source", "both") 
                }

        # 连续变量 
        elif cfg["type"] == "continuous":
            #连续变量先log1p再算zscore
            transform_type = cfg.get("transform", "log1p_zscore")
            
            for a in adatas.values():
                a.obs[cov_name] = pd.to_numeric(a.obs[cov_name], errors="coerce").astype(np.float32)
            
            train_vals = adata_train.obs[cov_name].values
            if transform_type == "log1p_zscore":
                train_vals_tf = np.log1p(train_vals)
            else:
                train_vals_tf = train_vals
                
            mu, std = float(np.nanmean(train_vals_tf)), float(np.nanstd(train_vals_tf))
            std = 1.0 if std == 0 or np.isnan(std) else std

            rulebook["continuous_stats"][cov_name] = {
                "mean": mu, 
                "std": std, 
                "transform": transform_type
            }

            for adata in adatas.values():
                vals = adata.obs[cov_name].values
                if transform_type == "log1p_zscore":
                    vals_tf = np.log1p(vals)
                else:
                    vals_tf = vals
                adata.obs[f"{cov_name}_scaled"] = (vals_tf - mu) / std
                
            if cfg.get("use_in_model"):
                rulebook["model_inputs"]["continuous"][cov_name] = {
                    "obs_col": f"{cov_name}_scaled",
                    "model_source": cfg.get("model_source", "both")
                }


    # 对于condition和cov的组合，用tuple来唯一地标识
    # (pertubation,cov1,cov2)这样
    schema = rulebook["condition_tuple_schema"]
    
    for split_name, adata in adatas.items():
        tuples_list = []
        for idx, row in adata.obs.iterrows():
            base_cond = str(row[condition_keys]) 
            current_tuple = [base_cond]
            
            for var in schema[1:]:
                cfg = cov_config[var]
                
                if split_name == "control" and cfg.get("condition_source") == "perturbed": #对于control 有些cov是没有的 用统一的占位符补充
                    if cfg["type"] == "categorical":
                        current_tuple.append("control")
                    elif cfg["type"] == "continuous":
                        current_tuple.append(0.0)
                else:
                    current_tuple.append(row[var])
            
            tuples_list.append(str(tuple(current_tuple)))
            
        adata.obs[condition_combined_keys] = tuples_list # 新增一个condition_combined_keys列，里面存放着组合后的condition的唯一标识


    for adata in adatas.values():
        adata.uns["global_rulebook"] = copy.deepcopy(rulebook)

    return adata_control, adata_train, adata_test
