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
    
    # 核心元数据结构
    cov_info = {
        "categorical_mappings": {},
        "continuous_stats": {},
        "model_inputs": {"categorical": [], "continuous": []},
        "stratification": {"control_groups": [], "perturbed_groups": []},
        "combined_logic": {"base": condition_keys, "constituents": []}
    }

    # 1. 基础协变量处理 (Categorical & Continuous)
    for cov_name, cfg in cov_config.items():
        # 记录分层逻辑
        if cfg.get("control_ot") == "groupwise":
            cov_info["stratification"]["control_groups"].append(cov_name)
        if cfg.get("perturbed_ot") == "groupwise":
            cov_info["stratification"]["perturbed_groups"].append(cov_name)
        
        # 处理参与 combined 的逻辑
        if cfg.get("contain_in_condition_combined_keys"):
            cov_info["combined_logic"]["constituents"].append(cov_name)

        # 类型转换与特征提取
        if cfg["type"] == "categorical":
            all_series = pd.concat([a.obs[cov_name] for a in adatas.values()]).astype(str)
            cat2idx = {cat: idx for idx, cat in enumerate(sorted(all_series.unique()))}
            cov_info["categorical_mappings"][cov_name] = cat2idx
            
            for adata in adatas.values():
                adata.obs[f"{cov_name}_idx"] = adata.obs[cov_name].astype(str).map(cat2idx).astype(np.int64)
            if cfg.get("use_in_model"):
                cov_info["model_inputs"]["categorical"].append(f"{cov_name}_idx")

        elif cfg["type"] == "continuous":
            # 统一转为 float32 并计算 train 统计量
            for a in adatas.values():
                a.obs[cov_name] = pd.to_numeric(a.obs[cov_name], errors="coerce").astype(np.float32)
            
            train_vals = np.log1p(adata_train.obs[cov_name].values)
            mu, std = float(train_vals.mean()), float(train_vals.std())
            std = 1.0 if std == 0 else std
            cov_info["continuous_stats"][cov_name] = {"mean": mu, "std": std}

            for adata in adatas.values():
                log_v = np.log1p(adata.obs[cov_name].values)
                adata.obs[f"{cov_name}_scaled"] = (log_v - mu) / std
            if cfg.get("use_in_model"):
                cov_info["model_inputs"]["continuous"].append(f"{cov_name}_scaled")

    # 2. 构造 condition_combined_keys (采样锚点)
    combine_vars = cov_info["combined_logic"]["constituents"]
    
    for split_name, adata in adatas.items():
        # 初始串：perturbation 列
        res = adata.obs[condition_keys].astype(str)
        
        for var in combine_vars:
            cfg = cov_config[var]
            if split_name == "control" and cfg.get("condition_combined_from") == "perturbed":
                # 对标剂量/时间，Control 组统一标记为 'control' 以便采样匹配
                vals = "control"
            else:
                # cell_line 等 both 类型，或者扰动组的剂量，保留真实值
                vals = adata.obs[var].astype(str)
            
            res = res + "_" + vals
        
        adata.obs[condition_combined_keys] = res

    # 3. 将元数据注入 uns 并返回
    for adata in adatas.values():
        adata.uns["covariate_info"] = copy.deepcopy(cov_info)

    return adata_control, adata_train, adata_test
