import os
import sys
import pickle
import gc  
import numpy as np
import torch
import ot
from scipy import sparse
from tqdm import tqdm
import random
import pandas as pd 

    

def compute_uot_plan_gpu(X_source, X_target,m_source=1,m_target=1, delta=1, reg_m=1, use_mini_batch_uot=False, group_number=5,draw=False):
    """
    如果数据量大了cpu跑不动 需要gpu版本
    mini_batch已实现
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        if isinstance(X_source, np.ndarray):
            X_source = torch.from_numpy(X_source).float().to(device)
        if isinstance(X_target, np.ndarray):
            X_target = torch.from_numpy(X_target).float().to(device)

        n_source, n_target = X_source.shape[0], X_target.shape[0]

        def get_cost_batch(xs, xt):
            dist = torch.cdist(xs, xt, p=2) 
            
            limit = torch.tensor(np.pi/2, device=device)
            term = torch.clamp(dist / (2 * delta), max=limit)
            cos_sq = torch.cos(term)**2
            
            epsilon = torch.tensor(1e-10, device=device)
            c_mat = -torch.log(torch.where(cos_sq == 0, epsilon, cos_sq))
            return c_mat

        a = torch.full((n_source,), m_source, device=device, dtype=torch.float)
        b = torch.full((n_target,), m_target, device=device, dtype=torch.float)
        
        if not use_mini_batch_uot:
            cost_matrix = get_cost_batch(X_source, X_target)
            G = ot.unbalanced.mm_unbalanced(a, b, cost_matrix, reg_m=reg_m, numItermax=1000)
            res_G = G.cpu().numpy()
            del cost_matrix, G
        else:
            res_G = np.zeros((n_source, n_target), dtype=np.float32) #直接放cpu上
    
            source_perm = torch.randperm(n_source, device=device)
            target_perm = torch.randperm(n_target, device=device)
    
            source_indices = torch.tensor_split(source_perm, group_number)
            target_indices = torch.tensor_split(target_perm, group_number)
    
            for src_idx, tgt_idx in zip(source_indices, target_indices): #只算对角
                
                sub_a = a[src_idx]
                sub_b = b[tgt_idx]
                
                sub_x_s = X_source[src_idx]
                sub_x_t = X_target[tgt_idx]
                sub_cost_matrix = get_cost_batch(sub_x_s, sub_x_t) 
                
                G_sub = ot.unbalanced.mm_unbalanced(
                    sub_a, sub_b, sub_cost_matrix, 
                    reg_m=reg_m,     
                    numItermax=1000
                )

                src_idx_cpu = src_idx.cpu().numpy()
                tgt_idx_cpu = tgt_idx.cpu().numpy()
                res_G[np.ix_(src_idx_cpu, tgt_idx_cpu)] = G_sub.cpu().numpy()
                del sub_cost_matrix, G_sub, sub_x_s, sub_x_t

        a_np = np.full((n_source,), m_source, dtype=np.float32)
        b_np = np.full((n_target,), m_target, dtype=np.float32)
    
        sum_1 = res_G.sum(axis=1, keepdims=True) 
        sum_0 = res_G.sum(axis=0, keepdims=True) 
    
        sum_1[sum_1 == 0] = 1.0
        sum_0[sum_0 == 0] = 1.0
    
        res_g0 = (a_np.reshape(-1, 1) / sum_1) * res_G
        res_g1 = (b_np.reshape(1, -1) / sum_0) * res_G


        if draw:
            import matplotlib.pyplot as plt
            source_pred = res_G.sum(axis=1)  # (n_source,)
            target_pred = res_G.sum(axis=0)  # (n_target,)
            
            sum1_flat = sum_1.squeeze(1)
            sum0_flat = sum_0.squeeze(0)
    
            fig = plt.figure(figsize=(15, 5))
            
            # 绘制 Source 侧
            plt.subplot(121)
            plt.plot(a_np, label='Target (m_source)', color='blue', alpha=0.6)
            plt.plot(sum1_flat, label='Pred (sum_i_give)', color='orange', alpha=0.6, linestyle='--')
            plt.title(f"Source Marginals (MSE: {np.mean((a_np - sum1_flat)**2):.4f})")
            plt.legend()
        
            # 绘制 Target 侧
            plt.subplot(122)
            plt.plot(b_np, label='Target (m_target)', color='blue', alpha=0.6)
            plt.plot(sum0_flat, label='Pred (sum_j_receive)', color='orange', alpha=0.6, linestyle='--')
            plt.title(f"Target Marginals (MSE: {np.mean((b_np - sum0_flat)**2):.4f})")
            plt.legend()
    
            plt.show()
    

    del X_source, X_target
    torch.cuda.empty_cache()

    return res_G, res_g0, res_g1



def pre_compute_wfr_ot(
        adata_control, 
        adata_treated, 
        save_path, 
        sample_rep='X_pca_scaled', 
        delta=1,
        reg_m=1,
        use_mini_batch_uot=True, 
        group_number=5,
        batch_save_size=10,
        draw=False
    ): 
    """
    支持多维 Covariant 分组的 WFR-OT 计算函数。
    直接通过读取 adata_treated.uns['covariate_info'] 进行动态匹配。
    """
    
    # 1. 解析协变量分组策略
    cov_info = adata_treated.uns.get('covariate_info')
    if cov_info is None or'stratification' not in cov_info:
        raise ValueError("adata_treated.uns 必须包含 'covariate_info['stratification']' 以执行多维分组。")
        
    control_groups_keys = list(cov_info['stratification']['control_groups'])
    perturbed_groups_keys = list(cov_info['stratification']['perturbed_groups'])
    
    # 2. 提取全局数据池
    m_source = adata_control.uns.get("normalized_m", 1)
    m_target = adata_treated.uns.get("normalized_m", 1)
    
    X_control_all = adata_control.obsm[sample_rep]
    obs_names_control_all = adata_control.obs_names.to_numpy()
    
    X_treated_all = adata_treated.obsm[sample_rep] 
    obs_names_treated_all = adata_treated.obs_names.to_numpy()

    # 3. 构建高效的分组索引字典 (Pandas GroupBy)
    control_gb = adata_control.obs.groupby(control_groups_keys)
    treated_gb = adata_treated.obs.groupby(perturbed_groups_keys)
    
    # 提取所有 Treated 组的 keys (如果有多列，则是 tuple；如果只有1列，则是 scalar)
    all_treated_keys = list(treated_gb.indices.keys())
    
    # 断点续传逻辑
    batch_dir = os.path.dirname(save_path) or '.'
    base_name = os.path.basename(save_path).replace('.pkl', '')
    completed_batches = []
    for f in os.listdir(batch_dir):
        if f.startswith(f"{base_name}_batch_") and f.endswith('.pkl'):
            batch_num = int(f.split('_batch_')[1].replace('.pkl', ''))
            completed_batches.append(batch_num)
    
    start_idx = max(completed_batches) if completed_batches else 0
    if start_idx > 0:
        print(f"从batch {start_idx}恢复，已完成 {start_idx}/{len(all_treated_keys)}")
    
    # 4. 主循环 - 按batch处理
    for batch_start in range(start_idx, len(all_treated_keys), batch_save_size):
        batch_end = min(batch_start + batch_save_size, len(all_treated_keys))
        
        batch_results = {
            "uot_plans": {},      
            "gamma0_plans": {},   
            "gamma1_plans": {},   
            "treat_obs_names": {}, 
            "control_obs_names": {} # 新增：现在每次 OT 对应的 Control 也是动态的
        }
        
        for i in tqdm(range(batch_start, batch_end)):
            treated_val = all_treated_keys[i]
            
            # (A) 规范化 Treated 值为 Tuple 及 Dict 形式
            treated_val_tuple = treated_val if isinstance(treated_val, tuple) else (treated_val,)
            treated_dict = dict(zip(perturbed_groups_keys, treated_val_tuple))
            
            # 生成结构化且可读的 Condition Name (例如: "cell_line:A549|dose_value:1.0|time:24.0")
            cur_condition_name = "|".join([f"{k}:{v}" for k, v in treated_dict.items()])
            
            # (B) 根据字典映射，反推出 Control 需要匹配的 key
            control_val_tuple = tuple(treated_dict[k] for k in control_groups_keys)
            control_key = control_val_tuple[0] if len(control_groups_keys) == 1 else control_val_tuple
            
            # (C) 检查 Control 池子中是否存在对应的细胞
            if control_key not in control_gb.indices:
                print(f"警告: 找不到匹配的 Control 数据用于 {cur_condition_name} (需要的Control条件为 {control_key})。跳过。")
                continue
                
            # (D) 提取当前的 Treated 和 Control 矩阵
            treat_indices = treated_gb.indices[treated_val]
            X_treat_cur = X_treated_all[treat_indices]
            obs_names_treat_cur = obs_names_treated_all[treat_indices]
            
            control_indices = control_gb.indices[control_key]
            X_control_cur = X_control_all[control_indices]
            obs_names_control_cur = obs_names_control_all[control_indices]
            
            # (E) 计算 OT
            gamma, g0, g1 = compute_uot_plan_gpu(
                X_control_cur, X_treat_cur, 
                m_source=m_source, m_target=m_target,
                delta=delta, reg_m=reg_m,
                use_mini_batch_uot=use_mini_batch_uot, 
                group_number=group_number,
                draw=draw
            )
            
            if hasattr(gamma, 'cpu'):
                gamma = gamma.cpu().numpy()
                g0 = g0.cpu().numpy()
                g1 = g1.cpu().numpy()
                torch.cuda.empty_cache()

            threshold = max(1e-4, 10 / X_treat_cur.shape[0])
            
            # (F) 保存结果
            batch_results["uot_plans"][cur_condition_name] = sparse.csr_matrix(gamma * (gamma >= threshold))
            batch_results["gamma0_plans"][cur_condition_name] = sparse.csr_matrix(g0 * (g0 >= threshold))
            batch_results["gamma1_plans"][cur_condition_name] = sparse.csr_matrix(g1 * (g1 >= threshold))
            batch_results["treat_obs_names"][cur_condition_name] = obs_names_treat_cur
            batch_results["control_obs_names"][cur_condition_name] = obs_names_control_cur
            
            del gamma, g0, g1, X_treat_cur, X_control_cur
            
        gc.collect()
        
        # 保存batch并清空内存
        batch_path = os.path.join(batch_dir, f"{base_name}_batch_{batch_end}.pkl")
        with open(batch_path, 'wb') as f:
            pickle.dump(batch_results, f)
        print(f"Batch saved: {batch_end}/{len(all_treated_keys)}")
        
        del batch_results
        gc.collect()
    
    # 5. 合并所有batch到最终结果
    print("合并所有batch...")
    final_results = {
        "all_conditions_computed": [], # 记录实际计算了的 condition string 列表
        "delta": delta,
        "uot_plans": {},
        "gamma0_plans": {},
        "gamma1_plans": {},
        "treat_obs_names": {},
        "control_obs_names": {} 
    }
    
    batch_files = sorted([f for f in os.listdir(batch_dir) 
                          if f.startswith(f"{base_name}_batch_")],
                         key=lambda x: int(x.split('_batch_')[1].replace('.pkl', '')))
    
    for batch_file in batch_files:
        with open(os.path.join(batch_dir, batch_file), 'rb') as f:
            batch_data = pickle.load(f)
        for key in ["uot_plans", "gamma0_plans", "gamma1_plans", "treat_obs_names", "control_obs_names"]:
            if key in batch_data:
                final_results[key].update(batch_data[key])
        del batch_data
        gc.collect()
        
    final_results["all_conditions_computed"] = list(final_results["uot_plans"].keys())
    
    # 保存最终结果
    with open(save_path, 'wb') as f:
        pickle.dump(final_results, f)
    
    # 删除所有batch文件
    for batch_file in batch_files:
        os.remove(os.path.join(batch_dir, batch_file))
    
    print(f"完成！最终结果: {save_path}")
    return final_results
    

def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    print(f"Global seed set to {seed}")