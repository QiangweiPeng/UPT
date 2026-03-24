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
        condition_keys,          
        condition_combined_keys, 
        save_path, 
        sample_rep='X_pca_scaled', 
        mass_deduct_keys=None, 
        delta=1,
        reg_m=1,
        use_mini_batch_uot=True, 
        group_number=5,
        batch_save_size=10,
        draw=False
    ): 
    """
    基于 Global Rulebook 和 Tuple 身份标识的 WFR-OT 计算函数。
    """
    
    # 在训练阶段，直接读取uns中的rulebook
    rulebook = adata_treated.uns.get('global_rulebook')
    if rulebook is None or'stratification' not in rulebook:
        raise ValueError("adata_treated.uns 必须包含 'global_rulebook'。请先运行 prepare_covariates。")
        
    control_groups_keys = list(rulebook['stratification']['control_groups'])
    perturbed_groups_keys = list(rulebook['stratification']['perturbed_groups'])
    
    treated_groupby_keys = [condition_keys] + perturbed_groups_keys # perturb的分组还要加上基础扰动

    # base_m_source提高flexibility，对array的数据后面会用孔数来现场计算调整比例
    base_m_source = adata_control.uns.get("normalized_m", 1.0)
    base_m_target = adata_treated.uns.get("normalized_m", 1.0)

    if mass_deduct_keys is not None:
        source_deduct_arr = adata_control.obs[mass_deduct_keys].astype(str).to_numpy()
        target_deduct_arr = adata_treated.obs[mass_deduct_keys].astype(str).to_numpy()
    
    X_control_all = adata_control.obsm[sample_rep]
    obs_names_control_all = adata_control.obs_names.to_numpy()
    
    X_treated_all = adata_treated.obsm[sample_rep] 
    obs_names_treated_all = adata_treated.obs_names.to_numpy()

    # 提前预分组
    if len(control_groups_keys) > 0:
        control_gb = adata_control.obs.groupby(control_groups_keys)
    else:
        control_gb = None
        
    treated_gb = adata_treated.obs.groupby(treated_groupby_keys)
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
    
    # 主循环按batch处理
    for batch_start in range(start_idx, len(all_treated_keys), batch_save_size):
        batch_end = min(batch_start + batch_save_size, len(all_treated_keys))
        
        batch_results = {
            "uot_plans": {},      
            "gamma0_plans": {},   
            "gamma1_plans": {},   
            "treat_obs_names": {}, 
            "control_obs_names": {} 
        }
        
        for i in tqdm(range(batch_start, batch_end)):
            treated_val = all_treated_keys[i]
            
            # 规范化为tuple后化为dict
            treated_val_tuple = treated_val if isinstance(treated_val, tuple) else (treated_val,)
            treated_dict = dict(zip(treated_groupby_keys, treated_val_tuple))
            
            # 反推control的ot_group
            if control_gb is None:
                # 如果 Control 是纯 Global，无需匹配，直接拿全体 Control
                control_indices = np.arange(len(adata_control))
                control_key = "global_control"
            else:
                try:
                    control_val_tuple = tuple(treated_dict[k] for k in control_groups_keys)
                except KeyError as e:
                    raise ValueError(f"配置冲突: 控制组需要的协变量 {e} 在扰动组中未定义，无法匹配。")
                    
                control_key = control_val_tuple[0] if len(control_groups_keys) == 1 else control_val_tuple
                
                if control_key not in control_gb.indices:
                    print(f"警告: 找不到匹配的 Control 数据用于 {treated_dict} (需要的Control条件为 {control_key})。跳过。")
                    continue
                control_indices = control_gb.indices[control_key]
                
            # 提取当前的 Treated 和 Control 矩阵
            treat_indices = treated_gb.indices[treated_val]
            X_treat_cur = X_treated_all[treat_indices]
            obs_names_treat_cur = obs_names_treated_all[treat_indices]
            
            X_control_cur = X_control_all[control_indices]
            obs_names_control_cur = obs_names_control_all[control_indices]

            cur_m_source = base_m_source
            cur_m_target = base_m_target
            
            if mass_deduct_keys is not None:
                # 统计当前切片下，独立的 plate_well 个数
                n_unique_source = len(np.unique(source_deduct_arr[control_indices]))
                n_unique_target = len(np.unique(target_deduct_arr[treat_indices]))
                
                # 对局部 mass 进行折扣 (防止除以 0)
                cur_m_source = base_m_source 
                cur_m_target = base_m_target * max(1, n_unique_source)/ max(1, n_unique_target)
                # print(max(1, n_unique_source)/ max(1, n_unique_target))
            
            # 计算 OT
            gamma, g0, g1 = compute_uot_plan_gpu(
                X_control_cur, X_treat_cur, 
                m_source=cur_m_source, m_target=cur_m_target,
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
            
            # 将算好的 OT Matrix 映射到它涵盖的所有 Tuple 身份上！
            # 获取当前 Treated 分组中，实际包含的所有 Tuple 字符串
            # 这种情况只发生在ot = global contain_in_condition=True的情况
            associated_tuples = adata_treated.obs.iloc[treat_indices][condition_combined_keys].unique()
            
            gamma_sparse = sparse.csr_matrix(gamma * (gamma >= threshold))
            g0_sparse = sparse.csr_matrix(g0 * (g0 >= threshold))
            g1_sparse = sparse.csr_matrix(g1 * (g1 >= threshold))
            
            # 无论这个 OT 覆盖了 1 个还是多个 Tuple 组合，统统挂载
            for t_str in associated_tuples:
                batch_results["uot_plans"][t_str] = gamma_sparse
                batch_results["gamma0_plans"][t_str] = g0_sparse
                batch_results["gamma1_plans"][t_str] = g1_sparse
                batch_results["treat_obs_names"][t_str] = obs_names_treat_cur
                batch_results["control_obs_names"][t_str] = obs_names_control_cur
            
            del gamma, g0, g1, X_treat_cur, X_control_cur
            
        gc.collect()
        
        # 保存batch并清空内存
        batch_path = os.path.join(batch_dir, f"{base_name}_batch_{batch_end}.pkl")
        with open(batch_path, 'wb') as f:
            pickle.dump(batch_results, f)
        print(f"Batch saved: {batch_end}/{len(all_treated_keys)}")
        
        del batch_results
        gc.collect()
    
    # 5. 合并所有batch到最终结果 (保持原样)
    print("合并所有batch...")
    final_results = {
        "all_conditions_computed": [], 
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
    
    with open(save_path, 'wb') as f:
        pickle.dump(final_results, f)
    
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