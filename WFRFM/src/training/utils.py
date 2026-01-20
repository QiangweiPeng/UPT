import os
import sys
import pickle
import gc  
import numpy as np
import torch
import ot
from scipy import sparse
from tqdm import tqdm

    

def compute_uot_plan_gpu(X_source, X_target, delta=1, reg_m=1, use_mini_batch_uot=False, group_number=5):
    """
    如果数据量大了cpu跑不动 需要gpu版本
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        if isinstance(X_source, np.ndarray):
            X_source = torch.from_numpy(X_source).float().to(device)
        if isinstance(X_target, np.ndarray):
            X_target = torch.from_numpy(X_target).float().to(device)

        n_source, n_target = X_source.shape[0], X_target.shape[0]

        # 3. 计算距离 (POT 库支持 GPU Tensor)
        norm_2_dist = ot.dist(X_source, X_target, metric='euclidean')

        # 4. 计算 Cost Matrix (全部在 GPU 完成)
        limit = torch.tensor(np.pi/2, device=device)
        term = torch.clamp(norm_2_dist / (2 * delta), max=limit)
        cos_sq = torch.cos(term)**2
        
        # 释放距离矩阵，腾出显存
        del norm_2_dist 
        
        epsilon = torch.tensor(1e-10, device=device)
        cost_matrix = -torch.log(torch.where(cos_sq == 0, epsilon, cos_sq))

        # 5. 求解 Unbalanced OT
        # 设置 numItermax 防止某些极端情况死循环
        if not use_mini_batch_uot:
            a = torch.ones(n_source, device=device)
            b = torch.ones(n_target, device=device)
            G = ot.unbalanced.mm_unbalanced(a, b, cost_matrix, reg_m=reg_m, numItermax=1000)
        else:
            pass
            # 后面再写
        

        # 6. 计算 Marginal Plans
        sum_1 = G.sum(1, keepdim=True)
        sum_1 = torch.where(sum_1 == 0, torch.tensor(1.0, device=device), sum_1)
        
        sum_0 = G.sum(0, keepdim=True)
        sum_0 = torch.where(sum_0 == 0, torch.tensor(1.0, device=device), sum_0)

        gamma0_plan = (a.view(-1, 1) / sum_1) * G
        gamma1_plan = (b.view(1, -1) / sum_0) * G

        # 7. 转回 CPU Numpy 并立即释放 GPU Tensor
        res_G = G.cpu().numpy()
        res_g0 = gamma0_plan.cpu().numpy()
        res_g1 = gamma1_plan.cpu().numpy()

    # 清理所有 GPU 临时变量
    del X_source, X_target, cost_matrix, G, gamma0_plan, gamma1_plan
    # 强制清理显存缓存
    torch.cuda.empty_cache()

    return res_G, res_g0, res_g1


def pre_compute_wfr_ot(adata_control, 
                       adata_treated, 
                       save_path, 
                       sample_rep='X_pca_scaled', 
                       condition_keys="target_gene",
                       delta=1,
                       reg_m=1,
                       use_mini_batch_uot=True, 
                       group_number=5,
                       batch_save_size=10): 
    
    X_control = adata_control.obsm[sample_rep]
    all_conditions = adata_treated.obs[condition_keys].unique().tolist()
    threshold = 1e-6
    
    # 检查已完成的batch
    batch_dir = os.path.dirname(save_path) or '.'
    base_name = os.path.basename(save_path).replace('.pkl', '')
    completed_batches = []
    for f in os.listdir(batch_dir):
        if f.startswith(f"{base_name}_batch_") and f.endswith('.pkl'):
            batch_num = int(f.split('_batch_')[1].replace('.pkl', ''))
            completed_batches.append(batch_num)
    
    start_idx = max(completed_batches) if completed_batches else 0
    if start_idx > 0:
        print(f"从batch {start_idx}恢复，已完成 {start_idx}/{len(all_conditions)}")
    
    # 主循环 - 按batch处理
    for batch_start in range(start_idx, len(all_conditions), batch_save_size):
        batch_end = min(batch_start + batch_save_size, len(all_conditions))
        
        # 每个batch独立的结果字典
        batch_results = {
            "all_conditions": [], 
            "control_obs_names": adata_control.obs_names.to_numpy(),
            "treat_obs_names": [],
            "uot_plans": [],
            "gamma0_plans": [],
            "gamma1_plans": [],
            "delta": delta
        }
        
        # 处理当前batch
        for i in tqdm(range(batch_start, batch_end)):
            cur_condition = all_conditions[i]
            adata_treat_cur = adata_treated[adata_treated.obs[condition_keys]==cur_condition]
            X_treat_cur = adata_treat_cur.obsm[sample_rep]
            
            gamma, g0, g1 = compute_uot_plan_gpu(
                X_control, X_treat_cur, 
                delta=delta, reg_m = reg_m,
                use_mini_batch_uot=use_mini_batch_uot, 
                group_number=group_number
            )
            
            if hasattr(gamma, 'cpu'):
                gamma = gamma.cpu().numpy()
                g0 = g0.cpu().numpy()
                g1 = g1.cpu().numpy()
                torch.cuda.empty_cache()
            
            batch_results["uot_plans"].append(sparse.csr_matrix(gamma * (gamma >= threshold)))
            batch_results["gamma0_plans"].append(sparse.csr_matrix(g0 * (g0 >= threshold)))
            batch_results["gamma1_plans"].append(sparse.csr_matrix(g1 * (g1 >= threshold)))
            batch_results["treat_obs_names"].append(adata_treat_cur.obs_names.to_numpy())
            batch_results["all_conditions"].append(cur_condition)
            
            del gamma, g0, g1, X_treat_cur, adata_treat_cur
            gc.collect()
        
        # 保存batch并清空内存
        batch_path = os.path.join(batch_dir, f"{base_name}_batch_{batch_end}.pkl")
        with open(batch_path, 'wb') as f:
            pickle.dump(batch_results, f)
        print(f"Batch saved: {batch_end}/{len(all_conditions)}")
        
        del batch_results
        gc.collect()
    
    # 合并所有batch到最终结果
    print("合并所有batch...")
    final_results = {
        "all_conditions": [], 
        "control_obs_names": adata_control.obs_names.to_numpy(),
        "treat_obs_names": [],
        "uot_plans": [],
        "gamma0_plans": [],
        "gamma1_plans": [],
        "delta": delta
    }
    
    batch_files = sorted([f for f in os.listdir(batch_dir) 
                          if f.startswith(f"{base_name}_batch_")],
                         key=lambda x: int(x.split('_batch_')[1].replace('.pkl', '')))
    
    for batch_file in batch_files:
        with open(os.path.join(batch_dir, batch_file), 'rb') as f:
            batch_data = pickle.load(f)
        for key in ["uot_plans", "gamma0_plans", "gamma1_plans", "treat_obs_names", "all_conditions"]:
            final_results[key].extend(batch_data[key])
        del batch_data
        gc.collect()
    
    # 保存最终结果
    with open(save_path, 'wb') as f:
        pickle.dump(final_results, f)
    
    # 删除所有batch文件
    for batch_file in batch_files:
        os.remove(os.path.join(batch_dir, batch_file))
    
    print(f"完成！最终结果: {save_path}")
    return final_results
