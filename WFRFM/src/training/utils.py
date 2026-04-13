import os
import sys
import pickle
import gc  
import numpy as np
import torch
import ot
import pandas as pd
from scipy import sparse
from tqdm import tqdm
import random

    
def _get_normalized_mass(adata):
    if "normalized_m" in adata.uns:
        return adata.uns["normalized_m"]
    return 1


def _resolve_marginal_time(marginal_value, marginal_time_map=None):
    if marginal_time_map is not None:
        if marginal_value not in marginal_time_map:
            raise KeyError(f"Missing time mapping for marginal value: {marginal_value}")
        return float(marginal_time_map[marginal_value])

    if isinstance(marginal_value, (int, float, np.integer, np.floating)):
        return float(marginal_value)

    raise ValueError(
        "marginal values are not numeric. Please provide marginal_time_map."
    )


def _get_ordered_marginals(available_values, marginal_order=None):
    available_values = list(available_values)
    if marginal_order is not None:
        available_set = set(available_values)
        ordered = [value for value in marginal_order if value in available_set]
        missing = available_set.difference(ordered)
        if missing:
            raise ValueError(
                f"marginal_order is missing values present in the data: {sorted(missing)}"
            )
        return ordered

    try:
        return sorted(available_values)
    except TypeError:
        return available_values


def _build_sparse_plan(plan, threshold):
    if sparse.issparse(plan):
        plan = plan.tocsr(copy=True)
        if threshold > 0:
            plan.data[plan.data < threshold] = 0
            plan.eliminate_zeros()
        return plan
    return sparse.csr_matrix(plan * (plan >= threshold))


def _compute_gamma_marginals(plan, m_source, m_target):
    if sparse.issparse(plan):
        plan = plan.tocsr()
        row_sum = np.asarray(plan.sum(axis=1)).reshape(-1)
        col_sum = np.asarray(plan.sum(axis=0)).reshape(-1)

        row_safe = row_sum.copy()
        col_safe = col_sum.copy()
        row_safe[row_safe == 0] = 1.0
        col_safe[col_safe == 0] = 1.0

        row_scale = (np.full(plan.shape[0], m_source, dtype=np.float32) / row_safe).astype(np.float32)
        col_scale = (np.full(plan.shape[1], m_target, dtype=np.float32) / col_safe).astype(np.float32)

        gamma0 = plan.multiply(row_scale[:, None]).tocsr()
        gamma1 = plan.tocsc().multiply(col_scale[None, :]).tocsr()
        return gamma0, gamma1, row_sum, col_sum

    a_np = np.full((plan.shape[0],), m_source, dtype=np.float32)
    b_np = np.full((plan.shape[1],), m_target, dtype=np.float32)
    sum_1 = plan.sum(axis=1, keepdims=True)
    sum_0 = plan.sum(axis=0, keepdims=True)
    sum_1[sum_1 == 0] = 1.0
    sum_0[sum_0 == 0] = 1.0
    gamma0 = (a_np.reshape(-1, 1) / sum_1) * plan
    gamma1 = (b_np.reshape(1, -1) / sum_0) * plan
    return gamma0, gamma1, sum_1.squeeze(1), sum_0.squeeze(0)


def compute_uot_plan_gpu(
    X_source,
    X_target,
    m_source=1,
    m_target=1,
    delta=1,
    reg_m=1,
    use_mini_batch_uot=False,
    group_number=5,
    sparse_threshold=1e-6,
    draw=False,
):
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

            limit = torch.tensor(np.pi / 2, device=device)
            term = torch.clamp(dist / (2 * delta), max=limit)
            cos_sq = torch.cos(term) ** 2

            epsilon = torch.tensor(1e-10, device=device)
            return -torch.log(torch.where(cos_sq == 0, epsilon, cos_sq))

        a = torch.full((n_source,), m_source, device=device, dtype=torch.float)
        b = torch.full((n_target,), m_target, device=device, dtype=torch.float)

        if not use_mini_batch_uot:
            cost_matrix = get_cost_batch(X_source, X_target)
            G = ot.unbalanced.mm_unbalanced(a, b, cost_matrix, reg_m=reg_m, numItermax=1000)
            res_G = G.cpu().numpy()
            del cost_matrix, G
        else:
            row_parts = []
            col_parts = []
            data_parts = []

            source_perm = torch.randperm(n_source, device=device)
            target_perm = torch.randperm(n_target, device=device)

            group_number = max(1, int(group_number))
            source_indices = torch.tensor_split(source_perm, group_number)
            target_indices = torch.tensor_split(target_perm, group_number)

            for src_idx, tgt_idx in zip(source_indices, target_indices):
                if src_idx.numel() == 0 or tgt_idx.numel() == 0:
                    continue

                sub_a = a[src_idx]
                sub_b = b[tgt_idx]
                sub_x_s = X_source[src_idx]
                sub_x_t = X_target[tgt_idx]
                sub_cost_matrix = get_cost_batch(sub_x_s, sub_x_t)

                G_sub = ot.unbalanced.mm_unbalanced(
                    sub_a,
                    sub_b,
                    sub_cost_matrix,
                    reg_m=reg_m,
                    numItermax=1000,
                )

                src_idx_cpu = src_idx.cpu().numpy()
                tgt_idx_cpu = tgt_idx.cpu().numpy()
                G_sub_np = G_sub.cpu().numpy().astype(np.float32, copy=False)
                keep = G_sub_np >= sparse_threshold
                if np.any(keep):
                    sub_rows, sub_cols = np.nonzero(keep)
                    row_parts.append(src_idx_cpu[sub_rows])
                    col_parts.append(tgt_idx_cpu[sub_cols])
                    data_parts.append(G_sub_np[sub_rows, sub_cols])
                del sub_cost_matrix, G_sub, sub_x_s, sub_x_t
                del G_sub_np

            if data_parts:
                rows = np.concatenate(row_parts)
                cols = np.concatenate(col_parts)
                data = np.concatenate(data_parts)
            else:
                rows = np.array([], dtype=np.int64)
                cols = np.array([], dtype=np.int64)
                data = np.array([], dtype=np.float32)

            res_G = sparse.csr_matrix(
                (data, (rows, cols)),
                shape=(n_source, n_target),
                dtype=np.float32,
            )

        res_g0, res_g1, row_sum, col_sum = _compute_gamma_marginals(
            res_G,
            m_source=m_source,
            m_target=m_target,
        )


        if draw:
            import matplotlib.pyplot as plt

            sum1_np = row_sum
            sum0_np = col_sum
            a_np = np.full((n_source,), m_source, dtype=np.float32)
            b_np = np.full((n_target,), m_target, dtype=np.float32)

            fig = plt.figure(figsize=(15, 5))
            plt.subplot(121)
            plt.plot(a_np, label='Target (m_source)', color='blue', alpha=0.6)
            plt.plot(sum1_np, label='Pred (sum_i_give)', color='orange', alpha=0.6, linestyle='--')
            plt.title(f"Source Marginals (MSE: {np.mean((a_np - sum1_np)**2):.4f})")
            plt.legend()
        
            plt.subplot(122)
            plt.plot(b_np, label='Target (m_target)', color='blue', alpha=0.6)
            plt.plot(sum0_np, label='Pred (sum_j_receive)', color='orange', alpha=0.6, linestyle='--')
            plt.title(f"Target Marginals (MSE: {np.mean((b_np - sum0_np)**2):.4f})")
            plt.legend()

            plt.show()

    del X_source, X_target
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
                       batch_save_size=10,
                        draw = False): 
    """
    让ai写了一个断点重连逻辑
    另外把result中储存方式改成了字典
    """
    m_source = _get_normalized_mass(adata_control)
    m_target = _get_normalized_mass(adata_treated)
    
    X_control = adata_control.obsm[sample_rep]
    control_obs_names = adata_control.obs_names.to_numpy()
    all_conditions = adata_treated.obs[condition_keys].unique().tolist()
    threshold = 1e-6

    grouped_indices = adata_treated.obs.groupby(condition_keys).indices 
    X_treated_all = adata_treated.obsm[sample_rep] 
    obs_names_treated_all = adata_treated.obs_names.to_numpy()
    
    # 检查已完成的batch
    batch_dir = os.path.dirname(save_path) or '.'
    base_name = os.path.basename(save_path).replace('.pkl', '')
    completed_batches = []
    for f in os.listdir(batch_dir):
        if f.startswith(f"{base_name}_batch_") and f.endswith('.pkl'):
            batch_num = int(f.rsplit('_batch_', 1)[1].replace('.pkl', ''))
            completed_batches.append(batch_num)
    
    start_idx = max(completed_batches) if completed_batches else 0
    if start_idx > 0:
        print(f"从batch {start_idx}恢复，已完成 {start_idx}/{len(all_conditions)}")
    
    # 主循环 - 按batch处理
    for batch_start in range(start_idx, len(all_conditions), batch_save_size):
        batch_end = min(batch_start + batch_save_size, len(all_conditions))
        
        # 每个batch独立的结果字典
        batch_results = {
            "uot_plans": {},      # {condition: sparse_matrix}
            "gamma0_plans": {},   # {condition: sparse_matrix}
            "gamma1_plans": {},   # {condition: sparse_matrix}
            "treat_obs_names": {} # {condition: [obs_names]}
        }
        
        # 处理当前batch
        for i in tqdm(range(batch_start, batch_end)):
            cur_condition = all_conditions[i]
            indices = grouped_indices[cur_condition]
            X_treat_cur = X_treated_all[indices]
            obs_names_cur = obs_names_treated_all[indices]
            
            gamma, g0, g1 = compute_uot_plan_gpu(
                X_control, X_treat_cur, 
                m_source=m_source, m_target=m_target,
                delta=delta, reg_m = reg_m,
                use_mini_batch_uot=use_mini_batch_uot, 
                group_number=group_number,
                draw = draw
            )
            
            if hasattr(gamma, 'cpu'):
                gamma = gamma.cpu().numpy()
                g0 = g0.cpu().numpy()
                g1 = g1.cpu().numpy()
                torch.cuda.empty_cache()
            
            batch_results["uot_plans"][cur_condition] = _build_sparse_plan(gamma, threshold)
            batch_results["gamma0_plans"][cur_condition] = _build_sparse_plan(g0, threshold)
            batch_results["gamma1_plans"][cur_condition] = _build_sparse_plan(g1, threshold)
            batch_results["treat_obs_names"][cur_condition] = obs_names_cur
            
            del gamma, g0, g1, X_treat_cur

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
        "all_conditions": all_conditions, 
        "control_obs_names": control_obs_names,
        "delta": delta,
        "uot_plans": {},
        "gamma0_plans": {},
        "gamma1_plans": {},
        "treat_obs_names": {}
    }
    
    batch_files = sorted([f for f in os.listdir(batch_dir) 
                          if f.startswith(f"{base_name}_batch_")],
                         key=lambda x: int(x.rsplit('_batch_', 1)[1].replace('.pkl', '')))
    
    for batch_file in batch_files:
        with open(os.path.join(batch_dir, batch_file), 'rb') as f:
            batch_data = pickle.load(f)
        for key in ["uot_plans", "gamma0_plans", "gamma1_plans", "treat_obs_names"]:
            if key in batch_data:
                final_results[key].update(batch_data[key])
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


def pre_compute_wfr_ot_multi_marginal(
    adata_control,
    adata_treated,
    save_path,
    marginal_key,
    sample_rep='X_pca_scaled',
    condition_keys="target_gene",
    marginal_order=None,
    marginal_time_map=None,
    control_time=0.0,
    delta=1,
    reg_m=1,
    use_mini_batch_uot=True,
    group_number=5,
    batch_save_size=10,
    draw=False,
):
    """
    multi-marginal 版本的 OT 预计算。

    对每个 condition，会按照 marginal_order / marginal_time_map 指定的顺序，
    计算 control -> marginal_1 -> marginal_2 -> ... 的相邻 OT plan。
    """

    if marginal_key not in adata_treated.obs:
        raise KeyError(f"{marginal_key} is not found in adata_treated.obs")
    if condition_keys not in adata_treated.obs:
        raise KeyError(f"{condition_keys} is not found in adata_treated.obs")

    m_control = _get_normalized_mass(adata_control)
    m_treated = _get_normalized_mass(adata_treated)

    X_control = adata_control.obsm[sample_rep]
    control_obs_names = adata_control.obs_names.to_numpy()
    X_treated_all = adata_treated.obsm[sample_rep]
    treated_obs_names_all = adata_treated.obs_names.to_numpy()
    threshold = 1e-6

    all_conditions = adata_treated.obs[condition_keys].unique().tolist()
    ordered_marginals = _get_ordered_marginals(
        adata_treated.obs[marginal_key].unique().tolist(),
        marginal_order=marginal_order,
    )
    grouped_indices = adata_treated.obs.groupby(
        [condition_keys, marginal_key],
        observed=True,
    ).indices

    batch_dir = os.path.dirname(save_path) or '.'
    base_name = os.path.basename(save_path).replace('.pkl', '')
    completed_batches = []
    for f in os.listdir(batch_dir):
        if f.startswith(f"{base_name}_batch_") and f.endswith('.pkl'):
            batch_num = int(f.rsplit('_batch_', 1)[1].replace('.pkl', ''))
            completed_batches.append(batch_num)

    start_idx = max(completed_batches) if completed_batches else 0
    if start_idx > 0:
        print(f"从batch {start_idx}恢复，已完成 {start_idx}/{len(all_conditions)}")

    for batch_start in range(start_idx, len(all_conditions), batch_save_size):
        batch_end = min(batch_start + batch_save_size, len(all_conditions))
        batch_results = {
            "pair_records": [],
            "condition_to_pair_ids": {},
            "uot_plans": {},
            "gamma0_plans": {},
            "gamma1_plans": {},
        }

        for i in tqdm(range(batch_start, batch_end)):
            cur_condition = all_conditions[i]
            available_marginals = [
                marginal_value
                for marginal_value in ordered_marginals
                if (cur_condition, marginal_value) in grouped_indices
            ]

            if not available_marginals:
                batch_results["condition_to_pair_ids"][cur_condition] = []
                continue

            prev_dataset = "control"
            prev_obs_names = control_obs_names
            prev_X = X_control
            prev_time = float(control_time)
            prev_mass = m_control
            prev_marginal = None
            condition_pair_ids = []

            for pair_idx, cur_marginal in enumerate(available_marginals):
                target_time = _resolve_marginal_time(cur_marginal, marginal_time_map)
                if target_time <= prev_time:
                    raise ValueError(
                        f"Non-increasing marginal time for condition={cur_condition}: "
                        f"{prev_time} -> {target_time}"
                    )

                indices = grouped_indices[(cur_condition, cur_marginal)]
                X_treat_cur = X_treated_all[indices]
                obs_names_cur = treated_obs_names_all[indices]

                gamma, g0, g1 = compute_uot_plan_gpu(
                    prev_X,
                    X_treat_cur,
                    m_source=prev_mass,
                    m_target=m_treated,
                    delta=delta,
                    reg_m=reg_m,
                    use_mini_batch_uot=use_mini_batch_uot,
                    group_number=group_number,
                    draw=draw,
                )

                if hasattr(gamma, 'cpu'):
                    gamma = gamma.cpu().numpy()
                    g0 = g0.cpu().numpy()
                    g1 = g1.cpu().numpy()
                    torch.cuda.empty_cache()

                pair_id = f"cond_{i:05d}_pair_{pair_idx:03d}"
                batch_results["pair_records"].append(
                    {
                        "pair_id": pair_id,
                        "condition": cur_condition,
                        "source_dataset": prev_dataset,
                        "target_dataset": "treated",
                        "source_marginal": prev_marginal,
                        "target_marginal": cur_marginal,
                        "source_time": prev_time,
                        "target_time": target_time,
                        "delta_t": target_time - prev_time,
                        "source_obs_names": prev_obs_names,
                        "target_obs_names": obs_names_cur,
                    }
                )
                batch_results["uot_plans"][pair_id] = _build_sparse_plan(gamma, threshold)
                batch_results["gamma0_plans"][pair_id] = _build_sparse_plan(g0, threshold)
                batch_results["gamma1_plans"][pair_id] = _build_sparse_plan(g1, threshold)
                condition_pair_ids.append(pair_id)

                prev_dataset = "treated"
                prev_obs_names = obs_names_cur
                prev_X = X_treat_cur
                prev_time = target_time
                prev_mass = m_treated
                prev_marginal = cur_marginal

                del gamma, g0, g1, X_treat_cur

            batch_results["condition_to_pair_ids"][cur_condition] = condition_pair_ids

        gc.collect()

        batch_path = os.path.join(batch_dir, f"{base_name}_batch_{batch_end}.pkl")
        with open(batch_path, 'wb') as f:
            pickle.dump(batch_results, f)
        print(f"Batch saved: {batch_end}/{len(all_conditions)}")

        del batch_results
        gc.collect()

    print("合并所有batch...")
    final_results = {
        "multi_marginal": True,
        "all_conditions": all_conditions,
        "condition_keys": condition_keys,
        "marginal_key": marginal_key,
        "ordered_marginals": ordered_marginals,
        "control_time": float(control_time),
        "delta": delta,
        "control_obs_names": control_obs_names,
        "pair_records": [],
        "condition_to_pair_ids": {},
        "uot_plans": {},
        "gamma0_plans": {},
        "gamma1_plans": {},
    }

    batch_files = sorted(
        [
            f for f in os.listdir(batch_dir)
            if f.startswith(f"{base_name}_batch_") and f.endswith('.pkl')
        ],
        key=lambda x: int(x.rsplit('_batch_', 1)[1].replace('.pkl', '')),
    )

    for batch_file in batch_files:
        with open(os.path.join(batch_dir, batch_file), 'rb') as f:
            batch_data = pickle.load(f)
        final_results["pair_records"].extend(batch_data.get("pair_records", []))
        final_results["condition_to_pair_ids"].update(
            batch_data.get("condition_to_pair_ids", {})
        )
        for key in ["uot_plans", "gamma0_plans", "gamma1_plans"]:
            final_results[key].update(batch_data.get(key, {}))
        del batch_data
        gc.collect()

    with open(save_path, 'wb') as f:
        pickle.dump(final_results, f)

    for batch_file in batch_files:
        os.remove(os.path.join(batch_dir, batch_file))

    print(f"完成！最终结果: {save_path}")
    return final_results


def _build_condition_embedding_map_from_adatas(
    adata_control,
    adata_treated,
    condition_keys,
    condition_rep_key,
):
    condition_emb_map = {}

    if condition_rep_key in adata_treated.obsm:
        treated_conditions = adata_treated.obs[condition_keys].astype(str).to_numpy()
        unique_treated = pd.unique(treated_conditions)
        for condition in unique_treated:
            idx = np.where(treated_conditions == condition)[0][0]
            condition_emb_map[str(condition)] = np.asarray(
                adata_treated.obsm[condition_rep_key][idx],
                dtype=np.float32,
            )

    if condition_rep_key in adata_control.obsm and condition_keys in adata_control.obs:
        control_conditions = adata_control.obs[condition_keys].astype(str).to_numpy()
        unique_control = pd.unique(control_conditions)
        for condition in unique_control:
            if str(condition) in condition_emb_map:
                continue
            idx = np.where(control_conditions == condition)[0][0]
            condition_emb_map[str(condition)] = np.asarray(
                adata_control.obsm[condition_rep_key][idx],
                dtype=np.float32,
            )

    return condition_emb_map


def pre_compute_wfr_ot_multi_marginal_independent_real_time(
    adata_control,
    adata_treated,
    save_path,
    marginal_key,
    sample_rep='X_pca_scaled',
    condition_keys="target_gene",
    condition_rep_key="gene_embeddings",
    marginal_order=None,
    marginal_time_map=None,
    delta=1,
    reg_m=1,
    use_mini_batch_uot=True,
    group_number=5,
    batch_save_size=10,
    draw=False,
):
    """
    Independent real-time multi-marginal OT for zebrafish.

    - control 轨迹只在 control 时间点内部连接
    - perturbation 轨迹只在各自 condition 的真实观测时间点内部连接
    - 所有 pair 都使用真实 source_time / target_time
    """
    if marginal_key not in adata_treated.obs:
        raise KeyError(f"{marginal_key} is not found in adata_treated.obs")
    if marginal_key not in adata_control.obs:
        raise KeyError(f"{marginal_key} is not found in adata_control.obs")
    if condition_keys not in adata_treated.obs:
        raise KeyError(f"{condition_keys} is not found in adata_treated.obs")
    if condition_keys not in adata_control.obs:
        raise KeyError(f"{condition_keys} is not found in adata_control.obs")

    threshold = 1e-6
    m_control = _get_normalized_mass(adata_control)
    m_treated = _get_normalized_mass(adata_treated)

    X_control_all = adata_control.obsm[sample_rep]
    X_treated_all = adata_treated.obsm[sample_rep]
    control_obs_names_all = adata_control.obs_names.to_numpy()
    treated_obs_names_all = adata_treated.obs_names.to_numpy()

    ordered_marginals = _get_ordered_marginals(
        list(
            set(adata_control.obs[marginal_key].unique().tolist())
            | set(adata_treated.obs[marginal_key].unique().tolist())
        ),
        marginal_order=marginal_order,
    )

    control_grouped_indices = adata_control.obs.groupby(
        marginal_key,
        observed=True,
    ).indices
    treated_grouped_indices = adata_treated.obs.groupby(
        [condition_keys, marginal_key],
        observed=True,
    ).indices

    control_condition_values = adata_control.obs[condition_keys].astype(str).unique().tolist()
    if len(control_condition_values) != 1:
        raise ValueError(
            "Independent real-time OT expects exactly one control condition label, got "
            + ", ".join(control_condition_values[:10])
        )
    control_condition = str(control_condition_values[0])

    condition_emb_map = _build_condition_embedding_map_from_adatas(
        adata_control=adata_control,
        adata_treated=adata_treated,
        condition_keys=condition_keys,
        condition_rep_key=condition_rep_key,
    )
    if control_condition not in condition_emb_map:
        raise KeyError(
            f"Missing control condition embedding for {control_condition}. "
            f"Expected {condition_rep_key} in adata_control.obsm or adata_treated.obsm."
        )

    perturbation_conditions = adata_treated.obs[condition_keys].astype(str).unique().tolist()
    all_conditions = [control_condition] + perturbation_conditions

    batch_dir = os.path.dirname(save_path) or '.'
    base_name = os.path.basename(save_path).replace('.pkl', '')
    completed_batches = []
    for f in os.listdir(batch_dir):
        if f.startswith(f"{base_name}_batch_") and f.endswith('.pkl'):
            batch_num = int(f.rsplit('_batch_', 1)[1].replace('.pkl', ''))
            completed_batches.append(batch_num)

    start_idx = max(completed_batches) if completed_batches else 0
    if start_idx > 0:
        print(f"从batch {start_idx}恢复，已完成 {start_idx}/{len(all_conditions)}")

    def _compute_pair_plan(X_source, X_target, m_source, m_target):
        gamma, g0, g1 = compute_uot_plan_gpu(
            X_source,
            X_target,
            m_source=m_source,
            m_target=m_target,
            delta=delta,
            reg_m=reg_m,
            use_mini_batch_uot=use_mini_batch_uot,
            group_number=group_number,
            draw=draw,
        )
        if hasattr(gamma, 'cpu'):
            gamma = gamma.cpu().numpy()
            g0 = g0.cpu().numpy()
            g1 = g1.cpu().numpy()
            torch.cuda.empty_cache()
        return gamma, g0, g1

    for batch_start in range(start_idx, len(all_conditions), batch_save_size):
        batch_end = min(batch_start + batch_save_size, len(all_conditions))
        batch_results = {
            "pair_records": [],
            "condition_to_pair_ids": {},
            "uot_plans": {},
            "gamma0_plans": {},
            "gamma1_plans": {},
        }

        for i in tqdm(range(batch_start, batch_end)):
            cur_condition = str(all_conditions[i])
            condition_pair_ids = []

            if cur_condition == control_condition:
                available_marginals = [
                    marginal_value
                    for marginal_value in ordered_marginals
                    if marginal_value in control_grouped_indices
                ]
                if len(available_marginals) < 2:
                    batch_results["condition_to_pair_ids"][cur_condition] = []
                    continue

                for pair_idx, (source_marginal, target_marginal) in enumerate(
                    zip(available_marginals[:-1], available_marginals[1:])
                ):
                    source_time = _resolve_marginal_time(source_marginal, marginal_time_map)
                    target_time = _resolve_marginal_time(target_marginal, marginal_time_map)
                    if target_time <= source_time:
                        raise ValueError(
                            f"Non-increasing control marginal time: {source_time} -> {target_time}"
                        )

                    source_indices = control_grouped_indices[source_marginal]
                    target_indices = control_grouped_indices[target_marginal]
                    X_source = X_control_all[source_indices]
                    X_target = X_control_all[target_indices]
                    source_obs_names = control_obs_names_all[source_indices]
                    target_obs_names = control_obs_names_all[target_indices]

                    gamma, g0, g1 = _compute_pair_plan(
                        X_source=X_source,
                        X_target=X_target,
                        m_source=m_control,
                        m_target=m_control,
                    )

                    pair_id = f"control_pair_{pair_idx:03d}"
                    batch_results["pair_records"].append(
                        {
                            "pair_id": pair_id,
                            "condition": cur_condition,
                            "source_dataset": "control",
                            "target_dataset": "control",
                            "source_marginal": source_marginal,
                            "target_marginal": target_marginal,
                            "source_time": source_time,
                            "target_time": target_time,
                            "delta_t": target_time - source_time,
                            "source_obs_names": source_obs_names,
                            "target_obs_names": target_obs_names,
                        }
                    )
                    batch_results["uot_plans"][pair_id] = _build_sparse_plan(gamma, threshold)
                    batch_results["gamma0_plans"][pair_id] = _build_sparse_plan(g0, threshold)
                    batch_results["gamma1_plans"][pair_id] = _build_sparse_plan(g1, threshold)
                    condition_pair_ids.append(pair_id)
                    del gamma, g0, g1, X_source, X_target
            else:
                available_marginals = [
                    marginal_value
                    for marginal_value in ordered_marginals
                    if (cur_condition, marginal_value) in treated_grouped_indices
                ]
                if len(available_marginals) < 2:
                    batch_results["condition_to_pair_ids"][cur_condition] = []
                    continue

                for pair_idx, (source_marginal, target_marginal) in enumerate(
                    zip(available_marginals[:-1], available_marginals[1:])
                ):
                    source_time = _resolve_marginal_time(source_marginal, marginal_time_map)
                    target_time = _resolve_marginal_time(target_marginal, marginal_time_map)
                    if target_time <= source_time:
                        raise ValueError(
                            f"Non-increasing perturbation marginal time for condition={cur_condition}: "
                            f"{source_time} -> {target_time}"
                        )

                    source_indices = treated_grouped_indices[(cur_condition, source_marginal)]
                    target_indices = treated_grouped_indices[(cur_condition, target_marginal)]
                    X_source = X_treated_all[source_indices]
                    X_target = X_treated_all[target_indices]
                    source_obs_names = treated_obs_names_all[source_indices]
                    target_obs_names = treated_obs_names_all[target_indices]

                    gamma, g0, g1 = _compute_pair_plan(
                        X_source=X_source,
                        X_target=X_target,
                        m_source=m_treated,
                        m_target=m_treated,
                    )

                    pair_id = f"cond_{i:05d}_pair_{pair_idx:03d}"
                    batch_results["pair_records"].append(
                        {
                            "pair_id": pair_id,
                            "condition": cur_condition,
                            "source_dataset": "treated",
                            "target_dataset": "treated",
                            "source_marginal": source_marginal,
                            "target_marginal": target_marginal,
                            "source_time": source_time,
                            "target_time": target_time,
                            "delta_t": target_time - source_time,
                            "source_obs_names": source_obs_names,
                            "target_obs_names": target_obs_names,
                        }
                    )
                    batch_results["uot_plans"][pair_id] = _build_sparse_plan(gamma, threshold)
                    batch_results["gamma0_plans"][pair_id] = _build_sparse_plan(g0, threshold)
                    batch_results["gamma1_plans"][pair_id] = _build_sparse_plan(g1, threshold)
                    condition_pair_ids.append(pair_id)
                    del gamma, g0, g1, X_source, X_target

            batch_results["condition_to_pair_ids"][cur_condition] = condition_pair_ids

        gc.collect()

        batch_path = os.path.join(batch_dir, f"{base_name}_batch_{batch_end}.pkl")
        with open(batch_path, 'wb') as f:
            pickle.dump(batch_results, f)
        print(f"Batch saved: {batch_end}/{len(all_conditions)}")

        del batch_results
        gc.collect()

    print("合并所有batch...")
    final_results = {
        "multi_marginal": True,
        "trajectory_mode": "independent_real_time",
        "all_conditions": all_conditions,
        "condition_keys": condition_keys,
        "marginal_key": marginal_key,
        "ordered_marginals": ordered_marginals,
        "control_condition": control_condition,
        "delta": delta,
        "condition_emb_map": condition_emb_map,
        "control_obs_names": control_obs_names_all,
        "pair_records": [],
        "condition_to_pair_ids": {},
        "uot_plans": {},
        "gamma0_plans": {},
        "gamma1_plans": {},
    }

    batch_files = sorted(
        [
            f for f in os.listdir(batch_dir)
            if f.startswith(f"{base_name}_batch_") and f.endswith('.pkl')
        ],
        key=lambda x: int(x.rsplit('_batch_', 1)[1].replace('.pkl', '')),
    )

    for batch_file in batch_files:
        with open(os.path.join(batch_dir, batch_file), 'rb') as f:
            batch_data = pickle.load(f)
        final_results["pair_records"].extend(batch_data.get("pair_records", []))
        final_results["condition_to_pair_ids"].update(
            batch_data.get("condition_to_pair_ids", {})
        )
        for key in ["uot_plans", "gamma0_plans", "gamma1_plans"]:
            final_results[key].update(batch_data.get(key, {}))
        del batch_data
        gc.collect()

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
