import os
import sys
import pickle

import ot
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

import sys

def compute_uot_plan(X_source, X_target, delta=1, use_mini_batch_uot=False, group_number=5, draw=False):

    n_source, n_target = X_source.shape[0], X_target.shape[0] 
    norm_2_dist = ot.dist(X_source, X_target, metric='euclidean')

    cos_sq = np.cos(np.minimum(norm_2_dist / (2 * delta), np.pi/2))**2
    cost_matrix = -np.log(np.where(cos_sq == 0, 1e-10, cos_sq))


    if not use_mini_batch_uot:
        a = np.ones(n_source)
        b = np.ones(n_target)
        G = ot.unbalanced.mm_unbalanced(a, b, cost_matrix, reg_m=[1.0,1.0])
    else:
        # Mini-batch UOT computation
        a = np.ones(n_source)
        b = np.ones(n_target)
        G = np.zeros((n_source, n_target))

        # 先打乱索引
        source_perm = np.arange(n_source)
        np.random.shuffle(source_perm)
        target_perm = np.arange(n_target)
        np.random.shuffle(target_perm)

        # 然后随机划分
        source_indices = np.array_split(source_perm, group_number)
        target_indices = np.array_split(target_perm, group_number)

        # for src_idx in source_indices:
        #     for tgt_idx in target_indices:
        for src_idx,tgt_idx in zip(source_indices,target_indices):
                sub_cost_matrix = cost_matrix[np.ix_(src_idx, tgt_idx)]
                sub_a = a[src_idx]
                sub_b = b[tgt_idx]
                G_sub = ot.unbalanced.mm_unbalanced(sub_a, sub_b, sub_cost_matrix, reg_m=[1.0,1.0])
                G[np.ix_(src_idx, tgt_idx)] = G_sub


    gamma0_plan = ((a / G.sum(1))[:, None]) * G
    gamma1_plan = (b/G.sum(0))*G


    print(((gamma1_plan- gamma0_plan)<0.0).any())
    
    if draw:
        source_pred = G.sum(1)
        tar_pred = G.sum(0)
        
        fig = plt.figure(figsize=(15, 5))

        plt.subplot(131)
        plt.plot(a, label = f'source_true')
        plt.plot(source_pred, label = f'source_pred')
        plt.legend()

        plt.subplot(132)
        plt.plot(b, label = f'target_true')
        plt.plot(tar_pred, label = f'target_pred')
        plt.legend()
        
        plt.subplot(133)
        plt.scatter(X_source[:,0],X_source[:,1],s=source_pred*10, alpha=0.5)
        plt.show()


    return G, gamma0_plan, gamma1_plan

def pre_compute_wfr_ot(adata_control, 
                       adata_treated, 
                       save_path, 
                       sample_rep='X_pca', 
                       condition_keys="target_gene",
                       delta = 1,
                       use_mini_batch_uot = True, 
                       group_number=5):
    
    uot_plans = []
    gamma0_plans = []
    gamma1_plans = []
    treat_obs_names = []

    X_control = adata_control.obsm[sample_rep]

    all_conditions = adata_treated.obs[condition_keys].unique().tolist()
    for cur_condition in tqdm(all_conditions, desc="Computing UOT plans..."):
        print(f"Pre-computing WFR-OT for condition: {cur_condition}")
        adata_treat_cur = adata_treated[adata_treated.obs[condition_keys]==cur_condition]
        X_treat_cur = adata_treat_cur.obsm[sample_rep]
        gamma_plan, gamma0_plan, gamma1_plan = compute_uot_plan(X_control, 
                                                       X_treat_cur, 
                                                       delta=delta, 
                                                       use_mini_batch_uot=use_mini_batch_uot, 
                                                       group_number=group_number, 
                                                       draw=False)
        uot_plans.append(gamma_plan)
        gamma0_plans.append(gamma0_plan)
        gamma1_plans.append(gamma1_plan)
        treat_obs_names.append(adata_treat_cur.obs_names.to_numpy())
    
    control_obs_name = adata_control.obs_names.to_numpy()

    ot_results = {"all_conditions": all_conditions,
                "control_obs_names": control_obs_name,
                "treat_obs_names": treat_obs_names,
                "uot_plans": uot_plans,
                "gamma0_plans": gamma0_plans,
                "gamma1_plans": gamma1_plans,
                "delta": delta}
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(ot_results, f)
    
    return ot_results