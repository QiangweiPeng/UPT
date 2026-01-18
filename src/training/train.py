import torch
from tqdm import tqdm
import pickle
import logging

import numpy as np

from .data import get_batch


def reorder_ot_results(ot_results, adata_control, adata_treated, condition_keys="target_gene"):
    """Reorder precomputed OT results to match the (possibly) new obs_names order."""

    # 当前的 control/treat 索引
    new_control_names = adata_control.obs_names.to_numpy()
    all_conditions = ot_results["all_conditions"]

    # 建立 control 索引映射：旧 -> 新的顺序
    old_control_names = ot_results["control_obs_names"]
    control_index_map = {name: i for i, name in enumerate(old_control_names)}
    reorder_control_idx = np.array([control_index_map[name] for name in new_control_names])

    new_uot_plans = []
    new_gamma0_plans = []
    new_gamma1_plans = []
    new_treat_obs_names = []

    # 针对每个 condition 分别处理
    for cond, uot, g0, g1, old_treat_names in zip(
        all_conditions, ot_results["uot_plans"], ot_results["gamma0_plans"], ot_results["gamma1_plans"], ot_results["treat_obs_names"]
    ):
        # 找到当前 condition 对应的 adata_treated 子集
        adata_treat_cur = adata_treated[adata_treated.obs[condition_keys] == cond]
        new_treat_names = adata_treat_cur.obs_names.to_numpy()

        # 建立 treat 索引映射：旧 -> 新的顺序
        treat_index_map = {name: i for i, name in enumerate(old_treat_names)}
        reorder_treat_idx = np.array([treat_index_map[name] for name in new_treat_names])

        # 重新排序矩阵
        uot_new = uot[np.ix_(reorder_control_idx, reorder_treat_idx)]
        g0_new = g0[reorder_control_idx]
        g1_new = g1[reorder_treat_idx]

        new_uot_plans.append(uot_new)
        new_gamma0_plans.append(g0_new)
        new_gamma1_plans.append(g1_new)
        new_treat_obs_names.append(new_treat_names)

    # 返回新的对齐结果
    reordered_ot_results = {
        "all_conditions": all_conditions,
        "control_obs_names": new_control_names,
        "treat_obs_names": new_treat_obs_names,
        "uot_plans": new_uot_plans,
        "gamma0_plans": new_gamma0_plans,
        "gamma1_plans": new_gamma1_plans,
        "delta": ot_results["delta"],
    }

    return reordered_ot_results


def train_model(adata_control,
          adata_treated,
          ot_results,
          model,
          optimizer,
          reorder_ot_results = False,
          n_iterations=10000,
          batch_size_per_condition = 256,
          batch_size_condition = 10,
          sample_rep = "X_pca_scaled", # X_ae; X_state
          condition_rep_keys = "gene_embeddings",
          device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          save_path=None,):
    
    
    if reorder_ot_results:
        ot_results = reorder_ot_results(ot_results, adata_control, adata_treated, condition_keys="target_gene")
    
    gamma0_plans = ot_results["gamma0_plans"]
    gamma1_plans = ot_results["gamma1_plans"]
    delta = ot_results["delta"]
    all_conditions = ot_results["all_conditions"]

    
    progress_bar = tqdm(range(n_iterations), desc="Begin flow and growth matching...", unit="epoch")
    vloss_list = []
    gloss_list = []
    loss_list = []

    for i in progress_bar:
        optimizer.zero_grad()
        t, xt, ut, gt, masst, cons = get_batch(adata_control, adata_treated, all_conditions,
                                              batch_size_per_condition, batch_size_condition,
                                              gamma0_plans, gamma1_plans, delta,
                                              sample_rep=sample_rep, condition_rep_keys=condition_rep_keys)
        
        # print(t.shape)
        # print(xt.shape)
        # print(ut.shape)
        # print(gt.shape)
        # print(masst.shape)
        # print(cons.shape)

        c = model.condition_encoder(cons)
        vt = model.v_net(t,xt,c)
        gt_pred = model.g_net(t,xt,c)

        vloss = torch.mean((vt - ut)**2 * masst)
        vloss_list.append(vloss.item())
        gloss = torch.mean((gt_pred - gt)**2 * masst)
        gloss_list.append(gloss.item())
        loss = vloss + gloss
        loss_list.append(loss.item())
        
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(v.parameters(), max_norm=0.1)
        # torch.nn.utils.clip_grad_norm_(g.parameters(), max_norm=0.1)

        optimizer.step()
        logging.info(f"Epoch {i}: loss={loss.item():.6f}, vloss={vloss.item():.6f}, gloss={gloss.item():.6f}")
        progress_bar.set_postfix({"loss": f"{loss.item():.6f}","vloss": f"{vloss.item():.6f}", "gloss": f"{gloss.item():.6f}"})