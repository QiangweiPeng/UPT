import torch
from tqdm import tqdm
import pickle
import logging

import numpy as np

from .data import get_batch


# 这个函数是干啥的，似乎用不到
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


def train_model(adata_control, adata_treated, adata_test,
                ot_results_train, ot_results_test,
                model,
                optimizer,
                scheduler = None,
                reorder_ot_results = False, #这个是干啥的
                n_iterations=10000,
                batch_size_per_condition = 256,
                batch_size_condition = 10,
                sample_rep = "X_pca_scaled", # X_ae; X_state
                condition_rep_keys = "gene_embeddings",
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                save_path=None,
                eval_interval = 10):
    
    
    if reorder_ot_results: 
        ot_results_train = reorder_ot_results(ot_results_train, adata_control, adata_treated, condition_keys="target_gene")
        ot_results_test = reorder_ot_results(ot_results_test, adata_control, adata_test, condition_keys = "target_gene")
    
    gamma0_plans = ot_results_train["gamma0_plans"]
    gamma1_plans = ot_results_train["gamma1_plans"]
    delta = ot_results_train["delta"]
    all_conditions = ot_results_train["all_conditions"]

    gamma0_plans_test = ot_results_test["gamma0_plans"]
    gamma1_plans_test = ot_results_test["gamma1_plans"]
    delta_test = ot_results_test["delta"]
    all_conditions_test = ot_results_test["all_conditions"]

    
    progress_bar = tqdm(range(n_iterations), desc="Begin flow and growth matching...", unit="epoch")
    vloss_list = []
    gloss_list = []
    loss_list = []
    test_loss_list = []

    for i in progress_bar:
        model.train()
        optimizer.zero_grad()
        t, xt, ut, gt, masst, cons = get_batch(adata_control, adata_treated, all_conditions,
                                              batch_size_per_condition, batch_size_condition,
                                              gamma0_plans, gamma1_plans, delta,
                                              sample_rep=sample_rep, condition_rep_keys=condition_rep_keys)


        vt, gt_pred = model(t, xt, cons)

        vloss = torch.mean((vt - ut)**2 * masst)
        vloss_list.append(vloss.item())
        gloss = torch.mean((gt_pred - gt)**2 * masst)
        gloss_list.append(gloss.item())
        loss = vloss + gloss
        loss_list.append(loss.item())


        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()


        if (i+1) % eval_interval == 0: # 10轮跑一下test
            model.eval()
            with torch.no_grad():
                if ot_results_test is not None:
                    t_test, xt_test, ut_test, gt_test, masst_test, cons_test = get_batch(
                        adata_control, adata_test, all_conditions_test,
                        batch_size_per_condition, batch_size_condition,
                        gamma0_plans_test, gamma1_plans_test, delta_test,
                        sample_rep=sample_rep, condition_rep_keys=condition_rep_keys)
                    vt_test, gt_pred_test = model(t_test, xt_test, cons_test)
                    test_vloss = torch.mean((vt_test - ut_test)**2 * masst_test)
                    test_gloss = torch.mean((gt_pred_test - gt_test)**2 * masst_test)
                    test_loss = test_vloss + test_gloss
                    test_loss_list.append(test_loss.item())
                    logging.info(f"Epoch {i}: loss={loss.item():.3f}, vloss={vloss.item():.3f}, gloss={gloss.item():.3f}, test_loss={test_loss.item():.3f}")
                    progress_bar.set_postfix({"loss": f"{loss.item():.3f}","vloss": f"{vloss.item():.3f}", 
                                              "gloss": f"{gloss.item():.3f}", "test_loss": f"{test_loss.item():.3f}"})
        else:
            logging.info(f"Epoch {i}: loss={loss.item():.3f}, vloss={vloss.item():.3f}, gloss={gloss.item():.3f}")
            progress_bar.set_postfix({"loss": f"{loss.item():.3f}","vloss": f"{vloss.item():.3f}", "gloss": f"{gloss.item():.3f}"})

        if (i+1) % 5000 == 0:
            if save_path is not None:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'vloss_list': vloss_list,
                    'gloss_list': gloss_list,
                    'loss_list': loss_list,
                    'test_loss_list': test_loss_list,
                }, f"{save_path}_epoch_{i+1}.pt")
                logging.info(f"Model and training state saved to {save_path}_epoch_{i+1}.pt")