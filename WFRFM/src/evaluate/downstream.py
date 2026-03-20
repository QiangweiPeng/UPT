import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns


def check_population_shift(
    adata_train,
    adata_control,
    target_cytokines,
    cell_type_key="cell_type",
    perturb_key="cytokine",
    control_name="PBS",
    label_threshold=0.03
):
    # train: 保留原始 cytokine
    train_obs = adata_train.obs[[perturb_key, cell_type_key]].copy()
    train_obs["condition"] = train_obs[perturb_key].astype(str)

    # control: 全部设成 PBS
    control_obs = adata_control.obs[[cell_type_key]].copy()
    control_obs["condition"] = control_name

    # 合并
    combined_obs = pd.concat([
        train_obs[["condition", cell_type_key]],
        control_obs[["condition", cell_type_key]]
    ], axis=0)

    # target_cytokines 可能是 Categorical
    all_conditions = list(target_cytokines)
    all_conditions.append(control_name)
    all_conditions = list(dict.fromkeys(all_conditions))  # 去重

    # 只保留目标条件
    combined_obs = combined_obs[combined_obs["condition"].isin(all_conditions)]

    # 统计绝对数量
    counts = combined_obs.groupby(["condition", cell_type_key]).size().unstack(fill_value=0)

    # 按指定顺序重排
    counts = counts.reindex(all_conditions, fill_value=0)

    # 统计相对比例
    fractions = counts.div(counts.sum(axis=1), axis=0)

    print("=== 细胞绝对数量 ===")
    print(counts)

    print("\n=== 细胞相对比例 ===")
    print(fractions)

    fig, ax = plt.subplots(figsize=(12, 16))

    fractions.plot(kind="barh", stacked=True, ax=ax)
    
    ax.set_title("Cell Type Composition Shift")
    ax.set_xlabel("Fraction")
    ax.set_ylabel("Condition")
    
    label_threshold = 0.08
    
    for i, condition in enumerate(fractions.index):
        left = 0
        for cell_type in fractions.columns:
            frac_value = fractions.loc[condition, cell_type]
    
            if frac_value >= label_threshold:
                ax.text(
                    left + frac_value / 2,
                    i,
                    f"{frac_value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8
                )
            left += frac_value
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


    return counts, fractions



def extract_v_and_g_metrics(
    inference_results,
    z0_tensor,
    adata_source,
    cell_type_key="cell_type",
    n_neighbors=15,
    knn_weights="distance"
):
    """
    inference_results: dict
        每个 condition 对应一个结果字典，至少包含:
        - 'z_pred': [N_cells, embedding_dim]
        - 'm_pred': [N_cells] or [N_cells, 1]

    z0_tensor:
        输入给模型的初始状态 [N_cells, embedding_dim]，这里也作为 KNN reference 的 PCA 空间坐标

    adata_source:
        控制组 AnnData，用来提取 reference 的 cell type 标签。
        默认假设它和 z0_tensor 的顺序一一对应。

    cell_type_key:
        adata_source.obs 里的细胞类型列名

    n_neighbors:
        KNN 邻居数

    knn_weights:
        'uniform' 或 'distance'
    """
    metrics_list = []

    # reference embedding: 初始细胞在 PCA 空间的位置
    z0_np = z0_tensor.cpu().numpy() if hasattr(z0_tensor, "cpu") else np.asarray(z0_tensor)

    # reference labels
    ref_cell_types = adata_source.obs[cell_type_key].values

    # 训练 KNN 分类器：用初始 control/reference 细胞作为 reference
    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=knn_weights,
        metric="euclidean"
    )
    knn.fit(z0_np, ref_cell_types)

    for cond, res in inference_results.items():
        z_pred = res["z_pred"]
        m_pred = res["m_pred"].flatten()

        # 如果 z_pred 是 tensor，转 numpy
        z_pred_np = z_pred.cpu().numpy() if hasattr(z_pred, "cpu") else np.asarray(z_pred)

        # 1. 计算状态转变距离 v
        distance = np.linalg.norm(z_pred_np - z0_np, axis=1)

        # 2. 计算质量变化 g
        log2_mass = np.log2(m_pred + 1e-6)

        # 3. 用 KNN 在 PCA 空间里给预测后状态重新判 cell type
        pred_cell_type = knn.predict(z_pred_np)

        # 4. 可选：输出 KNN 置信度（最大类别概率）
        pred_prob = knn.predict_proba(z_pred_np)
        pred_confidence = pred_prob.max(axis=1)

        df = pd.DataFrame({
            "cell_id": adata_source.obs_names,
            "cell_type": adata_source.obs[cell_type_key].values,   # 原始/起始 cell type
            "pred_cell_type": pred_cell_type,                      # 新增：KNN 判定的预测后 cell type
            "pred_cell_type_conf": pred_confidence,                # 可选：KNN 置信度
            "condition": cond,
            "distance_v": distance,
            "log2_mass_g": log2_mass,
            "mass_m": m_pred
        })
        metrics_list.append(df)

    return pd.concat(metrics_list, ignore_index=True)



def plot_predicted_mass_by_celltype(df_metrics, condition_name, target_cell_types):
    """
    df_metrics: 是我们上一步用 extract_v_and_g_metrics 提取的 DataFrame
    condition_name: 比如 'IFN-beta'
    target_cell_types: 我们重点关注的细胞，比如 ['NKT', 'cDC', 'CD14 Mono', 'B Naive']
    """
    # 筛选特定条件和特定细胞类型的预测结果
    df_cond = df_metrics[(df_metrics['condition'] == condition_name) & 
                         (df_metrics['cell_type'].isin(target_cell_types))]
    
    plt.figure(figsize=(10, 6))
    
    # 画小提琴图展示 m_pred 的分布
    sns.violinplot(data=df_cond, x='cell_type', y='mass_m', 
                   inner="quartile", palette="Set2", order=target_cell_types)
    plt.xticks(rotation=45) 
    
    # 画一条 m=1 的红线（代表不增不减）
    plt.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Mass Conservation (m=1)')
    
        
    plt.title(f'Predicted Mass Dynamics ($m_{{pred}}$) under {condition_name} Perturbation', fontsize=16)
    plt.ylabel('Predicted Mass ($m_{pred}$)', fontsize=14)
    plt.xlabel('Cell Type', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_predicted_mass_with_true_label(df_metrics, condition_name, target_cell_types,adata_control, adata_full, cytokine_key="cytokine", cell_type_key="cell_type"):
    """
    df_metrics: 提取出的包含 m_pred 的 DataFrame
    condition_name: 当前扰动名称 (e.g., 'IL-11')
    target_cell_types: 关注的细胞类型列表
    adata_full: 包含 PBS 和所有扰动真实数据的 AnnData (adata_train / adata_test)
    """
    # 1. 筛选特定条件和特定细胞类型的预测结果
    df_cond = df_metrics[(df_metrics['condition'] == condition_name) & 
                         (df_metrics['pred_cell_type'].isin(target_cell_types))]
    
    if df_cond.empty:
        print(f"No prediction data for {condition_name}")
        return

    # 2. 计算真实的存活/扩增率 (True Mass Ratio)
    # 获取 PBS 的真实数量 (记得除以 6 作为期望基线)
    pbs_subset = adata_control
    pbs_counts = pbs_subset.obs[cell_type_key].value_counts() / 6.0
    
    # 获取当前 Perturbation 的真实数量
    pert_subset = adata_full[adata_full.obs[cytokine_key] == condition_name]
    pert_counts = pert_subset.obs[cell_type_key].value_counts()
    
    # 计算每种细胞的 True Mass (真实数量 / 期望数量)
    true_mass_dict = {}
    for ct in target_cell_types:
        expected_n = pbs_counts.get(ct, 0)
        actual_n = pert_counts.get(ct, 0)
        if expected_n > 0: # 过滤掉极少量的噪点细胞
            true_mass_dict[ct] = actual_n / expected_n
        else:
            true_mass_dict[ct] = np.nan # 数量太少，不计算真实变化
            
    # 3. 开始画图
    plt.figure(figsize=(12, 6)) # 稍微加宽一点，防止拥挤
    
    # 画小提琴图展示 m_pred 的分布
    ax = sns.violinplot(data=df_cond, x='pred_cell_type', y='mass_m', 
                        inner="quartile", palette="Set2", order=target_cell_types)
    plt.xticks(rotation=45, ha='right') 
    
    # 画一条 m=1 的红线（代表不增不减）
    plt.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Mass Conservation (m=1)')
    
    # 4. 叠加 True Mass (真实的星星标签)
    x_coords = np.arange(len(target_cell_types))
    y_coords = [true_mass_dict[ct] for ct in target_cell_types]
    
    plt.scatter(x_coords, y_coords, color='black', marker='*', s=200, zorder=5, 
                edgecolor='white', linewidth=1, label='True Absolute Change ($\star$)')
    
    # 画线连接这些星星，方便看出真实的趋势
    valid_idx = [i for i, y in enumerate(y_coords) if not np.isnan(y)]
    valid_x = [x_coords[i] for i in valid_idx]
    valid_y = [y_coords[i] for i in valid_idx]
    plt.plot(valid_x, valid_y, color='black', linestyle=':', linewidth=1.5, alpha=0.5, zorder=4)
        
    plt.title(f'Predicted vs True Mass Dynamics under {condition_name} Perturbation', fontsize=16)
    plt.ylabel('Mass Multiplier ($m$)', fontsize=14)
    plt.xlabel('Cell Type', fontsize=14)
    
    # 调整图例
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()