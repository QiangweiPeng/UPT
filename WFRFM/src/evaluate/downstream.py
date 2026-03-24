import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns


from .evals_new import _get_matched_control

def check_population_shift(
    adata_train,
    adata_control,
    condition_combined_key, 
    cell_type_key="cell_type",
    control_tuple_name="('PBS', 'control')" 
):
    """
    对比控制组与训练组的细胞群体分布漂移，全面支持 Tuple 标识。
    """
    # 提取 obs 并统一列名
    train_obs = adata_train.obs[[condition_combined_key, cell_type_key]].copy()
    control_obs = adata_control.obs[[condition_combined_key, cell_type_key]].copy()
    
    # 合并数据
    combined_obs = pd.concat([train_obs, control_obs], axis=0)
    
    # 统计绝对数量
    counts = combined_obs.groupby([condition_combined_key, cell_type_key]).size().unstack(fill_value=0)
    
    # 统计相对比例
    fractions = counts.div(counts.sum(axis=1), axis=0)

    print("=== 细胞绝对数量 ===")
    print(counts)
    print("\n=== 细胞相对比例 ===")
    print(fractions)

    # 绘图优化：简化过长的 Tuple 标签
    plot_labels = [str(idx) if len(str(idx)) < 25 else str(idx)[:22]+"..." for idx in fractions.index]
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(fractions) * 0.5))) 
    
    fractions.plot(kind="barh", stacked=True, ax=ax)
    ax.set_yticklabels(plot_labels) # 使用简化标签
    
    ax.set_title("Cell Type Composition Shift (Multi-Covariate)", fontsize=14)
    ax.set_xlabel("Fraction")
    ax.set_ylabel("Condition Tuple")
    
    label_threshold = 0.08
    
    for i, condition in enumerate(fractions.index):
        left = 0
        for cell_type in fractions.columns:
            frac_value = fractions.loc[condition, cell_type]
            if frac_value >= label_threshold:
                ax.text(
                    left + frac_value / 2, i,
                    f"{frac_value:.0%}",
                    ha="center", va="center", fontsize=8
                )
            left += frac_value
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    return counts, fractions



def extract_v_and_g_metrics(
    inference_results,
    adata_source,         # 传入包含 sample_rep 的 AnnData
    rulebook, 
    sample_rep="X_pca_scaled", # 新增：指明 latent 空间名字
    cell_type_key="cell_type",
    n_neighbors=15,
    knn_weights="distance",
    use_groupwise_control=True # 新增：兼容精准 Control 匹配
):
    """
    提取预测结果的速度(v)和质量(g)变化，并将 Tuple 字符串拆解为独立特征列。
    """
    metrics_list = []

    # 1. 训练全局 KNN 分类器 (用全局数据训练，视野最广)
    z0_global_np = adata_source.obsm[sample_rep]
    ref_cell_types_global = adata_source.obs[cell_type_key].values
    
    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=knn_weights,
        metric="euclidean"
    )
    knn.fit(z0_global_np, ref_cell_types_global)

    for cond_tuple_str, res in inference_results.items():
        z_pred = res["z_pred"]
        m_pred = res["m_pred"].flatten()
        z_pred_np = z_pred.cpu().numpy() if hasattr(z_pred, "cpu") else np.asarray(z_pred)

        # ---------------------------------------------------------
        # 【核心修复】：获取与当前 z_pred 精准匹配的起始细胞 z0
        # ---------------------------------------------------------
        # 方案 A：如果在 run_batch_inference 已经存了，直接用
        if'z0_used' in res and'obs_names' in res:
            z0_np = res['z0_used']
            curr_obs_names = res['obs_names']
            # 从全局拿到对应的 origin cell types
            curr_origin_ct = adata_source[curr_obs_names].obs[cell_type_key].values
        
        # 方案 B：如果没存，使用法典动态复原切片
        else:
            curr_ctrl = _get_matched_control(adata_source, cond_tuple_str, rulebook, use_groupwise_control)
            z0_np = curr_ctrl.obsm[sample_rep]
            curr_obs_names = curr_ctrl.obs_names
            curr_origin_ct = curr_ctrl.obs[cell_type_key].values

        # 安全检查：确保维度完美对齐
        if len(z_pred_np) != len(z0_np):
            print(f"⚠️ 跳过 {cond_tuple_str}: 预测细胞数({len(z_pred_np)})与起始细胞数({len(z0_np)})不匹配！")
            continue
        # ---------------------------------------------------------

        # 2. 计算状态转变距离 v (现在维度严格一致了)
        distance = np.linalg.norm(z_pred_np - z0_np, axis=1)

        # 3. 用 KNN 在 PCA 空间里给预测后状态重新判 cell type
        pred_cell_type = knn.predict(z_pred_np)
        pred_confidence = knn.predict_proba(z_pred_np).max(axis=1)

        # 4. 组装 DataFrame (维度严格一致)
        df = pd.DataFrame({
            "cell_id": curr_obs_names,
            "origin_cell_type": curr_origin_ct,
            "pred_cell_type": pred_cell_type,
            "pred_cell_type_conf": pred_confidence,
            "condition_tuple": cond_tuple_str,
            "distance_v": distance,             
            "mass_m": m_pred
        })
        
        # 5. 将 Tuple 拆解回原始协变量
        try:
            tuple_vals = eval(cond_tuple_str)
            schema = rulebook.get("condition_tuple_schema", [])
            for i, col_name in enumerate(schema):
                if i < len(tuple_vals):
                    df[col_name] = tuple_vals[i]
        except Exception as e:
            pass
            
        metrics_list.append(df)

    if not metrics_list:
        return pd.DataFrame()
        
    return pd.concat(metrics_list, ignore_index=True)






def plot_predicted_mass_by_celltype(df_metrics, target_tuple_str, target_cell_types):
    """
    绘制特定 Tuple 条件下的预测 Mass 变化小提琴图。
    """
    # 筛选特定条件和特定细胞类型
    df_cond = df_metrics[(df_metrics['condition_tuple'] == target_tuple_str) & 
                         (df_metrics['pred_cell_type'].isin(target_cell_types))]
    
    plt.figure(figsize=(12, 6))
    
    # 画小提琴图
    sns.violinplot(data=df_cond, x='pred_cell_type', y='mass_m', 
                   inner="quartile", palette="Set2", order=target_cell_types)
    plt.xticks(rotation=45, ha='right') 
    
    # 画一条 m=1 的红线
    plt.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Mass Conservation (m=1)')
        
    plt.title(f'Predicted Mass Dynamics under\n{target_tuple_str}', fontsize=14)
    plt.ylabel('Predicted Mass ($m_{pred}$)', fontsize=14)
    plt.xlabel('Predicted Cell Type', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_predicted_mass_with_true_label(
    df_metrics, 
    target_tuple_str, 
    target_cell_types,
    adata_control, 
    adata_full, 
    rulebook=None,                           # 【新增】传入法典以匹配精准 Control
    use_groupwise_control=True,              # 【新增】控制开关
    condition_combined_key="condition_combined",
    cell_type_key="cell_type",
    mass_deduct_keys=None                    # 【核心修复】动态孔数计算键名 (e.g., 'plate_well')
):
    """
    对比模型预测的 Mass 与真实的群体扩增/死亡率。
    引入了 Groupwise Control 匹配和动态孔数(Well)密度校准，彻底消除硬编码。
    """
    # 1. 筛选特定条件和特定细胞类型的预测结果
    df_cond = df_metrics[(df_metrics['condition_tuple'] == target_tuple_str) & 
                         (df_metrics['pred_cell_type'].isin(target_cell_types))]
    
    if df_cond.empty:
        print(f"⚠️ 警告: 没有找到条件 {target_tuple_str} 的预测数据！")
        return

    # 2. 获取真实的 Pert 数据
    pert_subset = adata_full[adata_full.obs[condition_combined_key] == target_tuple_str]
    pert_counts = pert_subset.obs[cell_type_key].value_counts()
    
    # 3. 获取精准匹配的 Control 数据 (替代原先粗暴的 adata_control 全集)
    curr_ctrl = _get_matched_control(adata_control, target_tuple_str, rulebook, use_groupwise_control)
    ctrl_counts = curr_ctrl.obs[cell_type_key].value_counts()
    
    # 4. 【核心修复】：动态计算孔数密度 (Cells per well)
    if mass_deduct_keys is not None and mass_deduct_keys in pert_subset.obs and mass_deduct_keys in curr_ctrl.obs:
        n_pert_wells = len(pert_subset.obs[mass_deduct_keys].unique())
        n_ctrl_wells = len(curr_ctrl.obs[mass_deduct_keys].unique())
    else:
        n_pert_wells = 1.0
        n_ctrl_wells = 1.0
        if mass_deduct_keys is not None:
            print(f"⚠️ 警告: 列名 {mass_deduct_keys} 不存在，已回退到绝对细胞数对比。")
        
    # 防止除以 0 导致崩溃
    n_pert_wells = max(1.0, float(n_pert_wells))
    n_ctrl_wells = max(1.0, float(n_ctrl_wells))

    # 计算密度的期望值 (细胞数 / 孔数)
    pert_density = pert_counts / n_pert_wells
    ctrl_density = ctrl_counts / n_ctrl_wells
    
    true_mass_dict = {}
    for ct in target_cell_types:
        expected_n = ctrl_density.get(ct, 0)
        actual_n = pert_density.get(ct, 0)
        
        # 只有在 Control 组存在该细胞类型时，计算倍数才有意义
        if expected_n > 0: 
            true_mass_dict[ct] = actual_n / expected_n
        else:
            true_mass_dict[ct] = np.nan
            
    # 5. 开始画图
    plt.figure(figsize=(12, 6))
    
    # 画小提琴图 (模型预测的 m)
    ax = sns.violinplot(data=df_cond, x='pred_cell_type', y='mass_m', 
                        inner="quartile", palette="Set2", order=target_cell_types)
    plt.xticks(rotation=45, ha='right') 
    plt.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Mass Conservation (m=1)')
    
    # 6. 叠加 True Mass (真实的星星标签)
    x_coords = np.arange(len(target_cell_types))
    y_coords = [true_mass_dict[ct] for ct in target_cell_types]
    
    # 散点图画星星
    plt.scatter(x_coords, y_coords, color='black', marker='*', s=200, zorder=5, 
                edgecolor='white', linewidth=1, label='True Mass Multiplier ($\star$)')
    
    # 把有效的星星用虚线连起来，方便观察趋势
    valid_idx = [i for i, y in enumerate(y_coords) if not np.isnan(y)]
    valid_x = [x_coords[i] for i in valid_idx]
    valid_y = [y_coords[i] for i in valid_idx]
    plt.plot(valid_x, valid_y, color='black', linestyle=':', linewidth=1.5, alpha=0.5, zorder=4)
        
    plt.title(f'Predicted vs True Mass Dynamics\nCondition: {target_tuple_str}', fontsize=14)
    plt.ylabel('Mass Multiplier ($m$)', fontsize=14)
    plt.xlabel('Cell Type', fontsize=14)
    
    # 优化图例位置，防止挡住数据
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    plt.tight_layout()     
    plt.show()
