import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def plot_perturbation_umap(
    results_embedding, 
    adata_control, 
    adata_conditions, 
    target_genes, 
    control_indices,   # 你代码里的 indices
    condition_key="target_gene",
    rep_key="X_pca_scaled", # 你的 source_rep
    n_neighbors=30
):
    """
    绘制 Control vs Real vs Predict 的 UMAP，并融合 m_pred 信息
    """
    
    # 1. 获取 Control 数据 (所有基因共享)
    # 注意：这里需要拿原始的numpy数据，不要tensor
    X_control = adata_control.obsm[rep_key][control_indices]
    
    for gene in target_genes:
        if gene not in results_embedding:
            continue
            
        print(f"Plotting {gene}...")
        
        # 2. 获取 Real 数据
        subset = adata_conditions[adata_conditions.obs[condition_key] == gene]
        if subset.n_obs == 0:
            print(f"No real data for {gene}")
            continue
        # 随机采样 Real 数据以避免点太多遮挡 (可选，这里设为跟control一样多或者全部)
        idx_real = np.random.choice(range(subset.n_obs), size=min(subset.n_obs, len(control_indices)), replace=False)
        X_real = subset.obsm[rep_key][idx_real]
        
        # 3. 获取 Predict 数据和 Mass
        res = results_embedding[gene]
        X_pred = res['z_pred']  # [n_particles, dim]
        m_pred = res['m_pred'].flatten() # [n_particles,]
        
        # --- 数据预处理：构建联合 AnnData 做 UMAP ---
        # 拼接数据矩阵
        X_combined = np.vstack([X_control, X_real, X_pred])
        
        # 创建标签
        labels = (
            ['Control'] * len(X_control) + 
            ['Real'] * len(X_real) + 
            ['Predict'] * len(X_pred)
        )
        
        # 创建临时 AnnData
        adata_vis = sc.AnnData(X=X_combined)
        adata_vis.obs['condition'] = labels
        # 只要用原来的 rep 算 neighbor 即可，不需要再 PCA
        # scanpy 的 neighbors 默认用 .X，如果我们直接把 pca 放入 .X，就不要再用 use_rep
        sc.pp.neighbors(adata_vis, n_neighbors=n_neighbors, use_rep='X') 
        sc.tl.umap(adata_vis)
        
        # 提取 UMAP 坐标
        umap_coords = adata_vis.obsm['X_umap']
        umap_ctrl = umap_coords[:len(X_control)]
        umap_real = umap_coords[len(X_control):len(X_control)+len(X_real)]
        umap_pred = umap_coords[len(X_control)+len(X_real):]
        
        # --- 可视化 ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # === 图 1: 类别对比 (运用 m_pred 做 alpha) ===
        ax = axes[0]
        
        # 画 Control (灰色背景)
        ax.scatter(umap_ctrl[:, 0], umap_ctrl[:, 1], c='lightgrey', s=10, label='Control', alpha=0.5, rasterized=True)
        
        # 画 Real (橙色/红色目标)
        ax.scatter(umap_real[:, 0], umap_real[:, 1], c='#d62728', s=15, label='Real', alpha=0.6, rasterized=True)
        
        # 画 Predict (蓝色，根据 mass 调整透明度)
        # 归一化 mass 到 [0, 1] 区间用于显示 alpha
        # 注意：如果 m_pred 差异极大，建议先取 log
        # alpha = (m - min) / (max - min)
        # 为了视觉效果，可以设置最小 alpha 阈值，比如 0.05，否则完全看不见
        m_norm = (m_pred - m_pred.min()) / (m_pred.max() - m_pred.min() + 1e-9)
        alphas = m_norm # 保证最低有 0.1 的透明度
        alphas = np.clip(alphas, 0, 1)
        
        # Matplotlib scatter 支持 rgba 颜色，我们可以构建一个颜色数组
        # 基础颜色: tab:blue -> (0.12, 0.46, 0.7, 1.0)
        base_color = np.array(plt.cm.tab10(0)) # Blue
        rgba_colors = np.zeros((len(X_pred), 4))
        rgba_colors[:, :3] = base_color[:3]
        rgba_colors[:, 3] = alphas # 设置 Alpha 通道
        
        ax.scatter(umap_pred[:, 0], umap_pred[:, 1], c=rgba_colors, s=15, label='Predict', rasterized=True)
        
        ax.set_title(f"Perturbation: {gene}\n(Predict Alpha scaled by Mass)")
        ax.legend()
        ax.set_xticks([])
        ax.set_yticks([])
        
        # === 图 2: 仅展示 Predict 的 Mass 分布 ===
        ax = axes[1]
        
        # 为了背景参考，淡淡地画上 Control
        ax.scatter(umap_ctrl[:, 0], umap_ctrl[:, 1], c='lightgrey', s=5, alpha=0.2, rasterized=True)
        
        # 画 Predict，颜色映射 Mass
        sc_plot = ax.scatter(umap_pred[:, 0], umap_pred[:, 1], c=m_pred, cmap='viridis', s=15, rasterized=True)
        plt.colorbar(sc_plot, ax=ax, label='Predicted Mass (m_pred)')
        
        ax.set_title(f"Predicted Mass Distribution: {gene}")
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        plt.show()



import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def get_ground_truth_degs(
    adata_control: sc.AnnData,
    adata_real: sc.AnnData,
    target_gene: str,
    top_n: int = 5,
    condition_key: str = 'target_gene'
) -> list:
    """
    计算 Control 和 Real(Target) 之间的差异表达基因 (DEGs)。
    返回: [Target_Gene, DEG_1, DEG_2, ..., DEG_N]
    """
    # 1. 提取真实的 perturbation 数据
    subset_real = adata_real[adata_real.obs[condition_key] == target_gene].copy()
    
    if subset_real.n_obs < 5:
        print(f"Not enough real cells for {target_gene}")
        return [target_gene]

    # 2. 采样 Control 数据 (为了计算快一点，且保持平衡)
    # 注意：这里需要使用 raw counts 或者 log1p 后的数据，不要用 scale 后的
    n_samples = min(2000, adata_control.n_obs)
    idx_ctrl = np.random.choice(adata_control.n_obs, n_samples, replace=False)
    subset_ctrl = adata_control[idx_ctrl].copy()
    
    # 3. 临时合并数据用于差异分析
    subset_real.obs['temp_group'] = 'Perturb'
    subset_ctrl.obs['temp_group'] = 'Control'
    
    # 确保 var_names 一致
    common_genes = subset_real.var_names.intersection(subset_ctrl.var_names)
    adata_merged = subset_ctrl[:, common_genes].concatenate(subset_real[:, common_genes])
    
    # 4. 运行 Wilcoxon Rank-Sum test
    # 注意：如果你的 .X 是 scale 过的，结果可能不准，建议用 .raw 或 log1p 后的层
    sc.tl.rank_genes_groups(adata_merged, groupby='temp_group', reference='Control', method='wilcoxon')
    
    # 5. 提取 Top Genes (按得分为序)
    deg_df = sc.get.rank_genes_groups_df(adata_merged, group='Perturb')
    
    # 过滤掉 Target Gene 本身 (防止它占了 Top 1 的位置，虽然通常我们也想看它)
    # 我们手动把它加在列表第一个，所以这里取 Top N 个非 Target 的基因
    top_genes = deg_df[deg_df['names'] != target_gene].head(top_n)['names'].tolist()
    
    # 结果列表：Target Gene + Top Downstream Genes
    final_genes = [target_gene] + top_genes
    
    # 过滤掉不在 var_names 里的（以防万一）
    return [g for g in final_genes if g in adata_control.var_names]

def plot_top_degs_violin(
    reconstructed_data: dict,   # 你的 batch_reconstruct 结果
    adata_control: sc.AnnData,
    adata_real: sc.AnnData,
    target_gene: str,
    top_n: int = 5
):
    """
    自动化流程：找 DEG -> 加权重采样 -> 画图
    """
    print(f"--- Analyzing {target_gene} ---")
    
    # Step 1: 找出这个 perturbation 对应的 Top DEGs
    # 这一步确立了“Ground Truth”：真实的扰动到底改变了谁？
    genes_of_interest = get_ground_truth_degs(adata_control, adata_real, target_gene, top_n)
    
    print(f"Top DEGs found: {genes_of_interest}")

    # Step 2: 准备画图数据
    plot_data = []
    
    # --- A. Control Data ---
    # 随机采点
    idx_ctrl = np.random.choice(adata_control.n_obs, min(1000, adata_control.n_obs), replace=False)
    X_ctrl = adata_control[idx_ctrl, :].X
    if not isinstance(X_ctrl, np.ndarray): X_ctrl = X_ctrl.toarray()
    
    for gene in genes_of_interest:
        col_idx = adata_control.var_names.get_loc(gene)
        vals = X_ctrl[:, col_idx]
        for v in vals:
            plot_data.append({'Condition': 'Control', 'Gene': gene, 'Expression': v})

    # --- B. Real Data ---
    subset_real = adata_real[adata_real.obs['target_gene'] == target_gene]
    if subset_real.n_obs > 0:
        X_real = subset_real.X
        if not isinstance(X_real, np.ndarray): X_real = X_real.toarray()
        for gene in genes_of_interest:
            col_idx = subset_real.var_names.get_loc(gene)
            vals = X_real[:, col_idx]
            for v in vals:
                plot_data.append({'Condition': 'Real', 'Gene': gene, 'Expression': v})
    
    # --- C. Predict Data (Weighted) ---
    if target_gene in reconstructed_data:
        pred_ad = reconstructed_data[target_gene]
        mass = pred_ad.obs['mass'].values.flatten()
        
        # 处理全0 mass 的边缘情况
        if mass.sum() == 0: 
            probs = np.ones_like(mass) / len(mass)
        else:
            probs = mass / mass.sum()
            
        # 核心：根据 mass 重采样 2000 个细胞
        resample_idx = np.random.choice(len(mass), size=2000, p=probs, replace=True)
        X_pred = pred_ad.X[resample_idx]
        
        for gene in genes_of_interest:
            # 注意：reconstructed_data 的 var_names 和 ref_adata 一致
            if gene in pred_ad.var_names:
                col_idx = pred_ad.var_names.get_loc(gene)
                vals = X_pred[:, col_idx]
                for v in vals:
                    plot_data.append({'Condition': 'Predict', 'Gene': gene, 'Expression': v})

    # Step 3: 绘图
    df_plot = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(2 + 1.5 * len(genes_of_interest), 5))
    sns.violinplot(
        data=df_plot,
        x='Gene', y='Expression', hue='Condition',
        palette={'Control': 'lightgrey', 'Real': '#d62728', 'Predict': '#1f77b4'},
        scale='width', # 让宽度反映密度
        cut=0,         # 不显示数据范围外的推测
        linewidth=1
    )
    plt.title(f"Top DEGs for perturbation: {target_gene}\n(Predict weighted by mass)")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


