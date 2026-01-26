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




import numpy as np
import scipy.sparse as sp
from scipy.stats import pearsonr, wasserstein_distance  # 用于计算 PCC 和 分布距离
from sklearn.metrics import mean_squared_error  

# --- 补充缺失的辅助函数 ---

def to_dense(X):
    """
    通用转换工具：将稀疏矩阵或matrix对象转换为 numpy array
    """
    if sp.issparse(X):
        return X.toarray()
    if hasattr(X, "A"): # 处理 numpy matrix
        return X.A
    return np.array(X)


def get_weighted_mean(adata_pred):
    """
    计算预测数据的加权均值
    """
    X = to_dense(adata_pred.X)
    
    # 假设权重存储在 'mass' 列中
    if 'mass' in adata_pred.obs:
        mass = adata_pred.obs['mass'].values.flatten()
    else:
        # 如果没有 mass，默认均匀权重
        mass = np.ones(X.shape[0])
    
    # 归一化权重
    if mass.sum() == 0:
        weights = np.ones_like(mass) / len(mass)
    else:
        weights = mass / mass.sum()
        
    # 加权平均
    weighted_mean = np.average(X, axis=0, weights=weights)
    return weighted_mean, weights

def get_top_k_pred_diff(delta_vector, gene_names, top_k=20):
    """
    根据变化量幅度 (|Delta|)，找出变化最大的 Top K 基因名称
    """
    # 1. 取绝对值
    abs_delta = np.abs(delta_vector)
    
    # 2. 排序 (argsort 返回的是从小到大的索引)
    # 取最后 top_k 个，并倒序 ([::-1]) 变成从大到小
    if top_k > len(abs_delta):
        top_k = len(abs_delta)
        
    top_indices = np.argsort(abs_delta)[-top_k:][::-1]
    
    return gene_names[top_indices].tolist()



def evaluate_all_perturbations(
    reconstructed_data: dict,    # 你的预测结果字典 {gene: AnnData}
    adata_control: sc.AnnData,   # 对照组
    adata_real: sc.AnnData,      # 真实扰动组
    condition_key: str = 'target_gene',
    top_n_deg: int = 20,         # 评估 Top N 基因的重叠率
    compute_wasserstein: bool = True # 是否计算分布距离（较慢）
):
    """
    对所有扰动进行批量评估
    """
    
    # 1. 计算 Control 的基准均值
    # 建议使用 raw 或 normalized data，而不是 scale 过的
    ctrl_mean = np.mean(to_dense(adata_control.X), axis=0)
    var_names = np.array(adata_control.var_names)
    
    results_list = []
    
    # 获取共同的扰动目标
    pred_targets = list(reconstructed_data.keys())
    real_targets = adata_real.obs[condition_key].unique()
    valid_targets = [t for t in pred_targets if t in real_targets]
    
    print(f"Starting evaluation on {len(valid_targets)} perturbations...")
    
    for i, target in enumerate(valid_targets):
        if i % 10 == 0: print(f"Processing {i}/{len(valid_targets)}: {target}")
            
        # --- A. 准备真实数据 (Ground Truth) ---
        subset_real = adata_real[adata_real.obs[condition_key] == target]
        if subset_real.n_obs < 5: continue
        
        real_X = to_dense(subset_real.X)
        real_mean = np.mean(real_X, axis=0)
        
        # 真实变化量 (Delta)
        delta_real = real_mean - ctrl_mean
        
        # 获取真实的 DEGs (利用你提供的 rank_genes_groups 逻辑的简化版，或者直接用 mean shift)
        # 为了速度，这里用 |Mean Shift| 排序作为 Gold Standard
        # 如果需要更严格的 p-value，可以调用你原来的 get_ground_truth_degs，但这会很慢
        top_real_genes = get_top_k_pred_diff(delta_real, var_names, top_k=top_n_deg)
        
        
        # --- B. 准备预测数据 (Prediction) ---
        pred_ad = reconstructed_data[target]
        pred_mean, pred_weights = get_weighted_mean(pred_ad)
        
        # 预测变化量
        delta_pred = pred_mean - ctrl_mean
        
        # 获取预测认为变化最大的基因
        top_pred_genes = get_top_k_pred_diff(delta_pred, var_names, top_k=top_n_deg)
        
        
        # --- C. 计算指标 ---
        
        # 1. Global Metrics (全基因组表达量)
        # 关注 Delta 的相关性 (方向对不对)
        pcc_delta, _ = pearsonr(delta_real, delta_pred)
        # 关注 Delta 的误差 (幅度对不对)
        mse_delta = mean_squared_error(delta_real, delta_pred)
        
        # 2. DEG Overlap (Top N 基因重合度)
        # Jaccard Index
        set_real = set(top_real_genes)
        set_pred = set(top_pred_genes)
        overlap_count = len(set_real.intersection(set_pred))
        jaccard = overlap_count / len(set_real.union(set_pred))
        recall = overlap_count / len(set_real) # 找回了多少真实DEG
        
        metrics = {
            'Target': target,
            'N_Real_Cells': subset_real.n_obs,
            'MSE_Delta': mse_delta,
            'PCC_Delta': pcc_delta,
            'DEG_Recall': recall,
            'DEG_Jaccard': jaccard
        }
        
        # 3. Distribution Metrics (仅在 Target Gene 和 Top DEG 上计算)
        # 计算 Wassertein 距离看分布拟合得好不好
        if compute_wasserstein:
            # 为了计算分布距离，我们需要对预测数据进行重采样 (Resample)
            # 因为 Wasserstein 需要两个样本集
            resample_idx = np.random.choice(len(pred_weights), size=min(500, subset_real.n_obs), p=pred_weights)
            X_pred_resampled = to_dense(pred_ad.X)[resample_idx]
            X_real_sub = real_X[:len(resample_idx)] # 保持数量一致
            
            # 3.1 Target Gene 本身的分布距离
            if target in var_names:
                idx_t = np.where(var_names == target)[0][0]
                wd_target = wasserstein_distance(X_real_sub[:, idx_t], X_pred_resampled[:, idx_t])
                metrics['WD_Target'] = wd_target
            
            # 3.2 Top 5 Real DEGs 的平均分布距离
            wd_degs = []
            for deg in top_real_genes[:5]: # 只看前5个最显著的
                if deg in var_names:
                    idx_d = np.where(var_names == deg)[0][0]
                    wd = wasserstein_distance(X_real_sub[:, idx_d], X_pred_resampled[:, idx_d])
                    wd_degs.append(wd)
            metrics['WD_Top5_DEGs'] = np.mean(wd_degs) if wd_degs else np.nan

        results_list.append(metrics)

    # --- D. 汇总结果 ---
    results_df = pd.DataFrame(results_list)
    
    summary = {
        'Mean_MSE': results_df['MSE_Delta'].mean(),
        'Mean_PCC': results_df['PCC_Delta'].mean(),
        'Mean_DEG_Recall': results_df['DEG_Recall'].mean(),
        'Mean_WD_Target': results_df['WD_Target'].mean() if 'WD_Target' in results_df else None
    }
    
    print("\n=== Evaluation Summary ===")
    print(f"Evaluated {len(results_df)} perturbations.")
    print(f"Mean PCC (Delta): {summary['Mean_PCC']:.4f}")
    print(f"Mean MSE (Delta): {summary['Mean_MSE']:.4f}")
    print(f"Mean DEG Recall@{top_n_deg}: {summary['Mean_DEG_Recall']:.4f}")
    
    return results_df, summary