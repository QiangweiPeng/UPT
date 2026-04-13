import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from matplotlib.lines import Line2D

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


def _normalize_mass_weights_np(weights):
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    total = weights.sum()
    if total <= 0:
        return np.full(weights.shape[0], 1.0 / float(weights.shape[0]), dtype=np.float64)
    return weights / total


def resample_prediction_adata_by_mass(
    pred_adata: ad.AnnData,
    n_samples: int,
    weight_key: str = "mass",
    seed: int = 42,
):
    """Resample predicted particles with replacement according to mass weights."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if pred_adata.n_obs == 0:
        raise ValueError("pred_adata must contain at least one observation")

    if weight_key in pred_adata.obs.columns:
        weights = _normalize_mass_weights_np(pred_adata.obs[weight_key].to_numpy())
    else:
        weights = np.full(pred_adata.n_obs, 1.0 / float(pred_adata.n_obs), dtype=np.float64)

    rng = np.random.default_rng(seed)
    sampled_idx = rng.choice(pred_adata.n_obs, size=n_samples, replace=True, p=weights)
    sampled = pred_adata[sampled_idx].copy()
    sampled.obs["resampled_source_obs_name"] = pred_adata.obs_names[sampled_idx].astype(str)
    sampled.obs["resampled_source_index"] = sampled_idx.astype(np.int64)
    sampled.obs["resampled_rank"] = np.arange(n_samples, dtype=np.int64)
    sampled.obs_names = [
        f"{source_name}__rs{rank}"
        for rank, source_name in enumerate(sampled.obs["resampled_source_obs_name"].astype(str))
    ]
    return sampled


def _latent_to_vis_adata(latent, obs, prefix):
    vis_adata = ad.AnnData(X=np.asarray(latent, dtype=np.float32), obs=obs.copy())
    vis_adata.var_names = [f"{prefix}_{idx}" for idx in range(vis_adata.shape[1])]
    return vis_adata


def build_condition_real_prediction_umap_adata(
    real_adata: ad.AnnData,
    pred_adata: ad.AnnData,
    condition: str,
    timepoints: list,
    condition_key: str = "gene_target",
    time_key: str = "timepoint",
    latent_key: str = "X_pca_scaled",
    weight_key: str = "mass",
    seed: int = 42,
):
    """
    Build a condition-level AnnData for UMAP by matching sampled predicted counts to real counts.

    The predicted side is resampled with replacement using normalized mass weights for each
    ``(condition, timepoint)`` pair, so the sampled predicted count exactly matches the number
    of real cells in that pair.
    """
    condition = str(condition)
    timepoints = sorted({float(timepoint) for timepoint in timepoints})
    vis_adatas = []
    summary_rows = []

    for offset, timepoint in enumerate(timepoints):
        real_subset = real_adata[
            (real_adata.obs[condition_key].astype(str) == condition)
            & (real_adata.obs[time_key].astype(float) == timepoint)
        ].copy()
        pred_subset = pred_adata[
            (pred_adata.obs["target_condition"].astype(str) == condition)
            & (pred_adata.obs["target_timepoint"].astype(float) == timepoint)
        ].copy()

        if real_subset.n_obs == 0 or pred_subset.n_obs == 0:
            continue

        sampled_pred = resample_prediction_adata_by_mass(
            pred_adata=pred_subset,
            n_samples=int(real_subset.n_obs),
            weight_key=weight_key,
            seed=seed + offset,
        )

        real_obs = real_subset.obs.copy()
        real_obs["plot_source"] = "real"
        real_obs["plot_condition"] = condition
        real_obs["plot_timepoint"] = float(timepoint)
        real_obs["plot_group"] = f"real_t{float(timepoint):g}"
        real_vis = _latent_to_vis_adata(real_subset.obsm[latent_key], real_obs, prefix="latent")
        real_vis.obs_names = [
            f"real|{condition}|t{float(timepoint):g}|{obs_name}"
            for obs_name in real_subset.obs_names.astype(str)
        ]

        pred_obs = sampled_pred.obs.copy()
        pred_obs["plot_source"] = "pred_sampled"
        pred_obs["plot_condition"] = condition
        pred_obs["plot_timepoint"] = float(timepoint)
        pred_obs["plot_group"] = f"pred_t{float(timepoint):g}"
        pred_vis = _latent_to_vis_adata(sampled_pred.X, pred_obs, prefix="latent")
        pred_vis.obs_names = [
            f"pred|{condition}|t{float(timepoint):g}|{obs_name}"
            for obs_name in sampled_pred.obs_names.astype(str)
        ]

        vis_adatas.extend([real_vis, pred_vis])
        summary_rows.append(
            {
                "condition": condition,
                "timepoint": float(timepoint),
                "n_real": int(real_subset.n_obs),
                "n_pred_available": int(pred_subset.n_obs),
                "n_pred_sampled": int(sampled_pred.n_obs),
                "n_unique_pred_sampled": int(sampled_pred.obs["resampled_source_obs_name"].nunique()),
                "sampled_mass_mean": float(sampled_pred.obs[weight_key].astype(float).mean())
                if weight_key in sampled_pred.obs.columns
                else np.nan,
                "sampled_mass_sum": float(sampled_pred.obs[weight_key].astype(float).sum())
                if weight_key in sampled_pred.obs.columns
                else np.nan,
            }
        )

    if not vis_adatas:
        raise RuntimeError(f"No overlapping real/predicted pairs found for condition {condition}")

    combined = ad.concat(
        vis_adatas,
        axis=0,
        join="outer",
        merge="same",
        index_unique=None,
    )
    combined.uns["umap_sampling_metadata"] = {
        "condition": condition,
        "timepoints": timepoints,
        "latent_key": latent_key,
        "weight_key": weight_key,
        "sampling_scheme": "predictions resampled with replacement by normalized mass to match real pair counts",
    }
    return combined, pd.DataFrame(summary_rows)


def compute_latent_umap(
    adata_vis: ad.AnnData,
    n_neighbors: int = 30,
    min_dist: float = 0.3,
    random_state: int = 42,
):
    sc.pp.neighbors(adata_vis, n_neighbors=n_neighbors, use_rep="X")
    sc.tl.umap(adata_vis, min_dist=min_dist, random_state=random_state)
    return adata_vis


def plot_condition_real_prediction_umap(
    adata_vis: ad.AnnData,
    output_path,
    title: str | None = None,
    point_size: float = 6.0,
    alpha: float = 0.65,
):
    if "X_umap" not in adata_vis.obsm:
        raise KeyError("adata_vis.obsm['X_umap'] is required before plotting")

    coords = np.asarray(adata_vis.obsm["X_umap"], dtype=np.float32)
    plot_df = adata_vis.obs.copy()
    plot_df["UMAP1"] = coords[:, 0]
    plot_df["UMAP2"] = coords[:, 1]

    source_palette = {
        "real": "#d62728",
        "pred_sampled": "#1f77b4",
    }
    unique_timepoints = sorted(plot_df["plot_timepoint"].astype(float).unique().tolist())
    time_palette = dict(
        zip(unique_timepoints, sns.color_palette("viridis", n_colors=len(unique_timepoints)))
    )
    marker_map = {
        "real": "o",
        "pred_sampled": "^",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for source_name, color in source_palette.items():
        subset = plot_df[plot_df["plot_source"] == source_name]
        if subset.empty:
            continue
        ax.scatter(
            subset["UMAP1"],
            subset["UMAP2"],
            s=point_size,
            alpha=alpha,
            c=[color],
            marker=marker_map.get(source_name, "o"),
            label=source_name,
            linewidths=0.0,
            rasterized=True,
        )
    ax.set_title("Real vs Mass-Resampled Prediction")
    ax.legend(frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1]
    for timepoint in unique_timepoints:
        for source_name, marker in marker_map.items():
            subset = plot_df[
                (plot_df["plot_timepoint"].astype(float) == float(timepoint))
                & (plot_df["plot_source"] == source_name)
            ]
            if subset.empty:
                continue
            ax.scatter(
                subset["UMAP1"],
                subset["UMAP2"],
                s=point_size,
                alpha=alpha,
                c=[time_palette[timepoint]],
                marker=marker,
                label=f"{source_name}_t{float(timepoint):g}",
                linewidths=0.0,
                rasterized=True,
            )
    ax.set_title("By Timepoint")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _collapse_categories(series, max_categories=12):
    series = pd.Series(series).astype(str).fillna("Unknown")
    counts = series.value_counts(dropna=False)
    if max_categories is None or counts.shape[0] <= max_categories:
        return series, counts

    keep = counts.index[: max_categories - 1]
    collapsed = series.where(series.isin(keep), other="Other")
    return collapsed, collapsed.value_counts(dropna=False)


def plot_saved_condition_umap_with_real_categories(
    adata_vis: ad.AnnData,
    output_path,
    real_color_key: str = "major_group",
    secondary_real_color_key: str = "tissue",
    max_categories: int = 12,
    pred_alpha: float = 0.08,
    real_alpha: float = 0.85,
    pred_size: float = 2.0,
    real_size: float = 5.0,
):
    """
    Redraw a saved condition-level UMAP with reduced occlusion.

    Render a 2x2 redraw layout:
    - real_color_key over prediction background
    - secondary_real_color_key over prediction background
    - datatype
    - timepoint
    """
    if "X_umap" not in adata_vis.obsm:
        raise KeyError("adata_vis.obsm['X_umap'] is required before plotting")

    plot_df = adata_vis.obs.copy()
    coords = np.asarray(adata_vis.obsm["X_umap"], dtype=np.float32)
    plot_df["UMAP1"] = coords[:, 0]
    plot_df["UMAP2"] = coords[:, 1]
    plot_df["plot_source"] = plot_df["plot_source"].astype(str)

    real_df = plot_df[plot_df["plot_source"] == "real"].copy()
    pred_df = plot_df[plot_df["plot_source"] == "pred_sampled"].copy()
    if real_df.empty or pred_df.empty:
        raise RuntimeError("Both real and pred_sampled observations are required for redraw")

    for color_key in (real_color_key, secondary_real_color_key):
        if color_key not in real_df.columns:
            raise KeyError(f"{color_key} is not found in real observations")

    real_df["primary_category"], primary_counts = _collapse_categories(
        real_df[real_color_key],
        max_categories=max_categories,
    )
    real_df["secondary_category"], secondary_counts = _collapse_categories(
        real_df[secondary_real_color_key],
        max_categories=max_categories,
    )
    primary_order = primary_counts.index.tolist()
    primary_palette = dict(
        zip(primary_order, sns.color_palette("tab20", n_colors=len(primary_order)))
    )
    secondary_order = secondary_counts.index.tolist()
    secondary_palette = dict(
        zip(secondary_order, sns.color_palette("Set2", n_colors=len(secondary_order)))
    )

    pred_timepoints = sorted(pred_df["plot_timepoint"].astype(float).unique().tolist())
    time_palette = dict(
        zip(pred_timepoints, sns.color_palette("viridis", n_colors=len(pred_timepoints)))
    )

    datatype_palette = {
        "real": "#d62728",
        "pred_sampled": "#4c78a8",
    }
    marker_map = {
        "real": "o",
        "pred_sampled": "^",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    ax.scatter(
        pred_df["UMAP1"],
        pred_df["UMAP2"],
        s=pred_size,
        c="#b0b7c3",
        alpha=pred_alpha,
        linewidths=0.0,
        rasterized=True,
    )
    for category in primary_order:
        subset = real_df[real_df["primary_category"] == category]
        ax.scatter(
            subset["UMAP1"],
            subset["UMAP2"],
            s=real_size,
            c=[primary_palette[category]],
            alpha=real_alpha,
            linewidths=0.0,
            rasterized=True,
            label=category,
        )
    ax.set_title(str(real_color_key))
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[0, 1]
    ax.scatter(
        pred_df["UMAP1"],
        pred_df["UMAP2"],
        s=pred_size,
        c="#b0b7c3",
        alpha=pred_alpha,
        linewidths=0.0,
        rasterized=True,
    )
    for category in secondary_order:
        subset = real_df[real_df["secondary_category"] == category]
        ax.scatter(
            subset["UMAP1"],
            subset["UMAP2"],
            s=real_size,
            c=[secondary_palette[category]],
            alpha=real_alpha,
            linewidths=0.0,
            rasterized=True,
            label=category,
        )
    ax.set_title(str(secondary_real_color_key))
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1, 0]
    for source_name, color in datatype_palette.items():
        subset = plot_df[plot_df["plot_source"] == source_name]
        if subset.empty:
            continue
        ax.scatter(
            subset["UMAP1"],
            subset["UMAP2"],
            s=real_size if source_name == "real" else pred_size,
            c=[color],
            alpha=real_alpha if source_name == "real" else 0.18,
            marker=marker_map[source_name],
            linewidths=0.0,
            rasterized=True,
        )
    ax.set_title("Datatype")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1, 1]
    for source_name, marker in marker_map.items():
        subset_source = plot_df[plot_df["plot_source"] == source_name]
        if subset_source.empty:
            continue
        for timepoint in pred_timepoints:
            subset = subset_source[
                subset_source["plot_timepoint"].astype(float) == float(timepoint)
            ]
            if subset.empty:
                continue
            ax.scatter(
                subset["UMAP1"],
                subset["UMAP2"],
                s=real_size if source_name == "real" else pred_size,
                c=[time_palette[timepoint]],
                alpha=0.55 if source_name == "real" else 0.2,
                marker=marker,
                linewidths=0.0,
                rasterized=True,
            )
    ax.set_title("Timepoint")
    ax.set_xticks([])
    ax.set_yticks([])

    primary_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=primary_palette[category],
               markeredgecolor="none", markersize=6, label=str(category))
        for category in primary_order
    ]
    secondary_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=secondary_palette[category],
               markeredgecolor="none", markersize=6, label=str(category))
        for category in secondary_order
    ]
    time_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=time_palette[timepoint],
               markeredgecolor="none", markersize=6, label=f"t{float(timepoint):g}")
        for timepoint in pred_timepoints
    ]

    axes[0, 0].legend(
        handles=primary_handles,
        title=real_color_key,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    axes[0, 1].legend(
        handles=secondary_handles,
        title=secondary_real_color_key,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    datatype_handles = [
        Line2D([0], [0], marker=marker_map[source_name], linestyle="", markerfacecolor="#666666",
               markeredgecolor="none", markersize=6, label=source_name)
        for source_name in marker_map
    ]

    time_legend = axes[1, 1].legend(
        handles=time_handles,
        title="timepoint",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    axes[1, 1].add_artist(time_legend)
    axes[1, 0].legend(
        handles=datatype_handles,
        title="datatype",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )

    condition_name = str(plot_df["plot_condition"].astype(str).iloc[0])
    fig.suptitle(condition_name)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    primary_summary = (
        real_df.assign(category_key=real_color_key, category_value=real_df["primary_category"])
        .groupby(["category_key", "plot_timepoint", "category_value"], observed=True)
        .size()
        .reset_index(name="n_real")
    )
    secondary_summary = (
        real_df.assign(
            category_key=secondary_real_color_key,
            category_value=real_df["secondary_category"],
        )
        .groupby(["category_key", "plot_timepoint", "category_value"], observed=True)
        .size()
        .reset_index(name="n_real")
    )
    category_summary = pd.concat([primary_summary, secondary_summary], ignore_index=True)
    category_summary = category_summary.sort_values(
        ["category_key", "plot_timepoint", "n_real"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    return category_summary



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


def plot_mass_accuracy_summary(
    pair_df: pd.DataFrame,
    transition_df: pd.DataFrame,
    metrics: dict,
    output_path,
    title: str | None = None,
):
    if pair_df.empty:
        raise RuntimeError("pair_df must not be empty")
    if transition_df.empty:
        raise RuntimeError("transition_df must not be empty")

    condition_order = sorted(pair_df["target_condition"].astype(str).unique().tolist())
    condition_palette = dict(
        zip(condition_order, sns.color_palette("tab10", n_colors=len(condition_order)))
    )
    timepoints = sorted(pair_df["target_timepoint"].astype(float).unique().tolist())
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    time_markers = {
        float(timepoint): marker_cycle[idx % len(marker_cycle)]
        for idx, timepoint in enumerate(timepoints)
    }

    def _scatter_pairs(ax, xcol, ycol, xlabel, ylabel, subtitle):
        max_val = float(max(pair_df[xcol].max(), pair_df[ycol].max()))
        line_max = max(1.05 * max_val, 0.1)
        ax.plot([0, line_max], [0, line_max], linestyle="--", color="#666666", linewidth=1)
        for _, row in pair_df.iterrows():
            ax.scatter(
                row[xcol],
                row[ycol],
                s=60,
                c=[condition_palette[str(row["target_condition"])]],
                marker=time_markers[float(row["target_timepoint"])],
                alpha=0.9,
                linewidths=0.3,
                edgecolors="black",
                rasterized=True,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    ax = axes[0, 0]
    _scatter_pairs(
        ax,
        "real_ratio",
        "pred_ratio",
        "Real total mass ratio (N_t / N_control)",
        "Predicted total mass ratio (sum(m) / N_control)",
        "Raw Total Mass Ratio",
    )
    ax.text(
        0.03,
        0.97,
        (
            f"r = {metrics['total_ratio']['pearson_raw']:.3f}\n"
            f"MAE = {metrics['total_ratio']['mae_raw']:.3f}\n"
            f"RMSE = {metrics['total_ratio']['rmse_raw']:.3f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )

    ax = axes[0, 1]
    _scatter_pairs(
        ax,
        "real_ratio",
        "pred_ratio_calibrated",
        "Real total mass ratio (N_t / N_control)",
        "Calibrated predicted ratio",
        "Single-Scale Calibrated Total Mass Ratio",
    )
    ax.text(
        0.03,
        0.97,
        (
            f"scale = {metrics['calibration_factor']:.3f}\n"
            f"r = {metrics['total_ratio']['pearson_calibrated']:.3f}\n"
            f"MAE = {metrics['total_ratio']['mae_calibrated']:.3f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )

    ax = axes[1, 0]
    max_fc = float(max(transition_df["real_fc"].max(), transition_df["pred_fc"].max()))
    line_max = max(1.05 * max_fc, 0.1)
    ax.plot([0, line_max], [0, line_max], linestyle="--", color="#666666", linewidth=1)
    for _, row in transition_df.iterrows():
        ax.scatter(
            row["real_fc"],
            row["pred_fc"],
            s=60,
            c=[condition_palette[str(row["target_condition"])]],
            marker=time_markers[float(row["target_timepoint"])],
            alpha=0.9,
            linewidths=0.3,
            edgecolors="black",
            rasterized=True,
        )
    ax.set_xlabel("Real adjacent fold-change")
    ax.set_ylabel("Predicted adjacent fold-change")
    ax.set_title("Adjacent Mass Fold-Change")
    ax.text(
        0.03,
        0.97,
        (
            f"r = {metrics['transition_fold_change']['pearson']:.3f}\n"
            f"MAE = {metrics['transition_fold_change']['mae']:.3f}\n"
            f"RMSE = {metrics['transition_fold_change']['rmse']:.3f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )

    ax = axes[1, 1]
    linestyle_map = {"Real": "-", "Pred": "--", "Pred (cal.)": ":"}
    marker_map = {"Real": "o", "Pred": "s", "Pred (cal.)": "^"}
    for condition in condition_order:
        cond_df = pair_df[pair_df["target_condition"].astype(str) == condition].sort_values(
            "target_timepoint"
        )
        color = condition_palette[condition]
        ax.plot(
            cond_df["target_timepoint"],
            cond_df["real_ratio"],
            linestyle=linestyle_map["Real"],
            marker=marker_map["Real"],
            color=color,
            linewidth=2,
            markersize=5,
            alpha=0.95,
        )
        ax.plot(
            cond_df["target_timepoint"],
            cond_df["pred_ratio"],
            linestyle=linestyle_map["Pred"],
            marker=marker_map["Pred"],
            color=color,
            linewidth=1.7,
            markersize=4.5,
            alpha=0.95,
        )
        ax.plot(
            cond_df["target_timepoint"],
            cond_df["pred_ratio_calibrated"],
            linestyle=linestyle_map["Pred (cal.)"],
            marker=marker_map["Pred (cal.)"],
            color=color,
            linewidth=1.7,
            markersize=4.5,
            alpha=0.95,
        )
    ax.set_title("Per-Condition Mass Trajectories")
    ax.set_xlabel("Target timepoint")
    ax.set_ylabel("Mass ratio vs control")

    condition_handles = [
        Line2D([0], [0], color=condition_palette[condition], linewidth=2, label=condition)
        for condition in condition_order
    ]
    style_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=linestyle_map[label],
            marker=marker_map[label],
            linewidth=1.8,
            markersize=5,
            label=label,
        )
        for label in ["Real", "Pred", "Pred (cal.)"]
    ]
    cond_legend = ax.legend(
        handles=condition_handles,
        title="condition",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(cond_legend)
    ax.legend(
        handles=style_handles,
        title="series",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.38),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )

    time_handles = [
        Line2D(
            [0],
            [0],
            marker=time_markers[timepoint],
            linestyle="",
            color="black",
            markersize=6,
            label=f"t{float(timepoint):g}",
        )
        for timepoint in timepoints
    ]
    axes[0, 1].legend(
        handles=time_handles,
        title="timepoint",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.35),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )

    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mass_accuracy_condition_grid(
    pair_df: pd.DataFrame,
    metrics: dict,
    output_path,
    ncols: int = 2,
):
    if pair_df.empty:
        raise RuntimeError("pair_df must not be empty")

    condition_order = sorted(pair_df["target_condition"].astype(str).unique().tolist())
    n_conditions = len(condition_order)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n_conditions / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.8 * nrows), squeeze=False)

    for ax in axes.ravel():
        ax.set_visible(False)

    for idx, condition in enumerate(condition_order):
        ax = axes[idx // ncols, idx % ncols]
        ax.set_visible(True)
        cond_df = pair_df[pair_df["target_condition"].astype(str) == condition].sort_values(
            "target_timepoint"
        )
        ax.plot(
            cond_df["target_timepoint"],
            cond_df["real_ratio"],
            color="#d62728",
            linestyle="-",
            marker="o",
            linewidth=2,
            label="Real",
        )
        ax.plot(
            cond_df["target_timepoint"],
            cond_df["pred_ratio"],
            color="#1f77b4",
            linestyle="--",
            marker="s",
            linewidth=1.8,
            label="Pred",
        )
        ax.plot(
            cond_df["target_timepoint"],
            cond_df["pred_ratio_calibrated"],
            color="#2ca02c",
            linestyle=":",
            marker="^",
            linewidth=1.8,
            label="Pred (cal.)",
        )
        ax.set_title(condition)
        ax.set_xlabel("Target timepoint")
        ax.set_ylabel("Mass ratio vs control")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Per-Condition Mass Trajectories\n"
        f"single global calibration factor = {metrics['calibration_factor']:.3f}"
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_heatmap_grid(
    metrics_df: pd.DataFrame,
    output_path,
    condition_col: str = "target_condition",
    time_col: str = "target_timepoint",
    metric_order=None,
    title: str | None = None,
):
    if metrics_df.empty:
        raise RuntimeError("metrics_df must not be empty")

    if metric_order is None:
        metric_order = ["r2", "pcc", "mse", "mae", "w1", "w2"]

    conditions = sorted(metrics_df[condition_col].astype(str).unique().tolist())
    timepoints = sorted(metrics_df[time_col].astype(float).unique().tolist())
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 10))

    for ax, metric in zip(axes.ravel(), metric_order):
        pivot = (
            metrics_df.assign(
                _condition=metrics_df[condition_col].astype(str),
                _time=metrics_df[time_col].astype(float),
            )
            .pivot(index="_condition", columns="_time", values=metric)
            .reindex(index=conditions, columns=timepoints)
        )
        if metric in {"r2", "pcc"}:
            cmap = "vlag"
            vmin, vmax = -1.0, 1.0
        else:
            cmap = "mako"
            vmin = vmax = None

        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".3f",
            linewidths=0.5,
            linecolor="white",
            cbar=True,
            square=False,
        )
        ax.set_title(metric.upper())
        ax.set_xlabel("Target timepoint")
        ax.set_ylabel("Condition")

    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
