import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp

from scipy.spatial.distance import cosine

from sklearn.metrics import r2_score
from scipy.stats import rankdata
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from scipy.stats import wasserstein_distance


def adata_to_numpy(data, copy=True, dtype="float32"):
    """
    稳健地转换为 numpy.ndarray。
    兼容输入为: AnnData, scipy.sparse, 或 numpy.ndarray
    """
    # 1. 如果是 AnnData，取其 .X
    if hasattr(data, "X"):
        X = data.X
    else:
        # 如果不是 AnnData，假设它已经是矩阵（sparse 或 numpy）
        X = data

    # 2. 如果是稀疏矩阵，转为 dense
    if sp.issparse(X):
        X = X.toarray()

    # 3. 处理 numpy matrix 类型 (old style)
    if hasattr(X, "A"):
        X = X.A

    # 4. 确保是 ndarray
    X = np.asarray(X, dtype=dtype)

    if copy:
        return X.copy()
    return X


def get_embedding(adata, key):
    """
    辅助函数：从 AnnData 中获取 latent embedding。
    adata[key]
    """
    return adata.obsm[key]


def compute_edistance(X, Y, max_n=2000, seed=42):
    """
    实现文献中的 E-distance (Equation 2)。
    由于计算所有细胞对的距离矩阵非常耗时，按照文献通常做法进行下采样。
    """
    rng = np.random.default_rng(seed)
    
    if X.shape[0] > max_n:
        X = X[rng.choice(X.shape[0], max_n, replace=False)]
    if Y.shape[0] > max_n:
        Y = Y[rng.choice(Y.shape[0], max_n, replace=False)]
        
    d_xy = cdist(X, Y, metric='euclidean').mean()
    d_xx = cdist(X, X, metric='euclidean').mean()
    d_yy = cdist(Y, Y, metric='euclidean').mean()
    
    return 2 * d_xy - d_xx - d_yy


def evaluate_latent(
    results_embedding, 
    adata_treated, 
    adata_control, 
    pert_key="condition", 
    control_label="is_control",
    embedding_key="sample_rep_scaled"
):
    metrics_list = []
    
    ctrl_X = adata_to_numpy(adata_control)
    ctrl_mean_gene = np.nanmean(ctrl_X, axis=0)
    
    true_emb_ctrl = get_embedding(adata_control, embedding_key)
    ctrl_mean_latent = np.nanmean(true_emb_ctrl, axis=0)

    pert_names = list(results_embedding.keys())
    print(f"Start evaluating {len(pert_names)} perturbations (Pseudo-bulk)")

    for pert in pert_names:
        row = {'perturbation': pert}
        
        adata_true_pert = adata_treated[adata_treated.obs[pert_key] == pert]
        if adata_true_pert.n_obs == 0:
            print(f"{pert} no observation")
            continue
            
        # latent space
        Z_true = get_embedding(adata_true_pert, embedding_key)
        Z_pred = results_embedding[pert]['z_pred']
        Z_m_pred = results_embedding[pert]['m_pred'].flatten()
        Z_m_pred = np.asarray(Z_m_pred)
        Z_m_pred = Z_m_pred / Z_m_pred.sum()
        
        z_mean_true = np.mean(Z_true, axis=0)
        z_mean_pred = np.average(Z_pred, axis=0, weights = Z_m_pred)
        

        row['latent_mse'] = np.mean((z_mean_true - z_mean_pred)**2)
        row['latent_cosine'] = 1 - cosine(z_mean_true, z_mean_pred)
        
        delta_z_true = z_mean_true - ctrl_mean_latent
        delta_z_pred = z_mean_pred - ctrl_mean_latent
        row['latent_delta_cosine'] = 1 - cosine(delta_z_true, delta_z_pred)

        metrics_list.append(row)

    return pd.DataFrame(metrics_list)


def evaluate_population_average(
    results_genes, 
    adata_treated, 
    adata_control, 
    pert_key="condition", 
    control_label="is_control",
    embedding_key="sample_rep_scaled",
    Edistance_sample_num = 2000,
    random_seed = 42,
):
    metrics_list = []
    
    ctrl_X = adata_to_numpy(adata_control)
    ctrl_mean_gene = np.nanmean(ctrl_X, axis=0)
    
    true_emb_ctrl = get_embedding(adata_control, embedding_key)
    ctrl_mean_latent = np.nanmean(true_emb_ctrl, axis=0)

    pert_names = list(results_genes.keys())
    print(f"Start evaluating {len(pert_names)} perturbations (Pseudo-bulk)")

    for pert in pert_names:
        row = {'perturbation': pert}
        
        adata_true_pert = adata_treated[adata_treated.obs[pert_key] == pert]
        if adata_true_pert.n_obs == 0:
            print(f"{pert} no observation")
            continue
        X_true = adata_to_numpy(adata_true_pert)
        
        adata_pred_pert = results_genes[pert]
        X_pred = adata_to_numpy(adata_pred_pert)
        m_pred = adata_pred_pert.obs['mass']
        m_pred = np.asarray(m_pred)
        m_pred = m_pred / m_pred.sum()
        
        mean_true = np.nanmean(X_true, axis=0)
        mean_pred = np.average(X_pred, axis=0, weights = m_pred) # 根据 m_pred 加权

        # MSE
        mse = np.mean((mean_true - mean_pred) ** 2)
        row['mse_gene'] = mse

        # E-distance
        e_vals = []
        for seed in [random_seed*0, random_seed*1, random_seed*2, random_seed*3, random_seed*4]: # MC引入m_pred
            rng = np.random.default_rng(seed)
            idx = rng.choice(
                np.arange(X_pred.shape[0]),
                size=min(Edistance_sample_num, X_pred.shape[0]),
                replace=True,
                p=m_pred
            )
            X_pred_rs = X_pred[idx]
            e_vals.append(
                compute_edistance(X_true, X_pred_rs,max_n=Edistance_sample_num, seed=seed)
            )
        
        row['e_distance'] = np.mean(e_vals)
        row['e_distance_std'] = np.std(e_vals)

        # PCC delta
        delta_true = mean_true - ctrl_mean_gene
        delta_pred = mean_pred - ctrl_mean_gene
        if np.std(delta_true) == 0 or np.std(delta_pred) == 0:
            pcc_delta = 0.0
        else:
            pcc_delta, _ = pearsonr(delta_true, delta_pred)
        row['pcc_delta'] = pcc_delta


        metrics_list.append(row)

    return pd.DataFrame(metrics_list)



def _normalize_weights(w):
    w = np.asarray(w, dtype=float)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    s = w.sum()
    if s <= 0:
        return None
    return w / s


def _select_gene_indices(adata, max_genes=2000):
    """
    默认优先用 adata.var['highly_variable']（如果存在）。
    若不存在则取前 max_genes 个基因（或全部）。
    """
    n_genes = adata.shape[1]
    if 'highly_variable' in adata.var.columns:
        idx = np.where(adata.var['highly_variable'].values)[0]
        if idx.size == 0:
            idx = np.arange(n_genes)
    else:
        idx = np.arange(n_genes)

    if max_genes is not None and idx.size > max_genes:
        idx = idx[:max_genes]
    return idx


def _kl_divergence_hist(p_samples, q_samples, bins=50, eps=1e-12):
    """
    用共享 bins 的直方图估计 KL(P||Q) = sum p log(p/q)。
    """
    p_samples = np.asarray(p_samples, dtype=float)
    q_samples = np.asarray(q_samples, dtype=float)

    vmin = min(np.min(p_samples), np.min(q_samples))
    vmax = max(np.max(p_samples), np.max(q_samples))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return 0.0

    hist_p, edges = np.histogram(p_samples, bins=bins, range=(vmin, vmax), density=False)
    hist_q, _     = np.histogram(q_samples, bins=edges, density=False)

    p = hist_p.astype(float) + eps
    q = hist_q.astype(float) + eps
    p /= p.sum()
    q /= q.sum()

    return float(np.sum(p * np.log(p / q)))


def _rank_degs_top_names(adata_pert, adata_ctrl, top_n=200, seed=42):
    """
    用 Scanpy rank_genes_groups 的默认参数（除 groupby/groups/reference 必须指定）。
    返回 top_n 基因名列表。
    """
    # 合并成一个 adata：mapping key 已经包含了类别信息，所以不要再传 keys=
    adata_mix = ad.concat(
        {"pert": adata_pert, "ctrl": adata_ctrl},
        label="group",
        join="inner",
        merge="same"
    )

    sc.tl.rank_genes_groups(
        adata_mix,
        groupby="group",
        groups=["pert"],
        reference="ctrl"
    )

    df = sc.get.rank_genes_groups_df(adata_mix, group="pert")
    return df["names"].astype(str).tolist()[:top_n]


def evaluate_population_distribution(
    results_genes,
    adata_treated,
    adata_control,
    pert_key="condition",
    max_cells=2000,
    max_genes=2000,
    n_bins=50,
    top_n_degs=200,
    seed=42,
):
    """
    对每个 perturbation 计算：
      - wasserstein_mean：按基因的一维 Wasserstein 距离平均
      - kl_mean：按基因的 KL(P||Q) 平均（P=true, Q=pred）
      - common_degs：|TopDEG_true ∩ TopDEG_pred| / top_n_degs
    """
    rng = np.random.default_rng(seed)
    metrics_list = []

    # 选基因子集
    gene_idx = _select_gene_indices(adata_control, max_genes=max_genes)

    pert_names = list(results_genes.keys())
    print(f"Start evaluating {len(pert_names)} perturbations (Distribution metrics)")

    for pert in pert_names:
        row = {"perturbation": pert}

        # true pert cells
        adata_true_pert = adata_treated[adata_treated.obs[pert_key] == pert]
        if adata_true_pert.n_obs == 0:
            print(f"{pert} no observation")
            continue

        # predicted pert cells (AnnData)
        adata_pred_pert = results_genes[pert]

        # ---- subsample cells for distribution metrics ----
        n_true = adata_true_pert.n_obs
        n_pred = adata_pred_pert.n_obs

        idx_true = rng.choice(n_true, size=min(max_cells, n_true), replace=False)

        # pred 使用 mass 做加权重采样（若没有 mass，则均匀采样）
        if "mass" in adata_pred_pert.obs.columns:
            w = _normalize_weights(adata_pred_pert.obs["mass"].values)
        else:
            w = None

        if w is None:
            idx_pred = rng.choice(n_pred, size=min(max_cells, n_pred), replace=False if n_pred >= max_cells else True)
        else:
            idx_pred = rng.choice(n_pred, size=min(max_cells, n_pred), replace=True, p=w)

        X_true = adata_to_numpy(adata_true_pert[idx_true].X)[:, gene_idx]
        X_pred = adata_to_numpy(adata_pred_pert[idx_pred].X)[:, gene_idx]

        # ---- Wasserstein + KL (per gene -> mean) ----
        w1_vals = []
        kl_vals = []
        for j in range(X_true.shape[1]):
            xt = X_true[:, j]
            xp = X_pred[:, j]

            # Wasserstein
            try:
                w1 = wasserstein_distance(xt, xp)
            except Exception:
                w1 = np.nan
            w1_vals.append(w1)

            # KL(P||Q) via hist
            try:
                kl = _kl_divergence_hist(xt, xp, bins=n_bins)
            except Exception:
                kl = np.nan
            kl_vals.append(kl)

        row["wasserstein_mean"] = float(np.nanmean(w1_vals))
        row["wasserstein_std"]  = float(np.nanstd(w1_vals))
        row["kl_mean"]          = float(np.nanmean(kl_vals))
        row["kl_std"]           = float(np.nanstd(kl_vals))

        # ---- Common-DEGs ----
        # 预测 DEGs：用 idx_pred 的重采样子集（把 mass 体现进来）
        adata_pred_rs = adata_pred_pert[idx_pred].copy()

        # 注意：DEG 这里用原始 adata_control（你也可以改成 control 下采样）
        true_top = _rank_degs_top_names(adata_true_pert, adata_control, top_n=top_n_degs, seed=seed)
        pred_top = _rank_degs_top_names(adata_pred_rs,   adata_control, top_n=top_n_degs, seed=seed)

        overlap = len(set(true_top).intersection(set(pred_top)))
        row["common_degs"] = overlap / float(top_n_degs)

        metrics_list.append(row)

    return pd.DataFrame(metrics_list)
