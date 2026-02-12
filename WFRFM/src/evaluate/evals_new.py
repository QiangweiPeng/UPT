import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
import torch

from scipy.spatial.distance import cosine

from sklearn.metrics import r2_score
from scipy.stats import rankdata
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr 
from scipy.stats import wasserstein_distance

from scipy import sparse
import warnings  


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

def compute_edistance_torch(X, Y, max_n=2000, seed=42, device='cuda'):
    """
    使用 PyTorch 加速 E-distance 计算
    """
    rng = np.random.default_rng(seed)
    if X.shape[0] > max_n:
        X = X[rng.choice(X.shape[0], max_n, replace=False)]
    if Y.shape[0] > max_n:
        Y = Y[rng.choice(Y.shape[0], max_n, replace=False)]
        
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y, dtype=torch.float32)
        
    X = X.to(device)
    Y = Y.to(device)
    
    # 3. 计算距离 (p=2 代表欧几里得距离)
    d_xy = torch.cdist(X, Y, p=2).mean()
    d_xx = torch.cdist(X, X, p=2).mean()
    d_yy = torch.cdist(Y, Y, p=2).mean()
    
    result = 2 * d_xy - d_xx - d_yy
    return result.item()



def evaluate_latent(
    results_embedding, 
    adata_treated, 
    adata_control, 
    pert_key="condition", 
    control_label="is_control",
    embedding_key="sample_rep_scaled"
):
    metrics_list = []
    
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

    df = pd.DataFrame(metrics_list)
    mean_row = df.drop(columns=['perturbation']).mean(numeric_only=True).to_dict()
    mean_row['perturbation'] = 'mean'
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    
    return df


def evaluate_population_average(
    results_genes, 
    adata_treated, 
    adata_control, 
    pert_key="condition", 
    control_label="is_control",
    embedding_key="sample_rep_scaled", 
    Edistance_sample_num = 2000,
    random_seed = 42,
    detailed = True
):
    metrics_list = []
    
    X_ctrl = adata_to_numpy(adata_control)
    mean_ctrl = np.nanmean(X_ctrl, axis=0)
    
    # true_emb_ctrl = get_embedding(adata_control, embedding_key)
    # ctrl_mean_latent = np.nanmean(true_emb_ctrl, axis=0)

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

        if detailed:
            mse_true = np.mean((mean_true - mean_ctrl)**2)
            row['mse_true'] = mse_true

        # MAE
        mae = np.mean(np.abs(mean_true - mean_pred))
        row['mae_gene'] = mae


        #R^2
        var_true = np.var(mean_true) 
        r2 = 1 - (mse / var_true)
        row['r2_gene'] = r2

        if detailed:
            row['r2_true'] = 1 - ( mse_true / np.var(mean_ctrl))

        # deg 50  这里deg逻辑和scanpy不同
        diff_abs = np.abs(mean_true - mean_ctrl)
        top50_idx = np.argsort(diff_abs)[-50:]
        
        mean_true_deg = mean_true[top50_idx]
        mean_pred_deg = mean_pred[top50_idx]
        
        mse_deg = np.mean((mean_true_deg - mean_pred_deg) ** 2)
        var_true_deg = np.var(mean_true_deg)
        r2_deg = 1 - (mse_deg / var_true_deg)
        
        row['r2_gene_deg50'] = r2_deg
        row['mse_gene_deg50'] = mse_deg


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
                compute_edistance_torch(X_true, X_pred_rs,max_n=Edistance_sample_num, seed=seed)
            )
        
        row['e_distance'] = np.mean(e_vals)
        row['e_distance_std'] = np.std(e_vals)

        # PCC delta
        delta_true = mean_true - mean_ctrl
        delta_pred = mean_pred - mean_ctrl
        if np.std(delta_true) == 0 or np.std(delta_pred) == 0:
            pcc_delta = 0.0
            spearman_delta = 0.0
        else:
            pcc_delta, _ = pearsonr(delta_true, delta_pred)
            spearman_delta, _ = spearmanr(delta_true, delta_pred)
        row['pcc_delta'] = pcc_delta
        row['spearman_delta'] = spearman_delta

        metrics_list.append(row)

    df = pd.DataFrame(metrics_list)
    mean_row = df.drop(columns=['perturbation']).mean(numeric_only=True).to_dict()
    mean_row['perturbation'] = 'mean'
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    return df



def _normalize_weights(w):
    w = np.asarray(w, dtype=float)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    s = w.sum()
    if s <= 0:
        return None
    return w / s



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
    极简版差异基因计算。
    自动屏蔽警告，自动修复负值/NaN数据，防止 crash。
    """
    # 屏蔽所有 Warning，眼不见为净
    # ps 这个warning源自我们对权重的重采样
    # 另外这个warning实在是不知道怎么屏蔽了 无所谓了
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # 1. 直接合并 (index_unique="-" 保证了内部逻辑不冲突)
        adata = ad.concat(
            {"pert": adata_pert, "ctrl": adata_ctrl}, 
            label="group", join="inner", merge="same", index_unique="-"
        )

        # 2. 极简数据清洗 (转Dense -> 填补NaN -> 截断负值) 一气呵成
        # rank_genes_groups 必须用 Dense 且非负数据
        X = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
        adata.X = np.nan_to_num(X, nan=0.0).clip(min=0)

        # 3. 计算并返回
        try:
            sc.tl.rank_genes_groups(
                adata, groupby="group", groups=["pert"], reference="ctrl", 
                method='t-test', use_raw=False
            )
            # 链式调用直接拿结果
            return sc.get.rank_genes_groups_df(adata, group="pert")["names"][:top_n].tolist()
        except Exception:
            return []


def compute_metrics_batch_unequal(X_true, X_pred, n_bins=50, n_quantiles=100, eps=1e-12, device='cuda'):
    """
    [GPU 加速版 - 适配不等长数据]
    计算 Wasserstein (通过分位数近似) 和 KL 散度。
    
    参数:
    X_true: (N_true, n_features)
    X_pred: (N_pred, n_features) - N_pred 可以不等于 N_true
    n_bins: KL 散度的直方图箱数
    n_quantiles: Wasserstein 距离的采样点数 (通常 100 或 1000 足够精确)
    """
    
    # 1. 转换为 GPU Tensor
    # 使用 nan_to_num 填充 NaN，防止计算崩溃
    xt = torch.as_tensor(X_true, dtype=torch.float32, device=device)
    xp = torch.as_tensor(X_pred, dtype=torch.float32, device=device)
    
    xt = torch.nan_to_num(xt)
    xp = torch.nan_to_num(xp)

    N_t, D = xt.shape
    N_p, _ = xp.shape
    
    # =========================================================
    # Metric 1: Wasserstein Distance (基于 Quantile 采样)
    # =========================================================
    # 原理：当样本数不同时，我们在 [0, 1] 范围内取 n_quantiles 个等间距概率点
    # 计算两个分布在这些概率点上的值 (即分位数)，然后比较这些值的距离。
    
    # 生成概率网格: [0.0, 0.01, ..., 0.99, 1.0]
    # 注意：如果 PyTorch 版本较老(<1.7)，quantile 可能不支持 dim 参数，建议升级
    quantiles_grid = torch.linspace(0, 1, steps=n_quantiles, device=device)
    
    # 计算分位数 (Batch processing all features)
    # shape: (n_quantiles, D)
    q_t = torch.quantile(xt, quantiles_grid, dim=0)
    q_p = torch.quantile(xp, quantiles_grid, dim=0)
    
    # 计算分位数之间的 L1 距离作为 Wasserstein 的近似
    # shape: (D,)
    w1_per_gene = torch.abs(q_t - q_p).mean(dim=0)

    # =========================================================
    # Metric 2: KL Divergence (基于直方图)
    # =========================================================
    # 逻辑与等长版本相同，直方图会自动归一化，不受样本数影响
    
    # 1. 确定全局范围 (Global Min/Max)
    min_t, max_t = xt.min(dim=0)[0], xt.max(dim=0)[0]
    min_p, max_p = xp.min(dim=0)[0], xp.max(dim=0)[0]
    
    g_min = torch.minimum(min_t, min_p)
    g_max = torch.maximum(max_t, max_p)
    
    # 2. 计算 Bin 索引
    ranges = g_max - g_min
    ranges[ranges < eps] = 1.0 # 防止除零
    
    # 映射到 [0, n_bins-1]
    bin_idx_t = ((xt - g_min) / ranges * n_bins).long().clamp(0, n_bins - 1)
    bin_idx_p = ((xp - g_min) / ranges * n_bins).long().clamp(0, n_bins - 1)
    
    # 3. 扁平化索引技巧 (Offset trick)
    # 让不同列的数据落在不同的大桶里，以便一次性 bincount
    offset = torch.arange(D, device=device) * n_bins
    
    # 展平索引
    flat_idx_t = (bin_idx_t + offset).reshape(-1) # shape: (N_t * D)
    flat_idx_p = (bin_idx_p + offset).reshape(-1) # shape: (N_p * D)
    
    # 4. 统计频数
    # minlength 保证了即使某些 bin 为空，形状也是对的
    hist_t_flat = torch.bincount(flat_idx_t, minlength=D*n_bins).float()
    hist_p_flat = torch.bincount(flat_idx_p, minlength=D*n_bins).float()
    
    # 恢复形状 (D, n_bins)
    hist_t = hist_t_flat.view(D, n_bins)
    hist_p = hist_p_flat.view(D, n_bins)
    
    # 5. 归一化 (转换为概率分布)
    # 除以各自的样本数 (实际上是各自 histogram 的 sum)，这就解决了不等长问题
    prob_t = (hist_t + eps) / (hist_t.sum(dim=1, keepdim=True) + eps * n_bins)
    prob_p = (hist_p + eps) / (hist_p.sum(dim=1, keepdim=True) + eps * n_bins)
    
    # 6. 计算 KL
    kl_per_gene = (prob_t * torch.log(prob_t / prob_p)).sum(dim=1)
    
    return w1_per_gene.cpu().numpy(), kl_per_gene.cpu().numpy()


def _rank_degs_top_names_fast(adata_pert, adata_ctrl, top_n=200):
    """
    [极速 CPU 版] 不创建 AnnData，直接算 T-test。
    比 scanpy 快 10-50 倍。
    """
    # 1. 获取数据矩阵 (假设 X 是 (cells, genes))
    X_p = adata_pert.X
    X_c = adata_ctrl.X
    
    # 基因名列表
    gene_names = np.array(adata_pert.var_names)

    # 2. 如果是稀疏矩阵，计算均值和方差需要特定处理，或者转稠密
    # 考虑到差异分析通常只涉及几千个细胞，转稠密通常是最快的
    if sparse.issparse(X_p): X_p = X_p.toarray()
    if sparse.issparse(X_c): X_c = X_c.toarray()
    
    # 处理 NaN 和 负值 (与你原逻辑保持一致)
    X_p = np.nan_to_num(X_p, nan=0.0).clip(min=0)
    X_c = np.nan_to_num(X_c, nan=0.0).clip(min=0)

    # 3. 手写 Welch's t-test (不等方差 T 检验)
    # 计算均值、方差、样本数
    n_p = X_p.shape[0]
    n_c = X_c.shape[0]
    
    # 防止除以 0
    if n_p <= 1 or n_c <= 1:
        return []

    mean_p = X_p.mean(axis=0)
    mean_c = X_c.mean(axis=0)
    
    var_p = X_p.var(axis=0, ddof=1)
    var_c = X_c.var(axis=0, ddof=1)
    
    # 添加极小值防止分母为0
    epsilon = 1e-12
    denominator = np.sqrt((var_p / n_p) + (var_c / n_c)) + epsilon
    
    # T-statistics
    t_scores = (mean_p - mean_c) / denominator
    
    # 4. 获取 Top N (使用 argpartition 比全排序快)
    # 我们需要最大的 T 值 (正向富集)
    # 负号用于 argsort/argpartition 实现降序
    if top_n >= len(gene_names):
        top_indices = np.argsort(-t_scores)
    else:
        # argpartition 只排前 K 个，复杂度 O(N)
        top_indices = np.argpartition(-t_scores, top_n)[:top_n]
        # 由于 argpartition 内部不保证顺序，取出后再对这 top_n 个排一下序
        top_indices = top_indices[np.argsort(-t_scores[top_indices])]
        
    return gene_names[top_indices].tolist()


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
    gene_idx = np.arange(adata_control.shape[1])

    pert_names = list(results_genes.keys())
    print(f"Start evaluating {len(pert_names)} perturbations (Distribution metrics)")

    

    for pert in pert_names:
        row = {"perturbation": pert}

        # true pert cells
        adata_true_pert = adata_treated[adata_treated.obs[pert_key] == pert]
        if adata_true_pert.n_obs == 0:
            print(f"{pert} no observation")
            continue

        adata_pred_pert = results_genes[pert]

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

        # Wasserstein + KL
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        w1_vals, kl_vals = compute_metrics_batch_unequal(
            X_true, X_pred, n_bins=n_bins, n_quantiles=100, device=device
        )
        
        row["wasserstein_mean"] = float(np.nanmean(w1_vals))
        row["wasserstein_std"]  = float(np.nanstd(w1_vals))
        row["kl_mean"]          = float(np.nanmean(kl_vals))
        row["kl_std"]           = float(np.nanstd(kl_vals))

        # ---- Common-DEGs ----
        # 预测 DEGs：用 idx_pred 的重采样子集（把 mass 体现进来）
        adata_pred_rs = adata_pred_pert[idx_pred].copy()

        # 注意：DEG 这里用原始 adata_control（你也可以改成 control 下采样）
        true_top = _rank_degs_top_names_fast(adata_true_pert, adata_control, top_n=top_n_degs)
        pred_top = _rank_degs_top_names_fast(adata_pred_rs,   adata_control, top_n=top_n_degs)

        overlap = len(set(true_top).intersection(set(pred_top)))
        row["common_degs"] = overlap / float(top_n_degs)

        metrics_list.append(row)

    df = pd.DataFrame(metrics_list)
    mean_row = df.drop(columns=['perturbation']).mean(numeric_only=True).to_dict()
    mean_row['perturbation'] = 'mean'
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    return df


