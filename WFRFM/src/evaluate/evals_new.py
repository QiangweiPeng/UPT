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


def _as_torch_2d(X, device=None, dtype=torch.float32, name="X"):
    if torch.is_tensor(X):
        tensor = X.detach()
    else:
        tensor = torch.as_tensor(X)

    if tensor.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {tuple(tensor.shape)}")

    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def _normalize_sampling_weights(weights, n_samples, name):
    if weights is None:
        return torch.full((n_samples,), 1.0 / float(n_samples), dtype=torch.float64)

    weights = torch.as_tensor(weights, dtype=torch.float64).reshape(-1).detach().cpu()
    if weights.numel() != n_samples:
        raise ValueError(
            f"{name} must have length {n_samples}, got {weights.numel()}"
        )

    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    total = weights.sum()
    if total <= 0:
        raise ValueError(f"{name} must sum to a positive value after cleaning")
    return weights / total


def _sample_indices(weights, num_samples, target_device, generator=None):
    indices_cpu = torch.multinomial(
        weights,
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )
    return indices_cpu.to(device=target_device, non_blocking=True)


def _estimate_mean_pair_distance(
    X,
    Y,
    weights_x,
    weights_y,
    num_pairs,
    batch_size,
    generator=None,
):
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    total_distance = torch.zeros((), dtype=torch.float64, device=X.device)
    remaining = int(num_pairs)

    while remaining > 0:
        cur_batch = min(batch_size, remaining)
        idx_x = _sample_indices(weights_x, cur_batch, target_device=X.device, generator=generator)
        idx_y = _sample_indices(weights_y, cur_batch, target_device=Y.device, generator=generator)
        batch_distance = torch.linalg.vector_norm(
            X.index_select(0, idx_x) - Y.index_select(0, idx_y),
            dim=1,
        )
        total_distance += batch_distance.to(torch.float64).sum()
        remaining -= cur_batch

    return total_distance / float(num_pairs)


def compute_energy_distance_torch(
    X_real,
    X_pred,
    real_weights=None,
    pred_weights=None,
    num_pairs=100000,
    batch_size=2048,
    device=None,
    dtype=torch.float32,
    seed=42,
    clamp_min_zero=False,
    return_details=False,
):
    """
    Scalable Monte-Carlo Energy Distance for two empirical distributions.

    ``real_weights`` defaults to a uniform empirical measure over real cells.
    ``pred_weights`` should usually be the predicted particle mass. It is normalized
    internally to sum to one, so this metric compares distributional geometry only.
    Population-size / total-mass mismatch should be tracked separately.
    """
    if device is None:
        if torch.is_tensor(X_real):
            device = X_real.device
        elif torch.is_tensor(X_pred):
            device = X_pred.device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    X_real = _as_torch_2d(X_real, device=device, dtype=dtype, name="X_real")
    X_pred = _as_torch_2d(X_pred, device=device, dtype=dtype, name="X_pred")

    if X_real.shape[1] != X_pred.shape[1]:
        raise ValueError(
            "X_real and X_pred must have the same feature dimension, got "
            f"{X_real.shape[1]} and {X_pred.shape[1]}"
        )

    real_weights = _normalize_sampling_weights(real_weights, X_real.shape[0], "real_weights")
    pred_weights = _normalize_sampling_weights(pred_weights, X_pred.shape[0], "pred_weights")

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))

    e_xy = _estimate_mean_pair_distance(
        X_real,
        X_pred,
        real_weights,
        pred_weights,
        num_pairs=num_pairs,
        batch_size=batch_size,
        generator=generator,
    )
    e_xx = _estimate_mean_pair_distance(
        X_real,
        X_real,
        real_weights,
        real_weights,
        num_pairs=num_pairs,
        batch_size=batch_size,
        generator=generator,
    )
    e_yy = _estimate_mean_pair_distance(
        X_pred,
        X_pred,
        pred_weights,
        pred_weights,
        num_pairs=num_pairs,
        batch_size=batch_size,
        generator=generator,
    )

    energy = 2.0 * e_xy - e_xx - e_yy
    if clamp_min_zero:
        energy = torch.clamp(energy, min=0.0)

    if return_details:
        return {
            "energy_distance": float(energy.item()),
            "cross_term": float(e_xy.item()),
            "real_self_term": float(e_xx.item()),
            "pred_self_term": float(e_yy.item()),
            "num_pairs": int(num_pairs),
            "batch_size": int(batch_size),
            "device": str(device),
        }

    return float(energy.item())


def _load_dataframe_maybe(data):
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.read_csv(data)


def _safe_pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return np.nan
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan
    return float(pearsonr(x, y)[0])


def build_mass_accuracy_tables(
    mass_summary,
    real_counts,
    control_count=None,
    condition_col="target_condition",
    time_col="target_timepoint",
    real_condition_col="gene_target",
    real_time_col="timepoint",
    real_count_col="n_obs_full",
):
    """
    Build pair-level and transition-level mass evaluation tables.

    ``mass_summary`` should provide one row per ``(condition, target_timepoint)``
    with at least ``mass_sum`` and ``n_particles``.
    ``real_counts`` should provide the matched real cell count per pair.
    """
    mass_df = _load_dataframe_maybe(mass_summary)
    real_df = _load_dataframe_maybe(real_counts)

    merged = mass_df.merge(
        real_df,
        left_on=[condition_col, time_col],
        right_on=[real_condition_col, real_time_col],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise RuntimeError("No overlapping rows between mass_summary and real_counts")

    if control_count is None:
        if "n_particles" not in merged.columns:
            raise KeyError("control_count is None and mass_summary does not contain n_particles")
        control_count = int(merged["n_particles"].iloc[0])
    control_count = int(control_count)

    merged[condition_col] = merged[condition_col].astype(str)
    merged[time_col] = merged[time_col].astype(float)
    merged[real_count_col] = merged[real_count_col].astype(float)
    merged["mass_sum"] = merged["mass_sum"].astype(float)

    merged["control_count"] = control_count
    merged["real_ratio"] = merged[real_count_col] / float(control_count)
    merged["pred_ratio"] = merged["mass_sum"] / float(control_count)

    denom = float(np.square(merged["pred_ratio"]).sum())
    calibration_factor = (
        float(np.dot(merged["real_ratio"], merged["pred_ratio"]) / denom)
        if denom > 0
        else 1.0
    )
    merged["pred_ratio_calibrated"] = calibration_factor * merged["pred_ratio"]
    merged["raw_error"] = merged["pred_ratio"] - merged["real_ratio"]
    merged["calibrated_error"] = merged["pred_ratio_calibrated"] - merged["real_ratio"]
    merged["abs_raw_error"] = merged["raw_error"].abs()
    merged["abs_calibrated_error"] = merged["calibrated_error"].abs()
    merged["pair_id"] = (
        merged[condition_col].astype(str)
        + "|t"
        + merged[time_col].map(lambda x: f"{float(x):g}")
    )
    merged = merged.sort_values([condition_col, time_col]).reset_index(drop=True)

    transition_rows = []
    for condition, cond_df in merged.groupby(condition_col, sort=True):
        prev_real = float(control_count)
        prev_pred = float(control_count)
        prev_time = 0.0
        prev_label = "control"
        for row in cond_df.sort_values(time_col).itertuples(index=False):
            real_count = float(getattr(row, real_count_col))
            pred_count = float(row.mass_sum)
            cur_time = float(getattr(row, time_col))
            real_fc = real_count / prev_real if prev_real > 0 else np.nan
            pred_fc = pred_count / prev_pred if prev_pred > 0 else np.nan
            transition_rows.append(
                {
                    "target_condition": str(condition),
                    "target_timepoint": cur_time,
                    "previous_timepoint": float(prev_time),
                    "previous_label": prev_label,
                    "transition_label": f"{prev_label}->{cur_time:g}",
                    "real_count": real_count,
                    "pred_mass_sum": pred_count,
                    "real_fc": real_fc,
                    "pred_fc": pred_fc,
                    "fc_error": pred_fc - real_fc,
                    "abs_fc_error": abs(pred_fc - real_fc),
                }
            )
            prev_real = real_count
            prev_pred = pred_count
            prev_time = cur_time
            prev_label = f"{cur_time:g}"

    transitions = pd.DataFrame(transition_rows)
    metrics = {
        "control_count": control_count,
        "n_pairs": int(merged.shape[0]),
        "n_conditions": int(merged[condition_col].nunique()),
        "calibration_factor": calibration_factor,
        "total_ratio": {
            "pearson_raw": _safe_pearson(merged["real_ratio"], merged["pred_ratio"]),
            "pearson_calibrated": _safe_pearson(
                merged["real_ratio"], merged["pred_ratio_calibrated"]
            ),
            "mae_raw": float(merged["abs_raw_error"].mean()),
            "mae_calibrated": float(merged["abs_calibrated_error"].mean()),
            "rmse_raw": float(np.sqrt(np.mean(np.square(merged["raw_error"])))),
            "rmse_calibrated": float(
                np.sqrt(np.mean(np.square(merged["calibrated_error"])))
            ),
        },
        "transition_fold_change": {
            "pearson": _safe_pearson(transitions["real_fc"], transitions["pred_fc"]),
            "mae": float(transitions["abs_fc_error"].mean()),
            "rmse": float(np.sqrt(np.mean(np.square(transitions["fc_error"])))),
        },
    }
    return merged, transitions, metrics


def _weighted_choice_without_replacement(weights, n_samples, rng):
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    n_items = weights.shape[0]
    if n_samples > n_items:
        raise ValueError("n_samples cannot exceed the number of items when replace=False")

    positive_mask = weights > 0
    n_positive = int(positive_mask.sum())
    if n_positive == 0:
        return rng.choice(n_items, size=n_samples, replace=False)

    if n_positive >= n_samples:
        prob = weights / weights.sum()
        return rng.choice(n_items, size=n_samples, replace=False, p=prob)

    selected = np.flatnonzero(positive_mask)
    remaining = n_samples - selected.size
    zero_idx = np.flatnonzero(~positive_mask)
    extra = rng.choice(zero_idx, size=remaining, replace=False)
    return np.concatenate([selected, extra])


def compute_matched_sample_metrics(
    X_real,
    X_pred,
    pred_weights=None,
    n_match=None,
    seed=42,
):
    """
    Compare real and predicted latent samples after matched-size sampling.

    - Real cells are sampled uniformly without replacement.
    - Predicted particles are sampled without replacement using normalized mass weights.
    - ``R2/PCC/MSE/MAE`` are computed on pseudo-bulk mean vectors.
    - ``W1/W2`` are computed as the mean 1D Wasserstein distance across latent dimensions.
    """
    X_real = np.asarray(X_real, dtype=np.float32)
    X_pred = np.asarray(X_pred, dtype=np.float32)
    if X_real.ndim != 2 or X_pred.ndim != 2:
        raise ValueError("X_real and X_pred must both be 2D arrays")
    if X_real.shape[1] != X_pred.shape[1]:
        raise ValueError("X_real and X_pred must share the same feature dimension")

    n_real = X_real.shape[0]
    n_pred = X_pred.shape[0]
    if n_match is None:
        n_match = min(n_real, n_pred)
    n_match = int(n_match)
    if n_match <= 0:
        raise ValueError("n_match must be positive")
    if n_match > min(n_real, n_pred):
        raise ValueError("n_match cannot exceed min(n_real, n_pred) for replace=False")

    rng = np.random.default_rng(seed)
    idx_real = (
        np.arange(n_real, dtype=np.int64)
        if n_match == n_real
        else rng.choice(n_real, size=n_match, replace=False)
    )

    if pred_weights is None:
        idx_pred = (
            np.arange(n_pred, dtype=np.int64)
            if n_match == n_pred
            else rng.choice(n_pred, size=n_match, replace=False)
        )
    else:
        idx_pred = _weighted_choice_without_replacement(pred_weights, n_match, rng)

    X_real_match = X_real[idx_real]
    X_pred_match = X_pred[idx_pred]

    mean_real = X_real_match.mean(axis=0, dtype=np.float64)
    mean_pred = X_pred_match.mean(axis=0, dtype=np.float64)
    diff = mean_pred - mean_real

    if np.std(mean_real) == 0 or np.std(mean_pred) == 0:
        pcc = np.nan
    else:
        pcc = float(pearsonr(mean_real, mean_pred)[0])

    r2 = float(r2_score(mean_real, mean_pred))
    mse = float(np.mean(np.square(diff)))
    mae = float(np.mean(np.abs(diff)))

    X_real_sorted = np.sort(X_real_match.astype(np.float64), axis=0)
    X_pred_sorted = np.sort(X_pred_match.astype(np.float64), axis=0)
    sorted_diff = X_pred_sorted - X_real_sorted
    w1 = float(np.mean(np.mean(np.abs(sorted_diff), axis=0)))
    w2 = float(np.mean(np.sqrt(np.mean(np.square(sorted_diff), axis=0))))

    return {
        "n_real_available": int(n_real),
        "n_pred_available": int(n_pred),
        "n_match": int(n_match),
        "r2": r2,
        "pcc": pcc,
        "mse": mse,
        "mae": mae,
        "w1": w1,
        "w2": w2,
    }


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
