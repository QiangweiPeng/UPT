import ast
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import scanpy as sc
import torch

from scipy.stats import pearsonr, spearmanr, mannwhitneyu

from statsmodels.stats.multitest import multipletests
from sklearn.neighbors import KNeighborsClassifier


# =========================
# Basic helpers
# =========================

def adata_to_numpy(data, copy=True, dtype="float32"):
    """
    稳健地转换为 numpy.ndarray。
    兼容输入为: AnnData, scipy.sparse, 或 numpy.ndarray
    """
    if hasattr(data, "X"):
        X = data.X
    else:
        X = data

    if sp.issparse(X):
        X = X.toarray()

    if hasattr(X, "A"):
        X = X.A

    X = np.asarray(X, dtype=dtype)
    return X.copy() if copy else X


def get_embedding(adata, key):
    """
    从 AnnData.obsm 中获取 latent embedding。
    """
    return adata.obsm[key]


def _sanitize_mass(weights, n):
    if weights is None:
        return np.ones(n, dtype=float)

    w = np.asarray(weights, dtype=float).reshape(-1)
    if len(w) != n:
        raise ValueError(f"Weight length mismatch: got {len(w)}, expected {n}")

    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w[w < 0] = 0.0

    if w.sum() <= 0:
        return np.ones(n, dtype=float)

    return w


def _subsample_true(X, max_cells=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n <= max_cells:
        return X
    idx = rng.choice(n, size=max_cells, replace=False)
    return X[idx]


def _subsample_pred_with_mass(X, mass, max_cells=2000, seed=42):
    """
    按 mass 重采样预测细胞。
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n == 0:
        return X

    w = _sanitize_mass(mass, n)
    w = w / w.sum()

    sample_n = max(1, int(max_cells))
    idx = rng.choice(n, size=sample_n, replace=True, p=w)
    return X[idx]

def _resample_adata_with_mass(
    adata_obj,
    mass,
    sample_n=2000,
    seed=42,
):
    """
    按 mass 对 AnnData 细胞进行有放回重采样。
    用于需要把预测 mass 反映到下游统计（如 DEG）时。
    """
    n = adata_obj.n_obs
    if n == 0:
        return adata_obj.copy()

    w = _sanitize_mass(mass, n)
    w = w / w.sum()

    rng = np.random.default_rng(seed)
    sample_n = max(1, int(sample_n))
    idx = rng.choice(n, size=sample_n, replace=True, p=w)
    return adata_obj[idx].copy()


# =========================
# Control matching
# =========================

def _get_matched_control(adata_control, pert_tuple_str, rulebook, use_groupwise_control=True):
    """
    根据 rulebook 动态解析 tuple 字符串，从 adata_control 中筛选精准的 Control 子集。
    若匹配失败则回退到全局 Control。
    """
    if not use_groupwise_control or not rulebook:
        return adata_control

    control_groups = rulebook.get("stratification", {}).get("control_groups", [])
    if len(control_groups) == 0:
        return adata_control

    control_groups = list(control_groups)
    schema = list(rulebook.get("condition_tuple_schema", []))

    try:
        actual_tuple = ast.literal_eval(pert_tuple_str)
        mask = np.ones(adata_control.n_obs, dtype=bool)

        for i, col in enumerate(schema):
            if col in control_groups:
                mask &= (adata_control.obs[col].astype(str) == str(actual_tuple[i]))

        curr_ctrl_sub = adata_control[mask]

        if curr_ctrl_sub.n_obs > 0:
            return curr_ctrl_sub
        else:
            print(f"⚠️ 警告: 未找到 {pert_tuple_str} 对应的 Groupwise Control，已回退到全局 Control。")
            return adata_control
    except Exception as e:
        print(f"⚠️ 警告: 解析 Control Tuple {pert_tuple_str} 失败 ({e})，已回退到全局 Control。")
        return adata_control


# =========================
# Label transfer
# =========================

def transfer_cell_labels_knn(
    results_embedding,
    results_genes,
    adata_reference,
    cell_type_key="celltype",
    embedding_key="X_pca",
    n_neighbors=15,
    max_ref_cells=20000,
    seed=42,
):
    """
    用 reference 的 latent/embedding 训练 KNN，把 cell type 标签转移到生成细胞。
    """
    print(f"Training KNN (k={n_neighbors}) for label transfer using '{embedding_key}'...")
    rng = np.random.default_rng(seed)

    n_ref = adata_reference.n_obs
    if n_ref > max_ref_cells:
        idx = rng.choice(n_ref, max_ref_cells, replace=False)
        adata_ref_sub = adata_reference[idx]
    else:
        adata_ref_sub = adata_reference

    X_ref = adata_ref_sub.obsm[embedding_key]
    y_ref = adata_ref_sub.obs[cell_type_key].values

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", n_jobs=-1)
    knn.fit(X_ref, y_ref)

    predicted_labels_dict = {}
    for pert, res in results_embedding.items():
        Z_pred = res["z_pred"]
        y_pred = knn.predict(Z_pred)
        predicted_labels_dict[pert] = y_pred

        if results_genes is not None and pert in results_genes:
            results_genes[pert].obs[f"{cell_type_key}_pred"] = y_pred

    print("Label transfer completed.")
    return predicted_labels_dict





# =========================
# Aggregation helpers
# =========================

def _weighted_mean_safe(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if mask.sum() == 0:
        return np.nan

    v = values[mask]
    w = weights[mask]
    w_sum = w.sum()
    if w_sum <= 0:
        return np.nan

    return np.average(v, weights=w)


def summarize_stratified_metrics_two_stage(df, weight_col="n_true_ct"):
    """
    两阶段汇总:
    1) perturbation 内按 celltype 的 n_true_ct 加权
    2) across perturbation 简单平均
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    base_df = df.copy()

    if "perturbation" in base_df.columns:
        base_df = base_df[base_df["perturbation"] != "mean"].copy()

    numeric_cols = base_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = {weight_col, "n_ctrl_ct", "n_pred_ct"}
    metric_cols = [c for c in numeric_cols if c not in exclude_cols]

    per_pert_rows = []
    for pert, g in base_df.groupby("perturbation", sort=False):
        row = {"perturbation": pert, "cell_type": "all"}
        row[weight_col] = g[weight_col].sum()

        if "n_ctrl_ct" in g.columns:
            row["n_ctrl_ct"] = g["n_ctrl_ct"].sum()
        if "n_pred_ct" in g.columns:
            row["n_pred_ct"] = g["n_pred_ct"].sum()

        for col in metric_cols:
            row[col] = _weighted_mean_safe(g[col].values, g[weight_col].values)

        per_pert_rows.append(row)

    df_per_pert = pd.DataFrame(per_pert_rows)

    overall_row = {"perturbation": "mean", "cell_type": "all"}
    if not df_per_pert.empty:
        if weight_col in df_per_pert.columns:
            overall_row[weight_col] = df_per_pert[weight_col].sum()
        if "n_ctrl_ct" in df_per_pert.columns:
            overall_row["n_ctrl_ct"] = df_per_pert["n_ctrl_ct"].sum()
        if "n_pred_ct" in df_per_pert.columns:
            overall_row["n_pred_ct"] = df_per_pert["n_pred_ct"].sum()

        for col in metric_cols:
            overall_row[col] = df_per_pert[col].mean(skipna=True)

    df_overall = pd.DataFrame([overall_row])
    return df_per_pert, df_overall


# =========================
# Cell-type proportion metrics
# =========================

def compute_celltype_proportion_metrics(
    y_true,
    y_pred,
    pred_mass,
    true_mass=None,
    eps=1e-12,
):
    """
    比较真实 vs 预测 的 celltype mass proportion。
    """
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)

    w_true = _sanitize_mass(true_mass, len(y_true))
    w_pred = _sanitize_mass(pred_mass, len(y_pred))

    all_cts = sorted(set(y_true).union(set(y_pred)))

    true_df = pd.DataFrame({"cell_type": y_true, "mass": w_true})
    pred_df = pd.DataFrame({"cell_type": y_pred, "mass": w_pred})

    true_mass_by_ct = (
        true_df.groupby("cell_type")["mass"].sum().reindex(all_cts, fill_value=0.0)
    )
    pred_mass_by_ct = (
        pred_df.groupby("cell_type")["mass"].sum().reindex(all_cts, fill_value=0.0)
    )

    p_true = true_mass_by_ct.values.astype(float)
    p_pred = pred_mass_by_ct.values.astype(float)

    p_true = p_true / max(p_true.sum(), eps)
    p_pred = p_pred / max(p_pred.sum(), eps)

    abs_err = np.abs(p_true - p_pred)

    l1 = abs_err.sum()
    tv = 0.5 * l1
    rmse = np.sqrt(np.mean((p_true - p_pred) ** 2))

    m = 0.5 * (p_true + p_pred)
    js = (
        0.5 * np.sum(p_true * np.log((p_true + eps) / (m + eps)))
        + 0.5 * np.sum(p_pred * np.log((p_pred + eps) / (m + eps)))
    )

    true_present = p_true > 0
    offtarget_mass_ratio = p_pred[~true_present].sum()

    detail_df = pd.DataFrame({
        "cell_type": all_cts,
        "true_prop_mass": p_true,
        "pred_prop_mass": p_pred,
        "abs_prop_error_mass": abs_err,
    })

    metrics = {
        "celltype_prop_l1_mass": float(l1),
        "celltype_prop_tv_mass": float(tv),
        "celltype_prop_rmse_mass": float(rmse),
        "celltype_prop_js_mass": float(js),
        "celltype_offtarget_mass_ratio": float(offtarget_mass_ratio),
        "celltype_prop_max_abs_err_mass": float(abs_err.max() if len(abs_err) > 0 else np.nan),
    }
    return metrics, detail_df



# =========================
# wasserstein helper
# =========================

def _subsample_indices(n, max_cells=2000, seed=42, weights=None):
    """
    返回下采样索引。
    - weights=None: 均匀无放回采样
    - weights!=None: 按权重无放回采样
    """
    rng = np.random.default_rng(seed)

    if n <= max_cells:
        return np.arange(n)

    if weights is None:
        return rng.choice(n, size=max_cells, replace=False)

    w = _sanitize_mass(weights, n)
    w = w / w.sum()
    return rng.choice(n, size=max_cells, replace=False, p=w)


def _sinkhorn_wasserstein_torch(
    X,
    Y,
    a=None,
    b=None,
    reg=0.1,
    max_iter=200,
    tol=1e-6,
    squared_cost=False,
    device=None,
):
    """
    纯 PyTorch 的 entropic Sinkhorn OT.
    返回的是带 entropic regularization 的 transport cost。

    参数
    ----
    X: [n, d]
    Y: [m, d]
    a: X 上的权重，长度 n；若 None 则均匀
    b: Y 上的权重，长度 m；若 None 则均匀
    reg: Sinkhorn 正则强度
    squared_cost:
        False -> ground cost = Euclidean distance
        True  -> ground cost = squared Euclidean distance
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    else:
        X = X.to(dtype=torch.float32)

    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y, dtype=torch.float32)
    else:
        Y = Y.to(dtype=torch.float32)

    X = X.to(device)
    Y = Y.to(device)

    n = X.shape[0]
    m = Y.shape[0]

    if n == 0 or m == 0:
        return np.nan

    if a is None:
        a = torch.full((n,), 1.0 / n, dtype=torch.float32, device=device)
    else:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        a = np.clip(a, 0.0, None)
        if a.sum() <= 0:
            a = np.ones(n, dtype=np.float32) / n
        else:
            a = a / a.sum()
        a = torch.tensor(a, dtype=torch.float32, device=device)

    if b is None:
        b = torch.full((m,), 1.0 / m, dtype=torch.float32, device=device)
    else:
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        b = np.clip(b, 0.0, None)
        if b.sum() <= 0:
            b = np.ones(m, dtype=np.float32) / m
        else:
            b = b / b.sum()
        b = torch.tensor(b, dtype=torch.float32, device=device)

    # ground cost
    C = torch.cdist(X, Y, p=2)
    if squared_cost:
        C = C.pow(2)

    # 为了数值稳定，把 cost 缩放到一个合理量级
    cost_scale = torch.median(C.detach())
    if (not torch.isfinite(cost_scale)) or (cost_scale <= 0):
        cost_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    C_scaled = C / cost_scale

    log_a = torch.log(a.clamp_min(1e-12))
    log_b = torch.log(b.clamp_min(1e-12))

    # log K = -C / reg
    log_K = -C_scaled / reg

    log_u = torch.zeros_like(a)
    log_v = torch.zeros_like(b)

    for _ in range(max_iter):
        prev_log_u = log_u.clone()

        log_u = log_a - torch.logsumexp(log_K + log_v[None, :], dim=1)
        log_v = log_b - torch.logsumexp(log_K.T + log_u[None, :], dim=1)

        if torch.max(torch.abs(log_u - prev_log_u)) < tol:
            break

    log_P = log_u[:, None] + log_K + log_v[None, :]
    P = torch.exp(log_P)

    ot_cost = torch.sum(P * C_scaled)
    ot_cost = ot_cost * cost_scale

    return float(ot_cost.item())



# =========================
# Distribution metrics
# =========================

def compute_edistance_torch(X, Y, max_n=2000, seed=42, device="cuda"):
    """
    使用 PyTorch 计算 E-distance。
    使用平方欧式距离，与 CellFlow 口径一致。
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

    d_xy = torch.cdist(X, Y, p=2).pow(2).mean()
    d_xx = torch.cdist(X, X, p=2).pow(2).mean()
    d_yy = torch.cdist(Y, Y, p=2).pow(2).mean()

    result = 2 * d_xy - d_xx - d_yy
    return float(result.item())


def compute_wasserstein_latent(
    Z_true,
    Z_pred,
    pred_mass=None,
    true_mass=None,
    max_cells=2000,
    seed=42,
    reg=0.1,
    max_iter=200,
    tol=1e-6,
    squared_cost=False,
    device=None,
):
    """
    纯 PyTorch 版本 Wasserstein / Sinkhorn OT（latent 上）。

    口径：
    - true: 默认均匀权重；如果以后你想支持 true_mass，也可以传
    - pred: 直接使用 pred_mass 作为 OT 权重
    - 为了控制复杂度，对 true / pred 都做下采样
    """
    Z_true = np.asarray(Z_true, dtype=np.float32)
    Z_pred = np.asarray(Z_pred, dtype=np.float32)

    if Z_true.shape[0] == 0 or Z_pred.shape[0] == 0:
        return np.nan

    # true 侧：均匀下采样
    idx_true = _subsample_indices(
        Z_true.shape[0],
        max_cells=max_cells,
        seed=seed,
        weights=true_mass,
    )
    Z_true_sub = Z_true[idx_true]

    if true_mass is None:
        a = None
    else:
        true_mass = _sanitize_mass(true_mass, len(Z_true))
        a = true_mass[idx_true]
        a = a / a.sum()

    # pred 侧：按 pred_mass 加权下采样，并保留对应权重
    idx_pred = _subsample_indices(
        Z_pred.shape[0],
        max_cells=max_cells,
        seed=seed,
        weights=pred_mass,
    )
    Z_pred_sub = Z_pred[idx_pred]

    if pred_mass is None:
        b = None
    else:
        pred_mass = _sanitize_mass(pred_mass, len(Z_pred))
        b = pred_mass[idx_pred]
        b = b / b.sum()

    return _sinkhorn_wasserstein_torch(
        Z_true_sub,
        Z_pred_sub,
        a=a,
        b=b,
        reg=reg,
        max_iter=max_iter,
        tol=tol,
        squared_cost=squared_cost,
        device=device,
    )



# =========================
# DEG helper
# =========================

def _get_gene_indices(var_names, gene_names):
    """
    把 gene name 列表映射成 var index。
    """
    var_names = np.asarray(var_names)
    name_to_idx = {g: i for i, g in enumerate(var_names)}
    return [name_to_idx[g] for g in gene_names if g in name_to_idx]



def _get_deg_set_with_cutoff(
    adata_pert,
    adata_ctrl,
    padj_cutoff=0.01,
    fc_cutoff=2.0,
    top_n=None,
):
    """
    与 Scanpy 当前 rank_genes_groups(wilcoxon) 口径对齐。

    口径：
    - 检验：scanpy.tl.rank_genes_groups(method="wilcoxon")
    - 多重检验校正：Benjamini-Hochberg
    - FC：使用 Scanpy 风格的 approximate logfoldchanges
          log2((expm1(mean_log_pert) + 1e-9) / (expm1(mean_log_ctrl) + 1e-9))
      注意：这不是
          log2(mean(expm1(pert)) / mean(expm1(ctrl)))
    - 排序：沿用 Scanpy 返回顺序；
            当 rankby_abs=True 时，结果已按 |score| 排好序
    - 适用前提：adata.X 是 logarithmized / log1p 表达矩阵
    """
    if adata_pert.n_obs < 2 or adata_ctrl.n_obs < 2:
        return []

    def _mean_on_current_matrix(X):
        """按 Scanpy 的思路：直接在当前矩阵上取 mean(axis=0)。"""
        if sp.issparse(X):
            return np.asarray(X.mean(axis=0)).ravel()
        return np.asarray(X, dtype=np.float64).mean(axis=0)

    def _get_expm1_func(*adatas):
        """
        对齐 Scanpy:
        如果 adata.uns['log1p']['base'] 存在，则用对应 base 的逆变换；
        否则默认 np.expm1（对应自然对数 log1p）。
        """
        for a in adatas:
            base = a.uns.get("log1p", {}).get("base", None)
            if base is not None:
                return lambda x, base=base: np.expm1(x * np.log(base))
        return np.expm1

    # 1) 对齐基因，并强制使用同一顺序
    common_genes = adata_pert.var_names.intersection(adata_ctrl.var_names)
    if len(common_genes) == 0:
        return []

    adata_pert_use = adata_pert[:, common_genes].copy()
    adata_ctrl_use = adata_ctrl[:, common_genes].copy()

    # 2) 合并后跑 Scanpy Wilcoxon
    adata_deg = ad.concat(
        [adata_pert_use, adata_ctrl_use],
        axis=0,
        join="inner",
        merge="same",
        label="_deg_group",
        keys=["pert", "ctrl"],
        index_unique=None,
    )
    if not adata_deg.obs_names.is_unique:
        adata_deg.obs_names_make_unique()

    sc.tl.rank_genes_groups(
        adata_deg,
        groupby="_deg_group",
        groups=["pert"],
        reference="ctrl",
        method="wilcoxon",
        corr_method="benjamini-hochberg",
        tie_correct=True,
        use_raw=False,
        rankby_abs=True,
        n_genes=adata_deg.n_vars,
    )

    deg_df = sc.get.rank_genes_groups_df(adata_deg, group="pert").copy()
    if deg_df.empty:
        return []

    # 3) 使用 Scanpy 的 logfoldchanges；若当前版本/场景没给出，则按 Scanpy 源码口径回退重算
    if "logfoldchanges" not in deg_df.columns or deg_df["logfoldchanges"].isna().all():
        mean_pert = _mean_on_current_matrix(adata_pert_use.X)
        mean_ctrl = _mean_on_current_matrix(adata_ctrl_use.X)

        expm1_func = _get_expm1_func(adata_deg, adata_pert_use, adata_ctrl_use)
        eps = 1e-9
        log2fc = np.log2(
            (expm1_func(mean_pert) + eps) / (expm1_func(mean_ctrl) + eps)
        )
        log2fc_s = pd.Series(log2fc, index=adata_pert_use.var_names)
        deg_df["logfoldchanges"] = deg_df["names"].map(log2fc_s)

    # 4) 过滤：显著性 + FC cutoff
    # 这里保留你原来的接口：fc_cutoff=2.0 表示 fold change >= 2
    log2fc_thr = np.log2(fc_cutoff)

    deg_df = deg_df[
        deg_df["pvals_adj"].notna()
        & deg_df["logfoldchanges"].notna()
        & (deg_df["pvals_adj"] <= padj_cutoff)
        & (np.abs(deg_df["logfoldchanges"]) >= log2fc_thr)
    ].copy()

    if deg_df.empty:
        return []

    # 5) 不再手动重排
    # Scanpy 在 rankby_abs=True 时，返回结果本身就已经按 |score| 排序
    if top_n is not None:
        deg_df = deg_df.head(top_n)

    return deg_df["names"].tolist()







# =========================
# Main evaluation: latent
# =========================

def evaluate_latent(
    results_embedding,
    adata_treated,
    adata_control,
    pert_key="condition",
    embedding_key="sample_rep_scaled",
    rulebook=None,
    use_groupwise_control=True,
    distribution_sample_num=2000,
    random_seed=42,
):
    metrics_list = []
    pert_names = list(results_embedding.keys())
    print(f"Start evaluating {len(pert_names)} perturbations (Latent Space)")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for pert in pert_names:
        row = {"perturbation": pert}

        adata_true_pert = adata_treated[adata_treated.obs[pert_key] == pert]
        if adata_true_pert.n_obs == 0:
            print(f"{pert} no observation")
            continue

        curr_ctrl = _get_matched_control(adata_control, pert, rulebook, use_groupwise_control)
        true_emb_ctrl = get_embedding(curr_ctrl, embedding_key)
        ctrl_mean_latent = np.nanmean(true_emb_ctrl, axis=0)

        Z_true = get_embedding(adata_true_pert, embedding_key)
        Z_pred = results_embedding[pert]["z_pred"]

        Z_m_pred = _sanitize_mass(
            np.asarray(results_embedding[pert]["m_pred"]).flatten(),
            len(results_embedding[pert]["m_pred"]),
        )
        m_pred_sum = Z_m_pred.sum()
        m_pred_weights = Z_m_pred / m_pred_sum if m_pred_sum > 0 else np.ones_like(Z_m_pred) / len(Z_m_pred)

        # pseudo-bulk mean metrics
        z_mean_true = np.mean(Z_true, axis=0)
        z_mean_pred = np.average(Z_pred, axis=0, weights=m_pred_weights)

        mse = np.mean((z_mean_true - z_mean_pred) ** 2)
        row["latent_mse"] = mse

        var_true = np.var(z_mean_true)
        row["latent_r2"] = 1 - (mse / var_true) if var_true > 0 else 0.0

        delta_z_true = z_mean_true - ctrl_mean_latent
        delta_z_pred = z_mean_pred - ctrl_mean_latent

        if np.std(delta_z_true) == 0 or np.std(delta_z_pred) == 0:
            row["latent_pcc_delta"] = 0.0
        else:
            row["latent_pcc_delta"], _ = pearsonr(delta_z_true, delta_z_pred)

        # distribution metrics: E-distance + Wasserstein
        e_vals, w_vals = [], []
        for seed in [random_seed + i for i in range(3)]:
            rng = np.random.default_rng(seed)

            idx_true = rng.choice(
                Z_true.shape[0],
                size=min(distribution_sample_num, Z_true.shape[0]),
                replace=False,
            )
            Z_true_rs = Z_true[idx_true]

            idx_pred = rng.choice(
                Z_pred.shape[0],
                size=min(distribution_sample_num, Z_pred.shape[0]),
                replace=True,
                p=m_pred_weights,
            )
            Z_pred_rs = Z_pred[idx_pred]

            e_vals.append(
                compute_edistance_torch(
                    Z_true_rs,
                    Z_pred_rs,
                    max_n=distribution_sample_num,
                    seed=seed,
                    device=device,
                )
            )

            w_vals.append(
                compute_wasserstein_latent(
                    Z_true=Z_true,
                    Z_pred=Z_pred,
                    pred_mass=Z_m_pred,
                    max_cells=distribution_sample_num,
                    seed=seed,
                    reg=0.1,
                    max_iter=200,
                    tol=1e-6,
                    squared_cost=True, #用sqEucidean
                    device=device,
                )
            )


        row["latent_energy_distance"] = float(np.mean(e_vals))
        row["latent_wasserstein"] = float(np.mean(w_vals))

        metrics_list.append(row)

    df = pd.DataFrame(metrics_list)
    if not df.empty:
        mean_row = df.drop(columns=["perturbation"]).mean(numeric_only=True).to_dict()
        mean_row["perturbation"] = "mean"
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    return df


# =========================
# Main evaluation: pseudo-bulk genes
# =========================

def evaluate_population_average(
    results_genes,
    adata_treated,
    adata_control,
    pert_key="condition",
    detailed=True,
    rulebook=None,
    use_groupwise_control=True,
    mass_deduct_keys=None,
    deg_padj_cutoff=0.01,
    deg_fc_cutoff=1.0,
    deg_top_n_for_eval=50,
):
    metrics_list = []

    m_source_val = adata_control.uns.get("normalized_m", 1.0)
    m_target_val = adata_treated.uns.get("normalized_m", 1.0)

    pert_names = list(results_genes.keys())
    print(f"Start evaluating {len(pert_names)} perturbations (Pseudo-bulk)")

    for pert in pert_names:
        row = {"perturbation": pert}

        adata_true_pert = adata_treated[adata_treated.obs[pert_key] == pert]
        if adata_true_pert.n_obs == 0:
            print(f"{pert} no observation")
            continue

        curr_ctrl = _get_matched_control(adata_control, pert, rulebook, use_groupwise_control)
        if curr_ctrl.n_obs == 0:
            print(f"Warning: {pert} has no matching control cells. Skipping.")
            continue

        X_ctrl = adata_to_numpy(curr_ctrl)
        mean_ctrl = np.nanmean(X_ctrl, axis=0)

        if (
            mass_deduct_keys is not None
            and mass_deduct_keys in curr_ctrl.obs.columns
            and mass_deduct_keys in adata_true_pert.obs.columns
        ):
            n_unique_source = len(curr_ctrl.obs[mass_deduct_keys].unique())
            n_unique_target = len(adata_true_pert.obs[mass_deduct_keys].unique())
            adj_m_target = m_target_val * max(1, n_unique_source) / max(1, n_unique_target)
        else:
            adj_m_target = m_target_val

        m_true_sum = adata_true_pert.n_obs * adj_m_target
        m_ctrl_sum = curr_ctrl.n_obs * m_source_val
        m_true_change = m_true_sum / m_ctrl_sum if m_ctrl_sum > 0 else 1.0

        adata_pred_pert = results_genes[pert]
        X_pred = adata_to_numpy(adata_pred_pert)

        m_pred_raw = np.asarray(adata_pred_pert.obs["mass"], dtype=float)
        m_pred_raw = np.nan_to_num(m_pred_raw, nan=0.0, posinf=0.0, neginf=0.0)
        m_pred_raw[m_pred_raw < 0] = 0.0

        m_pred_sum = m_pred_raw.sum()
        # m_pred_change = m_pred_sum / m_ctrl_sum if m_ctrl_sum > 0 else np.nan
        m_pred_change = m_pred_sum / adata_pred_pert.n_obs # 暂时这么改 control出发应该都是1
        m_pred_weights = (
            m_pred_raw / m_pred_sum if m_pred_sum > 0 else np.ones_like(m_pred_raw) / len(m_pred_raw)
        )

        X_true = adata_to_numpy(adata_true_pert)
        mean_true = np.nanmean(X_true, axis=0)
        mean_pred = np.average(X_pred, axis=0, weights=m_pred_weights)

        mse = np.mean((mean_true - mean_pred) ** 2)
        row["mse_gene"] = mse

        if detailed:
            mse_true = np.mean((mean_true - mean_ctrl) ** 2)
            row["mse_identity"] = mse_true

        row["mae_gene"] = np.mean(np.abs(mean_true - mean_pred))

        var_true = np.var(mean_true)
        row["r2_gene"] = 1 - (mse / var_true) if var_true > 0 else 0.0

        if detailed:
            var_ctrl = np.var(mean_ctrl)
            row["r2_identity"] = 1 - (mse_true / var_true) if var_true > 0 else 0.0

        # 改成：真实 treated vs control 的 top-N DEGs
        true_deg50 = _get_deg_set_with_cutoff(
            adata_true_pert,
            curr_ctrl,
            padj_cutoff=deg_padj_cutoff,
            fc_cutoff=deg_fc_cutoff,
            top_n=deg_top_n_for_eval,
        )

        if len(true_deg50) == 0:
            row["mse_gene_deg50"] = np.nan
            row["r2_gene_deg50"] = np.nan
        else:
            deg50_idx = _get_gene_indices(adata_true_pert.var_names, true_deg50)

            mean_true_deg = mean_true[deg50_idx]
            mean_pred_deg = mean_pred[deg50_idx]

            mse_deg = np.mean((mean_true_deg - mean_pred_deg) ** 2)
            var_true_deg = np.var(mean_true_deg)

            row["mse_gene_deg50"] = mse_deg
            row["r2_gene_deg50"] = 1 - (mse_deg / var_true_deg) if var_true_deg > 0 else 0.0
            if detailed:
                mean_ctrl_deg = mean_ctrl[deg50_idx]
                mse_true_deg = np.mean((mean_true_deg - mean_ctrl_deg) ** 2)
                var_ctrl_deg = np.var(mean_ctrl_deg)
                row["mse_identity_deg50"] = mse_true_deg
                row["r2_identity_deg50"] = (
                    1 - (mse_true_deg / var_true_deg) if var_true_deg > 0 else 0.0
                )

        delta_true = mean_true - mean_ctrl
        delta_pred = mean_pred - mean_ctrl
        if np.std(delta_true) == 0 or np.std(delta_pred) == 0:
            row["pcc_delta"] = 0.0
            row["spearman_delta"] = 0.0
        else:
            row["pcc_delta"], _ = pearsonr(delta_true, delta_pred)
            row["spearman_delta"], _ = spearmanr(delta_true, delta_pred)

        row["m_true_change"] = m_true_change
        row["m_pred_change"] = m_pred_change

        metrics_list.append(row)

    df = pd.DataFrame(metrics_list)
    if not df.empty:
        mean_row = df.drop(columns=["perturbation"]).mean(numeric_only=True).to_dict()
        mean_row["perturbation"] = "mean"
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    return df



# =========================
# Main evaluation: Common-DEGs only
# =========================




def evaluate_population_distribution(
    results_genes,
    adata_treated,
    adata_control,
    pert_key="condition",
    max_cells=2000,
    top_n_degs=200,
    seed=42,
    rulebook=None,
    use_groupwise_control=True,
    deg_padj_cutoff=0.01,
    deg_fc_cutoff=1.0,
):
    metrics_list = []

    pert_names = list(results_genes.keys())
    print(f"Start evaluating {len(pert_names)} perturbations (DEG overlap, mass-aware pred)")

    for pert in pert_names:
        row = {"perturbation": pert}

        adata_true_pert = adata_treated[adata_treated.obs[pert_key] == pert]
        if adata_true_pert.n_obs == 0:
            print(f"{pert} no observation")
            continue

        curr_ctrl = _get_matched_control(
            adata_control, pert, rulebook, use_groupwise_control
        )

        adata_pred_pert = results_genes[pert]


        # ---- TRUE DEG ----
        true_deg = _get_deg_set_with_cutoff(
            adata_true_pert,
            curr_ctrl,
            padj_cutoff=deg_padj_cutoff,
            fc_cutoff=deg_fc_cutoff,
            top_n=top_n_degs,
        )

        # ---- pred mass resampling ----
        if "mass" in adata_pred_pert.obs.columns:
            pred_mass = np.asarray(adata_pred_pert.obs["mass"], dtype=float)
        else:
            pred_mass = np.ones(adata_pred_pert.n_obs, dtype=float)

        adata_pred_pert_rs = _resample_adata_with_mass(
            adata_pred_pert,
            mass=pred_mass,
            sample_n=max_cells,
            seed=seed,
        )


        # ---- PRED DEG ----
        pred_deg = _get_deg_set_with_cutoff(
            adata_pred_pert_rs,
            curr_ctrl,
            padj_cutoff=deg_padj_cutoff,
            fc_cutoff=deg_fc_cutoff,
            top_n=top_n_degs,
        )

        # ---- recall ----
        if len(true_deg) == 0:
            row["deg_recall"] = np.nan
        else:
            overlap = len(set(true_deg).intersection(set(pred_deg)))
            row["deg_recall"] = overlap / float(len(true_deg))

        metrics_list.append(row)

    df = pd.DataFrame(metrics_list)
    if not df.empty:
        mean_row = df.drop(columns=["perturbation"]).mean(numeric_only=True).to_dict()
        mean_row["perturbation"] = "mean"
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    return df


# =========================
# Main evaluation: stratified by cell type
# =========================

def evaluate_stratified_by_celltype(
    results_embedding,
    results_genes,
    predicted_labels_dict,
    adata_treated,
    adata_control,
    pert_key="condition",
    cell_type_key="celltype",
    embedding_key="sample_rep_scaled",
    rulebook=None,
    use_groupwise_control=True,
    min_cells_threshold=1,
    random_seed=42,
    distribution_sample_num=500,
    top_n_degs=200,
    deg_padj_cutoff=0.01,
    deg_fc_cutoff=1.0,
    deg_top_n_for_eval=50,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    metrics_list = []
    pert_level_celltype_metrics = []
    pert_names = list(results_embedding.keys())
    print(f"Start evaluating Cell-type specific metrics across {len(pert_names)} perturbations...")

    def _safe_pearson(x, y, default=0.0):
        x = np.asarray(x)
        y = np.asarray(y)
        if x.size == 0 or y.size == 0:
            return default
        if np.std(x) == 0 or np.std(y) == 0:
            return default
        return pearsonr(x, y)[0]

    def _safe_spearman(x, y, default=0.0):
        x = np.asarray(x)
        y = np.asarray(y)
        if x.size == 0 or y.size == 0:
            return default
        if np.std(x) == 0 or np.std(y) == 0:
            return default
        return spearmanr(x, y)[0]

    for pert in pert_names:
        adata_true_pert = adata_treated[adata_treated.obs[pert_key] == pert]
        if adata_true_pert.n_obs == 0:
            continue

        curr_ctrl = _get_matched_control(adata_control, pert, rulebook, use_groupwise_control)
        unique_cell_types = adata_true_pert.obs[cell_type_key].unique()

        Z_pred_global = results_embedding[pert]["z_pred"]
        Z_m_pred_global = _sanitize_mass(
            np.asarray(results_embedding[pert]["m_pred"]).flatten(),
            len(results_embedding[pert]["m_pred"]),
        )
        y_pred_global = np.asarray(predicted_labels_dict[pert]).astype(str)

        adata_pred_pert_global = results_genes[pert] if (results_genes is not None and pert in results_genes) else None

        y_true_global = adata_true_pert.obs[cell_type_key].astype(str).values
        true_mass_global = adata_true_pert.obs["mass"].values if "mass" in adata_true_pert.obs.columns else None

        ct_prop_metrics, _ = compute_celltype_proportion_metrics(
            y_true=y_true_global,
            y_pred=y_pred_global,
            pred_mass=Z_m_pred_global,
            true_mass=true_mass_global,
        )
        ct_prop_metrics["perturbation"] = pert
        pert_level_celltype_metrics.append(ct_prop_metrics)

        for ct in unique_cell_types:
            adata_true_ct = adata_true_pert[adata_true_pert.obs[cell_type_key] == ct]
            curr_ctrl_ct = curr_ctrl[curr_ctrl.obs[cell_type_key] == ct]

            mask_pred = (y_pred_global == str(ct))
            n_pred_ct = int(mask_pred.sum())

            if (
                adata_true_ct.n_obs < min_cells_threshold
                or curr_ctrl_ct.n_obs < min_cells_threshold
                or n_pred_ct < min_cells_threshold
            ):
                continue

            row = {
                "perturbation": pert,
                "cell_type": ct,
                "n_true_ct": int(adata_true_ct.n_obs),
                "n_ctrl_ct": int(curr_ctrl_ct.n_obs),
                "n_pred_ct": int(n_pred_ct),
            }

            rng = np.random.default_rng(random_seed)

            Z_pred_ct = Z_pred_global[mask_pred]
            m_pred_ct = _sanitize_mass(Z_m_pred_global[mask_pred], n_pred_ct)

            m_pred_sum = m_pred_ct.sum()
            m_weights_ct = (
                m_pred_ct / m_pred_sum if m_pred_sum > 0 else np.ones_like(m_pred_ct, dtype=float) / len(m_pred_ct)
            )

            Z_true_ct = get_embedding(adata_true_ct, embedding_key)
            Z_ctrl_ct = get_embedding(curr_ctrl_ct, embedding_key)

            z_mean_true = np.nanmean(Z_true_ct, axis=0)
            z_mean_ctrl = np.nanmean(Z_ctrl_ct, axis=0)
            z_mean_pred = np.average(Z_pred_ct, axis=0, weights=m_weights_ct)

            row["latent_mse_per_celltype"] = np.mean((z_mean_true - z_mean_pred) ** 2)

            latent_var_true = np.var(z_mean_true)
            row["latent_r2_per_celltype"] = (
                1 - (row["latent_mse_per_celltype"] / latent_var_true) if latent_var_true > 0 else 0.0
            )

            # identity-style latent metrics
            row["latent_mse_identity_per_celltype"] = np.mean((z_mean_true - z_mean_ctrl) ** 2)
            latent_var_ctrl = np.var(z_mean_ctrl)
            row["latent_r2_identity_per_celltype"] = (
                1 - (row["latent_mse_identity_per_celltype"] / latent_var_true) if latent_var_true > 0 else 0.0
            )

            delta_z_true = z_mean_true - z_mean_ctrl
            delta_z_pred = z_mean_pred - z_mean_ctrl
            row["latent_pcc_delta_per_celltype"] = _safe_pearson(delta_z_true, delta_z_pred)

            latent_e_vals = []
            for seed in [random_seed * (i + 1) for i in range(3)]:
                rng_local = np.random.default_rng(seed)

                sample_n_true = min(distribution_sample_num, Z_true_ct.shape[0])
                sample_n_pred = min(distribution_sample_num, Z_pred_ct.shape[0])

                idx_true = rng_local.choice(Z_true_ct.shape[0], size=sample_n_true, replace=False)
                idx_pred = rng_local.choice(
                    Z_pred_ct.shape[0],
                    size=sample_n_pred,
                    replace=True,
                    p=m_weights_ct,
                )

                Z_true_rs = Z_true_ct[idx_true]
                Z_pred_rs = Z_pred_ct[idx_pred]

                latent_e_vals.append(
                    compute_edistance_torch(
                        Z_true_rs,
                        Z_pred_rs,
                        max_n=distribution_sample_num,
                        seed=seed,
                        device=device,
                    )
                )

            row["latent_energy_dist_per_celltype"] = float(np.mean(latent_e_vals))

            if adata_pred_pert_global is not None:
                adata_pred_ct = adata_pred_pert_global[mask_pred]

                X_true_ct = adata_to_numpy(adata_true_ct)
                X_ctrl_ct = adata_to_numpy(curr_ctrl_ct)
                X_pred_ct = adata_to_numpy(adata_pred_ct)

                gene_mean_true = np.nanmean(X_true_ct, axis=0)
                gene_mean_ctrl = np.nanmean(X_ctrl_ct, axis=0)
                gene_mean_pred = np.average(X_pred_ct, axis=0, weights=m_weights_ct)

                gene_mse = np.mean((gene_mean_true - gene_mean_pred) ** 2)
                row["gene_mse_per_celltype"] = gene_mse
                row["gene_mae_per_celltype"] = np.mean(np.abs(gene_mean_true - gene_mean_pred))

                gene_var_true = np.var(gene_mean_true)
                row["gene_r2_per_celltype"] = 1 - (gene_mse / gene_var_true) if gene_var_true > 0 else 0.0

                # identity-style gene metrics
                row["gene_mse_identity_per_celltype"] = np.mean((gene_mean_true - gene_mean_ctrl) ** 2)
                gene_var_ctrl = np.var(gene_mean_ctrl)
                row["gene_r2_identity_per_celltype"] = (
                    1 - (row["gene_mse_identity_per_celltype"] / gene_var_true) if gene_var_true > 0 else 0.0
                )

                delta_g_true = gene_mean_true - gene_mean_ctrl
                delta_g_pred = gene_mean_pred - gene_mean_ctrl
                row["gene_pcc_delta_per_celltype"] = _safe_pearson(delta_g_true, delta_g_pred)
                row["gene_spearman_delta_per_celltype"] = _safe_spearman(delta_g_true, delta_g_pred)

                # 真实 celltype-specific top-N DEGs
                true_deg50 = _get_deg_set_with_cutoff(
                    adata_true_ct,
                    curr_ctrl_ct,
                    padj_cutoff=deg_padj_cutoff,
                    fc_cutoff=deg_fc_cutoff,
                    top_n=deg_top_n_for_eval,
                )

                if len(true_deg50) == 0:
                    row["gene_mse_deg50_per_celltype"] = np.nan
                    row["gene_r2_deg50_per_celltype"] = np.nan
                    row["gene_mse_identity_deg50_per_celltype"] = np.nan
                    row["gene_r2_identity_deg50_per_celltype"] = np.nan
                else:
                    deg50_idx = _get_gene_indices(adata_true_ct.var_names, true_deg50)

                    mean_true_deg50 = gene_mean_true[deg50_idx]
                    mean_pred_deg50 = gene_mean_pred[deg50_idx]
                    mean_ctrl_deg50 = gene_mean_ctrl[deg50_idx]

                    gene_mse_deg50 = np.mean((mean_true_deg50 - mean_pred_deg50) ** 2)
                    gene_var_deg50 = np.var(mean_true_deg50)

                    row["gene_mse_deg50_per_celltype"] = gene_mse_deg50
                    row["gene_r2_deg50_per_celltype"] = (
                        1 - (gene_mse_deg50 / gene_var_deg50) if gene_var_deg50 > 0 else 0.0
                    )

                    row["gene_mse_identity_deg50_per_celltype"] = np.mean(
                        (mean_true_deg50 - mean_ctrl_deg50) ** 2
                    )
                    gene_var_ctrl_deg50 = np.var(mean_ctrl_deg50)
                    row["gene_r2_identity_deg50_per_celltype"] = (
                        1 - (row["gene_mse_identity_deg50_per_celltype"] / gene_var_deg50)
                        if gene_var_deg50 > 0 else 0.0
                    )

                # adata_pred_ct_rs = _resample_adata_with_mass(
                #     adata_pred_ct,
                #     mass=m_weights_ct,
                #     sample_n=distribution_sample_num,
                #     seed=random_seed,
                # )

                # true_deg = _get_deg_set_with_cutoff(
                #     adata_true_ct,
                #     curr_ctrl_ct,
                #     padj_cutoff=deg_padj_cutoff,
                #     fc_cutoff=deg_fc_cutoff,
                #     top_n=top_n_degs
                # )
                # pred_deg = _get_deg_set_with_cutoff(
                #     adata_pred_ct_rs,
                #     curr_ctrl_ct,
                #     padj_cutoff=deg_padj_cutoff,
                #     fc_cutoff=deg_fc_cutoff,
                #     top_n=top_n_degs
                # )

                # if len(true_deg) == 0:
                #     row["gene_deg_recall_per_celltype"] = np.nan
                # else:
                #     overlap = len(set(true_deg).intersection(set(pred_deg)))
                #     row["gene_deg_recall_per_celltype"] = overlap / float(len(true_deg))

            metrics_list.append(row)

    df = pd.DataFrame(metrics_list)
    df_ctprop = pd.DataFrame(pert_level_celltype_metrics)

    if df.empty and df_ctprop.empty:
        return pd.DataFrame()

    if df.empty:
        df_per_pert = pd.DataFrame({"perturbation": df_ctprop["perturbation"], "cell_type": "all"})
        df_overall = pd.DataFrame([{"perturbation": "mean", "cell_type": "all"}])
    else:
        df_per_pert, df_overall = summarize_stratified_metrics_two_stage(df, weight_col="n_true_ct")

    if not df_ctprop.empty:
        df_per_pert = df_per_pert.merge(df_ctprop, on="perturbation", how="left")
        extra_cols = [c for c in df_ctprop.columns if c != "perturbation"]
        for col in extra_cols:
            df_overall.loc[df_overall.index[0], col] = df_ctprop[col].mean(skipna=True)

    final_df = pd.concat([df, df_per_pert, df_overall], ignore_index=True)
    return final_df



# =========================
# Baseline builders
# =========================

def _parse_pert_tuple_with_rulebook(pert_value, rulebook):
    """
    将 perturbation 值解析成 tuple，并结合 rulebook 返回 dict。
    """
    schema = list(rulebook.get("condition_tuple_schema", []))
    if len(schema) == 0:
        raise ValueError("rulebook['condition_tuple_schema'] is empty or missing.")

    if isinstance(pert_value, str):
        try:
            tuple_val = ast.literal_eval(pert_value)
        except Exception:
            raise ValueError(f"Failed to parse perturbation string: {pert_value}")
    elif isinstance(pert_value, (tuple, list, np.ndarray)):
        tuple_val = tuple(pert_value)
    else:
        raise ValueError(f"Unsupported perturbation type: {type(pert_value)}")

    if len(tuple_val) != len(schema):
        raise ValueError(f"Perturbation tuple length mismatch. tuple={tuple_val}, schema={schema}")

    meta = {k: tuple_val[i] for i, k in enumerate(schema)}
    return tuple(tuple_val), meta


def _get_X_and_Z_means(adata_obj, embedding_key):
    X = adata_to_numpy(adata_obj)
    Z = get_embedding(adata_obj, embedding_key)
    x_mean = np.nanmean(X, axis=0)
    z_mean = np.nanmean(Z, axis=0)
    return X, Z, x_mean, z_mean


def _make_pred_adata_from_control(
    ctrl_adata,
    X_pred,
    pert_value,
    donor_value,
    cytokine_value,
    pert_key,
    donor_key,
    cytokine_key,
    mass_value=1.0,
    copy_obs_meta=True,
):
    """
    基于 control population 的细胞数和 obs 生成 baseline 预测 AnnData。
    """
    if copy_obs_meta:
        obs = ctrl_adata.obs.copy()
    else:
        obs = pd.DataFrame(index=ctrl_adata.obs_names.copy())

    obs[pert_key] = pert_value
    obs[donor_key] = donor_value
    obs[cytokine_key] = cytokine_value
    obs["mass"] = float(mass_value)

    pred_adata = ad.AnnData(
        X=np.asarray(X_pred, dtype=np.float32),
        obs=obs,
        var=ctrl_adata.var.copy(),
    )
    return pred_adata


def _precompute_train_and_ctrl_means(
    adata_train,
    adata_control,
    pert_key,
    embedding_key,
    rulebook,
    use_groupwise_control,
):
    """
    预计算训练集中各 perturbation 及其 control 的 X/Z mean。
    """
    train_means = {}
    ctrl_means = {}

    unique_perts = pd.unique(adata_train.obs[pert_key])

    for pert in unique_perts:
        treated_sub = adata_train[adata_train.obs[pert_key] == pert]
        if treated_sub.n_obs == 0:
            continue

        ctrl_sub = _get_matched_control(
            adata_control,
            pert,
            rulebook,
            use_groupwise_control=use_groupwise_control,
        )
        if ctrl_sub.n_obs == 0:
            continue

        _, _, x_t, z_t = _get_X_and_Z_means(treated_sub, embedding_key)
        _, _, x_c, z_c = _get_X_and_Z_means(ctrl_sub, embedding_key)

        train_means[pert] = (x_t, z_t)
        ctrl_means[pert] = (x_c, z_c)

    return train_means, ctrl_means


def _get_unique_group_means_for_donor_baseline_fast(
    train_meta_df,
    train_means,
    ctrl_means,
    target_donor,
    target_cytokine,
    donor_key,
    cytokine_key,
    pert_key,
):
    valid_mask = (
        (train_meta_df[donor_key] == target_donor)
        & (train_meta_df[cytokine_key] != target_cytokine)
    )
    valid_perts = train_meta_df.loc[valid_mask, pert_key].unique()

    deltas_x, deltas_z = [], []
    for pert in valid_perts:
        if pert in train_means and pert in ctrl_means:
            x_t, z_t = train_means[pert]
            x_c, z_c = ctrl_means[pert]
            deltas_x.append(x_t - x_c)
            deltas_z.append(z_t - z_c)

    return deltas_x, deltas_z


def _get_unique_group_means_for_cytokine_baseline_fast(
    train_meta_df,
    train_means,
    ctrl_means,
    target_donor,
    target_cytokine,
    donor_key,
    cytokine_key,
    pert_key,
):
    valid_mask = (
        (train_meta_df[cytokine_key] == target_cytokine)
        & (train_meta_df[donor_key] != target_donor)
    )
    valid_perts = train_meta_df.loc[valid_mask, pert_key].unique()

    deltas_x, deltas_z = [], []
    for pert in valid_perts:
        if pert in train_means and pert in ctrl_means:
            x_t, z_t = train_means[pert]
            x_c, z_c = ctrl_means[pert]
            deltas_x.append(x_t - x_c)
            deltas_z.append(z_t - z_c)

    return deltas_x, deltas_z


def build_baseline_results(
    baseline_type,
    adata_control,
    adata_treated,
    adata_train,
    pert_key,
    donor_key,
    cytokine_key,
    embedding_key,
    rulebook,
    use_groupwise_control=True,
    mass_value=1.0,
    copy_obs_meta=True,
    min_reference_groups=1,
    fallback_to_identity=True,
    verbose=True,
):
    """
    构建 baseline results:
    - identity
    - mean_test_donor_response_across_cytokines
    - mean_test_cytokine_response_across_donors
    """
    valid_types = {
        "identity",
        "mean_test_donor_response_across_cytokines",
        "mean_test_cytokine_response_across_donors",
    }
    if baseline_type not in valid_types:
        raise ValueError(f"Unsupported baseline_type: {baseline_type}")

    for adata_obj in [adata_train, adata_control, adata_treated]:
        adata_obj.obs[donor_key] = adata_obj.obs[donor_key].astype(str)
        adata_obj.obs[cytokine_key] = adata_obj.obs[cytokine_key].astype(str)
        adata_obj.obs[pert_key] = adata_obj.obs[pert_key].astype(str)

    train_meta_df = adata_train.obs[[pert_key, donor_key, cytokine_key]].drop_duplicates()

    if baseline_type != "identity":
        if verbose:
            print("Precomputing all train/control group means...")
        train_means, ctrl_means = _precompute_train_and_ctrl_means(
            adata_train,
            adata_control,
            pert_key,
            embedding_key,
            rulebook,
            use_groupwise_control,
        )
    else:
        train_means, ctrl_means = {}, {}

    results_genes = {}
    results_embedding = {}

    pert_names = list(pd.unique(adata_treated.obs[pert_key]))
    if verbose:
        print(f"Building baseline '{baseline_type}' for {len(pert_names)} perturbations...")

    for pert in pert_names:
        try:
            _, pert_meta = _parse_pert_tuple_with_rulebook(pert, rulebook)
        except Exception as e:
            if verbose:
                print(f"⚠️ Skip {pert}: failed to parse perturbation ({e})")
            continue

        if donor_key not in pert_meta or cytokine_key not in pert_meta:
            raise ValueError(
                f"rulebook schema must include both donor_key='{donor_key}' and cytokine_key='{cytokine_key}'."
            )

        target_donor = str(pert_meta[donor_key])
        target_cytokine = str(pert_meta[cytokine_key])

        target_ctrl = _get_matched_control(
            adata_control,
            pert,
            rulebook,
            use_groupwise_control=use_groupwise_control,
        )
        if target_ctrl.n_obs == 0:
            if verbose:
                print(f"⚠️ Skip {pert}: no matched control found.")
            continue

        X_ctrl = adata_to_numpy(target_ctrl)
        Z_ctrl = get_embedding(target_ctrl, embedding_key)

        use_identity = False
        delta_x, delta_z = None, None

        if baseline_type == "identity":
            use_identity = True

        elif baseline_type == "mean_test_donor_response_across_cytokines":
            deltas_x, deltas_z = _get_unique_group_means_for_donor_baseline_fast(
                train_meta_df,
                train_means,
                ctrl_means,
                target_donor,
                target_cytokine,
                donor_key,
                cytokine_key,
                pert_key,
            )

            if len(deltas_x) >= min_reference_groups:
                delta_x = np.mean(np.stack(deltas_x, axis=0), axis=0)
                delta_z = np.mean(np.stack(deltas_z, axis=0), axis=0)
            elif fallback_to_identity:
                use_identity = True
            else:
                continue

        elif baseline_type == "mean_test_cytokine_response_across_donors":
            deltas_x, deltas_z = _get_unique_group_means_for_cytokine_baseline_fast(
                train_meta_df,
                train_means,
                ctrl_means,
                target_donor,
                target_cytokine,
                donor_key,
                cytokine_key,
                pert_key,
            )

            if len(deltas_x) >= min_reference_groups:
                delta_x = np.mean(np.stack(deltas_x, axis=0), axis=0)
                delta_z = np.mean(np.stack(deltas_z, axis=0), axis=0)
            elif fallback_to_identity:
                use_identity = True
            else:
                continue

        if use_identity:
            X_pred = X_ctrl.copy()
            Z_pred = np.asarray(Z_ctrl, dtype=np.float32).copy()
        else:
            X_pred = np.asarray(X_ctrl + delta_x[None, :], dtype=np.float32)
            Z_pred = np.asarray(Z_ctrl + delta_z[None, :], dtype=np.float32)

        pred_adata = _make_pred_adata_from_control(
            ctrl_adata=target_ctrl,
            X_pred=X_pred,
            pert_value=pert,
            donor_value=target_donor,
            cytokine_value=target_cytokine,
            pert_key=pert_key,
            donor_key=donor_key,
            cytokine_key=cytokine_key,
            mass_value=mass_value,
            copy_obs_meta=copy_obs_meta,
        )
        results_genes[pert] = pred_adata

        results_embedding[pert] = {
            "z_pred": np.asarray(Z_pred, dtype=np.float32),
            "m_pred": np.full(Z_pred.shape[0], float(mass_value), dtype=np.float32),
        }

    if verbose:
        print(
            f"Done. Built {len(results_genes)} gene baselines and "
            f"{len(results_embedding)} latent baselines."
        )

    return results_genes, results_embedding


