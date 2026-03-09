import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import cosine

import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import rankdata
import scipy.sparse as sp
from scipy.spatial.distance import cdist

import numpy as np
import scipy.sparse as sp

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



def calculate_r2_scores(preds, trues):
    """Calculate global R² and median per-protein R²."""

    trues = adata_to_numpy(trues)
    preds = adata_to_numpy(preds)

    # 全局R²：保留都不是NaN的点

    valid_idx = (~np.isnan(trues.flatten())) & (~np.isnan(preds.flatten()))
    if np.sum(valid_idx) > 1:
        global_r2 = r2_score(trues.flatten()[valid_idx], preds.flatten()[valid_idx])
    else:
        global_r2 = np.nan
    print(trues.flatten()[valid_idx].shape)
    per_protein_r2_scores = []
    for i in range(preds.shape[1]):
        yt = trues[:, i]
        yp = preds[:, i]
        valid_idx = (~np.isnan(yt)) & (~np.isnan(yp))
        if np.sum(valid_idx) > 1 and np.var(yt[valid_idx]) > 0:
            per_protein_r2_scores.append(
                r2_score(yt[valid_idx], yp[valid_idx])
            )
    median_per_protein_r2 = np.mean(per_protein_r2_scores) if per_protein_r2_scores else np.nan
    return global_r2, median_per_protein_r2, np.array(per_protein_r2_scores)


def calculate_mse(y_true, y_pred):
    """
    计算 MSE（Mean Squared Error）

    """
    y_true = adata_to_numpy(y_true)
    y_pred = adata_to_numpy(y_pred)
    
    if hasattr(y_true, "A"):  # A?
        y_true = y_true.A
    if hasattr(y_pred, "A"):
        y_pred = y_pred.A
    
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    
    diff = y_true[mask] - y_pred[mask]
    mse = np.mean(diff ** 2)

    return mse

def calculate_mae(y_true, y_pred):
    """
    计算 MAE
    """
    y_true = adata_to_numpy(y_true)
    y_pred = adata_to_numpy(y_pred)
    if hasattr(y_true, "A"): 
        y_true = y_true.A
    if hasattr(y_pred, "A"):
        y_pred = y_pred.A
    
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)

    diff = y_true[mask] - y_pred[mask]
    mae = np.mean(np.abs(diff))

    return mae


def compute_delta_mean(adata_ctrl, adata_pert):
    """
    根据 AnnData 计算扰动效应向量 delta_x (G 维)

    输入：
        adata_ctrl  : 对照组 AnnData
        adata_pert  : 扰动组 AnnData

    输出：
        delta_x : shape (G,) 的 numpy vector
    """
    Xc = adata_ctrl.X.A if hasattr(adata_ctrl.X, "A") else adata_ctrl.X
    Xp = adata_pert.X.A  if hasattr(adata_pert.X,  "A") else adata_pert.X

    mean_ctrl = np.nanmean(Xc, axis=0)
    mean_pert = np.nanmean(Xp, axis=0)

    return mean_pert - mean_ctrl


def compute_pcc(delta_true, delta_pred):
    """
    计算皮尔逊相关系数（PCC），完全按用户提供的公式实现。

    参数：
        delta_true : shape (G,) 或 (G,1)
        delta_pred : shape (G,) 或 (G,1)
    """


    if hasattr(delta_true, "A"):
        delta_true = delta_true.A.ravel()
    if hasattr(delta_pred, "A"):
        delta_pred = delta_pred.A.ravel()

    delta_true = np.asarray(delta_true, dtype=float).ravel()
    delta_pred = np.asarray(delta_pred, dtype=float).ravel()

    mask = ~np.isnan(delta_true) & ~np.isnan(delta_pred)
    dt = delta_true[mask]
    dp = delta_pred[mask]

    if dt.size < 2:
        return np.nan

    mu_true = dt.mean()
    mu_pred = dp.mean()

    numerator = np.sum((dt - mu_true) * (dp - mu_pred))
    denominator = np.sqrt(np.sum((dt - mu_true) ** 2)) * np.sqrt(np.sum((dp - mu_pred) ** 2))

    if denominator == 0:
        return np.nan

    return numerator / denominator


def calculate_pcc(adata_true, adata_pred,key="condition",key_control="ctrl",key_pert=None):
    """
    计算 pcc（单个扰动 p）
    """
    if key is None or key_control is None:
        raise ValueError("key and key_control should not be None")
    
    true_ctrl = adata_true[adata_true.obs[key]==key_control]
    pred_ctrl = adata_pred[adata_pred.obs[key]==key_control]
    if key_pert is not None:

        true_pert = adata_true[adata_true.obs[key]==key_pert]
        pred_pert = adata_pred[adata_pred.obs[key]==key_pert]

    else :
        true_pert = adata_true[adata_true.obs[key]!=key_control]
        pred_pert = adata_pred[adata_pred.obs[key]!=key_control]
    delta_true = compute_delta_mean(true_ctrl, true_pert)
    delta_pred = compute_delta_mean(pred_ctrl, pred_pert)

    pcc = compute_pcc(delta_true, delta_pred)

    return pcc




def compute_spearman(delta_true, delta_pred):
    """
    计算 Spearman 秩相关（完全按公式：先转秩，再算 Pearson）
    
    输入：
        delta_true : (G,)
        delta_pred : (G,)
    输出：
        spearman_r : float
    """

    if hasattr(delta_true, "A"):
        delta_true = delta_true.A.ravel()
    if hasattr(delta_pred, "A"):
        delta_pred = delta_pred.A.ravel()

    delta_true = np.asarray(delta_true, dtype=float).ravel()
    delta_pred = np.asarray(delta_pred, dtype=float).ravel()

    mask = ~np.isnan(delta_true) & ~np.isnan(delta_pred)
    dt = delta_true[mask]
    dp = delta_pred[mask]

    if dt.size < 2:
        return np.nan
    
    rt = rankdata(dt)
    rp = rankdata(dp)

    mu_t = rt.mean()
    mu_p = rp.mean()

    numerator = np.sum((rt - mu_t) * (rp - mu_p))
    denominator = (
        np.sqrt(np.sum((rt - mu_t)**2)) * 
        np.sqrt(np.sum((rp - mu_p)**2))
    )

    if denominator == 0:
        return np.nan

    return numerator / denominator

def calculate_spearman(adata_true, adata_pred, key="condition", key_control="ctrl", key_pert=None):
    """
    计算 Spearman 秩相关（单个扰动 p）
    """

    if key is None or key_control is None:
        raise ValueError("key and key_control should not be None")
    

    true_ctrl = adata_true[adata_true.obs[key] == key_control]
    pred_ctrl = adata_pred[adata_pred.obs[key] == key_control]

    if key_pert is not None:

        true_pert = adata_true[adata_true.obs[key]==key_pert]
        pred_pert = adata_pred[adata_pred.obs[key]==key_pert]

    else :
        true_pert = adata_true[adata_true.obs[key]!=key_control]
        pred_pert = adata_pred[adata_pred.obs[key]!=key_control]


    delta_true = compute_delta_mean(true_ctrl, true_pert)
    delta_pred = compute_delta_mean(pred_ctrl, pred_pert)

    return compute_spearman(delta_true, delta_pred)


def compute_pdisc(delta_true_dict, delta_pred_dict):
    pert_names = list(delta_true_dict.keys())
    T = len(pert_names)
    pdisc_dict = {}
    true_matrix = np.stack([delta_true_dict[p] for p in pert_names])
    
    for i, pert in enumerate(pert_names):
        pred_vec = delta_pred_dict[pert].reshape(1, -1)
        distances = cdist(pred_vec, true_matrix, metric='euclidean').flatten()
        r_t = np.sum(distances[np.arange(T) != i] < distances[i])
        pdisc_dict[pert] = r_t / T
    pdisc_mean = np.mean(list(pdisc_dict.values()))
    return pdisc_dict, pdisc_mean


def cauculate_pdiscn(
    adata_true, adata_pred, 
    key="condition", key_control="ctrl", key_pert_list=None
):
    """
    输入:
        adata_true : 实际 AnnData
        adata_pred : 预测 AnnData
        key : 存储条件的列名
        key_control : 对照组名称
        key_pert_list : 扰动列表 (如果 None 就用 adata_true 中唯一值)
    输出:
        PDISC
    """
    if key_pert_list is None:
        key_pert_list = adata_true.obs[key].unique().tolist()
        key_pert_list = [k for k in key_pert_list if k != key_control]

    delta_true_dict = {}
    delta_pred_dict = {}

    for pert in key_pert_list:
        true_ctrl = adata_true[adata_true.obs[key] == key_control]
        true_pert = adata_true[adata_true.obs[key] == pert]
        pred_ctrl = adata_pred[adata_pred.obs[key] == key_control]
        pred_pert = adata_pred[adata_pred.obs[key] == pert]

        delta_true = compute_delta_mean(true_ctrl, true_pert)
        delta_pred = compute_delta_mean(pred_ctrl, pred_pert)

        delta_true_dict[pert] = delta_true
        delta_pred_dict[pert] = delta_pred


        
    pdisc_dict, pdisc_mean = compute_pdisc(delta_true_dict, delta_pred_dict)


    return pdisc_dict, pdisc_mean




def get_embedding_matrix(adata, key):
    """
    辅助函数：从 AnnData 中获取 embedding。
    支持 adata.obsm['key'] 或 adata.layers['key'] 或 adata[key]
    """
    if key in adata.obsm.keys():
        return adata.obsm[key]
    elif key in adata.layers.keys():
        return adata.layers[key]
    elif key in adata.obs.keys():
        # 这种情况比较少见，通常 embedding 是矩阵
        return adata.obs[key].values
    else:
        # 尝试直接通过 adata[key] (用户提到的 adata_control[sample_rep_scaled])
        try:
            res = adata[key]
            if hasattr(res, "X"): # 如果返回的是 View
                return res.X
            return res
        except:
            raise ValueError(f"Could not find embedding key '{key}' in adata.")

def evaluate_all(
    results_genes, 
    results_embedding, 
    adata_train, 
    adata_control, 
    pert_key="condition", 
    control_label="ctrl",
    embedding_key="sample_rep_scaled"
):
    metrics_list = []
    
    # 1. 预计算 Control 的均值 (用于 Delta 计算)
    # 使用之前修复过的 adata_to_numpy
    ctrl_X = adata_to_numpy(adata_control)
    ctrl_mean_gene = np.nanmean(ctrl_X, axis=0)
    
    # Control 的 Latent 均值
    true_emb_ctrl = get_embedding_matrix(adata_control, embedding_key)
    ctrl_mean_latent = np.nanmean(true_emb_ctrl, axis=0)

    pert_names = list(results_genes.keys())
    print(f"Start evaluating {len(pert_names)} perturbations (Pseudo-bulk mode)...")

    for pert in pert_names:
        row = {'perturbation': pert}
        
        # =======================
        # A. 准备数据
        # =======================
        
        # 1. 获取 Gene Space 真实值
        adata_true_pert = adata_train[adata_train.obs[pert_key] == pert]
        if adata_true_pert.n_obs == 0:
            continue
        X_true = adata_to_numpy(adata_true_pert)
        
        # 2. 获取 Gene Space 预测值
        adata_pred_pert = results_genes[pert]
        X_pred = adata_to_numpy(adata_pred_pert)
        
        # =======================
        # B. Gene Space Metrics (Pseudo-bulk)
        # =======================
        # 核心修改：计算均值向量 (Shape: [n_genes])
        # 因为细胞数不匹配 (150 vs 3656)，必须对比均值
        mean_true = np.nanmean(X_true, axis=0)
        mean_pred = np.nanmean(X_pred, axis=0)
        
        # 1. Mean Gene MSE/MAE
        # 对比两个向量的差异
        row['gene_mse'] = np.mean((mean_true - mean_pred) ** 2)
        row['gene_mae'] = np.mean(np.abs(mean_true - mean_pred))
        
        # 2. Global R2 (Means correlation across genes)
        # 衡量预测的平均表达谱和真实的平均表达谱的相关性
        # 注意：这里不能算 per-gene R2，因为只有一个样本(均值)
        row['gene_r2_global'] = r2_score(mean_true, mean_pred)
        
        # 3. Delta Metrics (PCC, Spearman)
        # 均值改变量的相关性
        delta_pred = mean_pred - ctrl_mean_gene
        delta_true = mean_true - ctrl_mean_gene
        
        row['gene_pcc_delta'] = compute_pcc(delta_true, delta_pred)
        row['gene_spearman_delta'] = compute_spearman(delta_true, delta_pred)

        # =======================
        # C. Latent Space Metrics
        # =======================
        
        Z_true = get_embedding_matrix(adata_true_pert, embedding_key)
        Z_pred = results_embedding[pert]['z_pred']
        
        # 同样使用均值对比
        z_mean_true = np.mean(Z_true, axis=0)
        z_mean_pred = np.mean(Z_pred, axis=0)
        
        # Latent MSE (Means)
        row['latent_mse'] = np.mean((z_mean_true - z_mean_pred)**2)
        
        # Latent Cosine Similarity (Means)
        # 1 - cosine distance = cosine similarity
        row['latent_cosine'] = 1 - cosine(z_mean_true, z_mean_pred)
        
        # Latent Delta Cosine (Direction consistency)
        delta_z_true = z_mean_true - ctrl_mean_latent
        delta_z_pred = z_mean_pred - ctrl_mean_latent
        row['latent_delta_cosine'] = 1 - cosine(delta_z_true, delta_z_pred)

        metrics_list.append(row)

    return pd.DataFrame(metrics_list)

# --- 扩展：计算 PDISC (Separability) ---
# PDISC 需要所有扰动的 Delta 集合，因此不能在上面的单次循环中通过
def evaluate_pdisc(results_genes, adata_train, adata_control, pert_key="condition"):
    """
    专门计算 PDISC 的包装器
    """
    pert_names = list(results_genes.keys())
    ctrl_mean = np.nanmean(adata_to_numpy(adata_control), axis=0)
    
    delta_true_dict = {}
    delta_pred_dict = {}
    
    for pert in pert_names:
        # 获取真实数据
        adata_true = adata_train[adata_train.obs[pert_key] == pert]
        if adata_true.n_obs == 0: continue
        
        mean_true = np.nanmean(adata_to_numpy(adata_true), axis=0)
        mean_pred = np.nanmean(adata_to_numpy(results_genes[pert]), axis=0)
        
        delta_true_dict[pert] = mean_true - ctrl_mean
        delta_pred_dict[pert] = mean_pred - ctrl_mean
        
    pdisc_dict, pdisc_mean = compute_pdisc(delta_true_dict, delta_pred_dict)
    return pdisc_dict, pdisc_mean
