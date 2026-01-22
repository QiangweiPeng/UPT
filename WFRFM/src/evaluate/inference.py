import torch
import numpy as np
import anndata as ad
from tqdm import tqdm

@torch.no_grad()
def wfr_euler_solve(
    model,
    z: torch.Tensor,
    cond: torch.Tensor,
    n_steps: int,
    dt: float,
    clamp_g: float = 100.0
):
    """
    输入: z [B, D], cond [B, C]
    输出: z_final [B, D], m_final [B, 1]
    """
    B = z.shape[0]
    device = z.device
    m = torch.ones((B, 1), device=device, dtype=z.dtype)
    
    for k in range(n_steps):
        t_val = k * dt
        t = torch.full((B, 1), t_val, device=device, dtype=z.dtype)

        v, g = model(t, z, cond)
        
        if g.dim() == 1: g = g.unsqueeze(1)
        g = g.clamp(-clamp_g, clamp_g)

        z.add_(v * dt)
        m.mul_(torch.exp(g * dt))
        
    return z, m
    

@torch.no_grad()
def run_batch_inference(
    model,
    adata_source: torch.Tensor,       # 一般是adata_control #直接处理好放进来
    adata_conditions: ad.AnnData,   # adata_train or adata_test
    target_conditions: list,        # perturb什么gene
    condition_keys: str = "target_gene",
    embedding_key: str = "gene_embeddings",
    source_rep: str = "X_pca_scaled",
    n_steps: int = 50,
    device: str = "cuda",
    random_seed: int = 42
) -> dict:
    """
    输出: 一个字典 {'TP53': {'z_pred': [n_particles, embed_len], 'm_pred': [n_particles, 1]}, ...}
    实际上这个函数不是batch形式的，后面再改
    还可以有混合精度优化之类的
    """
    model.eval()
    model.to(device)
    dt = 1.0 / n_steps
    results = {}

    z0 = adata_source

    for cond_name in tqdm(target_conditions):
        subset = adata_conditions[adata_conditions.obs[condition_keys] == cond_name]
        if subset.n_obs == 0:
            print("no observation")
            continue
            
        cond_vec = torch.tensor(subset.obsm[embedding_key][0], dtype=torch.float32, device=device)
        cond_batch = cond_vec.unsqueeze(0).expand(z0.shape[0], -1) # Broadcast

        z_pred, m_pred = wfr_euler_solve(model, z0.clone(), cond_batch, n_steps, dt)

        results[cond_name] = {
            'z_pred': z_pred.cpu().numpy(),
            'm_pred': m_pred.cpu().numpy()
        }

    return results


def batch_reconstruct_pca(
    inference_results: dict,     
    ref_adata: ad.AnnData,       # 提供 PCA 参数
    store_as_anndata: bool = True
) -> dict:
    """
    输入: 'TP53': {'z_pred': [n_particles, embed_len], 'm_pred': [n_particles, 1]}
    输出: {'TP53': AnnData(X=gene_expression)}
    """


    PCs = ref_adata.varm["PCs"] # [n_genes, n_pcs]
    mean = ref_adata.varm["X_mean"]

    if isinstance(mean, np.ndarray) and mean.ndim > 1:
        mean = mean.flatten()

    var = np.asarray(ref_adata.uns["pca"]["variance"]) # 记得在preprocess我们进行了scale
    scale_factor = np.sqrt(var)

    reconstructed_dict = {}
    for cond_name, result_dict in tqdm(inference_results.items()):
        z_pca_scaled = result_dict['z_pred']
        m_pred = result_dict['m_pred'] 
        z_pca = z_pca_scaled * scale_factor
        X_recon = np.dot(z_pca, PCs.T) + mean
        
        if store_as_anndata:
            # 包装成 AnnData
            new_ad = ad.AnnData(X=X_recon, dtype=np.float32)
            new_ad.var_names = ref_adata.var_names
            new_ad.obs['condition'] = cond_name
            new_ad.obs['mass'] = m_pred.flatten()
            new_ad.uns['reconstruction_params'] = {
                'mean_mass': float(m_pred.mean()),
                'std_mass': float(m_pred.std()),
                'min_mass': float(m_pred.min()),
                'max_mass': float(m_pred.max())
            }
            reconstructed_dict[cond_name] = new_ad
        else:
            reconstructed_dict[cond_name] = {
                'X':X_recon,
                'mass':m_pred
            }
            
    return reconstructed_dict



def batch_reconstruct_scvi(
    inference_results: dict,     
    scvi_model,                  # 传入训练好的 scVI 模型对象 (model_ref)
    target_library_size: float = 1e4, # 用于标准化表达量，类似 normalize_total
    store_as_anndata: bool = True,
    batch_idx: int = 0           # 默认投射到第0个Batch (即 Control/Reference Batch)
) -> dict:
    """
    输入: 'TP53': {'z_pred': [n_particles, n_latent], 'm_pred': [n_particles, 1]}
    输出: {'TP53': AnnData(X=gene_expression)}
    """
    
    # 1. 准备模型状态
    # 确保模型在评估模式 (不更新权重)
    scvi_model.module.eval()
    device = scvi_model.device  # 获取模型所在的设备 (GPU/CPU)
    
    # 获取基因名称
    var_names = scvi_model.adata.var_names
    
    reconstructed_dict = {}

    # 不需要像 PCA 那样读取 mean/var，因为这些都在神经网络的权重里
    
    for cond_name, result_dict in tqdm(inference_results.items()):
        # 2. 准备输入数据 (转换为 PyTorch Tensor 并移至 GPU)
        z_pred = result_dict['z_pred'] # scVI 的 z 不需要再缩放，它本身就是标准化的
        m_pred = result_dict['m_pred'] # 这里假设 m_pred 是 log_library_size 或者 raw library size

        # 转换为 Tensor
        z_tensor = torch.tensor(z_pred, dtype=torch.float32, device=device)
        
        # 构造 Batch Index (全为0，模拟映射回 Control Batch)
        n_obs = z_pred.shape[0]
        batch_index = torch.full((n_obs, 1), batch_idx, dtype=torch.long, device=device)
        
        # 处理 Library Size (m_pred)
        # scVI 的 generative 函数通常需要 library 参数
        # 你的 m_pred 如果是 mass (total counts)，scVI 内部通常需要 log(total_counts)
        # 为了获得“标准化”后的表达量 (Normalized Expression)，我们通常不使用 m_pred，
        # 而是让模型输出 px_scale (即基因表达概率)，然后乘以一个固定的 target_sum (如 10,000)
        # 这样得到的表达量是可以直接对比的，消除了测序深度的影响。
        
        # 这里我们需要传入一个假的 library 占位符给 generative 函数，
        # 因为我们主要想要它的 'px_scale' 输出。
        library_tensor = torch.zeros((n_obs, 1), device=device) 

        # 3. 通过解码器 (Generative Pass)
        with torch.no_grad():
            # 调用 scVI 的生成过程
            # generative 返回: {'px_scale': ..., 'px_r': ..., 'px_rate': ..., 'px_dropout': ...}
            # 注意：不同版本的 scvi-tools 参数可能略有不同，但通常只需 z, library, batch_index
            outputs = scvi_model.module.generative(
                z=z_tensor, 
                library=library_tensor, 
                batch_index=batch_index
            )
            
            # px_scale 是 softmax 后的结果 (基因表达比例，sum=1)
            # 这相当于去除了 library size 影响的“纯”表达谱
            px_scale = outputs["px_scale"]
            
            # 还原到指定测序深度 (类似 Scanpy 的 normalize_total(1e4))
            # 这样输出的数据都在同一个尺度上
            X_recon_tensor = px_scale * target_library_size
            
            # 这一步通常得到的是 "Denoised Normalized Counts"
            # 如果你习惯看 log 空间的数据 (类似 PCA 的输出)，可以做 log1p
            # X_recon_tensor = torch.log1p(X_recon_tensor) 
            
            # 转回 Numpy
            X_recon = X_recon_tensor.cpu().numpy()

        # 4. 封装结果
        if store_as_anndata:
            new_ad = ad.AnnData(X=X_recon, dtype=np.float32)
            new_ad.var_names = var_names
            new_ad.obs['condition'] = cond_name
            
            # 保留原始的 mass 信息供参考
            if m_pred is not None:
                new_ad.obs['mass'] = m_pred.flatten()
                new_ad.uns['reconstruction_params'] = {
                    'mean_mass': float(m_pred.mean()),
                    'std_mass': float(m_pred.std()),
                }
            
            reconstructed_dict[cond_name] = new_ad
        else:
            reconstructed_dict[cond_name] = {
                'X': X_recon,
                'mass': m_pred
            }
            
    return reconstructed_dict