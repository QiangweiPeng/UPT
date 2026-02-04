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
    或许可以有perturb的batch之类
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
    scvi_model,                  # 传入训练好的 scVI 模型对象
    target_library_size: float = 1e4, # 用于标准化表达量 (Normalized Count)
    store_as_anndata: bool = True,
    batch_idx = None           
) -> dict:
    """
    输入: 'TP53': {'z_pred': [n_particles, n_latent], 'm_pred': [n_particles, 1]}
    输出: {'TP53': AnnData(X=normalized_gene_expression)}
    
    逻辑:
    1. 计算: X = px_scale * target_library_size (忽略 m_pred 对数值的影响)
    2. 存储: 将 m_pred 存入 obs['mass'] 和 uns 中 (保留元数据)
    """
    
    scvi_model.module.eval()
    device = scvi_model.device 
    var_names = scvi_model.adata.var_names
    
    reconstructed_dict = {}
    
    for cond_name, result_dict in tqdm(inference_results.items()):
        z_pred = result_dict['z_pred'] 
        m_pred = result_dict['m_pred']

        z_tensor = torch.tensor(z_pred, dtype=torch.float32, device=device)
        n_obs = z_pred.shape[0]

        if batch_idx is None:
            batch_idx = torch.full((n_obs, 1), 0, dtype=torch.long, device=device)
        
        library_tensor = torch.zeros((n_obs, 1), device=device) 

        scvi_model.module.eval()
        with torch.no_grad():
            outputs = scvi_model.module.generative(
                z=z_tensor, 
                library=library_tensor, 
                batch_index=batch_idx  
            )
            px = outputs["px"]
            px_scale = px.scale

            X_recon_tensor = px_scale * target_library_size
            
            X_recon = X_recon_tensor.cpu().numpy()

        if store_as_anndata:
            new_ad = ad.AnnData(X=X_recon, dtype=np.float32)
            new_ad.var_names = var_names
            new_ad.obs['condition'] = cond_name
            
            # 存入 m_pred 信息
            new_ad.obs['mass'] = m_pred.flatten()
            new_ad.uns['reconstruction_params'] = {
                'target_library_size': target_library_size, # 记录一下你是用多少标准化的
                'mean_mass': float(m_pred.mean()),
                'std_mass': float(m_pred.std()),
                'min_mass': float(m_pred.min()),
                'max_mass': float(m_pred.max())
            }
            
            reconstructed_dict[cond_name] = new_ad
        else:
            reconstructed_dict[cond_name] = {
                'X': X_recon,
                'mass': m_pred
            }
            
    return reconstructed_dict



def batch_reconstruct_flatvi(
    inference_results: dict,
    flatvi_model,  # FlatVIEmbedding对象
    target_library_size=1e4,
    store_as_anndata: bool = True,
    var_names = None
) -> dict:
    """
    使用FlatVI decoder从潜在空间重构到基因空间
    """
    reconstructed_dict = {}
    device = flatvi_model.device
    
    for cond_name, result_dict in tqdm(inference_results.items()):
        z_pred = result_dict['z_pred']
        m_pred = result_dict['m_pred']

        batch_size = z_pred.shape[0]
        
        # 使用FlatVI的decoder
        with torch.no_grad():
            z_tensor = torch.FloatTensor(z_pred).to(flatvi_model.device)
            
            # 解码

            lib_size_tensor = torch.full((batch_size,), float(target_library_size), device=device)

            decoder_output = flatvi_model.model.decode(z_tensor)
            decoder_output = flatvi_model.model._preprocess_decoder_output(
                decoder_output, 
                library_size=lib_size_tensor
            )
            
            # 获取重构的表达量（均值）
            X_recon = decoder_output['mu'].cpu().numpy()
        
        if store_as_anndata:
            new_ad = ad.AnnData(X=X_recon, dtype=np.float32)
            new_ad.var_names = var_names
            new_ad.obs['condition'] = cond_name
            new_ad.obs['mass'] = m_pred.flatten()
            new_ad.uns['reconstruction_params'] = {
                'mean_mass': float(m_pred.mean()),
                'std_mass': float(m_pred.std())
            }
            reconstructed_dict[cond_name] = new_ad
        else:
            reconstructed_dict[cond_name] = {
                'X': X_recon,
                'mass': m_pred
            }
    
    return reconstructed_dict



@torch.no_grad()
def batch_reconstruct_state(
    inference_results: dict,
    state_decoder,                     # 你的 NBDecoder（或同结构 decoder）
    var_names,                         # gene names（list/Index）
    target_library_size: float = 1e4,  # 标准化到多少 counts
    store_as_anndata: bool = True,
    device: str | None = None,
) -> dict:
    """
    输入 inference_results:
      {
        "TP53": {"z_pred": (n_obs, z_dim) 或 torch.Tensor, "m_pred": (n_obs,1) 可选},
        ...
      }

    输出:
      如果 store_as_anndata=True:
        {"TP53": AnnData(X=normalized_gene_expression), ...}
      否则:
        {"TP53": {"X": np.ndarray, "mass": m_pred}, ...}

    逻辑（对齐 scVI）:
      1) scale = softmax(decoder.scale_head(decoder.backbone(z)))
      2) X_recon = scale * target_library_size
      3) 可选把 m_pred 存到 obs['mass']
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state_decoder = state_decoder.to(device).eval()

    reconstructed_dict = {}

    for cond_name, result_dict in tqdm(inference_results.items()):
        z_pred = result_dict["z_pred"]
        m_pred = result_dict.get("m_pred", None)

        # --- z to tensor ---
        if isinstance(z_pred, np.ndarray):
            z = torch.from_numpy(z_pred).float().to(device)
        elif torch.is_tensor(z_pred):
            z = z_pred.float().to(device)
        else:
            z = torch.tensor(z_pred, dtype=torch.float32, device=device)

        n_obs = z.shape[0]

        # --- scVI-like decoder: get px_scale ---
        h = state_decoder.backbone(z)                           # (B,H)
        scale = torch.softmax(state_decoder.scale_head(h), -1)  # (B,G), row-sum=1

        X_recon = (scale * float(target_library_size)).cpu().numpy()  # (B,G)

        if store_as_anndata:
            new_ad = ad.AnnData(X=X_recon.astype(np.float32))
            new_ad.var_names = var_names
            new_ad.obs["condition"] = cond_name

            if m_pred is not None:
                new_ad.obs["mass"] = np.asarray(m_pred).reshape(-1)

                mp = np.asarray(m_pred).reshape(-1)
                new_ad.uns["reconstruction_params"] = {
                    "target_library_size": float(target_library_size),
                    "mean_mass": float(mp.mean()),
                    "std_mass": float(mp.std()),
                    "min_mass": float(mp.min()),
                    "max_mass": float(mp.min()),
                }
            else:
                new_ad.uns["reconstruction_params"] = {
                    "target_library_size": float(target_library_size),
                }

            reconstructed_dict[cond_name] = new_ad
        else:
            reconstructed_dict[cond_name] = {
                "X": X_recon,
                "mass": m_pred,
            }

    return reconstructed_dict
