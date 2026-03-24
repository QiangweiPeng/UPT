import torch
import numpy as np
import anndata as ad
from tqdm import tqdm
import scanpy as sc


@torch.no_grad()
def wfr_euler_solve(
    model,
    z: torch.Tensor,
    cond: torch.Tensor,
    cov_dict: dict,        
    n_steps: int,
    dt: float,
    clamp_g: float = 100.0,
):
    """
    Euler 求解器保持纯粹的数学逻辑，无需大改。
    输入: z [B, D], cond [B, C], cov_dict {key: [B]}
    输出: z_final [B, D], m_final [B, 1]
    """
    B = z.shape[0]
    device = z.device
    m = torch.ones((B, 1), device=device, dtype=z.dtype)
    
    for k in range(n_steps):
        t_val = k * dt
        t = torch.full((B, 1), t_val, device=device, dtype=z.dtype)

        v, g = model(t, z, cond, cov_dict)
        
        if g.dim() == 1: g = g.unsqueeze(1)
        g = g.clamp(-clamp_g, clamp_g)

        # 推荐使用非就地操作(out-of-place)避免潜在的视图覆盖问题
        z = z + v * dt
        m = m * torch.exp(g * dt)
        
    return z, m


@torch.no_grad()
def run_batch_inference(
    model,
    adata_source: ad.AnnData,       
    rulebook: dict,                 
    wishlist: list,                 
    condition_embeddings: dict,     
    condition_key_name: str = "perturbation", 
    sample_rep: str = "X_pca_scaled",
    n_steps: int = 50,
    device: str = "cuda"
) -> dict:
    
    model.eval()
    model.to(device)
    dt = 1.0 / n_steps
    results = {}

    print(f"准备为 {len(wishlist)} 个条件进行推理...")

    control_groups_keys = rulebook.get('stratification', {}).get('control_groups', [])

    for wish in tqdm(wishlist):
        # 1. 严格按照法典中的 schema 顺序组装 Tuple
        schema = rulebook["condition_tuple_schema"]
        current_tuple = []
        for var in schema:
            current_tuple.append(wish[var])
        cond_name = str(tuple(current_tuple)) 
            

        # 根据condition筛选起始control细胞
        if len(control_groups_keys) == 0:
            # 如果是 global OT，使用全部 Control
            adata_source_cur = adata_source
        else:
            # 如果是 groupwise OT，精准匹配
            mask = np.ones(len(adata_source), dtype=bool)
            for k in control_groups_keys:
                if k in wish:
                    # 统一转为 string 匹配，最安全可靠
                    mask = mask & (adata_source.obs[k].astype(str) == str(wish[k]))
            
            adata_source_cur = adata_source[mask]

        # 如果当前切片没有 Control 细胞 (例如预测从未见过的细胞系)，只能跳过
        if len(adata_source_cur) == 0:
            print(f"\n⚠️ 警告: 找不到匹配的 Control 细胞作为起点，已跳过条件: {cond_name}")
            continue

        # 动态提取当前批次的 z0 和 Batch Size
        z0_np = adata_source_cur.obsm[sample_rep]
        z0 = torch.tensor(z0_np, dtype=torch.float32, device=device)
        B = z0.shape[0] 

        
        # (A) 组装基础 Condition Embedding
        base_cond_val = wish[condition_key_name]
        cond_vec = condition_embeddings[base_cond_val]
        
        if not isinstance(cond_vec, torch.Tensor):
            cond_vec = torch.tensor(cond_vec, dtype=torch.float32, device=device)
            
        # 扩展到动态的 Batch Size: [B, C]
        cond_batch = cond_vec.unsqueeze(0).expand(B, -1)

        # (B) 根据法典组装 Covariates
        cov_dict_batch = {}
        
        # -- 处理 Categorical --
        for cov_name, info in rulebook["model_inputs"]["categorical"].items():
            m_source = info["model_source"]
            
            if m_source in ["control", "base_cell"]:
                obs_col = info["obs_col"]
                val_np = adata_source_cur.obs[obs_col].to_numpy(dtype=np.int64)
                cov_dict_batch[cov_name] = torch.tensor(val_np, device=device)
            else:
                raw_val = str(wish[cov_name])
                idx = rulebook["categorical_mappings"][cov_name].get(raw_val, -1)
                if idx == -1:
                    raise ValueError(f"严重错误: 愿望清单中的 {cov_name}={raw_val} 在法典中未找到映射！")
                
                cov_dict_batch[cov_name] = torch.full((B,), idx, dtype=torch.long, device=device)

        # -- 处理 Continuous --
        for cov_name, info in rulebook["model_inputs"]["continuous"].items():
            m_source = info["model_source"]
            
            if m_source in ["control", "base_cell"]:
                # 【核心修复 3】：同样从 adata_source_cur 剥离
                obs_col = info["obs_col"]
                val_np = adata_source_cur.obs[obs_col].to_numpy(dtype=np.float32)
                cov_dict_batch[cov_name] = torch.tensor(val_np, device=device)
            else:
                raw_val = float(wish[cov_name])
                stats = rulebook["continuous_stats"][cov_name]
                mu, std, transform = stats["mean"], stats["std"], stats["transform"]
                
                if transform == "log1p_zscore":
                    val_tf = np.log1p(raw_val)
                else:
                    val_tf = raw_val
                    
                scaled_val = (val_tf - mu) / std
                cov_dict_batch[cov_name] = torch.full((B,), scaled_val, dtype=torch.float32, device=device)

        # (C) 运行 ODE Solver
        z_pred, m_pred = wfr_euler_solve(
            model, 
            z0.clone(), 
            cond_batch, 
            cov_dict_batch, 
            n_steps, 
            dt
        )

        # (D) 保存结果
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
        # z_pca = z_pca_scaled * scale_factor
        z_pca = result_dict['z_pred']
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

            new_ad.layers["counts"] = new_ad.X.copy()
            sc.pp.log1p(new_ad) # scvi得到的是raw count
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
            new_ad.layers["counts"] = new_ad.X.copy()
            sc.pp.log1p(new_ad)
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

            new_ad.layers["counts"] = new_ad.X.copy()
            sc.pp.log1p(new_ad)
            reconstructed_dict[cond_name] = new_ad
        else:
            reconstructed_dict[cond_name] = {
                "X": X_recon,
                "mass": m_pred,
            }

    return reconstructed_dict
