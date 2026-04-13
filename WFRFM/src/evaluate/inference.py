import torch
import numpy as np
import anndata as ad
import pandas as pd
from tqdm import tqdm


VALID_CONDITION_MODES = ("full", "transport-only", "growth-only")
VALID_SOLVERS = ("euler", "rk4")


def _validate_condition_mode(condition_mode):
    if condition_mode not in VALID_CONDITION_MODES:
        raise ValueError(
            f"Unknown condition_mode '{condition_mode}'. "
            f"Expected one of {VALID_CONDITION_MODES}."
        )
    return condition_mode


def _broadcast_condition_batch(condition_vec, batch_size, reference_tensor):
    if condition_vec is None:
        raise ValueError("condition_vec is required for the requested condition_mode.")

    if torch.is_tensor(condition_vec):
        cond_batch = condition_vec.to(
            device=reference_tensor.device,
            dtype=reference_tensor.dtype,
        )
    else:
        cond_batch = torch.tensor(
            condition_vec,
            dtype=reference_tensor.dtype,
            device=reference_tensor.device,
        )

    if cond_batch.dim() == 1:
        cond_batch = cond_batch.unsqueeze(0)

    if cond_batch.shape[0] == 1:
        cond_batch = cond_batch.expand(batch_size, -1)
    elif cond_batch.shape[0] != batch_size:
        raise ValueError(
            f"condition batch row count must be 1 or {batch_size}, got {cond_batch.shape[0]}"
        )

    return cond_batch


def _resolve_head_condition_batches(cond, condition_mode="full", control_condition_vec=None):
    condition_mode = _validate_condition_mode(condition_mode)
    if condition_mode == "full":
        return None, None

    control_batch = _broadcast_condition_batch(
        condition_vec=control_condition_vec,
        batch_size=cond.shape[0],
        reference_tensor=cond,
    )
    if condition_mode == "transport-only":
        return None, control_batch
    return control_batch, None


def _validate_solver_name(solver: str) -> str:
    if solver not in VALID_SOLVERS:
        raise ValueError(
            f"Unknown solver '{solver}'. Expected one of {VALID_SOLVERS}."
        )
    return solver


def _evaluate_wfr_rhs(
    model,
    z: torch.Tensor,
    log_m: torch.Tensor,
    cond: torch.Tensor,
    t_val: float,
    clamp_g: float,
    cond_v,
    cond_g,
):
    del log_m  # The current model does not take mass as input.
    t = torch.full((z.shape[0], 1), t_val, device=z.device, dtype=z.dtype)
    v, g = model(t, z, cond, con_v=cond_v, con_g=cond_g)
    if g.dim() == 1:
        g = g.unsqueeze(1)
    g = g.clamp(-clamp_g, clamp_g)
    return v, g


def _wfr_euler_step(
    model,
    z: torch.Tensor,
    log_m: torch.Tensor,
    cond: torch.Tensor,
    dt: float,
    t_val: float,
    clamp_g: float,
    cond_v,
    cond_g,
):
    v, g = _evaluate_wfr_rhs(
        model=model,
        z=z,
        log_m=log_m,
        cond=cond,
        t_val=t_val,
        clamp_g=clamp_g,
        cond_v=cond_v,
        cond_g=cond_g,
    )
    z = z + dt * v
    log_m = log_m + dt * g
    return z, log_m


def _wfr_rk4_step(
    model,
    z: torch.Tensor,
    log_m: torch.Tensor,
    cond: torch.Tensor,
    dt: float,
    t_val: float,
    clamp_g: float,
    cond_v,
    cond_g,
):
    k1_z, k1_logm = _evaluate_wfr_rhs(
        model=model,
        z=z,
        log_m=log_m,
        cond=cond,
        t_val=t_val,
        clamp_g=clamp_g,
        cond_v=cond_v,
        cond_g=cond_g,
    )
    k2_z, k2_logm = _evaluate_wfr_rhs(
        model=model,
        z=z + 0.5 * dt * k1_z,
        log_m=log_m + 0.5 * dt * k1_logm,
        cond=cond,
        t_val=t_val + 0.5 * dt,
        clamp_g=clamp_g,
        cond_v=cond_v,
        cond_g=cond_g,
    )
    k3_z, k3_logm = _evaluate_wfr_rhs(
        model=model,
        z=z + 0.5 * dt * k2_z,
        log_m=log_m + 0.5 * dt * k2_logm,
        cond=cond,
        t_val=t_val + 0.5 * dt,
        clamp_g=clamp_g,
        cond_v=cond_v,
        cond_g=cond_g,
    )
    k4_z, k4_logm = _evaluate_wfr_rhs(
        model=model,
        z=z + dt * k3_z,
        log_m=log_m + dt * k3_logm,
        cond=cond,
        t_val=t_val + dt,
        clamp_g=clamp_g,
        cond_v=cond_v,
        cond_g=cond_g,
    )
    z = z + (dt / 6.0) * (k1_z + 2.0 * k2_z + 2.0 * k3_z + k4_z)
    log_m = log_m + (dt / 6.0) * (k1_logm + 2.0 * k2_logm + 2.0 * k3_logm + k4_logm)
    return z, log_m


def _wfr_integrate(
    model,
    z: torch.Tensor,
    cond: torch.Tensor,
    n_steps: int,
    dt: float,
    start_time: float = 0.0,
    clamp_g: float = 100.0,
    control_condition_vec=None,
    condition_mode: str = "full",
    solver: str = "euler",
    initial_log_m: torch.Tensor | None = None,
):
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    solver = _validate_solver_name(str(solver))
    cond_v, cond_g = _resolve_head_condition_batches(
        cond=cond,
        condition_mode=condition_mode,
        control_condition_vec=control_condition_vec,
    )

    if initial_log_m is None:
        log_m = torch.zeros((z.shape[0], 1), device=z.device, dtype=z.dtype)
    else:
        log_m = initial_log_m.to(device=z.device, dtype=z.dtype)

    step_fn = _wfr_euler_step if solver == "euler" else _wfr_rk4_step
    for k in range(n_steps):
        t_val = start_time + k * dt
        z, log_m = step_fn(
            model=model,
            z=z,
            log_m=log_m,
            cond=cond,
            dt=dt,
            t_val=t_val,
            clamp_g=clamp_g,
            cond_v=cond_v,
            cond_g=cond_g,
        )

    return z, log_m


@torch.no_grad()
def _wfr_euler_advance(
    model,
    z: torch.Tensor,
    m: torch.Tensor,
    cond: torch.Tensor,
    n_steps: int,
    dt: float,
    start_time: float = 0.0,
    clamp_g: float = 100.0,
    control_condition_vec=None,
    condition_mode: str = "full",
):
    """Advance latent states and masses across one interval without resetting mass."""
    log_m = torch.log(torch.clamp(m, min=torch.finfo(m.dtype).tiny))
    z, log_m = _wfr_integrate(
        model=model,
        z=z,
        cond=cond,
        n_steps=n_steps,
        dt=dt,
        start_time=start_time,
        clamp_g=clamp_g,
        control_condition_vec=control_condition_vec,
        condition_mode=condition_mode,
        solver="euler",
        initial_log_m=log_m,
    )
    m = torch.exp(log_m)
    return z, m


@torch.no_grad()
def wfr_euler_solve(
    model,
    z: torch.Tensor,
    cond: torch.Tensor,
    n_steps: int,
    dt: float,
    start_time: float = 0.0,
    clamp_g: float = 100.0,
    control_condition_vec=None,
    condition_mode: str = "full",
):
    """
    输入: z [B, D], cond [B, C]
    输出: z_final [B, D], m_final [B, 1]
    """
    return wfr_solve(
        model=model,
        z=z,
        cond=cond,
        n_steps=n_steps,
        dt=dt,
        start_time=start_time,
        clamp_g=clamp_g,
        control_condition_vec=control_condition_vec,
        condition_mode=condition_mode,
        solver="euler",
    )


@torch.no_grad()
def wfr_solve(
    model,
    z: torch.Tensor,
    cond: torch.Tensor,
    n_steps: int,
    dt: float,
    start_time: float = 0.0,
    clamp_g: float = 100.0,
    control_condition_vec=None,
    condition_mode: str = "full",
    solver: str = "euler",
):
    """
    输入: z [B, D], cond [B, C]
    输出: z_final [B, D], m_final [B, 1]
    """
    z, log_m = _wfr_integrate(
        model=model,
        z=z,
        cond=cond,
        n_steps=n_steps,
        dt=dt,
        start_time=start_time,
        clamp_g=clamp_g,
        control_condition_vec=control_condition_vec,
        condition_mode=condition_mode,
        solver=solver,
    )
    return z, torch.exp(log_m)


def _copy_source_obs_columns(source_adata, obs_columns):
    obs = pd.DataFrame(index=source_adata.obs_names.astype(str))
    for column in obs_columns:
        if column not in source_adata.obs.columns:
            continue
        values = source_adata.obs[column]
        if hasattr(values, "cat") or str(values.dtype) == "object":
            obs[f"source_{column}"] = pd.Categorical(values.astype(str))
        else:
            obs[f"source_{column}"] = values.to_numpy()
    return obs


def build_latent_prediction_adata(
    source_adata: ad.AnnData,
    z_pred: np.ndarray,
    m_pred: np.ndarray,
    target_condition: str,
    target_timepoint: float,
    source_timepoint: float = 0.0,
    obs_columns=None,
    latent_key: str = "X_pca_scaled",
    record_kind: str = "predicted",
    is_model_output: bool = True,
):
    """
    Save latent-space predictions as an AnnData object.

    ``X`` stores the predicted latent representation directly to avoid duplicating it in ``obsm``.
    Source metadata columns are preserved with a ``source_`` prefix.
    """
    if obs_columns is None:
        obs_columns = []

    z_pred = np.asarray(z_pred, dtype=np.float32)
    m_pred = np.asarray(m_pred, dtype=np.float32).reshape(-1)
    if z_pred.shape[0] != source_adata.n_obs:
        raise ValueError("z_pred row count must match source_adata.n_obs")
    if m_pred.shape[0] != source_adata.n_obs:
        raise ValueError("m_pred row count must match source_adata.n_obs")

    obs = _copy_source_obs_columns(source_adata, obs_columns)
    obs["source_obs_name"] = source_adata.obs_names.astype(str)
    obs["target_condition"] = str(target_condition)
    obs["target_timepoint"] = float(target_timepoint)
    obs["source_model_timepoint"] = float(source_timepoint)
    obs["mass"] = m_pred
    obs["latent_key"] = latent_key
    obs["record_kind"] = str(record_kind)
    obs["is_model_output"] = bool(is_model_output)
    obs.index = [
        f"{target_condition}|t{float(target_timepoint):g}|{obs_name}"
        for obs_name in source_adata.obs_names.astype(str)
    ]

    pred_adata = ad.AnnData(X=z_pred, obs=obs)
    pred_adata.var_names = [f"latent_{idx}" for idx in range(z_pred.shape[1])]
    pred_adata.uns["prediction_metadata"] = {
        "latent_key": latent_key,
        "target_condition": str(target_condition),
        "target_timepoint": float(target_timepoint),
        "source_model_timepoint": float(source_timepoint),
        "n_source_obs": int(source_adata.n_obs),
        "record_kind": str(record_kind),
        "is_model_output": bool(is_model_output),
    }
    return pred_adata
    

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
    solver: str = "euler",
    device: str = "cuda",
    random_seed: int = 42,
    show_progress: bool = True,
    condition_mode: str = "full",
    control_condition_vec=None,
) -> dict:
    """
    输出: 一个字典 {'TP53': {'z_pred': [n_particles, embed_len], 'm_pred': [n_particles, 1]}, ...}
    或许可以有perturb的batch之类
    还可以有混合精度优化之类的
    """
    model.eval()
    model.to(device)
    dt = 1.0 / n_steps
    solver = _validate_solver_name(str(solver))
    results = {}

    z0 = adata_source

    iterator = tqdm(target_conditions, disable=not show_progress)
    for cond_name in iterator:
        subset = adata_conditions[adata_conditions.obs[condition_keys] == cond_name]
        if subset.n_obs == 0:
            print("no observation")
            continue
            
        cond_vec = torch.tensor(subset.obsm[embedding_key][0], dtype=torch.float32, device=device)
        cond_batch = cond_vec.unsqueeze(0).expand(z0.shape[0], -1) # Broadcast

        z_pred, m_pred = wfr_solve(
            model=model,
            z=z0.clone(),
            cond=cond_batch,
            n_steps=n_steps,
            dt=dt,
            control_condition_vec=control_condition_vec,
            condition_mode=condition_mode,
            solver=solver,
        )

        results[cond_name] = {
            'z_pred': z_pred.cpu().numpy(),
            'm_pred': m_pred.cpu().numpy()
        }

    return results


@torch.no_grad()
def run_timepoint_batch_inference(
    model,
    adata_source: ad.AnnData,
    source_rep: str,
    adata_conditions: ad.AnnData,
    target_specs: list,
    condition_keys: str = "target_gene",
    time_key: str = "timepoint",
    embedding_key: str = "gene_embeddings",
    source_timepoint: float = 0.0,
    n_steps: int = 50,
    solver: str = "euler",
    device: str = "cuda",
    show_progress: bool = True,
    condition_mode: str = "full",
    control_condition_vec=None,
):
    """
    Run absolute-time inference for observed ``(condition, target_timepoint)`` pairs.

    ``target_specs`` is a list of ``(condition_name, target_timepoint)`` tuples.
    """
    solver = _validate_solver_name(str(solver))
    model.eval()
    model.to(device)
    z0 = torch.from_numpy(np.asarray(adata_source.obsm[source_rep], dtype=np.float32)).to(device)
    results = {}

    iterator = tqdm(target_specs, disable=not show_progress)
    for cond_name, target_timepoint in iterator:
        target_timepoint = float(target_timepoint)
        subset = adata_conditions[
            (adata_conditions.obs[condition_keys].astype(str) == str(cond_name))
            & (adata_conditions.obs[time_key].astype(float) == target_timepoint)
        ]
        if subset.n_obs == 0:
            continue

        dt = (target_timepoint - float(source_timepoint)) / float(n_steps)
        if dt <= 0:
            raise ValueError(
                f"target_timepoint must be larger than source_timepoint: "
                f"{target_timepoint} <= {source_timepoint}"
            )

        cond_vec = torch.tensor(
            subset.obsm[embedding_key][0],
            dtype=torch.float32,
            device=device,
        )
        cond_batch = cond_vec.unsqueeze(0).expand(z0.shape[0], -1)
        z_pred, m_pred = wfr_solve(
            model=model,
            z=z0.clone(),
            cond=cond_batch,
            control_condition_vec=control_condition_vec,
            condition_mode=condition_mode,
            n_steps=n_steps,
            dt=dt,
            start_time=float(source_timepoint),
            solver=solver,
        )
        results[(str(cond_name), target_timepoint)] = {
            "z_pred": z_pred.cpu().numpy(),
            "m_pred": m_pred.cpu().numpy(),
            "n_true_obs": int(subset.n_obs),
            "source_timepoint": float(source_timepoint),
            "target_timepoint": target_timepoint,
            "condition": str(cond_name),
        }

    return results


@torch.no_grad()
def run_condition_timepoint_trajectory_inference(
    model,
    adata_source: ad.AnnData,
    source_rep: str,
    adata_conditions: ad.AnnData,
    condition_name: str,
    target_timepoints: list,
    condition_keys: str = "target_gene",
    time_key: str = "timepoint",
    embedding_key: str = "gene_embeddings",
    source_timepoint: float = 0.0,
    n_steps: int = 50,
    solver: str = "euler",
    device: str = "cuda",
    condition_mode: str = "full",
    control_condition_vec=None,
):
    """
    Run one continuous trajectory per condition and cache snapshots at requested timepoints.

    ``n_steps`` is interpreted as the number of Euler steps for each observed interval.
    """
    if not target_timepoints:
        return {}

    solver = _validate_solver_name(str(solver))

    model.eval()
    model.to(device)

    condition_name = str(condition_name)
    condition_subset = adata_conditions[
        adata_conditions.obs[condition_keys].astype(str) == condition_name
    ]
    if condition_subset.n_obs == 0:
        return {}

    ordered_timepoints = sorted({float(timepoint) for timepoint in target_timepoints})
    z_state = torch.from_numpy(
        np.asarray(adata_source.obsm[source_rep], dtype=np.float32)
    ).to(device)
    m_state = torch.ones((z_state.shape[0], 1), device=device, dtype=z_state.dtype)

    cond_vec = torch.tensor(
        condition_subset.obsm[embedding_key][0],
        dtype=torch.float32,
        device=device,
    )
    cond_batch = cond_vec.unsqueeze(0).expand(z_state.shape[0], -1)

    current_time = float(source_timepoint)
    results = {}
    for target_timepoint in ordered_timepoints:
        if target_timepoint <= current_time:
            raise ValueError(
                "target_timepoints must be strictly larger than source_timepoint "
                f"and sorted in increasing order, got {target_timepoint} after {current_time}"
            )

        interval_dt = (target_timepoint - current_time) / float(n_steps)
        z_state, log_m_state = _wfr_integrate(
            model=model,
            z=z_state,
            cond=cond_batch,
            n_steps=n_steps,
            dt=interval_dt,
            start_time=current_time,
            clamp_g=100.0,
            control_condition_vec=control_condition_vec,
            condition_mode=condition_mode,
            solver=solver,
            initial_log_m=torch.log(torch.clamp(m_state, min=torch.finfo(m_state.dtype).tiny)),
        )
        m_state = torch.exp(log_m_state)

        true_subset = condition_subset[
            condition_subset.obs[time_key].astype(float) == target_timepoint
        ]
        if true_subset.n_obs == 0:
            current_time = target_timepoint
            continue

        results[(condition_name, target_timepoint)] = {
            "z_pred": z_state.detach().cpu().numpy().copy(),
            "m_pred": m_state.detach().cpu().numpy().copy(),
            "n_true_obs": int(true_subset.n_obs),
            "source_timepoint": float(source_timepoint),
            "target_timepoint": target_timepoint,
            "condition": condition_name,
        }
        current_time = target_timepoint

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



# 在 src/evaluate/inference.py 中添加

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
