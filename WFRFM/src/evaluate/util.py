import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import gc
import scvi


def draw_loss(load_path = None,eval_interval = 10,cut = 0,save_path = None):

    checkpoint = torch.load(load_path, map_location='cpu')
    
    loss_list = checkpoint['loss_list'][cut:]   
    vloss_list = checkpoint['vloss_list'][cut:] 
    gloss_list = checkpoint['gloss_list'][cut:]
    test_loss_list = checkpoint['test_loss_list'][cut//eval_interval:]
    test_vloss_list = checkpoint['test_vloss_list'][cut//eval_interval:]
    test_gloss_list = checkpoint['test_gloss_list'][cut//eval_interval:]
    
    
    print(f"训练步数: {len(loss_list)}")
    print(f"最后一步 Loss: {loss_list[-1]:.6f}")
    print(f"最后一步 vLoss: {vloss_list[-1]:.6f}")
    print(f"最后一步 gLoss: {gloss_list[-1]:.6f}")
    print(f"最后一步 test_Loss: {test_loss_list[-1]:.6f}")
    print(f"最后一步 test_vLoss: {test_vloss_list[-1]:.6f}")
    print(f"最后一步 test_gLoss: {test_gloss_list[-1]:.6f}")
    

    plt.figure(figsize=(15, 5))

    # train loss vloss gloss
    plt.subplot(1, 2, 1)
    plt.plot(loss_list, label='Total Train Loss', color='blue', alpha=0.6)
    plt.plot(vloss_list, label='Velocity Loss (train)', color='orange', alpha=0.7)
    plt.plot(gloss_list, label='Growth Loss (train)', color='green', alpha=0.7)
    plt.title('Total Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # test loss vloss gloss
    plt.subplot(1, 2, 2)
    plt.plot(test_loss_list, label='Total Test Loss', color='blue', alpha=0.6)
    plt.plot(test_vloss_list, label='Velocity Loss (test)', color='orange', alpha=0.7)
    plt.plot(test_gloss_list, label='Growth Loss (test)', color='green', alpha=0.7)
    plt.title('Total Test Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    #plt.tight_layout()

    file_name = os.path.basename(load_path).replace(".pt","")
    if save_path is not None:
        save_root = os.path.join(save_path,f"loss_{file_name}_all.png")
        plt.savefig(save_root, dpi=300, bbox_inches="tight")
    
    # gloss
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(gloss_list, label='Train gLoss', color='green', alpha=0.6)
    #plt.plot(test_gloss_list, label='Test gLoss', color='blue', alpha=0.6)
    plt.title('gLoss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    #plt.plot(gloss_list, label='Train gLoss', color='green', alpha=0.6)
    plt.plot(test_gloss_list, label='Test gLoss', color='blue', alpha=0.6)
    plt.title('gLoss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    #plt.tight_layout()

    if save_path is not None:
        save_root = os.path.join(save_path,f"loss_{file_name}_gloss.png")
        plt.savefig(save_root, dpi=300, bbox_inches="tight")
    plt.show()



def get_origin_expression(adata, model, sample_rep, target_library_size=1e4, batch_size=1024):
    """
    分批次重建基因表达，防止显存爆炸 (OOM)。
    """
    device = model.device
    
    # 1. 获取 Embedding (建议先放在 CPU 上，循环时再搬运到 GPU)
    # 使用 .copy() 确保在内存中是连续的
    z_all = adata.obsm[sample_rep].copy()
    n_obs = z_all.shape[0]
    
    recon_list = []
    
    print(f"Starting reconstruction for {n_obs} cells in batches of {batch_size}...")
    
    model.module.eval()
    for i in range(0, n_obs, batch_size):
        end_idx = min(i + batch_size, n_obs)
        z_batch = torch.tensor(z_all[i:end_idx], device=device).float()
        current_batch_size = z_batch.shape[0]
        
        library_batch = torch.zeros((current_batch_size, 1), device=device)
        batch_idx_batch = torch.full((current_batch_size, 1), 0, dtype=torch.long, device=device)
        
        with torch.no_grad():
            outputs = model.module.generative(
                z=z_batch, 
                library=library_batch, 
                batch_index=batch_idx_batch
            )

            px = outputs["px"]
            px_scale = px.scale
            batch_recon = (px_scale * target_library_size).cpu().numpy()
            recon_list.append(batch_recon)
            

        del z_batch, library_batch, batch_idx_batch, outputs, px_scale
        torch.cuda.empty_cache()

    print("Concatenating results...")
    X_recon = np.vstack(recon_list)
    
    adata.X = X_recon
    print(f"Reconstruction finished. Adata X shape: {adata.X.shape}")
    del recon_list
    if 'log1p' in adata.uns:
        del adata.uns['log1p']
    gc.collect()


def classify_perturbations(train_conds, test_conds, ctrl_tag='ctrl'):
    # 1. 构建训练集基因词表 (Vocabulary)
    # 提取训练集中出现过的所有非ctrl的基因/扰动
    train_genes_vocab = set()
    for cond in train_conds:
        parts = cond.split('+')
        for p in parts:
            if p != ctrl_tag:
                train_genes_vocab.add(p)
    
    print(f"训练集中包含的唯一扰动源数量: {len(train_genes_vocab)}")
    
    # 2. 初始化结果字典
    results = {
        "single_new": [],      # 单扰动：train中未出现的基因
        "single_seen": [],     # 单扰动：train中出现的基因
        "double_0": [],        # 双扰动：包含0个train基因 (两个都是新基因)
        "double_1": [],        # 双扰动：包含1个train基因 (一新一旧)
        "double_2": [],        # 双扰动：包含2个train基因 (两个都是旧基因，但组合可能是新的)
        "control": []          # 纯对照 (ctrl+ctrl)
    }
    
    # 3. 遍历测试集进行分类
    for cond in test_conds:
        # 分割并移除 ctrl 标记
        parts = cond.split('+')
        genes = [p for p in parts if p != ctrl_tag]
        
        # 判断类型
        if len(genes) == 0:
            results["control"].append(cond)
            
        elif len(genes) == 1:
            # 单扰动逻辑
            gene = genes[0]
            if gene in train_genes_vocab:
                results["single_seen"].append(cond)
            else:
                results["single_new"].append(cond)
                
        elif len(genes) == 2:
            # 双扰动逻辑：计算有几个基因在训练集中出现过
            seen_count = sum(1 for g in genes if g in train_genes_vocab)
            if seen_count == 0:
                results["double_0"].append(cond)
            elif seen_count == 1:
                results["double_1"].append(cond)
            elif seen_count == 2:
                results["double_2"].append(cond)
                
    return results, train_genes_vocab



import pandas as pd
from .inference import run_batch_inference
from .inference import batch_reconstruct_pca, batch_reconstruct_scvi, batch_reconstruct_flatvi, batch_reconstruct_state
from .evals_new import evaluate_latent,evaluate_population_average,evaluate_population_distribution
import anndata as ad

def _ensure_perturbation_col(df: pd.DataFrame, key="perturbation") -> pd.DataFrame:
    """Ensure df has a 'perturbation' column; if it's in index, move it to a column."""
    if df is None:
        return None
    df = df.copy()
    if key not in df.columns:
        # 常见：perturbation 在 index
        if df.index.name == key:
            df = df.reset_index()
        else:
            # 如果 index 看起来就是 perturbation（但没命名），也尝试兜底
            if df.index.dtype == object:
                df = df.reset_index().rename(columns={"index": key})
            else:
                raise ValueError(f"Cannot find '{key}' column or index in df.")
    return df

def _merge_on_perturbation(dfs, key="perturbation") -> pd.DataFrame:
    """Outer-merge a list of dfs on perturbation and avoid duplicate column collisions."""
    merged = None
    for i, df in enumerate(dfs):
        if df is None:
            continue
        df = _ensure_perturbation_col(df, key=key)

        if merged is None:
            merged = df
            continue

        # 若有重名列（除了 key），保留已有的，新增的加后缀
        overlap = (set(merged.columns) & set(df.columns)) - {key}
        if overlap:
            df = df.rename(columns={c: f"{c}__dup{i}" for c in overlap})

        merged = merged.merge(df, on=key, how="outer")

    if merged is None:
        merged = pd.DataFrame(columns=[key])

    return merged

    
import os
import torch
import numpy as np
import anndata as ad
import scvi


def evaluate_all(
    *,
    model,
    adata_control,
    adata_test,
    wishlist,            # 【核心修改 1】：从 target_conditions 改为 wishlist (List of Dicts)
    condition_keys,      # 代表基础扰动的列名，如 'perturbation'
    control_key,
    condition_rep_keys,  # e.g., 'gene_embeddings'
    condition_combined_keys,
    sample_rep,
    device,
    results_save_path,
    mass_deduct_keys = None,
    n_particles: int = 10000,
    random_seed: int = 42,
    n_steps: int = 50,
    scvi_model_load_path: str = None,
    state_model_load_path: str = None,
    run_origin_expression: bool = True,
    origin_expression_batch_size: int = 2048,
    do_log1p_when_not_pca: bool = True,
    inplace_log1p: bool = False,
    Edistance_sample_num: int = 200,
    dist_max_cells: int = 200,
    dist_max_genes: int = 100000,
    dist_n_bins: int = 50,
    dist_top_n_degs: int = 50,
    save_each: bool = True,
    save_final: bool = True,
    final_filename: str = "final_metrics.csv",
    detailed: bool = True,
    use_groupwise_control: bool = True,
):
    if adata_test is None:
        raise ValueError("adata_test is None, but evaluation needs treated data (adata_treated).")

    if not wishlist:
        print("wishlist is empty")
        return [], []
        
    os.makedirs(results_save_path, exist_ok=True)
    
    # ---- 1. 解析法典与构建 Embedding 字典 ----
    rulebook = adata_test.uns.get('global_rulebook')
    if rulebook is None:
        raise ValueError("adata_test.uns 缺少 'global_rulebook'。请先运行 prepare_covariates。")

    # 自动从 Control 和 Test 集中收集所有出现过的基础扰动 (如药物) 的 Embedding
    condition_embeddings = {}
    for adata in [adata_control, adata_test]:
        if condition_rep_keys in adata.obsm:
            for pert in adata.obs[condition_keys].unique():
                if pert not in condition_embeddings:
                    idx = np.where(adata.obs[condition_keys] == pert)[0][0]
                    condition_embeddings[pert] = adata.obsm[condition_rep_keys][idx]

    # ---- 2. 采样起点细胞 X_0 ----
    all_indices = np.arange(adata_control.n_obs)
    rng = np.random.default_rng(random_seed)
    indices = (
        rng.choice(all_indices, n_particles, replace=False)
        if n_particles < len(all_indices)
        else all_indices
    )

    # 【核心修改 3】：不再剥离为 Tensor！直接传 AnnData 切片，保留 .obs 供推理器抽取固有属性！
    adata_source_subset = adata_control[indices].copy()

    # ---- 3. Inference (凭空造物) ----
    results_embedding = run_batch_inference(
        model=model,
        adata_source=adata_source_subset,  # 传入 AnnData 格式的 X_0
        rulebook=rulebook,                 # 传入法典
        wishlist=wishlist,                 # 传入愿望清单
        condition_embeddings=condition_embeddings, # 传入提取好的特征库
        condition_key_name=condition_keys,
        sample_rep=sample_rep,
        n_steps=n_steps,
        device=device
    )

    # ---- 4. Reconstruct (保持原样) ----
    results_genes = None
    model_ref = None
    model_train = None
    model_test = None

    if sample_rep in ["X_pca_scaled", "X_pca"]:
        results_genes = batch_reconstruct_pca(
            inference_results=results_embedding,  
            ref_adata=adata_control        
        )
    elif sample_rep in ["X_scVI"]:
        model_ref = scvi.model.SCVI.load(f"{scvi_model_load_path}_ref", adata=adata_control)
        model_train = scvi.model.SCVI.load(f"{scvi_model_load_path}_train", adata=adata_control)
        model_test = scvi.model.SCVI.load(f"{scvi_model_load_path}_test", adata=adata_control)
        results_genes = batch_reconstruct_scvi(
            inference_results=results_embedding,
            scvi_model=model_ref,  
            target_library_size=1e4, 
            batch_idx=None, 
        )
    elif sample_rep == "X_flatvi":
        model_ref = scvi.model.SCVI.load(f"{scvi_model_load_path}_ref", adata=adata_control)
        results_genes = batch_reconstruct_flatvi(
            inference_results=results_embedding,
            flatvi_model=model_ref,
            target_library_size=1e4,
            var_names=adata_control.var_names
        )
    elif sample_rep == "X_state":
        from src.preprocessing import NBDecoder, NBDecoderTrainer
        z_dim = adata_control.obsm["X_state"].shape[1]
        n_genes = adata_control.n_vars
        state_decoder = NBDecoder(z_dim=z_dim, n_genes=n_genes, hidden=(1024,2048,4096), dropout=0.1)
        trainer = NBDecoderTrainer(state_decoder, device="cuda", use_amp=False)
        trainer.load(state_model_load_path)  
        state_decoder = trainer.decoder
        state_decoder.eval()
        
        results_genes = batch_reconstruct_state(
            inference_results=results_embedding,
            state_decoder=state_decoder,
            target_library_size=1e4,
            var_names=adata_control.var_names
        )

    # ---- 5. Evaluate (评估) ----
    # 🚨 注意：由于现在的 results_embedding 的 key 是从 wishlist 拼出来的字符串 (如 "DrugA_MCF7_1.0")
    # 下游的 evaluate_latent 等函数切片 adata_treated 时，匹配的 column 需要对应得上。
    # 建议将 pert_key 传为你生成的 condition_combined_keys，或者在评估函数内部适配字符串解析。
    
    latent_df = evaluate_latent(
        results_embedding=results_embedding,
        adata_treated=adata_test,
        adata_control=adata_control,
        pert_key=condition_combined_keys,  # 确保传入的是 Tuple 字符串列名
        control_label=control_key,
        embedding_key=sample_rep,
        rulebook=rulebook,                 # 传入法典
        use_groupwise_control=use_groupwise_control       # 开启精细化筛选选项
    )
    avg_df = evaluate_population_average(
        results_genes=results_genes,
        adata_treated=adata_test,
        adata_control=adata_control,
        pert_key=condition_combined_keys,
        control_label=control_key,
        embedding_key=sample_rep,
        Edistance_sample_num=Edistance_sample_num,
        random_seed=random_seed,
        detailed=detailed,
        rulebook=rulebook,                 # 传入法典
        use_groupwise_control= use_groupwise_control,      # 开启精细化筛选选项
        mass_deduct_keys= mass_deduct_keys,
    )
    dist_df = evaluate_population_distribution(
        results_genes=results_genes,
        adata_treated=adata_test,
        adata_control=adata_control,
        pert_key=condition_combined_keys,
        max_cells=dist_max_cells,
        max_genes=dist_max_genes,
        n_bins=dist_n_bins,
        top_n_degs=dist_top_n_degs,
        seed=random_seed,
        rulebook=rulebook,                 # 传入法典
        use_groupwise_control= use_groupwise_control        # 开启精细化筛选选项
    )

    # ---- 6. Save (保存) ----
    if save_each:
        _ensure_perturbation_col(latent_df).to_csv(os.path.join(results_save_path, "latent_metrics.csv"), index=False)
        _ensure_perturbation_col(avg_df).to_csv(os.path.join(results_save_path, "average_metrics.csv"), index=False)
        _ensure_perturbation_col(dist_df).to_csv(os.path.join(results_save_path, "distribution_metrics.csv"), index=False)

    final_df = _merge_on_perturbation([latent_df, avg_df, dist_df], key="perturbation")
    if detailed:
        final_df = final_df.sort_values("perturbation").reset_index(drop=True)
    else:
        final_df = final_df[final_df["perturbation"] == "mean"].reset_index(drop=True)

    if save_final:
        final_path = os.path.join(results_save_path, final_filename)
        final_df.to_csv(final_path, index=False)

    artifacts = {
        "indices": indices,
        "results_embedding": results_embedding,
        "results_genes": results_genes,
        "latent_df": latent_df,
        "avg_df": avg_df,
        "dist_df": dist_df,
    }
    return final_df, artifacts
