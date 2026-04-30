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


    


import os
import numpy as np
import scvi

import numpy as np

def get_condition_embedding(
    adata,
    condition,
    condition_rep_keys,
    condition_keys=None,
    strict=True,
):
    condition = str(condition).strip()

    # 新逻辑：优先从 uns 的 dict 查
    if condition_rep_keys in adata.uns:
        rep_dict = adata.uns[condition_rep_keys]
        if condition in rep_dict:
            return np.asarray(rep_dict[condition], dtype=np.float32)

        if strict:
            raise KeyError(
                f"Condition '{condition}' not found in adata.uns['{condition_rep_keys}']"
            )
        return None

    # 兼容旧逻辑：只有在 obsm 还存在时才回退
    if condition_rep_keys in adata.obsm:
        if condition_keys is None or condition_keys not in adata.obs.columns:
            if strict:
                raise ValueError(
                    f"Cannot recover condition embedding from obsm because "
                    f"'{condition_keys}' is not in adata.obs"
                )
            return None

        cond_series = adata.obs[condition_keys].astype(str).str.strip()
        idx = np.where(cond_series.values == condition)[0]
        if len(idx) == 0:
            if strict:
                raise KeyError(
                    f"Condition '{condition}' not found in adata.obs['{condition_keys}']"
                )
            return None

        return adata.obsm[condition_rep_keys][idx[0]]

    if strict:
        raise KeyError(
            f"Neither adata.uns['{condition_rep_keys}'] nor adata.obsm['{condition_rep_keys}'] exists"
        )
    return None



def run_inference_and_reconstruct(
    model,
    adata_control,
    adata_test,
    wishlist,
    condition_keys,
    condition_rep_keys,
    sample_rep,
    device,
    n_particles: int = 10000,
    random_seed: int = 42,
    n_steps: int = 50,
    scvi_model_load_path: str = None,
    state_model_load_path: str = None,
    source_celltype_key: str | None = None,
):

    if not wishlist:
        print("wishlist is empty")
        return None, None, None

    # ---- 1. 解析法典与构建 Embedding 字典 ----
    rulebook = adata_test.uns.get("global_rulebook")
    if rulebook is None:
        raise ValueError("adata_test.uns 缺少 'global_rulebook'。请先运行 prepare_covariates。")

    condition_embeddings = {}

    for adata in [adata_control, adata_test]:
        if adata is None:
            continue
        if condition_keys not in adata.obs.columns:
            continue

        for pert in adata.obs[condition_keys].astype(str).str.strip().unique():
            if pert in condition_embeddings:
                continue

            emb = get_condition_embedding(
                adata=adata,
                condition=pert,
                condition_rep_keys=condition_rep_keys,
                condition_keys=condition_keys,
                strict=False,
            )

            if emb is not None:
                condition_embeddings[pert] = emb

    # ---- 2. 采样起点细胞 X_0 ----
    all_indices = np.arange(adata_control.n_obs)
    rng = np.random.default_rng(random_seed)
    indices = (
        rng.choice(all_indices, n_particles, replace=False)
        if n_particles < len(all_indices)
        else all_indices
    )
    adata_source_subset = adata_control[indices].copy()

    # ---- 3. Inference (Latent Space) ----
    print(f"Starting inference for {len(wishlist)} conditions...")

    results_embedding = run_batch_inference(
        model=model,
        adata_source=adata_source_subset,
        rulebook=rulebook,
        wishlist=wishlist,
        condition_embeddings=condition_embeddings,
        condition_key_name=condition_keys,
        sample_rep=sample_rep,
        n_steps=n_steps,
        device=device,
        source_celltype_key=source_celltype_key,
    )

    # ---- 4. Reconstruct (Gene Expression) ----
    print("Starting gene expression reconstruction...")
    results_genes = None

    if sample_rep in ["X_pca_scaled", "X_pca"]:
        results_genes = batch_reconstruct_pca(
            inference_results=results_embedding,
            ref_adata=adata_control,
        )

    elif sample_rep in ["X_scVI"]:
        model_ref = scvi.model.SCVI.load(f"{scvi_model_load_path}_ref", adata=adata_control)
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
            var_names=adata_control.var_names,
        )

    elif sample_rep == "X_state":
        from src.preprocessing import NBDecoder, NBDecoderTrainer

        z_dim = adata_control.obsm["X_state"].shape[1]
        n_genes = adata_control.n_vars
        state_decoder = NBDecoder(
            z_dim=z_dim,
            n_genes=n_genes,
            hidden=(1024, 2048, 4096),
            dropout=0.1,
        )
        trainer = NBDecoderTrainer(state_decoder, device=device, use_amp=False)
        trainer.load(state_model_load_path)
        state_decoder = trainer.decoder
        state_decoder.eval()

        results_genes = batch_reconstruct_state(
            inference_results=results_embedding,
            state_decoder=state_decoder,
            target_library_size=1e4,
            var_names=adata_control.var_names,
        )

    print("Inference and reconstruction completed.")
    return results_embedding, results_genes, indices



import numpy as np
import pandas as pd


def _get_summary_base_df(df):
    """
    只保留原始 perturbation 行，避免把已有的 mean / mean_xxx 行再次纳入汇总。
    """
    out = df.copy()

    if "perturbation" not in out.columns:
        return out

    pert_str = out["perturbation"].astype(str)
    mask_summary = (pert_str == "mean") | (pert_str.str.startswith("mean_"))
    return out.loc[~mask_summary].copy()


def _get_metric_cols_for_mean(df, weight_col="n_true_ct"):
    """
    需要做平均的 metric 列：
    - 所有 numeric 列
    - 排除计数列
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = {weight_col, "n_ctrl_ct", "n_pred_ct"}
    metric_cols = [c for c in numeric_cols if c not in exclude_cols]
    return metric_cols


def append_celltype_mean_rows(final_df, weight_col="n_true_ct"):
    df = final_df.copy()

    if "cell_type" not in df.columns:
        raise ValueError("final_df must contain a 'cell_type' column")

    # 只用原始 perturbation 行做 summary，避免重复纳入 mean / mean_xxx
    base_df = _get_summary_base_df(df)
    metric_cols = _get_metric_cols_for_mean(base_df, weight_col=weight_col)

    mean_rows = []
    for ct, g in base_df.groupby("cell_type", sort=True, dropna=False):
        row = {
            "perturbation": f"mean_{ct}",
            "cell_type": ct
        }

        # 计数列做 sum
        if weight_col in g.columns:
            row[weight_col] = g[weight_col].sum()
        if "n_ctrl_ct" in g.columns:
            row["n_ctrl_ct"] = g["n_ctrl_ct"].sum()
        if "n_pred_ct" in g.columns:
            row["n_pred_ct"] = g["n_pred_ct"].sum()

        # 所有 metric 列直接简单平均
        for col in metric_cols:
            row[col] = g[col].mean(skipna=True)

        mean_rows.append(row)

    mean_df = pd.DataFrame(mean_rows)

    # 对齐列顺序；没有的列自动补 NaN
    for col in df.columns:
        if col not in mean_df.columns:
            mean_df[col] = np.nan
    mean_df = mean_df[df.columns]

    out = pd.concat([df, mean_df], ignore_index=True)
    return out


def make_celltype_mean_df(final_df, weight_col="n_true_ct"):
    df = final_df.copy()

    if "cell_type" not in df.columns:
        raise ValueError("final_df must contain a 'cell_type' column")

    # 只用原始 perturbation 行做 summary，避免重复纳入 mean / mean_xxx
    base_df = _get_summary_base_df(df)
    metric_cols = _get_metric_cols_for_mean(base_df, weight_col=weight_col)

    rows = []
    for ct, g in base_df.groupby("cell_type", sort=True, dropna=False):
        row = {
            "perturbation": f"mean_{ct}",
            "cell_type": ct
        }

        # 计数列做 sum
        if weight_col in g.columns:
            row[weight_col] = g[weight_col].sum()
        if "n_ctrl_ct" in g.columns:
            row["n_ctrl_ct"] = g["n_ctrl_ct"].sum()
        if "n_pred_ct" in g.columns:
            row["n_pred_ct"] = g["n_pred_ct"].sum()

        # 所有 metric 列直接简单平均
        for col in metric_cols:
            row[col] = g[col].mean(skipna=True)

        rows.append(row)

    out = pd.DataFrame(rows)

    # 为了和上游结构一致，补齐原列
    for col in df.columns:
        if col not in out.columns:
            out[col] = np.nan
    out = out[df.columns]

    return out




import pandas as pd
from .inference import run_batch_inference
from .inference import batch_reconstruct_pca, batch_reconstruct_scvi, batch_reconstruct_flatvi, batch_reconstruct_state
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
import pandas as pd
from .evals_new import evaluate_latent,evaluate_population_average,evaluate_population_distribution
from .evals_new import transfer_cell_labels_knn,evaluate_stratified_by_celltype


def evaluate_generated_results(
    *,
    results_embedding,
    results_genes,
    adata_control,
    adata_test,
    condition_combined_keys,
    control_key,
    sample_rep,
    results_save_path,
    mass_deduct_keys=None,
    random_seed: int = 42,
    Edistance_sample_num: int = 200,
    dist_max_cells: int = 200,
    dist_top_n_degs: int = 50,
    deg_padj_cutoff: float = 0.01,
    deg_fc_cutoff: float = 2.0,
    deg_top_n_for_eval: int = 50,
    save_each: bool = True,
    save_final: bool = True,
    final_filename: str = "final_metrics.csv",
    detailed_perturbation: bool = True,
    use_groupwise_control: bool = True,
    cell_type_key:str | None =None,  
    celltype_transition_rules: dict | None =None,
    celltype_transition_unknown_policy: str ="same_only",
    detailed_celltype: bool = False,
):
    """
    模块二：基于生成的 Latent 和 Genes 字典，执行多维度的指标评估并保存结果。
    """
    if adata_test is None:
        raise ValueError("adata_test is None, but evaluation needs treated data (adata_treated).")
    if not results_embedding or not results_genes:
        print("No inference results provided. Skipping evaluation.")
        return pd.DataFrame(), {}

    os.makedirs(results_save_path, exist_ok=True)

    # 解析法典
    rulebook = adata_test.uns.get("global_rulebook")
    if rulebook is None:
        raise ValueError("adata_test.uns 缺少 'global_rulebook'。请先运行 prepare_covariates。")

    # ---- 1. Evaluate ----
    print("Evaluating Latent metrics (MSE, R2, PCC Delta, Wasserstein/Sinkhorn, etc.)...")
    latent_df = evaluate_latent(
        results_embedding=results_embedding,
        adata_treated=adata_test,
        adata_control=adata_control,
        pert_key=condition_combined_keys,
        embedding_key=sample_rep,
        rulebook=rulebook,
        use_groupwise_control=use_groupwise_control,
        distribution_sample_num=Edistance_sample_num,
        random_seed=random_seed,
    )




    print("Evaluating Population Average metrics...")
    avg_df = evaluate_population_average(
        results_genes=results_genes,
        adata_treated=adata_test,
        adata_control=adata_control,
        pert_key=condition_combined_keys,
        detailed = True,
        rulebook=rulebook,
        use_groupwise_control=use_groupwise_control,
        mass_deduct_keys=mass_deduct_keys,
        deg_padj_cutoff=deg_padj_cutoff,
        deg_fc_cutoff=deg_fc_cutoff,
        deg_top_n_for_eval=deg_top_n_for_eval,
    )


    print("Evaluating Population Distribution metrics...")
    dist_df = evaluate_population_distribution(
        results_genes=results_genes,
        adata_treated=adata_test,
        adata_control=adata_control,
        pert_key=condition_combined_keys,
        max_cells=dist_max_cells,
        top_n_degs=dist_top_n_degs,
        seed=random_seed,
        rulebook=rulebook,
        use_groupwise_control=use_groupwise_control,
        deg_padj_cutoff=deg_padj_cutoff,
        deg_fc_cutoff=deg_fc_cutoff,
    )

    if cell_type_key is not None:
        print("\n--- Starting Stratified Evaluation ---")

        # 1. 标签转移
        predicted_labels_dict = transfer_cell_labels_knn(
            results_embedding=results_embedding,
            results_genes=results_genes,
            adata_reference=adata_control,
            cell_type_key=cell_type_key,
            embedding_key=sample_rep,
            n_neighbors=15,
            max_ref_cells=50000,
        )

        # 2. 分层评估
        stratified_df = evaluate_stratified_by_celltype(
            results_embedding=results_embedding,
            results_genes=results_genes,
            predicted_labels_dict=predicted_labels_dict,
            adata_treated=adata_test,
            adata_control=adata_control,
            pert_key=condition_combined_keys,
            cell_type_key=cell_type_key,
            embedding_key=sample_rep,
            rulebook=rulebook,
            use_groupwise_control=use_groupwise_control,
            min_cells_threshold=10,
            random_seed=random_seed,
            distribution_sample_num=Edistance_sample_num,
            top_n_degs=dist_top_n_degs,
            deg_padj_cutoff=deg_padj_cutoff,
            deg_fc_cutoff=deg_fc_cutoff,
            deg_top_n_for_eval=deg_top_n_for_eval,
            celltype_transition_rules=celltype_transition_rules,
            celltype_transition_unknown_policy=celltype_transition_unknown_policy,
        )

        # 3. 保存
        if save_each:
            _ensure_perturbation_col(latent_df).to_csv(
                os.path.join(results_save_path, "latent_metrics.csv"), index=False
            )
            _ensure_perturbation_col(avg_df).to_csv(
                os.path.join(results_save_path, "average_metrics.csv"), index=False
            )
            _ensure_perturbation_col(dist_df).to_csv(
                os.path.join(results_save_path, "distribution_metrics.csv"), index=False
            )
            _ensure_perturbation_col(stratified_df).to_csv(
                os.path.join(results_save_path, "stratified_celltype_metrics.csv"), index=False
            )

        final_df = _merge_on_perturbation(
            [latent_df, avg_df, dist_df, stratified_df],
            key="perturbation",
        )

        if detailed_celltype and detailed_perturbation:
            final_df = append_celltype_mean_rows(final_df, weight_col="n_true_ct")
            final_df = final_df.sort_values(["perturbation", "cell_type"]).reset_index(drop=True)

        elif detailed_celltype and (not detailed_perturbation):
            final_df = make_celltype_mean_df(final_df, weight_col="n_true_ct")
            final_df = final_df.sort_values("cell_type").reset_index(drop=True)

        elif (not detailed_celltype) and detailed_perturbation:
            final_df = final_df[final_df["cell_type"] == "all"].reset_index(drop=True)

        else:
            final_df = final_df[final_df["perturbation"] == "mean"].reset_index(drop=True)

        if save_final:
            final_path = os.path.join(results_save_path, final_filename)
            final_df.to_csv(final_path, index=False)
            print(f"All evaluations completed. Results saved to {final_path}")

        artifacts = {
            "latent_df": latent_df,
            "avg_df": avg_df,
            "dist_df": dist_df,
            "stratified_df": stratified_df,
        }

    else:
        if save_each:
            _ensure_perturbation_col(latent_df).to_csv(
                os.path.join(results_save_path, "latent_metrics.csv"), index=False
            )
            _ensure_perturbation_col(avg_df).to_csv(
                os.path.join(results_save_path, "average_metrics.csv"), index=False
            )
            _ensure_perturbation_col(dist_df).to_csv(
                os.path.join(results_save_path, "distribution_metrics.csv"), index=False
            )

        final_df = _merge_on_perturbation([latent_df, avg_df, dist_df], key="perturbation")
        if detailed_perturbation:
            final_df = final_df.sort_values("perturbation").reset_index(drop=True)
        else:
            final_df = final_df[final_df["perturbation"] == "mean"].reset_index(drop=True)

        if save_final:
            final_path = os.path.join(results_save_path, final_filename)
            final_df.to_csv(final_path, index=False)
            print(f"All evaluations completed. Results saved to {final_path}")

        artifacts = {
            "latent_df": latent_df,
            "avg_df": avg_df,
            "dist_df": dist_df,
        }

    return final_df, artifacts


from .wfr_multitime_umap_helper import (
    run_batch_inference_with_traj,
    fit_umap_reducer,
)

def run_inference_only(
    model,
    adata_control,
    adata_test,
    wishlist,
    condition_keys,
    condition_rep_keys,
    sample_rep,
    device,
    n_particles: int = 10000,
    random_seed: int = 42,
    n_steps: int = 50,
    scvi_model_load_path: str = None,   # 保留接口兼容，实际不再使用
    state_model_load_path: str = None,  # 保留接口兼容，实际不再使用
    source_celltype_key: str | None = None,

    # latent trajectory / UMAP
    return_traj: bool = True,
    save_times: list[float] | tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    t_destiny: float = 1.0,
    return_reducer: bool = False,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.35,
    umap_random_state: int = 0,

    # 新增：支持 all / v_only / g_only
    inference_modes: tuple[str, ...] = ("all", "v_only", "g_only"),
):
    """
    精简版：
    - 只做 latent space inference
    - 可返回多时间点 trajectory
    - 可返回 UMAP reducer
    - 不做 gene reconstruction

    Returns
    -------
    若 return_traj or return_reducer:
        results_embedding, None, indices, reducer

    否则:
        results_embedding, None, indices
    """
    import numpy as np

    if not wishlist:
        print("wishlist is empty")
        if return_traj or return_reducer:
            return None, None, None, None
        return None, None, None

    rulebook = adata_test.uns.get("global_rulebook")
    if rulebook is None:
        raise ValueError("adata_test.uns 缺少 'global_rulebook'。请先运行 prepare_covariates。")

    # ---- 1. 构建 perturbation embedding 字典 ----
    condition_embeddings = {}

    for adata in [adata_control, adata_test]:
        if adata is None:
            continue
        if condition_keys not in adata.obs.columns:
            continue

        for pert in adata.obs[condition_keys].astype(str).str.strip().unique():
            if pert in condition_embeddings:
                continue

            emb = get_condition_embedding(
                adata=adata,
                condition=pert,
                condition_rep_keys=condition_rep_keys,
                condition_keys=condition_keys,
                strict=False,
            )

            if emb is not None:
                condition_embeddings[pert] = emb

    # ---- 2. 采样 control cells 作为起点 ----
    all_indices = np.arange(adata_control.n_obs)
    rng = np.random.default_rng(random_seed)

    indices = (
        rng.choice(all_indices, n_particles, replace=False)
        if n_particles < len(all_indices)
        else all_indices
    )
    adata_source_subset = adata_control[indices].copy()

    # ---- 3. latent inference ----
    print(f"Starting latent inference for {len(wishlist)} conditions...")

    if return_traj:
        results_embedding = run_batch_inference_with_traj(
            model=model,
            adata_source=adata_source_subset,
            rulebook=rulebook,
            wishlist=wishlist,
            condition_embeddings=condition_embeddings,
            condition_key_name=condition_keys,
            sample_rep=sample_rep,
            n_steps=n_steps,
            device=device,
            source_celltype_key=source_celltype_key,
            t_destiny=t_destiny,
            save_times=save_times,
            modes=inference_modes,
        )
    else:
        results_embedding = run_batch_inference(
            model=model,
            adata_source=adata_source_subset,
            rulebook=rulebook,
            wishlist=wishlist,
            condition_embeddings=condition_embeddings,
            condition_key_name=condition_keys,
            sample_rep=sample_rep,
            n_steps=n_steps,
            device=device,
            source_celltype_key=source_celltype_key,
        )

    # ---- 4. fit UMAP reducer（可选）----
    reducer = None
    if return_reducer:
        reducer = fit_umap_reducer(
            adata_source=adata_source_subset,
            sample_rep=sample_rep,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            random_state=umap_random_state,
        )

    print("Latent inference completed.")

    # 不再做 gene reconstruction，为了兼容旧接口，第二个返回值仍然给 None
    if return_traj or return_reducer:
        return results_embedding, None, indices, reducer

    return results_embedding, None, indices