import os
import numpy as np
import pandas as pd
import anndata as ad

from .evals_new import evaluate_population_average, evaluate_population_distribution


def _ensure_perturbation_col(df: pd.DataFrame, key="perturbation") -> pd.DataFrame:
    if df is None:
        return None
    df = df.copy()
    if key not in df.columns:
        if df.index.name == key:
            df = df.reset_index()
        else:
            if df.index.dtype == object:
                df = df.reset_index().rename(columns={"index": key})
            else:
                raise ValueError(f"Cannot find '{key}' column or index in df.")
    return df


def _merge_on_perturbation(dfs, key="perturbation") -> pd.DataFrame:
    merged = None
    for i, df in enumerate(dfs):
        if df is None:
            continue
        df = _ensure_perturbation_col(df, key=key)

        if merged is None:
            merged = df
            continue

        overlap = (set(merged.columns) & set(df.columns)) - {key}
        if overlap:
            df = df.rename(columns={c: f"{c}__dup{i}" for c in overlap})

        merged = merged.merge(df, on=key, how="outer")

    if merged is None:
        merged = pd.DataFrame(columns=[key])

    return merged


def _ensure_mass_in_results_genes_dict(
    results_genes_dict: dict,
    mass_key: str = "mass",
    inplace: bool = False,
) -> dict:
    """
    确保每个 AnnData 都有 obs[mass_key]。
    若缺失则补 1.0（等权）。
    """
    if results_genes_dict is None or len(results_genes_dict) == 0:
        raise ValueError("results_genes_dict is None or empty.")

    out = results_genes_dict if inplace else {}
    for pert, a in results_genes_dict.items():
        if a is None:
            continue
        if not isinstance(a, ad.AnnData):
            raise TypeError(f"results_genes_dict['{pert}'] is not an AnnData.")

        aa = a if inplace else a.copy()

        if mass_key not in aa.obs.columns:
            aa.obs[mass_key] = np.ones(aa.n_obs, dtype=np.float32)

        # 转成 float 并做一次健壮性处理：负数/全0 -> 等权
        m = np.asarray(aa.obs[mass_key], dtype=np.float64)
        if np.any(~np.isfinite(m)) or np.any(m < 0) or m.sum() <= 0:
            m = np.ones(aa.n_obs, dtype=np.float64)
            aa.obs[mass_key] = m

        if not inplace:
            out[pert] = aa

    return out


def evaluate_gene_reconstruction_only(
    *,
    results_genes_dict: dict,          # Dict[str, AnnData], condition -> AnnData(X=pred)
    adata_control: ad.AnnData,
    adata_treated: ad.AnnData,         # treated 的真实数据（用于对比）
    pert_key: str,                     # adata_treated.obs 里标识条件的列名
    control_label,                     # 透传给你的 eval 函数（即使内部不用）
    results_save_path: str,
    embedding_key: str = "unused",     # 透传给你的 eval 函数（即使内部不用）
    mass_key: str = "mass",
    random_seed: int = 42,
    Edistance_sample_num: int = 2000,
    dist_max_cells: int = 200,
    dist_max_genes: int = 100000,
    dist_n_bins: int = 50,
    dist_top_n_degs: int = 50,
    save_each: bool = False,
    save_final: bool = True,
    final_filename: str = "final_metrics.csv",
    inplace_mass: bool = False,        # True: 直接在原 AnnData 上补 mass；False: copy 后补
):
    """
    只依赖 AnnData.X 的 gene 重建评估：
    - Pseudo-bulk（average）
    - 分布（distribution）

    约定：
    - results_genes_dict 的 key 必须能在 adata_treated.obs[pert_key] 中匹配到
    - 每个 AnnData 如果没有 obs[mass_key]，自动补等权
    """
    if adata_treated is None:
        raise ValueError("adata_treated is None.")
    if adata_control is None:
        raise ValueError("adata_control is None.")
    if pert_key is None:
        raise ValueError("pert_key is None.")
    if results_save_path is None:
        raise ValueError("results_save_path is None.")

    os.makedirs(results_save_path, exist_ok=True)

    # 1) 补齐 mass
    results_genes = _ensure_mass_in_results_genes_dict(
        results_genes_dict,
        mass_key=mass_key,
        inplace=inplace_mass
    )

    # 2) 两项评估
    avg_df = evaluate_population_average(
        results_genes=results_genes,
        adata_treated=adata_treated,
        adata_control=adata_control,
        pert_key=pert_key,
        control_label=control_label,
        embedding_key=embedding_key,
        Edistance_sample_num=Edistance_sample_num,
        random_seed=random_seed,
    )

    dist_df = evaluate_population_distribution(
        results_genes=results_genes,
        adata_treated=adata_treated,
        adata_control=adata_control,
        pert_key=pert_key,
        max_cells=dist_max_cells,
        max_genes=dist_max_genes,
        n_bins=dist_n_bins,
        top_n_degs=dist_top_n_degs,
        seed=random_seed,
    )

    # 3) 保存每项
    if save_each:
        _ensure_perturbation_col(avg_df).to_csv(
            os.path.join(results_save_path, "average_metrics.csv"),
            index=False
        )
        _ensure_perturbation_col(dist_df).to_csv(
            os.path.join(results_save_path, "distribution_metrics.csv"),
            index=False
        )

    # 4) 合并 final
    final_df = _merge_on_perturbation([avg_df, dist_df], key="perturbation")
    final_df = final_df.sort_values("perturbation").reset_index(drop=True)

    if save_final:
        final_df.to_csv(os.path.join(results_save_path, final_filename), index=False)

    artifacts = {
        "results_genes": results_genes,
        "avg_df": avg_df,
        "dist_df": dist_df,
    }
    return final_df, artifacts
