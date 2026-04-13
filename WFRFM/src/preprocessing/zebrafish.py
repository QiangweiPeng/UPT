import anndata as ad
import numpy as np
import pandas as pd


DEFAULT_ZEBRAFISH_TIMEPOINTS = [18.0, 24.0, 36.0, 48.0, 72.0]


def build_zebrafish_condition_rep_dict(
    obs_df,
    gene_embeddings,
    condition_key="gene_target",
    gene1_key="gene_target_1",
    gene2_key="gene_target_2",
):
    """Build condition embeddings by averaging the two gene embeddings."""
    required = [condition_key, gene1_key, gene2_key]
    missing = [key for key in required if key not in obs_df.columns]
    if missing:
        raise KeyError(f"Missing zebrafish condition columns: {missing}")

    unique_rows = obs_df[required].drop_duplicates()
    condition_rep_dict = {}
    missing_genes = set()

    for row in unique_rows.itertuples(index=False):
        condition_name, gene_1, gene_2 = row
        if gene_1 not in gene_embeddings:
            missing_genes.add(gene_1)
            continue
        if gene_2 not in gene_embeddings:
            missing_genes.add(gene_2)
            continue

        vec_1 = np.asarray(gene_embeddings[gene_1], dtype=np.float32)
        vec_2 = np.asarray(gene_embeddings[gene_2], dtype=np.float32)
        condition_rep_dict[str(condition_name)] = ((vec_1 + vec_2) / 2.0).astype(np.float32)

    if missing_genes:
        raise KeyError(
            "Missing embeddings for zebrafish genes: "
            + ", ".join(sorted(str(gene) for gene in missing_genes))
        )

    return condition_rep_dict


def attach_condition_embeddings(
    adata,
    condition_rep_dict,
    condition_key="gene_target",
    embedding_key="gene_embeddings",
):
    """Attach per-condition embeddings to ``adata.obsm``."""
    if condition_key not in adata.obs:
        raise KeyError(f"{condition_key} is not found in adata.obs")

    condition_series = adata.obs[condition_key].astype(str)
    missing_conditions = sorted(set(condition_series.unique()).difference(condition_rep_dict))
    if missing_conditions:
        raise KeyError(
            "Missing condition embeddings for: "
            + ", ".join(missing_conditions[:10])
        )

    adata.obsm[embedding_key] = np.stack(
        [condition_rep_dict[condition] for condition in condition_series],
        axis=0,
    ).astype(np.float32)
    return adata


def scale_representation_by_control(
    adata_control,
    adata_treated,
    source_rep="X_pca",
    target_rep=None,
    min_std=1e-6,
):
    """Scale an existing representation using control-group standard deviation."""
    if source_rep not in adata_control.obsm:
        raise KeyError(f"{source_rep} is not found in adata_control.obsm")
    if source_rep not in adata_treated.obsm:
        raise KeyError(f"{source_rep} is not found in adata_treated.obsm")

    if target_rep is None:
        target_rep = f"{source_rep}_scaled"

    control_rep = np.asarray(adata_control.obsm[source_rep], dtype=np.float32)
    treated_rep = np.asarray(adata_treated.obsm[source_rep], dtype=np.float32)
    scale = control_rep.std(axis=0, dtype=np.float64)
    scale = np.where(scale < min_std, 1.0, scale).astype(np.float32)

    adata_control.obsm[target_rep] = control_rep / scale
    adata_treated.obsm[target_rep] = treated_rep / scale
    adata_control.uns[f"{target_rep}_scale"] = scale
    adata_treated.uns[f"{target_rep}_scale"] = scale
    return target_rep


def build_zebrafish_control_condition_embedding(
    data_path,
    adata_control,
    condition_key="gene_target",
):
    """
    Build the explicit control-condition embedding used for zebrafish inference.

    Returns ``(condition_name, embedding)`` where ``condition_name`` is typically
    ``"control_control"``.
    """
    if adata_control.n_obs == 0:
        raise ValueError("adata_control is empty; cannot build the control condition embedding.")

    backed = ad.read_h5ad(data_path, backed="r")
    try:
        control_rep_dict = build_zebrafish_condition_rep_dict(
            obs_df=adata_control.obs,
            gene_embeddings=backed.uns["gene_embeddings"],
            condition_key=condition_key,
            gene1_key="gene_target_1",
            gene2_key="gene_target_2",
        )
    finally:
        backed.file.close()

    if "control_control" in control_rep_dict:
        condition_name = "control_control"
    elif len(control_rep_dict) == 1:
        condition_name = next(iter(control_rep_dict))
    else:
        raise KeyError(
            "Failed to identify a unique zebrafish control condition embedding. "
            f"Available control conditions: {sorted(control_rep_dict)}"
        )

    return str(condition_name), np.asarray(control_rep_dict[condition_name], dtype=np.float32)


def _downsample_positions(positions, max_cells, rng):
    if max_cells is None or len(positions) <= max_cells:
        return np.asarray(positions, dtype=np.int64)
    return np.sort(rng.choice(np.asarray(positions, dtype=np.int64), size=max_cells, replace=False))


def _build_lightweight_adata(backed_adata, positions, rep_keys, obs_columns):
    obs = backed_adata.obs.iloc[positions][obs_columns].copy()
    placeholder_x = np.zeros((len(positions), 1), dtype=np.float32)
    subset = ad.AnnData(X=placeholder_x, obs=obs)
    subset.var_names = ["placeholder"]

    for rep_key in rep_keys:
        subset.obsm[rep_key] = np.asarray(backed_adata.obsm[rep_key][positions], dtype=np.float32)

    return subset


def load_zebrafish_training_data(
    data_path,
    rep_keys=None,
    condition_key="gene_target",
    marginal_key="timepoint",
    control_key="is_control",
    first_control_key="first_t_control",
    use_first_control=True,
    max_conditions=None,
    selected_conditions=None,
    control_max_cells=None,
    treated_max_cells_per_group=None,
    extra_obs_columns=None,
    seed=42,
):
    """
    Load a lightweight zebrafish AnnData for training.

    Only ``obs`` and requested ``obsm`` representations are materialized to memory.
    """
    if rep_keys is None:
        rep_keys = ["X_pca"]
    if extra_obs_columns is None:
        extra_obs_columns = []

    backed = ad.read_h5ad(data_path, backed="r")
    try:
        obs_df = backed.obs.copy()
        obs_df[condition_key] = obs_df[condition_key].astype(str)
        obs_df["_obs_position"] = np.arange(obs_df.shape[0], dtype=np.int64)
        requested_obs_columns = [
            condition_key,
            marginal_key,
            control_key,
            first_control_key,
            "gene_target_1",
            "gene_target_2",
            "condition",
            *list(extra_obs_columns),
        ]
        obs_columns = []
        for column in requested_obs_columns:
            if column in obs_df.columns and column not in obs_columns:
                obs_columns.append(column)

        treated_mask = ~obs_df[control_key].astype(bool)
        treated_obs = obs_df.loc[treated_mask].copy()
        all_conditions = sorted(treated_obs[condition_key].unique().tolist())
        if selected_conditions is not None:
            selected_conditions = [str(condition) for condition in selected_conditions]
            missing_conditions = sorted(set(selected_conditions).difference(all_conditions))
            if missing_conditions:
                raise KeyError(
                    "Unknown zebrafish conditions requested: "
                    + ", ".join(missing_conditions[:10])
                )
            treated_obs = treated_obs[treated_obs[condition_key].isin(selected_conditions)].copy()
        elif max_conditions is not None:
            selected_conditions = all_conditions[:max_conditions]
            treated_obs = treated_obs[treated_obs[condition_key].isin(selected_conditions)].copy()
        else:
            selected_conditions = all_conditions

        rng = np.random.default_rng(seed)
        treated_positions = []
        grouped = treated_obs.groupby(
            [condition_key, marginal_key],
            observed=True,
            sort=False,
        )
        for _, group_df in grouped:
            positions = group_df["_obs_position"].to_numpy(dtype=np.int64, copy=False)
            treated_positions.append(
                _downsample_positions(positions, treated_max_cells_per_group, rng)
            )
        if treated_positions:
            treated_positions = np.concatenate(treated_positions)
            treated_positions.sort()
        else:
            treated_positions = np.array([], dtype=np.int64)

        if use_first_control and first_control_key in obs_df.columns:
            control_mask = obs_df[first_control_key].astype(bool)
        else:
            control_mask = obs_df[control_key].astype(bool)
        control_positions = np.flatnonzero(control_mask.to_numpy())
        control_positions = _downsample_positions(control_positions, control_max_cells, rng)

        adata_control = _build_lightweight_adata(backed, control_positions, rep_keys, obs_columns)
        adata_treated = _build_lightweight_adata(backed, treated_positions, rep_keys, obs_columns)

        condition_rep_dict = build_zebrafish_condition_rep_dict(
            obs_df=adata_treated.obs,
            gene_embeddings=backed.uns["gene_embeddings"],
            condition_key=condition_key,
            gene1_key="gene_target_1",
            gene2_key="gene_target_2",
        )
    finally:
        backed.file.close()

    return adata_control, adata_treated, condition_rep_dict, selected_conditions
