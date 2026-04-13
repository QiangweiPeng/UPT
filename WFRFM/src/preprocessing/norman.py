from __future__ import annotations

import pickle
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse


DEFAULT_NORMAN_CONTROL_CONDITION = "ctrl"


def load_norman_condition_rep_dict(condition_rep_path):
    """Load Norman condition embeddings from a pickle file."""
    with open(condition_rep_path, "rb") as handle:
        raw = pickle.load(handle)

    if not isinstance(raw, dict):
        raise TypeError("Norman condition embedding file must contain a dict.")

    return {
        str(condition): np.asarray(embedding, dtype=np.float32)
        for condition, embedding in raw.items()
    }


def load_norman_split_conditions(split_path):
    """Load the train/val/test condition split definition."""
    with open(split_path, "rb") as handle:
        split_dict = pickle.load(handle)

    required_splits = ("train", "val", "test")
    missing = [split_name for split_name in required_splits if split_name not in split_dict]
    if missing:
        raise KeyError(f"Missing Norman split keys: {missing}")

    return {
        split_name: [str(condition) for condition in split_dict[split_name]]
        for split_name in required_splits
    }


def _downsample_positions(positions, max_cells, rng):
    positions = np.asarray(positions, dtype=np.int64)
    if max_cells is None or positions.size <= max_cells:
        return positions
    return np.sort(rng.choice(positions, size=max_cells, replace=False))


def _build_empty_adata(var_df, obs_columns):
    obs = {column: np.array([], dtype=object) for column in obs_columns}
    adata = ad.AnnData(
        X=sparse.csr_matrix((0, var_df.shape[0]), dtype=np.float32),
        obs=obs,
        var=var_df.copy(),
    )
    return adata


def _build_lightweight_adata(backed_adata, obs_df, positions, obs_columns):
    positions = np.asarray(positions, dtype=np.int64)
    if positions.size == 0:
        return _build_empty_adata(backed_adata.var, obs_columns)

    obs = obs_df.iloc[positions][obs_columns].copy()
    X = backed_adata.X[positions]
    if sparse.issparse(X):
        X = X.tocsr().astype(np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)

    subset = ad.AnnData(X=X, obs=obs, var=backed_adata.var.copy())
    if "control" in subset.obs.columns and "is_control" not in subset.obs.columns:
        subset.obs["is_control"] = subset.obs["control"].astype(bool)
    return subset


def _collect_treated_positions(
    obs_df,
    condition_key,
    split_conditions,
    max_cells_per_condition,
    rng,
):
    positions = []
    treated_obs = obs_df.loc[~obs_df["_is_control"]]
    for condition in split_conditions:
        condition_positions = treated_obs.loc[
            treated_obs[condition_key] == condition,
            "_obs_position",
        ].to_numpy(dtype=np.int64, copy=False)
        positions.append(
            _downsample_positions(condition_positions, max_cells_per_condition, rng)
        )

    if not positions:
        return np.array([], dtype=np.int64)

    merged = np.concatenate(positions)
    merged.sort()
    return merged


def _normalize_split_conditions(
    raw_split_conditions,
    observed_treated_conditions,
    condition_rep_dict,
    control_condition,
):
    observed_set = set(observed_treated_conditions)
    embedding_conditions = set(condition_rep_dict)
    normalized = {}
    missing_in_data = {}
    missing_in_embeddings = {}

    for split_name, conditions in raw_split_conditions.items():
        selected = []
        missing_data = []
        missing_embeddings = []

        for condition in conditions:
            if condition == control_condition:
                continue
            if condition not in observed_set:
                missing_data.append(condition)
                continue
            if condition not in embedding_conditions:
                missing_embeddings.append(condition)
                continue
            if condition not in selected:
                selected.append(condition)

        normalized[split_name] = selected
        missing_in_data[split_name] = missing_data
        missing_in_embeddings[split_name] = missing_embeddings

    if any(missing_in_data.values()):
        raise KeyError(
            "Norman split contains conditions missing from the dataset: "
            + ", ".join(
                f"{split_name}={missing}"
                for split_name, missing in missing_in_data.items()
                if missing
            )
        )

    if any(missing_in_embeddings.values()):
        raise KeyError(
            "Norman split contains conditions missing from condition_rep_dict: "
            + ", ".join(
                f"{split_name}={missing}"
                for split_name, missing in missing_in_embeddings.items()
                if missing
            )
        )

    return normalized


def load_norman_training_data(
    data_path,
    split_path,
    condition_rep_path,
    condition_key="condition",
    control_key="control",
    control_condition=DEFAULT_NORMAN_CONTROL_CONDITION,
    control_max_cells=None,
    treated_max_cells_per_condition=None,
    max_train_conditions=None,
    max_val_conditions=None,
    max_test_conditions=None,
    materialize_test=False,
    extra_obs_columns=None,
    seed=42,
):
    """
    Load lightweight Norman AnnData subsets for training and validation.

    The returned AnnData objects keep ``adata.X`` as the processed gene-expression
    input and only materialize the selected subsets into memory.
    """
    if extra_obs_columns is None:
        extra_obs_columns = ["condition_name", "cell_type", "dose_val", control_key]

    raw_split_conditions = load_norman_split_conditions(split_path)
    full_condition_rep_dict = load_norman_condition_rep_dict(condition_rep_path)
    rng = np.random.default_rng(seed)

    backed = ad.read_h5ad(data_path, backed="r")
    try:
        obs_df = backed.obs.copy()
        obs_df[condition_key] = obs_df[condition_key].astype(str)
        obs_df["_obs_position"] = np.arange(obs_df.shape[0], dtype=np.int64)
        obs_df["_is_control"] = obs_df[control_key].astype(bool).to_numpy()

        observed_treated_conditions = sorted(
            obs_df.loc[~obs_df["_is_control"], condition_key].unique().tolist()
        )
        selected_split_conditions = _normalize_split_conditions(
            raw_split_conditions=raw_split_conditions,
            observed_treated_conditions=observed_treated_conditions,
            condition_rep_dict=full_condition_rep_dict,
            control_condition=control_condition,
        )

        if max_train_conditions is not None:
            selected_split_conditions["train"] = selected_split_conditions["train"][:max_train_conditions]
        if max_val_conditions is not None:
            selected_split_conditions["val"] = selected_split_conditions["val"][:max_val_conditions]
        if max_test_conditions is not None:
            selected_split_conditions["test"] = selected_split_conditions["test"][:max_test_conditions]

        requested_obs_columns = [condition_key, control_key, *list(extra_obs_columns)]
        obs_columns = []
        for column in requested_obs_columns:
            if column in obs_df.columns and column not in obs_columns:
                obs_columns.append(column)

        control_positions = np.flatnonzero(obs_df["_is_control"].to_numpy())
        control_positions = _downsample_positions(control_positions, control_max_cells, rng)
        train_positions = _collect_treated_positions(
            obs_df=obs_df,
            condition_key=condition_key,
            split_conditions=selected_split_conditions["train"],
            max_cells_per_condition=treated_max_cells_per_condition,
            rng=rng,
        )
        val_positions = _collect_treated_positions(
            obs_df=obs_df,
            condition_key=condition_key,
            split_conditions=selected_split_conditions["val"],
            max_cells_per_condition=treated_max_cells_per_condition,
            rng=rng,
        )
        test_positions = _collect_treated_positions(
            obs_df=obs_df,
            condition_key=condition_key,
            split_conditions=selected_split_conditions["test"],
            max_cells_per_condition=treated_max_cells_per_condition,
            rng=rng,
        )

        adata_control = _build_lightweight_adata(backed, obs_df, control_positions, obs_columns)
        adata_train = _build_lightweight_adata(backed, obs_df, train_positions, obs_columns)
        adata_val = _build_lightweight_adata(backed, obs_df, val_positions, obs_columns)
        adata_test = None
        if materialize_test:
            adata_test = _build_lightweight_adata(backed, obs_df, test_positions, obs_columns)
    finally:
        backed.file.close()

    used_conditions = {control_condition}
    for conditions in selected_split_conditions.values():
        used_conditions.update(conditions)
    condition_rep_dict = {
        condition: full_condition_rep_dict[condition]
        for condition in sorted(used_conditions)
        if condition in full_condition_rep_dict
    }

    split_cell_counts = {}
    control_mask = obs_df["_is_control"].to_numpy()
    for split_name, conditions in selected_split_conditions.items():
        if not conditions:
            split_cell_counts[split_name] = 0
            continue
        split_mask = (~control_mask) & obs_df[condition_key].isin(conditions).to_numpy()
        split_cell_counts[split_name] = int(split_mask.sum())

    assigned_conditions = set().union(*selected_split_conditions.values()) if selected_split_conditions else set()
    metadata = {
        "data_path": str(Path(data_path)),
        "split_path": str(Path(split_path)),
        "condition_rep_path": str(Path(condition_rep_path)),
        "condition_key": condition_key,
        "control_key": control_key,
        "control_condition": control_condition,
        "raw_split_conditions": raw_split_conditions,
        "selected_split_conditions": selected_split_conditions,
        "observed_treated_conditions": observed_treated_conditions,
        "unassigned_treated_conditions": sorted(
            set(observed_treated_conditions).difference(assigned_conditions)
        ),
        "split_cell_counts": split_cell_counts,
        "materialized_n_obs": {
            "control": int(control_positions.size),
            "train": int(train_positions.size),
            "val": int(val_positions.size),
            "test": int(test_positions.size),
        },
    }

    return adata_control, adata_train, adata_val, adata_test, condition_rep_dict, metadata


def build_pca_reference_adata(ref_adata):
    """Create a lightweight AnnData object that keeps only PCA reconstruction stats."""
    required_varm = ("PCs", "X_mean")
    missing_varm = [key for key in required_varm if key not in ref_adata.varm]
    if missing_varm:
        raise KeyError(f"Missing PCA reference fields in ref_adata.varm: {missing_varm}")
    if "pca" not in ref_adata.uns or "variance" not in ref_adata.uns["pca"]:
        raise KeyError("Missing ref_adata.uns['pca']['variance'] for PCA reconstruction.")

    reference = ad.AnnData(
        X=sparse.csr_matrix((0, ref_adata.n_vars), dtype=np.float32),
        var=ref_adata.var.copy(),
    )
    reference.varm["PCs"] = np.asarray(ref_adata.varm["PCs"], dtype=np.float32)
    reference.varm["X_mean"] = np.asarray(ref_adata.varm["X_mean"], dtype=np.float32)

    pca_uns = {}
    for key, value in ref_adata.uns["pca"].items():
        if isinstance(value, np.ndarray):
            pca_uns[key] = value.astype(np.float32, copy=False)
        else:
            pca_uns[key] = value
    reference.uns["pca"] = pca_uns
    reference.uns["pca_reference"] = {
        "n_vars": int(ref_adata.n_vars),
        "n_comps": int(reference.varm["PCs"].shape[1]),
    }
    return reference
