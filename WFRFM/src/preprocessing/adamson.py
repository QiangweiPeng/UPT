from __future__ import annotations

import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


DEFAULT_ADAMSON_CONTROL_CONDITION = "ctrl"


def load_adamson_gene_rep_dict(condition_rep_path):
    """
    Load Adamson gene embeddings from CSV or pickle.

    Supported formats:
    - CSV with one identifier column named `gene_symbol` or `gene`, plus `emb_*` columns
    - Pickle containing a dict mapping gene symbol to embedding vector
    """
    path = Path(condition_rep_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
        if "gene_symbol" in df.columns:
            gene_col = "gene_symbol"
        elif "gene" in df.columns:
            gene_col = "gene"
        else:
            raise KeyError(
                f"{path} must contain a 'gene_symbol' or 'gene' column."
            )
        embedding_cols = [col for col in df.columns if col != gene_col]
        if not embedding_cols:
            raise ValueError(f"{path} does not contain embedding columns.")
        return {
            str(row[gene_col]): row[embedding_cols].to_numpy(dtype=np.float32, copy=True)
            for _, row in df.iterrows()
        }

    if suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as handle:
            raw = pickle.load(handle)
        if not isinstance(raw, dict):
            raise TypeError(f"{path} must contain a dict when using pickle format.")
        return {
            str(gene): np.asarray(embedding, dtype=np.float32)
            for gene, embedding in raw.items()
        }

    raise ValueError(f"Unsupported Adamson embedding file format: {path.suffix}")


def _coerce_embedding_dict(embedding_dict):
    return {
        str(key): np.asarray(value, dtype=np.float32)
        for key, value in embedding_dict.items()
    }


def load_adamson_split_conditions(split_path):
    """Load the Adamson train/val/test condition split definition."""
    with open(split_path, "rb") as handle:
        split_dict = pickle.load(handle)

    required_splits = ("train", "val", "test")
    missing = [split_name for split_name in required_splits if split_name not in split_dict]
    if missing:
        raise KeyError(f"Missing Adamson split keys: {missing}")

    return {
        split_name: [str(condition) for condition in split_dict[split_name]]
        for split_name in required_splits
    }


def _extract_condition_genes(condition_name, control_condition):
    parts = [part.strip() for part in str(condition_name).split("+") if part.strip()]
    return [part for part in parts if part != control_condition]


def build_adamson_condition_rep_dict(
    condition_names,
    go_path=None,
    gene_rep_dict=None,
    control_condition=DEFAULT_ADAMSON_CONTROL_CONDITION,
    normalize=True,
):
    """
    Build Adamson condition embeddings from GO-derived source-target importance profiles.

    Each perturbation condition is represented by the mean of its perturbed gene vectors.
    The control condition receives the zero vector.
    """
    unique_conditions = []
    seen_conditions = set()
    for condition in condition_names:
        condition = str(condition)
        if condition not in seen_conditions:
            unique_conditions.append(condition)
            seen_conditions.add(condition)

    perturbed_genes = sorted(
        {
            gene
            for condition in unique_conditions
            for gene in _extract_condition_genes(condition, control_condition)
        }
    )

    embedding_metadata = {
        "normalize": bool(normalize),
    }
    if gene_rep_dict is None:
        if go_path is None:
            raise ValueError("Either go_path or gene_rep_dict must be provided for Adamson embeddings.")
        go_df = pd.read_csv(go_path, usecols=["source", "target", "importance"])
        go_df["source"] = go_df["source"].astype(str)
        go_df["target"] = go_df["target"].astype(str)
        go_df = go_df.loc[go_df["source"].isin(perturbed_genes)].copy()
        if go_df.empty:
            raise ValueError("Adamson GO table did not yield any rows for the requested perturbation genes.")

        target_vocab = sorted(go_df["target"].unique().tolist())
        target_index = {target: idx for idx, target in enumerate(target_vocab)}

        built_gene_rep_dict = {}
        missing_genes = []
        for gene in perturbed_genes:
            gene_df = go_df.loc[go_df["source"] == gene]
            if gene_df.empty:
                missing_genes.append(gene)
                continue

            vector = np.zeros(len(target_vocab), dtype=np.float32)
            indices = gene_df["target"].map(target_index).to_numpy(dtype=np.int64, copy=False)
            values = gene_df["importance"].to_numpy(dtype=np.float32, copy=False)
            np.add.at(vector, indices, values)
            if normalize:
                norm = float(np.linalg.norm(vector))
                if norm > 0:
                    vector /= norm
            built_gene_rep_dict[gene] = vector

        if missing_genes:
            raise KeyError(
                "Missing Adamson GO profiles for perturbation genes: "
                + ", ".join(sorted(missing_genes))
            )
        gene_rep_dict = built_gene_rep_dict
        embedding_metadata.update(
            {
                "embedding_source": "go_csv",
                "go_path": str(Path(go_path)),
                "n_go_targets": len(target_vocab),
            }
        )
    else:
        gene_rep_dict = {
            str(gene): np.asarray(embedding, dtype=np.float32)
            for gene, embedding in gene_rep_dict.items()
        }
        missing_genes = sorted(set(perturbed_genes).difference(gene_rep_dict))
        if missing_genes:
            raise KeyError(
                "Missing Adamson gene embeddings for perturbation genes: "
                + ", ".join(missing_genes)
            )
        embedding_metadata.update(
            {
                "embedding_source": "gene_rep_dict",
                "n_embedding_dims": int(next(iter(gene_rep_dict.values())).shape[0]),
            }
        )

    if not gene_rep_dict:
        raise ValueError("Adamson gene_rep_dict is empty after embedding construction.")
    zero_vector = np.zeros_like(next(iter(gene_rep_dict.values())), dtype=np.float32)
    condition_rep_dict = {control_condition: zero_vector.copy()}
    for condition in unique_conditions:
        genes = _extract_condition_genes(condition, control_condition)
        if not genes:
            condition_rep_dict[condition] = zero_vector.copy()
            continue
        vectors = [gene_rep_dict[gene] for gene in genes]
        condition_rep_dict[condition] = np.mean(vectors, axis=0).astype(np.float32)

    metadata = {
        "n_conditions": len(unique_conditions),
        "n_perturbed_genes": len(perturbed_genes),
    }
    metadata.update(embedding_metadata)
    return condition_rep_dict, metadata


def _downsample_positions(positions, max_cells, rng):
    positions = np.asarray(positions, dtype=np.int64)
    if max_cells is None or positions.size <= max_cells:
        return positions
    return np.sort(rng.choice(positions, size=max_cells, replace=False))


def _build_empty_adata(var_df, obs_columns):
    obs = {column: np.array([], dtype=object) for column in obs_columns}
    return ad.AnnData(
        X=sparse.csr_matrix((0, var_df.shape[0]), dtype=np.float32),
        obs=obs,
        var=var_df.copy(),
    )


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
            "Adamson split contains conditions missing from the dataset: "
            + ", ".join(
                f"{split_name}={missing}"
                for split_name, missing in missing_in_data.items()
                if missing
            )
        )

    if any(missing_in_embeddings.values()):
        raise KeyError(
            "Adamson split contains conditions missing from condition_rep_dict: "
            + ", ".join(
                f"{split_name}={missing}"
                for split_name, missing in missing_in_embeddings.items()
                if missing
            )
        )

    return normalized


def load_adamson_training_data(
    data_path,
    split_path,
    go_path=None,
    condition_rep_path=None,
    condition_key="condition",
    control_key="control",
    control_condition=DEFAULT_ADAMSON_CONTROL_CONDITION,
    control_max_cells=None,
    treated_max_cells_per_condition=None,
    max_train_conditions=None,
    max_val_conditions=None,
    max_test_conditions=None,
    materialize_test=False,
    extra_obs_columns=None,
    seed=42,
):
    """Load lightweight Adamson AnnData subsets and GO-based condition embeddings."""
    if extra_obs_columns is None:
        extra_obs_columns = ["condition_name", "cell_type", "dose_val", control_key]

    raw_split_conditions = load_adamson_split_conditions(split_path)
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
        split_requested_conditions = []
        split_seen = set()
        for split_name in ("train", "val", "test"):
            for condition in raw_split_conditions[split_name]:
                condition = str(condition)
                if condition == control_condition or condition in split_seen:
                    continue
                split_requested_conditions.append(condition)
                split_seen.add(condition)

        gene_rep_dict = None
        direct_condition_rep_dict = None
        if condition_rep_path is not None:
            loaded_rep_dict = load_adamson_gene_rep_dict(condition_rep_path)
            if control_condition in loaded_rep_dict or any(
                "+" in key for key in loaded_rep_dict
            ):
                direct_condition_rep_dict = _coerce_embedding_dict(loaded_rep_dict)
                if control_condition not in direct_condition_rep_dict:
                    zero_template = np.zeros_like(
                        next(iter(direct_condition_rep_dict.values())),
                        dtype=np.float32,
                    )
                    direct_condition_rep_dict[control_condition] = zero_template
                embedding_metadata = {
                    "embedding_source": "condition_rep_path",
                    "condition_rep_path": str(Path(condition_rep_path)),
                    "n_embedding_dims": int(next(iter(direct_condition_rep_dict.values())).shape[0]),
                }
                full_condition_rep_dict = direct_condition_rep_dict
            else:
                gene_rep_dict = loaded_rep_dict
                full_condition_rep_dict, embedding_metadata = build_adamson_condition_rep_dict(
                    go_path=go_path,
                    gene_rep_dict=gene_rep_dict,
                    condition_names=[control_condition, *split_requested_conditions],
                    control_condition=control_condition,
                )
        else:
            full_condition_rep_dict, embedding_metadata = build_adamson_condition_rep_dict(
                go_path=go_path,
                gene_rep_dict=gene_rep_dict,
                condition_names=[control_condition, *split_requested_conditions],
                control_condition=control_condition,
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
        "go_path": None if go_path is None else str(Path(go_path)),
        "condition_rep_path": None if condition_rep_path is None else str(Path(condition_rep_path)),
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
        "embedding_metadata": embedding_metadata,
    }

    return adata_control, adata_train, adata_val, adata_test, condition_rep_dict, metadata
