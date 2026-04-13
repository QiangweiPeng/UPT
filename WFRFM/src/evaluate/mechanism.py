from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROLLOUT_MODE_ORDER = ["full", "transport-only", "growth-only"]


def load_prediction_collection(path_or_adata, selected_conditions=None):
    if isinstance(path_or_adata, ad.AnnData):
        if selected_conditions is None:
            return path_or_adata
        return path_or_adata[
            path_or_adata.obs["target_condition"].astype(str).isin(
                [str(condition) for condition in selected_conditions]
            )
        ].copy()

    path = Path(path_or_adata)
    backed = ad.read_h5ad(path, backed="r")
    try:
        obs_df = backed.obs.copy()
        if selected_conditions is not None:
            selected_conditions = [str(condition) for condition in selected_conditions]
            mask = obs_df["target_condition"].astype(str).isin(selected_conditions)
            positions = np.flatnonzero(mask.to_numpy())
        else:
            positions = np.arange(obs_df.shape[0], dtype=np.int64)

        subset_obs = obs_df.iloc[positions].copy()
        subset_x = np.asarray(backed.X[positions], dtype=np.float32)
        subset = ad.AnnData(X=subset_x, obs=subset_obs)
        subset.var_names = backed.var_names.copy()
        if "prediction_collection" in backed.uns:
            subset.uns["prediction_collection"] = dict(backed.uns["prediction_collection"])
        return subset
    finally:
        backed.file.close()


def normalize_label_series(series, missing_label="Unknown"):
    values = pd.Series(series).astype(str).replace({"nan": missing_label, "None": missing_label})
    values = values.fillna(missing_label)
    return values


def compute_weighted_label_distribution(labels, weights, label_order=None):
    labels = normalize_label_series(labels)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if labels.shape[0] != weights.shape[0]:
        raise ValueError("labels and weights must share the same length")

    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    if label_order is None:
        label_order = sorted(labels.unique().tolist())
    label_order = [str(label) for label in label_order]

    totals = {label: 0.0 for label in label_order}
    for label, weight in zip(labels.to_numpy(), weights, strict=False):
        label = str(label)
        if label not in totals:
            totals[label] = 0.0
            label_order.append(label)
        totals[label] += float(weight)

    total_mass = float(sum(totals.values()))
    if total_mass <= 0:
        return {label: 0.0 for label in label_order}
    return {label: float(totals[label] / total_mass) for label in label_order}


def total_variation_distance_from_dict(comp_a, comp_b, label_order=None):
    if label_order is None:
        label_order = sorted(set(comp_a).union(comp_b))
    label_order = [str(label) for label in label_order]
    return 0.5 * sum(
        abs(float(comp_a.get(label, 0.0)) - float(comp_b.get(label, 0.0)))
        for label in label_order
    )


def infer_label_order(real_adata, label_key):
    if label_key not in real_adata.obs.columns:
        raise KeyError(f"{label_key} is not found in real_adata.obs")
    return sorted(normalize_label_series(real_adata.obs[label_key]).unique().tolist())


def attach_terminal_labels_knn(
    pred_adata: ad.AnnData,
    real_adata: ad.AnnData,
    latent_key: str,
    label_key: str,
    output_key: str,
    k: int = 15,
):
    if pred_adata.n_obs == 0:
        pred_adata.obs[output_key] = pd.Categorical([])
        return pred_adata
    if real_adata.n_obs == 0:
        raise ValueError("real_adata is empty; cannot transfer terminal labels")
    if label_key not in real_adata.obs.columns:
        raise KeyError(f"{label_key} is not found in real_adata.obs")

    X_real = np.asarray(real_adata.obsm[latent_key], dtype=np.float32)
    X_pred = np.asarray(pred_adata.X, dtype=np.float32)
    if X_real.ndim != 2 or X_pred.ndim != 2:
        raise ValueError("Both real and predicted latent arrays must be 2D")
    if X_real.shape[1] != X_pred.shape[1]:
        raise ValueError("Real and predicted latent dimensions must match")

    labels_real = normalize_label_series(real_adata.obs[label_key]).to_numpy()
    tree = cKDTree(X_real)
    query_k = int(min(max(1, k), X_real.shape[0]))
    distances, indices = tree.query(X_pred, k=query_k)

    if query_k == 1:
        distances = np.asarray(distances, dtype=np.float64).reshape(-1, 1)
        indices = np.asarray(indices, dtype=np.int64).reshape(-1, 1)
    else:
        distances = np.asarray(distances, dtype=np.float64)
        indices = np.asarray(indices, dtype=np.int64)

    transferred = []
    confidence = []
    for row_dist, row_idx in zip(distances, indices, strict=False):
        row_labels = labels_real[row_idx]
        # Inverse-distance voting; exact matches dominate safely.
        votes = {}
        for dist, label in zip(row_dist, row_labels, strict=False):
            label = str(label)
            weight = 1.0 if float(dist) == 0.0 else 1.0 / float(dist)
            votes[label] = votes.get(label, 0.0) + weight
        best_label, best_vote = max(votes.items(), key=lambda item: item[1])
        total_vote = float(sum(votes.values()))
        transferred.append(best_label)
        confidence.append(best_vote / total_vote if total_vote > 0 else 0.0)

    pred_adata = pred_adata.copy()
    pred_adata.obs[output_key] = pd.Categorical(transferred)
    pred_adata.obs[f"{output_key}_confidence"] = np.asarray(confidence, dtype=np.float32)
    pred_adata.uns.setdefault("terminal_label_transfer", {})
    pred_adata.uns["terminal_label_transfer"][output_key] = {
        "latent_key": latent_key,
        "label_key": label_key,
        "k": int(query_k),
        "n_reference": int(real_adata.n_obs),
    }
    return pred_adata


def build_real_endpoint_adata(
    real_adata: ad.AnnData,
    condition: str,
    timepoint: float,
    condition_key: str,
    time_key: str,
    latent_key: str,
    label_key: str,
):
    subset = real_adata[
        (real_adata.obs[condition_key].astype(str) == str(condition))
        & (real_adata.obs[time_key].astype(float) == float(timepoint))
    ].copy()
    if subset.n_obs == 0:
        return subset

    endpoint = ad.AnnData(
        X=np.asarray(subset.obsm[latent_key], dtype=np.float32),
        obs=subset.obs.copy(),
    )
    endpoint.var_names = [f"latent_{idx}" for idx in range(endpoint.n_vars)]
    endpoint.obsm[latent_key] = np.asarray(endpoint.X, dtype=np.float32)
    endpoint.obs["terminal_label"] = pd.Categorical(
        normalize_label_series(endpoint.obs[label_key]).to_numpy()
    )
    endpoint.obs["rollout_mode"] = "real"
    endpoint.obs["datatype"] = "real"
    endpoint.obs["target_condition"] = str(condition)
    endpoint.obs["target_timepoint"] = float(timepoint)
    endpoint.obs["mass"] = np.ones(endpoint.n_obs, dtype=np.float32)
    return endpoint


def summarize_global_validation(
    real_endpoint: ad.AnnData,
    pred_by_mode: dict[str, ad.AnnData],
    condition: str,
    timepoint: float,
    n_source_total: int,
    label_order,
    terminal_label_key: str = "terminal_label",
):
    rows = []
    real_comp = compute_weighted_label_distribution(
        labels=real_endpoint.obs[terminal_label_key],
        weights=np.ones(real_endpoint.n_obs, dtype=np.float32),
        label_order=label_order,
    )
    real_ratio = float(real_endpoint.n_obs) / float(n_source_total)

    rows.append(
        {
            "target_condition": str(condition),
            "target_timepoint": float(timepoint),
            "rollout_mode": "real",
            "n_source_total": int(n_source_total),
            "n_real": int(real_endpoint.n_obs),
            "real_count_ratio": real_ratio,
            "pred_mass_sum": float(real_endpoint.n_obs),
            "pred_mass_ratio": real_ratio,
            "mass_abs_error": 0.0,
            "composition_tvd": 0.0,
            **{f"comp__{label}": float(real_comp[label]) for label in label_order},
        }
    )

    for rollout_mode in ROLLOUT_MODE_ORDER:
        pred_adata = pred_by_mode[rollout_mode]
        pred_mass_sum = float(np.asarray(pred_adata.obs["mass"], dtype=np.float64).sum())
        pred_ratio = pred_mass_sum / float(n_source_total)
        pred_comp = compute_weighted_label_distribution(
            labels=pred_adata.obs[terminal_label_key],
            weights=pred_adata.obs["mass"].astype(float).to_numpy(),
            label_order=label_order,
        )
        rows.append(
            {
                "target_condition": str(condition),
                "target_timepoint": float(timepoint),
                "rollout_mode": rollout_mode,
                "n_source_total": int(n_source_total),
                "n_real": int(real_endpoint.n_obs),
                "real_count_ratio": real_ratio,
                "pred_mass_sum": pred_mass_sum,
                "pred_mass_ratio": pred_ratio,
                "mass_abs_error": abs(pred_ratio - real_ratio),
                "composition_tvd": total_variation_distance_from_dict(
                    pred_comp,
                    real_comp,
                    label_order=label_order,
                ),
                **{f"comp__{label}": float(pred_comp[label]) for label in label_order},
            }
        )

    return pd.DataFrame(rows)


def _build_source_counts(source_obs, source_cohort_key):
    source_labels = normalize_label_series(source_obs[source_cohort_key])
    counts = source_labels.value_counts().sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def summarize_cohort_scores(
    pred_by_mode: dict[str, ad.AnnData],
    source_obs: pd.DataFrame,
    condition: str,
    timepoint: float,
    source_cohort_key: str,
    terminal_label_key: str,
    label_order,
    dominance_threshold: float = 0.6,
):
    source_counts = _build_source_counts(source_obs, source_cohort_key)
    rows = []

    for source_cohort, n_source in source_counts.items():
        mode_metrics = {}
        for rollout_mode, pred_adata in pred_by_mode.items():
            subset = pred_adata[
                normalize_label_series(pred_adata.obs[source_cohort_key]).to_numpy() == str(source_cohort)
            ].copy()
            weights = subset.obs["mass"].astype(float).to_numpy()
            mass_sum = float(weights.sum())
            retention = mass_sum / float(n_source) if n_source > 0 else np.nan
            composition = compute_weighted_label_distribution(
                labels=subset.obs[terminal_label_key],
                weights=weights,
                label_order=label_order,
            )
            mode_metrics[rollout_mode] = {
                "n_particles": int(subset.n_obs),
                "mass_sum": mass_sum,
                "retention": float(retention),
                "composition": composition,
            }

        d_trans_comp = total_variation_distance_from_dict(
            mode_metrics["transport-only"]["composition"],
            mode_metrics["full"]["composition"],
            label_order=label_order,
        )
        d_growth_comp = total_variation_distance_from_dict(
            mode_metrics["growth-only"]["composition"],
            mode_metrics["full"]["composition"],
            label_order=label_order,
        )
        drift_support = (
            d_growth_comp / (d_trans_comp + d_growth_comp + 1e-8)
            if np.isfinite(d_trans_comp + d_growth_comp)
            else np.nan
        )

        d_trans_mass = abs(
            mode_metrics["transport-only"]["retention"] - mode_metrics["full"]["retention"]
        )
        d_growth_mass = abs(
            mode_metrics["growth-only"]["retention"] - mode_metrics["full"]["retention"]
        )
        growth_support = (
            d_trans_mass / (d_trans_mass + d_growth_mass + 1e-8)
            if np.isfinite(d_trans_mass + d_growth_mass)
            else np.nan
        )

        if drift_support >= dominance_threshold and growth_support < dominance_threshold:
            mechanism_label = "drift-dominant"
        elif growth_support >= dominance_threshold and drift_support < dominance_threshold:
            mechanism_label = "growth-dominant"
        else:
            mechanism_label = "mixed"

        row = {
            "target_condition": str(condition),
            "target_timepoint": float(timepoint),
            "source_cohort": str(source_cohort),
            "n_source": int(n_source),
            "drift_support": float(drift_support),
            "growth_support": float(growth_support),
            "drift_comp_distance_transport": float(d_trans_comp),
            "drift_comp_distance_growth": float(d_growth_comp),
            "growth_mass_distance_transport": float(d_trans_mass),
            "growth_mass_distance_growth": float(d_growth_mass),
            "mechanism_label": mechanism_label,
        }
        for rollout_mode in ROLLOUT_MODE_ORDER:
            row[f"retention__{rollout_mode}"] = float(mode_metrics[rollout_mode]["retention"])
            row[f"mass_sum__{rollout_mode}"] = float(mode_metrics[rollout_mode]["mass_sum"])
            row[f"n_particles__{rollout_mode}"] = int(mode_metrics[rollout_mode]["n_particles"])
            for label in label_order:
                row[f"comp__{rollout_mode}__{label}"] = float(
                    mode_metrics[rollout_mode]["composition"].get(label, 0.0)
                )
        rows.append(row)

    return pd.DataFrame(rows)


def combine_endpoint_analysis_adata(
    real_endpoint: ad.AnnData,
    pred_by_mode: dict[str, ad.AnnData],
    source_cohort_key: str,
    terminal_label_key: str,
):
    adatas = []

    real_copy = real_endpoint.copy()
    real_copy.obs["source_cohort"] = "real_unknown"
    real_copy.obs["terminal_label"] = pd.Categorical(
        normalize_label_series(real_copy.obs[terminal_label_key]).to_numpy()
    )
    real_copy.obs["source_cohort_key"] = source_cohort_key
    real_copy.obs["terminal_label_key"] = terminal_label_key
    adatas.append(real_copy)

    for rollout_mode in ROLLOUT_MODE_ORDER:
        pred_adata = pred_by_mode[rollout_mode].copy()
        pred_adata.obs["rollout_mode"] = rollout_mode
        pred_adata.obs["datatype"] = "predicted"
        pred_adata.obs["source_cohort"] = pd.Categorical(
            normalize_label_series(pred_adata.obs[source_cohort_key]).to_numpy()
        )
        pred_adata.obs["terminal_label"] = pd.Categorical(
            normalize_label_series(pred_adata.obs[terminal_label_key]).to_numpy()
        )
        pred_adata.obs["source_cohort_key"] = source_cohort_key
        pred_adata.obs["terminal_label_key"] = terminal_label_key
        adatas.append(pred_adata)

    combined = ad.concat(adatas, axis=0, join="outer", merge="same", index_unique=None)
    combined.uns.setdefault("mechanism_analysis", {})
    combined.uns["mechanism_analysis"]["source_cohort_key"] = source_cohort_key
    combined.uns["mechanism_analysis"]["terminal_label_key"] = terminal_label_key
    combined.uns["mechanism_analysis"]["rollout_modes"] = list(ROLLOUT_MODE_ORDER)
    return combined


def build_mechanism_summary(global_validation_df, cohort_scores_df):
    summary = {
        "n_pairs": int(
            global_validation_df[
                global_validation_df["rollout_mode"].astype(str) == "real"
            ].shape[0]
        ),
        "n_cohort_rows": int(cohort_scores_df.shape[0]),
        "rollout_modes": list(ROLLOUT_MODE_ORDER),
        "global_validation": {},
    }

    for rollout_mode in ROLLOUT_MODE_ORDER:
        subset = global_validation_df[
            global_validation_df["rollout_mode"].astype(str) == rollout_mode
        ]
        summary["global_validation"][rollout_mode] = {
            "mean_mass_abs_error": float(subset["mass_abs_error"].mean()),
            "median_mass_abs_error": float(subset["mass_abs_error"].median()),
            "mean_composition_tvd": float(subset["composition_tvd"].mean()),
            "median_composition_tvd": float(subset["composition_tvd"].median()),
        }

    mechanism_counts = (
        cohort_scores_df["mechanism_label"].astype(str).value_counts().sort_index().to_dict()
    )
    summary["mechanism_label_counts"] = {
        str(key): int(value) for key, value in mechanism_counts.items()
    }
    return summary


def save_summary_json(path, payload):
    path = Path(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
