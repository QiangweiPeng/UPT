import argparse
import json
import re
import sys
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate.evals_plot import resample_prediction_adata_by_mass  # noqa: E402
from src.evaluate.mechanism import attach_terminal_labels_knn  # noqa: E402


MODE_ORDER = ["real", "full", "transport-only", "growth-only"]
MODE_COLORS = {
    "real": "#d62728",
    "full": "#1f77b4",
    "transport-only": "#ff7f0e",
    "growth-only": "#2ca02c",
}
MECHANISM_COLORS = {
    "transport-dominant": "#1f77b4",
    "growth-dominant": "#ff7f0e",
    "mixed": "#7f7f7f",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a shared UMAP for zebrafish endpoint mode comparison and save "
            "datatype / celltype views."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--condition", type=str, required=True)
    parser.add_argument("--timepoint", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--label-key", type=str, default="cell_type_broad")
    parser.add_argument("--source-filter-key", type=str, default=None)
    parser.add_argument(
        "--source-filter-values",
        nargs="*",
        default=None,
        help="Optional allowed values for source filtering. Supports comma-separated values.",
    )
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--control-max-cells",
        type=int,
        default=4000,
        help="Maximum time-matched control cells to retain for the shared UMAP.",
    )
    parser.add_argument(
        "--pair-max-cells",
        type=int,
        default=None,
        help="Optional cap on real/predicted cells per mode after resampling.",
    )
    parser.add_argument(
        "--mass-alpha-calibration",
        choices=["global", "per-panel"],
        default="global",
        help="How to calibrate prediction mass before mapping it to alpha in mode_compare_massalpha.png.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_list_args(raw_items):
    if not raw_items:
        return None
    items = []
    for item in raw_items:
        parts = [part.strip() for part in item.split(",") if part.strip()]
        items.extend(parts)
    return items or None


def sanitize_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def resolve_output_dir(run_dir, explicit_output_dir, condition, timepoint):
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return run_dir / "mode_umap" / f"{sanitize_name(condition)}_t{float(timepoint):g}"


def resolve_prediction_paths(run_dir):
    paths = {
        "full": run_dir / "real_source_eval" / "predictions_real_source_latent.h5ad",
        "transport-only": run_dir / "real_source_eval_transport_only" / "predictions_real_source_latent.h5ad",
        "growth-only": run_dir / "real_source_eval_growth_only" / "predictions_real_source_latent.h5ad",
    }
    missing = [mode for mode, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing cached prediction files for modes: " + ", ".join(missing)
        )
    return paths


def _sample_positions(positions, max_cells, rng):
    positions = np.asarray(positions, dtype=np.int64)
    if max_cells is None or positions.shape[0] <= max_cells:
        return positions
    return np.sort(rng.choice(positions, size=max_cells, replace=False))


def load_scaled_real_subsets(config, condition, timepoint, label_key, control_max_cells=4000, seed=42):
    condition_key = config["condition_key"]
    marginal_key = config["marginal_key"]
    control_key = config.get("control_key", "is_control")
    sample_rep = config["sample_rep"]
    scaled_rep = config["scaled_rep"]

    backed = ad.read_h5ad(config["data_path"], backed="r")
    try:
        requested_obs_columns = []
        for column in [
            condition_key,
            marginal_key,
            control_key,
            label_key,
            "cell_type_sub",
            "cell_type_broad",
            "tissue",
            "major_group",
        ]:
            if column in backed.obs.columns and column not in requested_obs_columns:
                requested_obs_columns.append(column)

        obs_df = backed.obs[requested_obs_columns].copy()
        obs_df[condition_key] = obs_df[condition_key].astype(str)
        obs_df[marginal_key] = obs_df[marginal_key].astype(float)
        obs_df["_obs_position"] = np.arange(obs_df.shape[0], dtype=np.int64)

        control_positions_all = np.flatnonzero(obs_df[control_key].astype(bool).to_numpy())
        control_rep_all = np.asarray(backed.obsm[sample_rep][control_positions_all], dtype=np.float32)
        scale = control_rep_all.std(axis=0, dtype=np.float64)
        scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)

        rng = np.random.default_rng(seed)
        control_mask = (
            obs_df[control_key].astype(bool)
            & (obs_df[marginal_key].astype(float) == float(timepoint))
        )
        control_positions = _sample_positions(
            np.flatnonzero(control_mask.to_numpy()),
            max_cells=control_max_cells,
            rng=rng,
        )
        real_mask = (
            (~obs_df[control_key].astype(bool))
            & (obs_df[condition_key].astype(str) == str(condition))
            & (obs_df[marginal_key].astype(float) == float(timepoint))
        )
        real_positions = np.flatnonzero(real_mask.to_numpy())

        if control_positions.shape[0] == 0:
            raise RuntimeError(f"No control cells found at timepoint {float(timepoint):g}")
        if real_positions.shape[0] == 0:
            raise RuntimeError(
                f"No real treated cells found for {condition} at timepoint {float(timepoint):g}"
            )

        control_obs = obs_df.iloc[control_positions][requested_obs_columns].copy()
        real_obs = obs_df.iloc[real_positions][requested_obs_columns].copy()
        control_latent = np.asarray(backed.obsm[sample_rep][control_positions], dtype=np.float32) / scale
        real_latent = np.asarray(backed.obsm[sample_rep][real_positions], dtype=np.float32) / scale
    finally:
        backed.file.close()

    control_adata = ad.AnnData(X=control_latent, obs=control_obs)
    control_adata.var_names = [f"latent_{idx}" for idx in range(control_adata.n_vars)]
    control_adata.obsm[scaled_rep] = control_latent.copy()
    real_adata = ad.AnnData(X=real_latent, obs=real_obs)
    real_adata.var_names = [f"latent_{idx}" for idx in range(real_adata.n_vars)]
    real_adata.obsm[scaled_rep] = real_latent.copy()
    real_adata.obs["terminal_label"] = pd.Categorical(real_adata.obs[label_key].astype(str))
    return control_adata, real_adata


def load_prediction_subset(path, condition, timepoint):
    backed = ad.read_h5ad(path, backed="r")
    try:
        obs_df = backed.obs.copy()
        mask = (
            (obs_df["target_condition"].astype(str) == str(condition))
            & (obs_df["record_kind"].astype(str) == "predicted_from_real_source")
            & (obs_df["target_timepoint"].astype(float) == float(timepoint))
        )
        positions = np.flatnonzero(mask.to_numpy())
        if positions.shape[0] == 0:
            raise RuntimeError(
                f"No predicted rows found in {path} for {condition} @ {float(timepoint):g}"
            )

        subset = ad.AnnData(
            X=np.asarray(backed.X[positions], dtype=np.float32),
            obs=obs_df.iloc[positions].copy(),
        )
        subset.var_names = backed.var_names.copy()
        if "prediction_collection" in backed.uns:
            subset.uns["prediction_collection"] = dict(backed.uns["prediction_collection"])
        return subset
    finally:
        backed.file.close()


def filter_prediction_subset(pred_adata, filter_key=None, filter_values=None):
    if filter_key is None or filter_values is None:
        return pred_adata.copy()
    if filter_key not in pred_adata.obs.columns:
        raise KeyError(f"{filter_key} is not found in prediction obs")
    mask = pred_adata.obs[filter_key].astype(str).isin([str(value) for value in filter_values])
    filtered = pred_adata[mask].copy()
    if filtered.n_obs == 0:
        raise RuntimeError("Prediction subset is empty after source filtering")
    return filtered


def compute_mode_global_calibration_factors(config, prediction_paths, filter_key=None, filter_values=None):
    condition_key = config["condition_key"]
    marginal_key = config["marginal_key"]
    control_key = config.get("control_key", "is_control")

    backed_real = ad.read_h5ad(config["data_path"], backed="r")
    try:
        real_obs = backed_real.obs[[condition_key, marginal_key, control_key]].copy()
        real_obs[condition_key] = real_obs[condition_key].astype(str)
        real_obs[marginal_key] = real_obs[marginal_key].astype(float)
        real_obs = real_obs[~real_obs[control_key].astype(bool)].copy()
        real_counts = (
            real_obs.groupby([condition_key, marginal_key], observed=True)
            .size()
            .rename("real_count")
            .reset_index()
            .rename(columns={condition_key: "target_condition", marginal_key: "target_timepoint"})
        )
    finally:
        backed_real.file.close()

    factors = {}
    for mode_name, path in prediction_paths.items():
        backed_pred = ad.read_h5ad(path, backed="r")
        try:
            pred_obs = backed_pred.obs.copy()
            pred_obs["target_condition"] = pred_obs["target_condition"].astype(str)
            pred_obs["target_timepoint"] = pred_obs["target_timepoint"].astype(float)
            pred_obs["record_kind"] = pred_obs["record_kind"].astype(str)
            pred_obs = pred_obs[pred_obs["record_kind"] == "predicted_from_real_source"].copy()
            if filter_key is not None and filter_values is not None:
                if filter_key not in pred_obs.columns:
                    raise KeyError(f"{filter_key} is not found in prediction obs")
                pred_obs = pred_obs[
                    pred_obs[filter_key].astype(str).isin([str(value) for value in filter_values])
                ].copy()

            pred_summary = (
                pred_obs.groupby(["target_condition", "target_timepoint"], observed=True)["mass"]
                .sum()
                .reset_index(name="pred_mass_sum")
            )
        finally:
            backed_pred.file.close()

        merged = real_counts.merge(
            pred_summary,
            on=["target_condition", "target_timepoint"],
            how="inner",
            validate="one_to_one",
        )
        denom = float(np.square(merged["pred_mass_sum"].astype(float)).sum())
        factor = (
            float(
                np.dot(
                    merged["real_count"].astype(float).to_numpy(),
                    merged["pred_mass_sum"].astype(float).to_numpy(),
                )
                / denom
            )
            if denom > 0
            else 1.0
        )
        factors[str(mode_name)] = factor
    return factors


def build_shared_umap_adata(
    control_adata,
    real_adata,
    pred_by_mode,
    label_key,
    seed=42,
    pair_max_cells=None,
    mass_alpha_calibration="global",
    mode_global_display_scales=None,
):
    if pair_max_cells is None:
        target_n = int(real_adata.n_obs)
    else:
        target_n = int(min(real_adata.n_obs, pair_max_cells))
    if target_n <= 0:
        raise ValueError("target_n must be positive")

    rng = np.random.default_rng(seed)
    if control_adata.n_obs > target_n:
        control_idx = rng.choice(control_adata.n_obs, size=target_n, replace=False)
        control_vis = control_adata[control_idx].copy()
    else:
        control_vis = control_adata.copy()

    if real_adata.n_obs > target_n:
        real_idx = rng.choice(real_adata.n_obs, size=target_n, replace=False)
        real_vis = real_adata[real_idx].copy()
    else:
        real_vis = real_adata.copy()

    real_vis.obs["datatype"] = "real"
    real_vis.obs["celltype_label"] = pd.Categorical(real_vis.obs[label_key].astype(str))
    real_vis.obs["display_mass"] = np.full(real_vis.n_obs, 1.0, dtype=np.float32)
    control_vis.obs["datatype"] = "control"
    control_vis.obs["celltype_label"] = pd.Categorical(control_vis.obs[label_key].astype(str))
    control_vis.obs["display_mass"] = np.full(control_vis.n_obs, 1.0, dtype=np.float32)

    vis_adatas = [control_vis, real_vis]
    summary_rows = [
        {"datatype": "control", "n_obs": int(control_vis.n_obs)},
        {"datatype": "real", "n_obs": int(real_vis.n_obs)},
    ]

    for offset, (mode_name, pred_adata) in enumerate(pred_by_mode.items(), start=1):
        sampled = resample_prediction_adata_by_mass(
            pred_adata=pred_adata,
            n_samples=target_n,
            weight_key="mass",
            seed=seed + offset,
        )
        sampled.obs["datatype"] = str(mode_name)
        sampled.obs["celltype_label"] = pd.Categorical(
            sampled.obs["terminal_label"].astype(str)
        )
        raw_mass = sampled.obs["mass"].astype(float).to_numpy()
        raw_mass_sum = float(raw_mass.sum())
        if mass_alpha_calibration == "global":
            display_scale = float(mode_global_display_scales.get(str(mode_name), 1.0))
        else:
            display_scale = float(target_n / raw_mass_sum) if raw_mass_sum > 0 else 1.0
        sampled.obs["display_mass"] = (raw_mass * display_scale).astype(np.float32)
        vis_adatas.append(sampled)
        summary_rows.append(
            {
                "datatype": str(mode_name),
                "n_obs": int(sampled.n_obs),
                "n_unique_pred_sampled": int(sampled.obs["resampled_source_obs_name"].nunique()),
                "mass_mean": float(sampled.obs["mass"].astype(float).mean()),
                "mass_sum": raw_mass_sum,
                "display_mass_scale": float(display_scale),
                "display_mass_mean": float(sampled.obs["display_mass"].astype(float).mean()),
                "display_mass_sum": float(sampled.obs["display_mass"].astype(float).sum()),
            }
        )

    combined = ad.concat(vis_adatas, join="outer", merge="same", index_unique=None)
    combined.uns["mode_umap_metadata"] = {
        "label_key": label_key,
        "sampling_target_n": int(target_n),
        "mode_order": MODE_ORDER,
        "mass_display_mode": str(mass_alpha_calibration),
    }
    return combined, pd.DataFrame(summary_rows)


def compute_shared_umap(adata_vis, n_neighbors=30, min_dist=0.3, random_state=42):
    sc.pp.neighbors(adata_vis, n_neighbors=n_neighbors, use_rep="X")
    sc.tl.umap(adata_vis, min_dist=min_dist, random_state=random_state)
    return adata_vis


def plot_mode_datatype_panels(adata_vis, output_path, condition, timepoint):
    coords = np.asarray(adata_vis.obsm["X_umap"], dtype=np.float32)
    plot_df = adata_vis.obs.copy()
    plot_df["UMAP1"] = coords[:, 0]
    plot_df["UMAP2"] = coords[:, 1]

    control_df = plot_df[plot_df["datatype"].astype(str) == "control"].copy()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, mode_name in zip(axes, MODE_ORDER, strict=False):
        if not control_df.empty:
            ax.scatter(
                control_df["UMAP1"],
                control_df["UMAP2"],
                s=3.0,
                c="#c7c7c7",
                alpha=0.20,
                linewidths=0.0,
                rasterized=True,
            )
        subset = plot_df[plot_df["datatype"].astype(str) == mode_name].copy()
        ax.scatter(
            subset["UMAP1"],
            subset["UMAP2"],
            s=5.0,
            c=[MODE_COLORS[mode_name]],
            alpha=0.70 if mode_name == "real" else 0.55,
            linewidths=0.0,
            rasterized=True,
        )
        ax.set_title(mode_name)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{condition} @ {float(timepoint):g} hpf: control background vs mode endpoint")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mode_celltype_panels(adata_vis, output_path, condition, timepoint, label_key):
    coords = np.asarray(adata_vis.obsm["X_umap"], dtype=np.float32)
    plot_df = adata_vis.obs.copy()
    plot_df["UMAP1"] = coords[:, 0]
    plot_df["UMAP2"] = coords[:, 1]
    control_df = plot_df[plot_df["datatype"].astype(str) == "control"].copy()

    label_series = plot_df.loc[
        plot_df["datatype"].astype(str) != "control",
        "celltype_label",
    ].astype(str)
    label_counts = label_series.value_counts()
    top_labels = label_counts.index[:12].tolist()
    plot_df["celltype_plot"] = plot_df["celltype_label"].astype(str).where(
        plot_df["celltype_label"].astype(str).isin(top_labels),
        other="Other",
    )
    label_order = sorted(plot_df["celltype_plot"].astype(str).unique().tolist())
    palette = dict(zip(label_order, sns.color_palette("tab20", n_colors=len(label_order))))

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    for ax, mode_name in zip(axes, MODE_ORDER, strict=False):
        if not control_df.empty:
            ax.scatter(
                control_df["UMAP1"],
                control_df["UMAP2"],
                s=2.5,
                c="#d0d0d0",
                alpha=0.10,
                linewidths=0.0,
                rasterized=True,
            )
        subset = plot_df[plot_df["datatype"].astype(str) == mode_name].copy()
        for label in label_order:
            label_subset = subset[subset["celltype_plot"].astype(str) == label]
            if label_subset.empty:
                continue
            ax.scatter(
                label_subset["UMAP1"],
                label_subset["UMAP2"],
                s=5.0,
                c=[palette[label]],
                alpha=0.80 if mode_name == "real" else 0.55,
                linewidths=0.0,
                rasterized=True,
                label=label,
            )
        ax.set_title(mode_name)
        ax.set_xticks([])
        ax.set_yticks([])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title=label_key,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=8,
            title_fontsize=9,
        )
    fig.suptitle(f"{condition} @ {float(timepoint):g} hpf: celltype-colored endpoints")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _build_celltype_palette(plot_df):
    label_series = plot_df.loc[
        plot_df["datatype"].astype(str) != "control",
        "celltype_label",
    ].astype(str)
    label_counts = label_series.value_counts()
    top_labels = label_counts.index[:12].tolist()
    plot_df = plot_df.copy()
    plot_df["celltype_plot"] = plot_df["celltype_label"].astype(str).where(
        plot_df["celltype_label"].astype(str).isin(top_labels),
        other="Other",
    )
    label_order = sorted(plot_df["celltype_plot"].astype(str).unique().tolist())
    palette = dict(zip(label_order, sns.color_palette("tab20", n_colors=len(label_order))))
    return plot_df, label_order, palette


def _build_source_palette(plot_df):
    source_key = None
    for candidate in ["source_cell_type_broad", "source_cell_type_sub", "source_major_group"]:
        if candidate in plot_df.columns:
            source_key = candidate
            break
    plot_df = plot_df.copy()
    if source_key is None:
        plot_df["source_plot"] = "Unknown"
        return plot_df, ["Unknown"], "unknown"

    source_series = plot_df.loc[
        plot_df["datatype"].astype(str).isin(["full", "transport-only", "growth-only"]),
        source_key,
    ].astype(str)
    top_labels = source_series.value_counts().index[:10].tolist()
    plot_df["source_plot"] = plot_df[source_key].astype(str).where(
        plot_df[source_key].astype(str).isin(top_labels),
        other="Other",
    )
    label_order = sorted(plot_df["source_plot"].astype(str).unique().tolist())
    return plot_df, label_order, source_key


def _build_joint_palette(source_label_order, terminal_label_order):
    all_labels = []
    for label in list(source_label_order) + list(terminal_label_order):
        if label not in all_labels:
            all_labels.append(label)
    colors = sns.color_palette("tab20", n_colors=max(len(all_labels), 3))
    return dict(zip(all_labels, colors[: len(all_labels)]))


def plot_mode_massalpha_panels(adata_vis, output_path, condition, timepoint):
    coords = np.asarray(adata_vis.obsm["X_umap"], dtype=np.float32)
    plot_df = adata_vis.obs.copy()
    plot_df["UMAP1"] = coords[:, 0]
    plot_df["UMAP2"] = coords[:, 1]
    plot_df, label_order, _ = _build_celltype_palette(plot_df)
    plot_df, source_label_order, source_label_key = _build_source_palette(plot_df)
    joint_palette = _build_joint_palette(source_label_order, label_order)
    pred_df = plot_df[
        plot_df["datatype"].astype(str).isin(["full", "transport-only", "growth-only"])
    ].copy()

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    panel_order = ["full", "transport-only", "growth-only"]
    axes_top = axes[0]
    axes_mid = axes[1]
    axes_bottom = axes[2]
    norm = plt.Normalize(vmin=-2.5, vmax=2.5)
    cmap = plt.cm.coolwarm

    for ax, mode_name in zip(axes_top, panel_order, strict=False):
        subset = plot_df[plot_df["datatype"].astype(str) == mode_name].copy()
        for label in source_label_order:
            label_subset = subset[subset["source_plot"].astype(str) == label].copy()
            if label_subset.empty:
                continue
            ax.scatter(
                label_subset["UMAP1"],
                label_subset["UMAP2"],
                s=5.0,
                c=[joint_palette[label]],
                alpha=0.75,
                linewidths=0.0,
                rasterized=True,
            )
        ax.set_title(f"{mode_name}: source")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax, mode_name in zip(axes_mid, panel_order, strict=False):
        subset = plot_df[plot_df["datatype"].astype(str) == mode_name].copy()
        for label in label_order:
            label_subset = subset[subset["celltype_plot"].astype(str) == label].copy()
            if label_subset.empty:
                continue
            ax.scatter(
                label_subset["UMAP1"],
                label_subset["UMAP2"],
                s=5.0,
                c=[joint_palette[label]],
                alpha=0.70,
                linewidths=0.0,
                rasterized=True,
            )
        ax.set_title(f"{mode_name}: terminal")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax, mode_name in zip(axes_bottom, panel_order, strict=False):
        subset = plot_df[plot_df["datatype"].astype(str) == mode_name].copy()
        masses = (
            subset["display_mass"].astype(float).to_numpy()
            if "display_mass" in subset.columns
            else np.array([], dtype=np.float32)
        )
        if masses.size == 0:
            zscores = np.array([], dtype=np.float32)
        else:
            log_mass = np.log1p(np.clip(masses, a_min=0.0, a_max=None))
            lower = float(np.quantile(log_mass, 0.01))
            upper = float(np.quantile(log_mass, 0.99))
            clipped = np.clip(log_mass, lower, upper)
            mean = float(clipped.mean())
            std = float(clipped.std())
            if std < 1e-8:
                zscores = np.zeros_like(clipped, dtype=np.float32)
            else:
                zscores = ((clipped - mean) / std).astype(np.float32)
            zscores = np.clip(zscores, -2.5, 2.5)
        ax.scatter(
            subset["UMAP1"],
            subset["UMAP2"],
            s=5.0,
            c=zscores,
            cmap=cmap,
            norm=norm,
            alpha=0.85,
            linewidths=0.0,
            rasterized=True,
        )
        ax.set_title(f"{mode_name}: mass z-score")
        ax.set_xticks([])
        ax.set_yticks([])

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=joint_palette[label],
            markeredgecolor="none",
            markersize=6,
            label=str(label),
        )
        for label in label_order
    ]
    fig.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=joint_palette[label],
                markeredgecolor="none",
                markersize=6,
                label=str(label),
            )
            for label in source_label_order
        ],
        title=f"source ({source_label_key})",
        loc="center left",
        bbox_to_anchor=(1.01, 0.84),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    fig.legend(
        handles=legend_handles,
        title="terminal celltype",
        loc="center left",
        bbox_to_anchor=(1.01, 0.53),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes_bottom.tolist(),
        fraction=0.03,
        pad=0.04,
    )
    cbar.set_label("mass z-score per mode (log1p, 1%-99% clipped)")
    fig.suptitle(
        f"{condition} @ {float(timepoint):g} hpf: source / terminal / mass z-score"
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mode_mass_zscore_panels(adata_vis, output_path, condition, timepoint):
    coords = np.asarray(adata_vis.obsm["X_umap"], dtype=np.float32)
    plot_df = adata_vis.obs.copy()
    plot_df["UMAP1"] = coords[:, 0]
    plot_df["UMAP2"] = coords[:, 1]
    panel_order = ["full", "transport-only", "growth-only"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=-2.5, vmax=2.5)

    for ax, mode_name in zip(axes, panel_order, strict=False):
        subset = plot_df[plot_df["datatype"].astype(str) == mode_name].copy()
        masses = (
            subset["display_mass"].astype(float).to_numpy()
            if "display_mass" in subset.columns
            else np.array([], dtype=np.float32)
        )
        if masses.size == 0:
            zscores = np.array([], dtype=np.float32)
        else:
            log_mass = np.log1p(np.clip(masses, a_min=0.0, a_max=None))
            lower = float(np.quantile(log_mass, 0.01))
            upper = float(np.quantile(log_mass, 0.99))
            clipped = np.clip(log_mass, lower, upper)
            mean = float(clipped.mean())
            std = float(clipped.std())
            if std < 1e-8:
                zscores = np.zeros_like(clipped, dtype=np.float32)
            else:
                zscores = ((clipped - mean) / std).astype(np.float32)
            zscores = np.clip(zscores, -2.5, 2.5)

        ax.scatter(
            subset["UMAP1"],
            subset["UMAP2"],
            s=5.0,
            c=zscores,
            cmap=cmap,
            norm=norm,
            alpha=0.85,
            linewidths=0.0,
            rasterized=True,
        )
        ax.set_title(f"{mode_name}")
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes.tolist(),
        fraction=0.03,
        pad=0.04,
    )
    cbar.set_label("mass z-score per mode (log1p, 1%-99% clipped)")
    fig.suptitle(f"{condition} @ {float(timepoint):g} hpf: mass z-score by mode")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize_celltype_mechanism(adata_vis, dominance_threshold=0.6):
    coords = np.asarray(adata_vis.obsm["X_umap"], dtype=np.float32)
    plot_df = adata_vis.obs.copy()
    plot_df["UMAP1"] = coords[:, 0]
    plot_df["UMAP2"] = coords[:, 1]
    plot_df, _, _ = _build_celltype_palette(plot_df)

    real_df = plot_df[plot_df["datatype"].astype(str) == "real"].copy()
    pred_df = plot_df[plot_df["datatype"].astype(str).isin(["full", "transport-only", "growth-only"])].copy()
    rows = []
    for celltype in sorted(real_df["celltype_plot"].astype(str).unique().tolist()):
        centroid_subset = real_df[real_df["celltype_plot"].astype(str) == celltype]
        full_mass = float(
            pred_df[
                (pred_df["datatype"].astype(str) == "full")
                & (pred_df["celltype_plot"].astype(str) == celltype)
            ]["mass"].astype(float).sum()
        )
        trans_mass = float(
            pred_df[
                (pred_df["datatype"].astype(str) == "transport-only")
                & (pred_df["celltype_plot"].astype(str) == celltype)
            ]["mass"].astype(float).sum()
        )
        growth_mass = float(
            pred_df[
                (pred_df["datatype"].astype(str) == "growth-only")
                & (pred_df["celltype_plot"].astype(str) == celltype)
            ]["mass"].astype(float).sum()
        )
        d_trans = abs(full_mass - trans_mass)
        d_growth = abs(full_mass - growth_mass)
        transport_support = d_growth / (d_trans + d_growth + 1e-8)
        growth_support = d_trans / (d_trans + d_growth + 1e-8)

        if transport_support >= dominance_threshold and growth_support < dominance_threshold:
            mechanism_label = "transport-dominant"
        elif growth_support >= dominance_threshold and transport_support < dominance_threshold:
            mechanism_label = "growth-dominant"
        else:
            mechanism_label = "mixed"

        rows.append(
            {
                "celltype_label": str(celltype),
                "centroid_x": float(centroid_subset["UMAP1"].mean()),
                "centroid_y": float(centroid_subset["UMAP2"].mean()),
                "n_real": int(centroid_subset.shape[0]),
                "mass_full": full_mass,
                "mass_transport": trans_mass,
                "mass_growth": growth_mass,
                "transport_support": float(transport_support),
                "growth_support": float(growth_support),
                "mechanism_label": mechanism_label,
            }
        )
    return pd.DataFrame(rows)


def plot_celltype_mechanism_overlay(adata_vis, mechanism_df, output_path, condition, timepoint):
    coords = np.asarray(adata_vis.obsm["X_umap"], dtype=np.float32)
    plot_df = adata_vis.obs.copy()
    plot_df["UMAP1"] = coords[:, 0]
    plot_df["UMAP2"] = coords[:, 1]
    plot_df, label_order, palette = _build_celltype_palette(plot_df)
    real_df = plot_df[plot_df["datatype"].astype(str) == "real"].copy()

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for label in label_order:
        subset = real_df[real_df["celltype_plot"].astype(str) == label]
        if subset.empty:
            continue
        ax.scatter(
            subset["UMAP1"],
            subset["UMAP2"],
            s=5.0,
            c=[palette[label]],
            alpha=0.18,
            linewidths=0.0,
            rasterized=True,
        )

    max_mass = float(mechanism_df["mass_full"].max()) if not mechanism_df.empty else 1.0
    for row in mechanism_df.itertuples(index=False):
        size = 80.0 + 220.0 * (float(row.mass_full) / (max_mass + 1e-8))
        ax.scatter(
            row.centroid_x,
            row.centroid_y,
            s=size,
            c=[MECHANISM_COLORS[str(row.mechanism_label)]],
            alpha=0.95,
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )
        ax.text(
            row.centroid_x,
            row.centroid_y,
            str(row.celltype_label),
            fontsize=7,
            ha="center",
            va="center",
            zorder=4,
        )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=8,
            label=label,
        )
        for label, color in MECHANISM_COLORS.items()
    ]
    ax.legend(
        handles=legend_handles,
        title="mechanism",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
    )
    ax.set_title(f"{condition} @ {float(timepoint):g} hpf: celltype-level mechanism overlay")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize_terminal_celltype_mass(real_adata, pred_by_mode, label_key, mode_global_display_scales, top_k=12):
    real_labels = real_adata.obs[label_key].astype(str)
    real_counts = real_labels.value_counts()
    top_labels = real_counts.index[:top_k].tolist()

    def _collapse(series):
        return series.astype(str).where(series.astype(str).isin(top_labels), other="Other")

    rows = []
    real_collapsed = _collapse(real_labels)
    for celltype, value in real_collapsed.value_counts().items():
        rows.append(
            {
                "celltype_plot": str(celltype),
                "datatype": "real",
                "display_mass_sum": float(value),
            }
        )

    for mode_name, pred_adata in pred_by_mode.items():
        labels = _collapse(pred_adata.obs["terminal_label"].astype(str))
        masses = pred_adata.obs["mass"].astype(float).to_numpy()
        scale = float(mode_global_display_scales.get(str(mode_name), 1.0))
        calibrated = masses * scale
        mode_df = pd.DataFrame(
            {
                "celltype_plot": labels.to_numpy(dtype=object),
                "display_mass_sum": calibrated,
            }
        )
        grouped = (
            mode_df.groupby("celltype_plot", observed=True)["display_mass_sum"]
            .sum()
            .reset_index()
        )
        for row in grouped.itertuples(index=False):
            rows.append(
                {
                    "celltype_plot": str(row.celltype_plot),
                    "datatype": str(mode_name),
                    "display_mass_sum": float(row.display_mass_sum),
                }
            )

    label_order = real_collapsed.value_counts().index.tolist()
    if "Other" not in label_order:
        label_order.append("Other")
    full_index = pd.MultiIndex.from_product(
        [label_order, MODE_ORDER],
        names=["celltype_plot", "datatype"],
    )
    summary = (
        pd.DataFrame(rows)
        .set_index(["celltype_plot", "datatype"])
        .reindex(full_index, fill_value=0.0)
        .reset_index()
    )
    summary["celltype_plot"] = pd.Categorical(
        summary["celltype_plot"],
        categories=label_order,
        ordered=True,
    )
    summary = summary.sort_values(["celltype_plot", "datatype"]).reset_index(drop=True)
    return summary


def plot_terminal_celltype_mass_bars(summary_df, output_path, condition, timepoint):
    celltypes = summary_df["celltype_plot"].cat.categories.tolist()
    x = np.arange(len(celltypes), dtype=np.float32)
    width = 0.20

    fig, ax = plt.subplots(1, 1, figsize=(max(14, 1.1 * len(celltypes)), 6))
    for idx, mode_name in enumerate(MODE_ORDER):
        subset = (
            summary_df[summary_df["datatype"].astype(str) == mode_name]
            .set_index("celltype_plot")
            .reindex(celltypes, fill_value=0.0)
        )
        ax.bar(
            x + (idx - 1.5) * width,
            subset["display_mass_sum"].to_numpy(dtype=np.float32),
            width=width,
            color=MODE_COLORS[mode_name],
            alpha=0.88,
            label=mode_name,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(celltypes, rotation=40, ha="right")
    ax.set_ylabel("calibrated terminal mass")
    ax.set_title(
        f"{condition} @ {float(timepoint):g} hpf: terminal celltype mass by mode"
    )
    ax.legend(frameon=False, ncol=4, loc="upper right")
    sns.despine(ax=ax)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = resolve_output_dir(run_dir, args.output_dir, args.condition, args.timepoint)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(run_dir / "config.json")
    prediction_paths = resolve_prediction_paths(run_dir)
    source_filter_values = parse_list_args(args.source_filter_values)
    mode_global_display_scales = compute_mode_global_calibration_factors(
        config=config,
        prediction_paths=prediction_paths,
        filter_key=args.source_filter_key,
        filter_values=source_filter_values,
    )

    control_adata, real_adata = load_scaled_real_subsets(
        config=config,
        condition=args.condition,
        timepoint=args.timepoint,
        label_key=args.label_key,
        control_max_cells=args.control_max_cells,
        seed=args.seed,
    )

    pred_by_mode = {}
    for mode_name, path in prediction_paths.items():
        pred_subset = load_prediction_subset(path, args.condition, args.timepoint)
        pred_subset = filter_prediction_subset(
            pred_subset,
            filter_key=args.source_filter_key,
            filter_values=source_filter_values,
        )
        pred_subset = attach_terminal_labels_knn(
            pred_adata=pred_subset,
            real_adata=real_adata,
            latent_key=config["scaled_rep"],
            label_key="terminal_label",
            output_key="terminal_label",
            k=15,
        )
        pred_by_mode[mode_name] = pred_subset

    adata_vis, sampling_summary = build_shared_umap_adata(
        control_adata=control_adata,
        real_adata=real_adata,
        pred_by_mode=pred_by_mode,
        label_key=args.label_key,
        seed=args.seed,
        pair_max_cells=args.pair_max_cells,
        mass_alpha_calibration=args.mass_alpha_calibration,
        mode_global_display_scales=mode_global_display_scales,
    )
    compute_shared_umap(
        adata_vis,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.seed,
    )

    adata_vis.uns["mode_umap_metadata"] = {
        "condition": str(args.condition),
        "timepoint": float(args.timepoint),
        "label_key": str(args.label_key),
        "source_filter_key": args.source_filter_key,
        "source_filter_values": source_filter_values,
        "prediction_paths": {mode: str(path) for mode, path in prediction_paths.items()},
        "mass_alpha_calibration": str(args.mass_alpha_calibration),
        "mode_global_display_scales": {
            mode: float(scale) for mode, scale in mode_global_display_scales.items()
        },
    }
    mechanism_df = summarize_celltype_mechanism(adata_vis)
    terminal_mass_df = summarize_terminal_celltype_mass(
        real_adata=real_adata,
        pred_by_mode=pred_by_mode,
        label_key=args.label_key,
        mode_global_display_scales=mode_global_display_scales,
    )
    adata_vis.uns["mode_umap_metadata"]["mechanism_summary"] = {
        "dominance_threshold": 0.6,
        "n_celltypes": int(mechanism_df.shape[0]),
    }
    adata_vis.write_h5ad(output_dir / "mode_compare_umap.h5ad")
    sampling_summary.to_csv(output_dir / "mode_compare_sampling_summary.csv", index=False)
    mechanism_df.to_csv(output_dir / "mode_compare_celltype_mechanism.csv", index=False)
    terminal_mass_df.to_csv(output_dir / "mode_compare_terminal_celltype_mass.csv", index=False)
    plot_mode_datatype_panels(
        adata_vis=adata_vis,
        output_path=output_dir / "mode_compare_datatype.png",
        condition=args.condition,
        timepoint=args.timepoint,
    )
    plot_mode_celltype_panels(
        adata_vis=adata_vis,
        output_path=output_dir / "mode_compare_celltype.png",
        condition=args.condition,
        timepoint=args.timepoint,
        label_key=args.label_key,
    )
    plot_mode_massalpha_panels(
        adata_vis=adata_vis,
        output_path=output_dir / "mode_compare_massalpha.png",
        condition=args.condition,
        timepoint=args.timepoint,
    )
    plot_mode_mass_zscore_panels(
        adata_vis=adata_vis,
        output_path=output_dir / "mode_compare_mass_zscore.png",
        condition=args.condition,
        timepoint=args.timepoint,
    )
    plot_celltype_mechanism_overlay(
        adata_vis=adata_vis,
        mechanism_df=mechanism_df,
        output_path=output_dir / "mode_compare_celltype_mechanism.png",
        condition=args.condition,
        timepoint=args.timepoint,
    )
    plot_terminal_celltype_mass_bars(
        summary_df=terminal_mass_df,
        output_path=output_dir / "mode_compare_terminal_celltype_mass.png",
        condition=args.condition,
        timepoint=args.timepoint,
    )
    print(f"Saved mode-compare UMAP outputs to {output_dir}")


if __name__ == "__main__":
    main()
