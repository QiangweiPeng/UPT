import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.zebrafish import (  # noqa: E402
    build_zebrafish_condition_rep_dict,
    build_zebrafish_control_condition_embedding,
)
from src.training.model import FNet  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether zebrafish transport/growth heads are condition-sensitive "
            "on cached shared-UMAP prediction subsets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--mode-umap-paths",
        nargs="*",
        default=None,
        help="Optional explicit mode_compare_umap.h5ad paths. Defaults to all cached mode-UMAP files under <run-dir>/mode_umap/.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--top-celltypes", type=int, default=8)
    parser.add_argument(
        "--reference-datatype",
        type=str,
        default="full",
        choices=["full", "transport-only", "growth-only"],
        help="Which sampled prediction subset to use as the common evaluation state set.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def resolve_device(requested_device):
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable. Falling back to cpu for head diagnostics.")
        return "cpu"
    return requested_device


def resolve_output_dir(run_dir, explicit_output_dir):
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return run_dir / "head_diagnostics"


def resolve_mode_umap_paths(run_dir, explicit_paths):
    if explicit_paths:
        paths = [Path(path).resolve() for path in explicit_paths]
    else:
        paths = sorted((run_dir / "mode_umap").glob("*/mode_compare_umap.h5ad"))
    if not paths:
        raise FileNotFoundError("No mode_compare_umap.h5ad files were found.")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing mode UMAP files:\n" + "\n".join(missing))
    return paths


def load_run_artifacts(run_dir):
    config_path = run_dir / "config.json"
    final_path = run_dir / "final.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    if not final_path.exists():
        raise FileNotFoundError(f"Missing final model file: {final_path}")
    return load_json(config_path), final_path


def build_model(config, model_path, device):
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model = FNet(
        in_out_dim=config["in_out_dim"],
        hidden_dim_v=config["hidden_dim_v"],
        n_hiddens_v=config["n_hiddens_v"],
        hidden_dim_g=config["hidden_dim_g"],
        n_hiddens_g=config["n_hiddens_g"],
        condition_dim=config["condition_dim"],
        con_embedding_dim=config["con_embedding_dim"],
        hidden_dim_con=config["hidden_dim_con"],
        time_dim=config["time_dim"],
        time_embedding_dim=config["time_embedding_dim"],
        hidden_dim_time=config["hidden_dim_time"],
        bottle_dim=config["bottle_dim"],
        activation=config["activation"],
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def ensure_model_dims(config, example_path):
    if "in_out_dim" in config and "condition_dim" in config:
        return config

    adata = ad.read_h5ad(example_path)
    try:
        condition_dim = int(adata.uns["mode_umap_metadata"]["condition_dim"]) if "condition_dim" in adata.uns.get("mode_umap_metadata", {}) else None
    except Exception:
        condition_dim = None
    in_out_dim = int(adata.n_vars)
    patched = dict(config)
    patched["in_out_dim"] = int(in_out_dim)
    if condition_dim is not None:
        patched["condition_dim"] = int(condition_dim)
    else:
        patched["condition_dim"] = int(config.get("condition_dim", 2560))
    return patched


def load_condition_rep_dict(data_path, condition_names, condition_key):
    required_conditions = {str(name) for name in condition_names}
    backed = ad.read_h5ad(data_path, backed="r")
    try:
        obs_df = backed.obs[[condition_key, "gene_target_1", "gene_target_2"]].copy()
        obs_df[condition_key] = obs_df[condition_key].astype(str)
        obs_df = obs_df[obs_df[condition_key].isin(required_conditions)].copy()
        if obs_df.empty:
            raise RuntimeError("Failed to load any zebrafish rows for the requested conditions.")
        condition_rep_dict = build_zebrafish_condition_rep_dict(
            obs_df=obs_df,
            gene_embeddings=backed.uns["gene_embeddings"],
            condition_key=condition_key,
            gene1_key="gene_target_1",
            gene2_key="gene_target_2",
        )
    finally:
        backed.file.close()

    missing = sorted(required_conditions.difference(condition_rep_dict))
    if missing:
        raise KeyError("Missing condition embeddings for: " + ", ".join(missing))
    return {condition: np.asarray(condition_rep_dict[condition], dtype=np.float32) for condition in sorted(required_conditions)}


def load_control_condition(data_path, condition_key, control_key):
    backed = ad.read_h5ad(data_path, backed="r")
    try:
        obs_df = backed.obs[[condition_key, control_key, "gene_target_1", "gene_target_2"]].copy()
        control_obs = obs_df[obs_df[control_key].astype(bool)].copy()
    finally:
        backed.file.close()
    adata_control = ad.AnnData(
        X=np.zeros((control_obs.shape[0], 1), dtype=np.float32),
        obs=control_obs,
    )
    adata_control.var_names = ["placeholder"]
    return build_zebrafish_control_condition_embedding(
        data_path=data_path,
        adata_control=adata_control,
        condition_key=condition_key,
    )


def _robust_clip(values, lower_q=0.01, upper_q=0.99):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    lower = float(np.quantile(values, lower_q))
    upper = float(np.quantile(values, upper_q))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower = float(values.min())
        upper = float(values.max()) if values.size else lower + 1.0
    return np.clip(values, lower, upper), lower, upper


def _top_label_order(series, top_n):
    counts = series.astype(str).value_counts()
    labels = counts.index[:top_n].tolist()
    return labels


def compute_head_differences(model, x, timepoint, condition_vec, control_vec, batch_size, device):
    x = np.asarray(x, dtype=np.float32)
    condition_vec = np.asarray(condition_vec, dtype=np.float32)
    control_vec = np.asarray(control_vec, dtype=np.float32)

    transport_abs = np.zeros(x.shape[0], dtype=np.float32)
    transport_rel = np.zeros(x.shape[0], dtype=np.float32)
    growth_abs = np.zeros(x.shape[0], dtype=np.float32)
    growth_rel = np.zeros(x.shape[0], dtype=np.float32)
    v_full_norm = np.zeros(x.shape[0], dtype=np.float32)
    g_full_abs = np.zeros(x.shape[0], dtype=np.float32)
    g_ctrl_abs = np.zeros(x.shape[0], dtype=np.float32)

    cond_tensor = torch.as_tensor(condition_vec, dtype=torch.float32, device=device)
    ctrl_tensor = torch.as_tensor(control_vec, dtype=torch.float32, device=device)

    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            end = min(start + batch_size, x.shape[0])
            xb = torch.as_tensor(x[start:end], dtype=torch.float32, device=device)
            tb = torch.full((xb.shape[0],), float(timepoint), dtype=torch.float32, device=device)
            cond_batch = cond_tensor.unsqueeze(0).expand(xb.shape[0], -1)
            ctrl_batch = ctrl_tensor.unsqueeze(0).expand(xb.shape[0], -1)

            v_full, g_full = model(tb, xb, cond_batch)
            v_ctrl, _ = model(tb, xb, cond_batch, con_v=ctrl_batch, con_g=cond_batch)
            _, g_ctrl = model(tb, xb, cond_batch, con_v=cond_batch, con_g=ctrl_batch)

            v_full_np = v_full.detach().cpu().numpy().astype(np.float32)
            v_ctrl_np = v_ctrl.detach().cpu().numpy().astype(np.float32)
            g_full_np = g_full.detach().cpu().numpy().reshape(-1).astype(np.float32)
            g_ctrl_np = g_ctrl.detach().cpu().numpy().reshape(-1).astype(np.float32)

            transport_delta = np.linalg.norm(v_full_np - v_ctrl_np, axis=1)
            v_norm = np.linalg.norm(v_full_np, axis=1)
            growth_delta = np.abs(g_full_np - g_ctrl_np)
            g_norm = np.abs(g_full_np) + np.abs(g_ctrl_np)

            transport_abs[start:end] = transport_delta
            transport_rel[start:end] = transport_delta / (v_norm + 1e-6)
            growth_abs[start:end] = growth_delta
            growth_rel[start:end] = growth_delta / (g_norm + 1e-6)
            v_full_norm[start:end] = v_norm
            g_full_abs[start:end] = np.abs(g_full_np)
            g_ctrl_abs[start:end] = np.abs(g_ctrl_np)

    return {
        "transport_effect_abs": transport_abs,
        "transport_effect_rel": transport_rel,
        "growth_effect_abs": growth_abs,
        "growth_effect_rel": growth_rel,
        "v_full_norm": v_full_norm,
        "g_full_abs": g_full_abs,
        "g_ctrl_abs": g_ctrl_abs,
    }


def build_scored_reference_adata(mode_umap_path, reference_datatype, top_celltypes):
    adata = ad.read_h5ad(mode_umap_path)
    ref = adata[adata.obs["datatype"].astype(str) == reference_datatype].copy()
    if ref.n_obs == 0:
        raise RuntimeError(f"No '{reference_datatype}' rows found in {mode_umap_path}")
    coords = np.asarray(ref.obsm["X_umap"], dtype=np.float32)
    ref.obs["UMAP1"] = coords[:, 0]
    ref.obs["UMAP2"] = coords[:, 1]
    label_col = "terminal_label" if "terminal_label" in ref.obs.columns else "celltype_label"
    ref.obs["terminal_plot"] = ref.obs[label_col].astype(str)
    top_labels = _top_label_order(ref.obs["terminal_plot"], top_celltypes)
    ref.obs["terminal_plot_top"] = ref.obs["terminal_plot"].where(
        ref.obs["terminal_plot"].isin(top_labels),
        other="Other",
    )
    return ref


def summarize_scores(ref, condition, timepoint, source_filter_key=None, source_filter_values=None):
    summary = {
        "condition": str(condition),
        "timepoint": float(timepoint),
        "n_reference_points": int(ref.n_obs),
        "source_filter_key": source_filter_key,
        "source_filter_values": None if source_filter_values is None else [str(x) for x in source_filter_values],
        "metrics": {},
    }
    for key in [
        "transport_effect_abs",
        "transport_effect_rel",
        "growth_effect_abs",
        "growth_effect_rel",
        "v_full_norm",
        "g_full_abs",
        "g_ctrl_abs",
    ]:
        values = ref.obs[key].astype(float)
        summary["metrics"][key] = {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "p90": float(values.quantile(0.90)),
            "p95": float(values.quantile(0.95)),
            "max": float(values.max()),
        }
    summary["transport_vs_growth_median_ratio"] = float(
        summary["metrics"]["growth_effect_rel"]["median"]
        / (summary["metrics"]["transport_effect_rel"]["median"] + 1e-6)
    )
    return summary


def summarize_by_terminal_celltype(ref):
    group_cols = ["terminal_plot_top"]
    grouped = ref.obs.groupby(group_cols, observed=True, sort=False)
    rows = []
    for label, group_df in grouped:
        label_str = label if isinstance(label, str) else label[0]
        rows.append(
            {
                "terminal_celltype": str(label_str),
                "n_points": int(group_df.shape[0]),
                "transport_effect_rel_median": float(group_df["transport_effect_rel"].astype(float).median()),
                "growth_effect_rel_median": float(group_df["growth_effect_rel"].astype(float).median()),
                "transport_effect_rel_mean": float(group_df["transport_effect_rel"].astype(float).mean()),
                "growth_effect_rel_mean": float(group_df["growth_effect_rel"].astype(float).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["n_points", "terminal_celltype"],
        ascending=[False, True],
    ).reset_index(drop=True)


def annotate_mechanism_pattern(ref, top_celltypes, dominance_threshold=0.12):
    ref = ref.copy()
    mechanism_score = np.log1p(ref.obs["growth_effect_rel"].astype(float)) - np.log1p(
        ref.obs["transport_effect_rel"].astype(float)
    )
    ref.obs["mechanism_score"] = mechanism_score.astype(np.float32)
    mechanism_label = np.where(
        mechanism_score >= dominance_threshold,
        "growth-dominant",
        np.where(mechanism_score <= -dominance_threshold, "transport-dominant", "mixed"),
    )
    ref.obs["mechanism_label"] = pd.Categorical(
        mechanism_label,
        categories=["transport-dominant", "mixed", "growth-dominant"],
    )

    label_col = "terminal_plot_top" if "terminal_plot_top" in ref.obs.columns else "terminal_plot"
    top_labels = _top_label_order(ref.obs[label_col], top_celltypes)
    ref.obs["terminal_plot_top"] = ref.obs[label_col].astype(str).where(
        ref.obs[label_col].astype(str).isin(top_labels),
        other="Other",
    )

    grouped = ref.obs.groupby("terminal_plot_top", observed=True, sort=False)
    rows = []
    for label, group_df in grouped:
        label_str = str(label)
        x = float(ref.obs.loc[group_df.index, "UMAP1"].astype(float).mean())
        y = float(ref.obs.loc[group_df.index, "UMAP2"].astype(float).mean())
        mode_counts = (
            group_df["mechanism_label"]
            .astype(str)
            .value_counts()
            .reindex(["transport-dominant", "mixed", "growth-dominant"], fill_value=0)
        )
        rows.append(
            {
                "terminal_celltype": label_str,
                "n_points": int(group_df.shape[0]),
                "umap1": x,
                "umap2": y,
                "mechanism_score_median": float(group_df["mechanism_score"].astype(float).median()),
                "transport_dominant_fraction": float(mode_counts["transport-dominant"] / group_df.shape[0]),
                "mixed_fraction": float(mode_counts["mixed"] / group_df.shape[0]),
                "growth_dominant_fraction": float(mode_counts["growth-dominant"] / group_df.shape[0]),
                "dominant_label": str(mode_counts.idxmax()),
            }
        )
    celltype_pattern_df = pd.DataFrame(rows).sort_values(
        ["n_points", "terminal_celltype"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return ref, celltype_pattern_df


def plot_condition_diagnostics(ref, summary, celltype_df, output_path):
    transport_plot, _, _ = _robust_clip(np.log1p(ref.obs["transport_effect_rel"].astype(float).to_numpy()))
    growth_plot, _, _ = _robust_clip(np.log1p(ref.obs["growth_effect_rel"].astype(float).to_numpy()))

    label_order = celltype_df["terminal_celltype"].tolist()
    palette = dict(zip(label_order, sns.color_palette("tab20", n_colors=max(len(label_order), 3))[: len(label_order)]))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax = axes[0, 0]
    sc_transport = ax.scatter(
        ref.obs["UMAP1"],
        ref.obs["UMAP2"],
        c=transport_plot,
        cmap="viridis",
        s=6.0,
        alpha=0.85,
        linewidths=0.0,
        rasterized=True,
    )
    ax.set_title("Transport Sensitivity")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(sc_transport, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    sc_growth = ax.scatter(
        ref.obs["UMAP1"],
        ref.obs["UMAP2"],
        c=growth_plot,
        cmap="magma",
        s=6.0,
        alpha=0.85,
        linewidths=0.0,
        rasterized=True,
    )
    ax.set_title("Growth Sensitivity")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(sc_growth, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 2]
    top_df = celltype_df.head(min(8, len(celltype_df))).copy()
    for _, row in top_df.iterrows():
        label = row["terminal_celltype"]
        ax.scatter(
            row["transport_effect_rel_median"],
            row["growth_effect_rel_median"],
            s=max(30.0, row["n_points"] / 10.0),
            c=[palette[label]],
            alpha=0.85,
            linewidths=0.5,
            edgecolors="black",
        )
        ax.text(
            row["transport_effect_rel_median"],
            row["growth_effect_rel_median"],
            str(label),
            fontsize=8,
            ha="left",
            va="bottom",
        )
    ax.set_xlabel("Median transport sensitivity")
    ax.set_ylabel("Median growth sensitivity")
    ax.set_title("Top Terminal Celltypes")

    ax = axes[1, 0]
    sns.histplot(ref.obs["transport_effect_rel"].astype(float), bins=40, color="#1f77b4", ax=ax)
    ax.set_title("Transport Sensitivity Distribution")
    ax.set_xlabel("relative effect")

    ax = axes[1, 1]
    sns.histplot(ref.obs["growth_effect_rel"].astype(float), bins=40, color="#ff7f0e", ax=ax)
    ax.set_title("Growth Sensitivity Distribution")
    ax.set_xlabel("relative effect")

    ax = axes[1, 2]
    ax.axis("off")
    text = "\n".join(
        [
            f"condition: {summary['condition']}",
            f"timepoint: {summary['timepoint']:g}",
            f"n_points: {summary['n_reference_points']}",
            f"transport median: {summary['metrics']['transport_effect_rel']['median']:.4f}",
            f"transport p95: {summary['metrics']['transport_effect_rel']['p95']:.4f}",
            f"growth median: {summary['metrics']['growth_effect_rel']['median']:.4f}",
            f"growth p95: {summary['metrics']['growth_effect_rel']['p95']:.4f}",
            f"growth/transport median ratio: {summary['transport_vs_growth_median_ratio']:.4f}",
        ]
    )
    ax.text(0.0, 1.0, text, fontsize=10, va="top", ha="left", family="monospace")

    fig.suptitle(
        f"{summary['condition']} @ {summary['timepoint']:g} hpf: head condition sensitivity on {summary['n_reference_points']} full-reference particles"
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mechanism_pattern(ref, celltype_pattern_df, output_path):
    mech_clip, _, _ = _robust_clip(ref.obs["mechanism_score"].astype(float).to_numpy(), 0.02, 0.98)
    abs_max = max(0.12, float(np.max(np.abs(mech_clip))))
    norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)
    mech_palette = {
        "transport-dominant": "#1f77b4",
        "mixed": "#7f7f7f",
        "growth-dominant": "#d62728",
    }

    label_order = celltype_pattern_df["terminal_celltype"].tolist()
    colors = sns.color_palette("tab20", n_colors=max(len(label_order), 3))[: len(label_order)]
    celltype_palette = dict(zip(label_order, colors))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    for label in label_order:
        subset = ref.obs[ref.obs["terminal_plot_top"].astype(str) == label]
        if subset.empty:
            continue
        ax.scatter(
            subset["UMAP1"],
            subset["UMAP2"],
            s=6.0,
            c=[celltype_palette[label]],
            alpha=0.80,
            linewidths=0.0,
            rasterized=True,
        )
    ax.set_title("Terminal Celltype")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[0, 1]
    sc_cont = ax.scatter(
        ref.obs["UMAP1"],
        ref.obs["UMAP2"],
        c=mech_clip,
        cmap="coolwarm",
        norm=norm,
        s=6.0,
        alpha=0.85,
        linewidths=0.0,
        rasterized=True,
    )
    ax.set_title("Mechanism Score\nlog(1+growth) - log(1+transport)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(sc_cont, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    for label in ["transport-dominant", "mixed", "growth-dominant"]:
        subset = ref.obs[ref.obs["mechanism_label"].astype(str) == label]
        if subset.empty:
            continue
        ax.scatter(
            subset["UMAP1"],
            subset["UMAP2"],
            s=6.0,
            c=[mech_palette[label]],
            alpha=0.80,
            linewidths=0.0,
            rasterized=True,
            label=label,
        )
    ax.set_title("Dominant Mechanism")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=8, loc="best")

    ax = axes[1, 1]
    for _, row in celltype_pattern_df.head(min(10, len(celltype_pattern_df))).iterrows():
        ax.scatter(
            row["umap1"],
            row["umap2"],
            s=max(40.0, row["n_points"] / 6.0),
            c=[mech_palette[row["dominant_label"]]],
            alpha=0.90,
            linewidths=0.6,
            edgecolors="black",
        )
        ax.text(
            row["umap1"],
            row["umap2"],
            row["terminal_celltype"],
            fontsize=8,
            ha="left",
            va="bottom",
        )
    ax.set_title("Celltype Centroids")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.suptitle("Mechanism pattern on shared UMAP")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_overview(summary_df, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.scatterplot(
        data=summary_df,
        x="transport_effect_rel_median",
        y="growth_effect_rel_median",
        hue="condition",
        size="n_reference_points",
        sizes=(60, 220),
        ax=ax,
    )
    for _, row in summary_df.iterrows():
        ax.text(
            row["transport_effect_rel_median"],
            row["growth_effect_rel_median"],
            f"{row['condition']}@{float(row['timepoint']):g}",
            fontsize=8,
            ha="left",
            va="bottom",
        )
    ax.set_xlabel("Median transport sensitivity")
    ax.set_ylabel("Median growth sensitivity")
    ax.set_title("Head sensitivity overview across cached mode-UMAP subsets")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_root = resolve_output_dir(run_dir, args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    config, final_path = load_run_artifacts(run_dir)
    mode_umap_paths = resolve_mode_umap_paths(run_dir, args.mode_umap_paths)
    config = ensure_model_dims(config, mode_umap_paths[0])
    model = build_model(config, final_path, device)

    conditions = set()
    metadata_rows = []
    for path in mode_umap_paths:
        adata_meta = ad.read_h5ad(path, backed="r")
        try:
            meta = adata_meta.uns.get("mode_umap_metadata", {})
            condition = str(meta.get("condition", adata_meta.obs["target_condition"].astype(str).iloc[0]))
            timepoint = float(meta.get("timepoint", adata_meta.obs["target_timepoint"].astype(float).iloc[0]))
            source_filter_key = meta.get("source_filter_key")
            source_filter_values = meta.get("source_filter_values")
            if source_filter_values is not None:
                source_filter_values = [str(v) for v in np.asarray(source_filter_values).tolist()]
            metadata_rows.append(
                {
                    "path": path,
                    "condition": condition,
                    "timepoint": timepoint,
                    "source_filter_key": source_filter_key,
                    "source_filter_values": source_filter_values,
                }
            )
            conditions.add(condition)
        finally:
            if getattr(adata_meta, "isbacked", False):
                adata_meta.file.close()

    condition_rep_dict = load_condition_rep_dict(
        data_path=config["data_path"],
        condition_names=conditions,
        condition_key=config["condition_key"],
    )
    control_condition_name, control_condition_vec = load_control_condition(
        data_path=config["data_path"],
        condition_key=config["condition_key"],
        control_key=config.get("control_key", "is_control"),
    )

    summary_rows = []
    for meta in metadata_rows:
        path = meta["path"]
        condition = meta["condition"]
        timepoint = meta["timepoint"]
        condition_dir = output_root / path.parent.name
        condition_dir.mkdir(parents=True, exist_ok=True)

        ref = build_scored_reference_adata(
            mode_umap_path=path,
            reference_datatype=args.reference_datatype,
            top_celltypes=args.top_celltypes,
        )
        scores = compute_head_differences(
            model=model,
            x=np.asarray(ref.X, dtype=np.float32),
            timepoint=timepoint,
            condition_vec=condition_rep_dict[condition],
            control_vec=control_condition_vec,
            batch_size=args.batch_size,
            device=device,
        )
        for key, values in scores.items():
            ref.obs[key] = np.asarray(values, dtype=np.float32)
        ref, celltype_pattern_df = annotate_mechanism_pattern(
            ref=ref,
            top_celltypes=args.top_celltypes,
        )
        ref.uns["head_diagnostics"] = {
            "condition": condition,
            "timepoint": float(timepoint),
            "reference_datatype": args.reference_datatype,
            "control_condition_name": control_condition_name,
            "source_filter_key": meta["source_filter_key"],
            "source_filter_values": meta["source_filter_values"],
        }

        summary = summarize_scores(
            ref=ref,
            condition=condition,
            timepoint=timepoint,
            source_filter_key=meta["source_filter_key"],
            source_filter_values=meta["source_filter_values"],
        )
        celltype_df = summarize_by_terminal_celltype(ref)
        plot_condition_diagnostics(
            ref=ref,
            summary=summary,
            celltype_df=celltype_df,
            output_path=condition_dir / "head_diagnostics.png",
        )
        plot_mechanism_pattern(
            ref=ref,
            celltype_pattern_df=celltype_pattern_df,
            output_path=condition_dir / "head_mechanism_pattern.png",
        )

        ref.write(condition_dir / "head_diagnostics_points.h5ad", compression=None)
        celltype_df.to_csv(condition_dir / "head_diagnostics_celltypes.csv", index=False)
        celltype_pattern_df.to_csv(condition_dir / "head_mechanism_celltypes.csv", index=False)
        save_json(condition_dir / "head_diagnostics_summary.json", summary)

        summary_rows.append(
            {
                "condition": condition,
                "timepoint": float(timepoint),
                "mode_umap_path": str(path),
                "reference_datatype": args.reference_datatype,
                "n_reference_points": summary["n_reference_points"],
                "transport_effect_rel_median": summary["metrics"]["transport_effect_rel"]["median"],
                "transport_effect_rel_p95": summary["metrics"]["transport_effect_rel"]["p95"],
                "growth_effect_rel_median": summary["metrics"]["growth_effect_rel"]["median"],
                "growth_effect_rel_p95": summary["metrics"]["growth_effect_rel"]["p95"],
                "transport_vs_growth_median_ratio": summary["transport_vs_growth_median_ratio"],
                "source_filter_key": meta["source_filter_key"],
                "source_filter_values": ",".join(meta["source_filter_values"]) if meta["source_filter_values"] else "",
            }
        )

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(summary_rows).sort_values(["condition", "timepoint"]).reset_index(drop=True)
    summary_df.to_csv(output_root / "head_diagnostics_overview.csv", index=False)
    plot_overview(summary_df, output_root / "head_diagnostics_overview.png")


if __name__ == "__main__":
    main()
