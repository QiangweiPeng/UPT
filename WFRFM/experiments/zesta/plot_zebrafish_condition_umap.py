import argparse
import json
import re
import sys
from pathlib import Path

import anndata as ad
import pandas as pd

import matplotlib

matplotlib.use("Agg")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate.evals_plot import (  # noqa: E402
    build_condition_real_prediction_umap_adata,
    compute_latent_umap,
    plot_condition_real_prediction_umap,
)
from src.preprocessing.zebrafish import (  # noqa: E402
    load_zebrafish_training_data,
    scale_representation_by_control,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build per-condition UMAP visualizations by combining full real zebrafish cells and mass-resampled predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory containing config.json.",
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=None,
        help="Latent prediction AnnData. Defaults to <run-dir>/basic_eval/predictions_latent.h5ad.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory used to save per-condition UMAP h5ad and plots. Defaults to <predictions parent>/condition_umap.",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Optional condition list. Supports repeated values or comma-separated values.",
    )
    parser.add_argument(
        "--timepoints",
        nargs="*",
        default=None,
        help="Optional target timepoint list. Supports repeated values or comma-separated values.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=30,
        help="Number of neighbors used for UMAP graph construction.",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.3,
        help="UMAP min_dist parameter.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for weighted resampling and UMAP.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_condition_args(raw_conditions):
    if not raw_conditions:
        return None
    conditions = []
    for item in raw_conditions:
        parts = [part.strip() for part in item.split(",") if part.strip()]
        conditions.extend(parts)
    return conditions or None


def parse_timepoint_args(raw_timepoints):
    if not raw_timepoints:
        return None
    timepoints = []
    for item in raw_timepoints:
        parts = [part.strip() for part in item.split(",") if part.strip()]
        for part in parts:
            timepoints.append(float(part))
    return timepoints or None


def resolve_predictions_path(run_dir, explicit_path=None):
    if explicit_path is not None:
        return explicit_path
    default_path = run_dir / "basic_eval" / "predictions_latent.h5ad"
    if not default_path.exists():
        raise FileNotFoundError(
            f"Missing predictions file: {default_path}. Pass --predictions-path explicitly."
        )
    return default_path


def resolve_real_obs_columns(data_path):
    backed = ad.read_h5ad(data_path, backed="r")
    try:
        return list(backed.obs.columns)
    finally:
        backed.file.close()


def sanitize_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def select_conditions(explicit_conditions, predictions_adata):
    available = sorted(predictions_adata.obs["target_condition"].astype(str).unique().tolist())
    if explicit_conditions is None:
        return available
    missing = [condition for condition in explicit_conditions if condition not in available]
    if missing:
        raise KeyError(f"Unknown conditions in predictions file: {missing}")
    return explicit_conditions


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = load_json(run_dir / "config.json")

    predictions_path = resolve_predictions_path(run_dir, args.predictions_path)
    predictions_adata = ad.read_h5ad(predictions_path)
    selected_conditions = select_conditions(parse_condition_args(args.conditions), predictions_adata)
    selected_timepoints = parse_timepoint_args(args.timepoints)

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (predictions_path.resolve().parent / "condition_umap")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    real_obs_columns = resolve_real_obs_columns(config["data_path"])

    adata_control, adata_treated, _, _ = load_zebrafish_training_data(
        data_path=config["data_path"],
        rep_keys=[config["sample_rep"]],
        condition_key=config["condition_key"],
        marginal_key=config["marginal_key"],
        control_key=config["control_key"],
        first_control_key=config["first_control_key"],
        use_first_control=not config.get("use_all_control", False),
        selected_conditions=selected_conditions,
        control_max_cells=None,
        treated_max_cells_per_group=None,
        extra_obs_columns=real_obs_columns,
        seed=config["seed"],
    )
    scale_representation_by_control(
        adata_control=adata_control,
        adata_treated=adata_treated,
        source_rep=config["sample_rep"],
        target_rep=config["scaled_rep"],
    )

    summary_tables = []
    index_rows = []
    for condition in selected_conditions:
        condition_pred = predictions_adata[
            predictions_adata.obs["target_condition"].astype(str) == str(condition)
        ]
        if condition_pred.n_obs == 0:
            continue

        timepoints = sorted(
            condition_pred.obs["target_timepoint"].astype(float).unique().tolist()
        )
        if selected_timepoints is not None:
            timepoints = [timepoint for timepoint in timepoints if timepoint in selected_timepoints]
        if not timepoints:
            continue

        vis_adata, sampling_summary = build_condition_real_prediction_umap_adata(
            real_adata=adata_treated,
            pred_adata=predictions_adata,
            condition=condition,
            timepoints=timepoints,
            condition_key=config["condition_key"],
            time_key=config["marginal_key"],
            latent_key=config["scaled_rep"],
            weight_key="mass",
            seed=args.seed,
        )
        compute_latent_umap(
            vis_adata,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            random_state=args.seed,
        )

        condition_name = sanitize_name(condition)
        condition_dir = output_dir / condition_name
        condition_dir.mkdir(parents=True, exist_ok=True)

        vis_adata.uns["condition_umap_metadata"] = {
            "condition": str(condition),
            "predictions_path": str(predictions_path),
            "run_dir": str(run_dir),
            "n_neighbors": int(args.n_neighbors),
            "min_dist": float(args.min_dist),
            "seed": int(args.seed),
            "sampling_scheme": "predicted particles resampled with replacement by normalized mass to match full real pair counts",
        }
        vis_adata.write_h5ad(condition_dir / f"{condition_name}_umap.h5ad", compression="gzip")
        sampling_summary.to_csv(condition_dir / f"{condition_name}_sampling_summary.csv", index=False)
        plot_condition_real_prediction_umap(
            vis_adata,
            output_path=condition_dir / f"{condition_name}_umap.png",
            title=f"{condition}",
        )

        summary_tables.append(sampling_summary)
        index_rows.append(
            {
                "condition": str(condition),
                "timepoints": ",".join(f"{timepoint:g}" for timepoint in timepoints),
                "umap_h5ad": str(condition_dir / f"{condition_name}_umap.h5ad"),
                "umap_png": str(condition_dir / f"{condition_name}_umap.png"),
                "sampling_summary_csv": str(condition_dir / f"{condition_name}_sampling_summary.csv"),
                "n_obs_vis": int(vis_adata.n_obs),
            }
        )
        print(f"Saved UMAP outputs for {condition} to {condition_dir}")

    if summary_tables:
        pd.concat(summary_tables, axis=0, ignore_index=True).to_csv(
            output_dir / "sampling_summary_all_conditions.csv",
            index=False,
        )
    pd.DataFrame(index_rows).to_csv(output_dir / "condition_umap_index.csv", index=False)
    print(f"Saved condition UMAP index to {output_dir / 'condition_umap_index.csv'}")


if __name__ == "__main__":
    main()
