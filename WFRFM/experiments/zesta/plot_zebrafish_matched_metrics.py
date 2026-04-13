import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import compute_matched_sample_metrics, plot_metric_heatmap_grid  # noqa: E402
from src.preprocessing.zebrafish import (  # noqa: E402
    load_zebrafish_training_data,
    scale_representation_by_control,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate zebrafish predictions with matched-size no-replacement sampling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Evaluation directory containing predictions_latent.h5ad.",
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=None,
        help="Optional explicit path to predictions_latent.h5ad.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Optional explicit path to zebrafish_processed.h5ad. Defaults to run config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <eval-dir>/matched_metrics_no_replacement.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for matched sampling.",
    )
    return parser.parse_args()


def load_run_config(eval_dir: Path):
    candidate = eval_dir.parent / "config.json"
    if candidate.exists():
        with open(candidate, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def summarize_metrics(metrics_df: pd.DataFrame):
    summary = {}
    for metric in ["r2", "pcc", "mse", "mae", "w1", "w2"]:
        values = metrics_df[metric].astype(float)
        summary[metric] = {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def main():
    args = parse_args()
    eval_dir = args.eval_dir.resolve()
    config = load_run_config(eval_dir)

    predictions_path = (
        args.predictions_path.resolve()
        if args.predictions_path is not None
        else (eval_dir / "predictions_latent.h5ad").resolve()
    )
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions_latent.h5ad: {predictions_path}")

    default_data_path = config.get(
        "data_path",
        str(PROJECT_ROOT / "data" / "zesta" / "zebrafish_processed.h5ad"),
    )
    data_path = (
        args.data_path.resolve()
        if args.data_path is not None
        else Path(default_data_path).resolve()
    )
    if not data_path.exists():
        raise FileNotFoundError(f"Missing zebrafish data file: {data_path}")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (eval_dir / "matched_metrics_no_replacement")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rep = str(config.get("sample_rep", "X_pca"))
    scaled_rep = str(config.get("scaled_rep", f"{sample_rep}_scaled"))
    condition_key = str(config.get("condition_key", "gene_target"))
    marginal_key = str(config.get("marginal_key", "timepoint"))
    control_key = str(config.get("control_key", "is_control"))
    first_control_key = str(config.get("first_control_key", "first_t_control"))
    use_first_control = not bool(config.get("use_all_control", False))

    pred_adata = ad.read_h5ad(predictions_path)
    pair_df = (
        pred_adata.obs[["target_condition", "target_timepoint"]]
        .drop_duplicates()
        .sort_values(["target_condition", "target_timepoint"])
        .reset_index(drop=True)
    )
    selected_conditions = sorted(pair_df["target_condition"].astype(str).unique().tolist())

    adata_control, adata_treated, _, _ = load_zebrafish_training_data(
        data_path=str(data_path),
        rep_keys=[sample_rep],
        condition_key=condition_key,
        marginal_key=marginal_key,
        control_key=control_key,
        first_control_key=first_control_key,
        use_first_control=use_first_control,
        selected_conditions=selected_conditions,
        control_max_cells=None,
        treated_max_cells_per_group=None,
        seed=args.seed,
    )
    scale_representation_by_control(
        adata_control=adata_control,
        adata_treated=adata_treated,
        source_rep=sample_rep,
        target_rep=scaled_rep,
    )

    metric_rows = []
    for idx, row in pair_df.iterrows():
        condition = str(row["target_condition"])
        timepoint = float(row["target_timepoint"])
        pred_subset = pred_adata[
            (pred_adata.obs["target_condition"].astype(str) == condition)
            & (pred_adata.obs["target_timepoint"].astype(float) == timepoint)
        ]
        real_subset = adata_treated[
            (adata_treated.obs[condition_key].astype(str) == condition)
            & (adata_treated.obs[marginal_key].astype(float) == timepoint)
        ]
        if pred_subset.n_obs == 0 or real_subset.n_obs == 0:
            continue

        n_match = int(min(pred_subset.n_obs, real_subset.n_obs))
        metrics = compute_matched_sample_metrics(
            X_real=real_subset.obsm[scaled_rep],
            X_pred=np.asarray(pred_subset.X, dtype=np.float32),
            pred_weights=pred_subset.obs["mass"].to_numpy(dtype=np.float64)
            if "mass" in pred_subset.obs.columns
            else None,
            n_match=n_match,
            seed=args.seed + idx,
        )
        metrics.update(
            {
                "target_condition": condition,
                "target_timepoint": timepoint,
            }
        )
        metric_rows.append(metrics)

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["target_condition", "target_timepoint"]
    ).reset_index(drop=True)
    metrics_df.to_csv(output_dir / "matched_sample_metrics.csv", index=False)

    summary = {
        "sampling_scheme": (
            "For each (condition, timepoint), sample min(n_real, n_pred) cells "
            "without replacement; real is uniform, prediction is mass-weighted."
        ),
        "predictions_path": str(predictions_path),
        "data_path": str(data_path),
        "sample_rep": sample_rep,
        "scaled_rep": scaled_rep,
        "seed": int(args.seed),
        "n_pairs_evaluated": int(metrics_df.shape[0]),
        "metrics": summarize_metrics(metrics_df),
    }
    with open(output_dir / "matched_sample_metrics_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    plot_metric_heatmap_grid(
        metrics_df=metrics_df,
        output_path=output_dir / "matched_sample_metrics_heatmap.png",
        title="Matched-Size No-Replacement Latent Metrics",
    )

    pd.DataFrame(
        [
            {
                "predictions_path": str(predictions_path),
                "data_path": str(data_path),
                "metrics_csv": str(output_dir / "matched_sample_metrics.csv"),
                "summary_json": str(output_dir / "matched_sample_metrics_summary.json"),
                "heatmap_png": str(output_dir / "matched_sample_metrics_heatmap.png"),
            }
        ]
    ).to_csv(output_dir / "matched_sample_metrics_index.csv", index=False)
    print(f"Saved matched-size metrics to {output_dir}")


if __name__ == "__main__":
    main()
