import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.condition_benchmark import (  # noqa: E402
    _align_adatas,
    _get_matrix,
    _get_weights,
    _safe_pearson,
    _subset_condition,
    _weighted_mean,
)


NON_W_METRICS = ("r2", "pcc", "pearson_delta", "mse", "mae")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark simulation predictions using only non-Wasserstein metrics. "
            "This keeps the CellFlow-comparable summary focused on "
            "r2 / pcc / pearson_delta / mse / mae."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--truth-adata", type=Path, required=True)
    parser.add_argument("--pred-adata", type=Path, required=True)
    parser.add_argument("--ctrl-adata", type=Path, required=True)
    parser.add_argument("--condition-key", type=str, default=None)
    parser.add_argument("--truth-condition-key", type=str, default=None)
    parser.add_argument("--pred-condition-key", type=str, default=None)
    parser.add_argument("--truth-matrix", type=str, default="X")
    parser.add_argument("--pred-matrix", type=str, default="X")
    parser.add_argument("--ctrl-matrix", type=str, default="X")
    parser.add_argument("--weight-key", type=str, default="mass")
    parser.add_argument("--model-name", type=str, default="wfrfm")
    parser.add_argument("--dataset-name", type=str, default="dataset")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--reference-summary-csv", type=Path, default=None)
    parser.add_argument("--reference-model", type=str, default=None)
    parser.add_argument("--compare-output-csv", type=Path, default=None)
    parser.add_argument("--compare-output-json", type=Path, default=None)
    return parser.parse_args()


def evaluate_conditionwise_non_w(
    truth_adata: ad.AnnData,
    pred_adata: ad.AnnData,
    ctrl_adata: ad.AnnData,
    *,
    truth_condition_key: str,
    pred_condition_key: str,
    truth_matrix: str,
    pred_matrix: str,
    ctrl_matrix: str,
    weight_key: str | None,
    model_name: str,
    dataset_name: str,
) -> pd.DataFrame:
    truth_adata = truth_adata.copy()
    pred_adata = pred_adata.copy()
    alignment_key = "__benchmark_condition__"
    truth_adata.obs[alignment_key] = truth_adata.obs[truth_condition_key].astype(str).to_numpy()
    pred_adata.obs[alignment_key] = pred_adata.obs[pred_condition_key].astype(str).to_numpy()

    truth_adata, pred_adata = _align_adatas(
        truth_adata=truth_adata,
        pred_adata=pred_adata,
        condition_key=alignment_key,
    )
    ctrl_adata = ctrl_adata[:, truth_adata.var_names].copy()
    ctrl_weights = _get_weights(ctrl_adata, weight_key=None)
    ctrl_mean = _weighted_mean(_get_matrix(ctrl_adata, ctrl_matrix), ctrl_weights)

    truth_conditions = set(truth_adata.obs[alignment_key].astype(str))
    pred_conditions = set(pred_adata.obs[alignment_key].astype(str))
    common_conditions = sorted(truth_conditions & pred_conditions)
    if not common_conditions:
        raise ValueError("No overlapping conditions between truth and prediction.")

    rows = []
    for condition in common_conditions:
        truth_cond = _subset_condition(truth_adata, alignment_key, condition)
        pred_cond = _subset_condition(pred_adata, alignment_key, condition)
        if truth_cond.n_obs == 0 or pred_cond.n_obs == 0:
            continue

        truth_weights = _get_weights(truth_cond, weight_key=None)
        pred_weights = _get_weights(pred_cond, weight_key=weight_key)
        truth_metric_matrix = _get_matrix(truth_cond, truth_matrix)
        pred_metric_matrix = _get_matrix(pred_cond, pred_matrix)
        if truth_metric_matrix.shape[1] != pred_metric_matrix.shape[1]:
            raise ValueError(
                f"Condition '{condition}' has mismatched feature counts: "
                f"{truth_metric_matrix.shape[1]} vs {pred_metric_matrix.shape[1]}."
            )

        truth_mean = _weighted_mean(truth_metric_matrix, truth_weights)
        pred_mean = _weighted_mean(pred_metric_matrix, pred_weights)
        rows.append(
            {
                "model": model_name,
                "dataset": str(dataset_name),
                "condition": str(condition),
                "n_truth_cells": int(truth_cond.n_obs),
                "n_pred_cells": int(pred_cond.n_obs),
                "r2": float(r2_score(truth_mean, pred_mean)),
                "pcc": _safe_pearson(truth_mean, pred_mean),
                "pearson_delta": _safe_pearson(truth_mean - ctrl_mean, pred_mean - ctrl_mean),
                "mse": float(mean_squared_error(truth_mean, pred_mean)),
                "mae": float(mean_absolute_error(truth_mean, pred_mean)),
            }
        )

    return pd.DataFrame(rows).sort_values(["model", "condition"]).reset_index(drop=True)


def summarise_non_w(results: pd.DataFrame) -> pd.DataFrame:
    summary = results.groupby("model")[list(NON_W_METRICS)].agg(["mean", "median", "std"]).reset_index()
    summary.columns = [
        "model" if col == ("model", "") else f"{col[0]}_{col[1]}"
        for col in summary.columns.to_flat_index()
    ]
    return summary


def build_compare_payload(summary_df, reference_summary_df, model_name, reference_model):
    summary_row = summary_df.set_index("model").loc[model_name]
    ref_row = reference_summary_df.set_index("model").loc[reference_model]
    compare_rows = []
    for metric in NON_W_METRICS:
        metric_key = f"{metric}_mean"
        ours = float(summary_row[metric_key])
        reference = float(ref_row[metric_key])
        compare_rows.append(
            {
                "metric": metric,
                "ours": ours,
                "reference": reference,
                "delta_ours_minus_reference": ours - reference,
                "direction": "higher_is_better" if metric in {"r2", "pcc", "pearson_delta"} else "lower_is_better",
                "is_better_or_equal": (ours >= reference) if metric in {"r2", "pcc", "pearson_delta"} else (ours <= reference),
            }
        )
    compare_df = pd.DataFrame(compare_rows)
    payload = {
        "model_name": model_name,
        "reference_model": reference_model,
        "n_metrics_not_worse": int(compare_df["is_better_or_equal"].sum()),
        "metrics": compare_rows,
    }
    return compare_df, payload


if __name__ == "__main__":
    args = parse_args()

    truth_adata = ad.read_h5ad(args.truth_adata)
    pred_adata = ad.read_h5ad(args.pred_adata)
    ctrl_adata = ad.read_h5ad(args.ctrl_adata)

    results = evaluate_conditionwise_non_w(
        truth_adata=truth_adata,
        pred_adata=pred_adata,
        ctrl_adata=ctrl_adata,
        truth_condition_key=args.truth_condition_key or args.condition_key or "condition",
        pred_condition_key=args.pred_condition_key or args.condition_key or "condition",
        truth_matrix=args.truth_matrix,
        pred_matrix=args.pred_matrix,
        ctrl_matrix=args.ctrl_matrix,
        weight_key=args.weight_key,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
    )
    summary = summarise_non_w(results)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)

    if args.reference_summary_csv is not None and args.reference_model is not None:
        ref_summary = pd.read_csv(args.reference_summary_csv)
        compare_df, compare_payload = build_compare_payload(
            summary_df=summary,
            reference_summary_df=ref_summary,
            model_name=args.model_name,
            reference_model=args.reference_model,
        )
        if args.compare_output_csv is not None:
            args.compare_output_csv.parent.mkdir(parents=True, exist_ok=True)
            compare_df.to_csv(args.compare_output_csv, index=False)
        if args.compare_output_json is not None:
            args.compare_output_json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.compare_output_json, "w", encoding="utf-8") as handle:
                json.dump(compare_payload, handle, indent=2, sort_keys=True)

    print(results)
    print("\nSummary:")
    print(summary)
