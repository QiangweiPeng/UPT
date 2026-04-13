import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import (  # noqa: E402
    ROLLOUT_MODE_ORDER,
    attach_terminal_labels_knn,
    build_mechanism_summary,
    build_real_endpoint_adata,
    combine_endpoint_analysis_adata,
    infer_label_order,
    load_prediction_collection,
    plot_condition_mechanism_summary,
    plot_mechanism_heatmap_grid,
    plot_mechanism_overview,
    save_summary_json,
    summarize_cohort_scores,
    summarize_global_validation,
)
from src.preprocessing.zebrafish import (  # noqa: E402
    scale_representation_by_control,
)


DEFAULT_PILOT_CONDITIONS = [
    "mafba_control",
    "noto_control",
    "epha4a_control",
    "tfap2a_control",
    "hgfa_control",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Cohort-level mechanism analysis for zebrafish real-source rollouts "
            "using full / transport-only / growth-only cached predictions."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory containing config.json and cached real_source_eval* outputs.",
    )
    parser.add_argument(
        "--full-predictions-path",
        type=Path,
        default=None,
        help="Optional full rollout prediction file. Defaults to <run-dir>/real_source_eval/predictions_real_source_latent.h5ad.",
    )
    parser.add_argument(
        "--transport-predictions-path",
        type=Path,
        default=None,
        help="Optional transport-only rollout prediction file.",
    )
    parser.add_argument(
        "--growth-predictions-path",
        type=Path,
        default=None,
        help="Optional growth-only rollout prediction file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory used to save mechanism-analysis outputs.",
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
        "--all-conditions",
        action="store_true",
        help="Analyze all conditions in the run summary instead of the pilot subset.",
    )
    parser.add_argument(
        "--label-key",
        type=str,
        default="major_group",
        help="Real endpoint annotation used as the terminal label.",
    )
    parser.add_argument(
        "--source-cohort-key",
        type=str,
        default="source_major_group",
        help="Prediction obs column used as the source cohort definition.",
    )
    parser.add_argument(
        "--source-filter-key",
        type=str,
        default=None,
        help="Optional prediction/source obs column used to restrict cohort analysis to a source subset.",
    )
    parser.add_argument(
        "--source-filter-values",
        nargs="*",
        default=None,
        help="Optional allowed values for --source-filter-key. Supports repeated values or comma-separated values.",
    )
    parser.add_argument(
        "--latent-key",
        type=str,
        default=None,
        help="Optional explicit latent key. Defaults to scaled_rep from config.json.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=15,
        help="Number of real neighbors used for terminal label transfer.",
    )
    parser.add_argument(
        "--dominance-threshold",
        type=float,
        default=0.6,
        help="Threshold used to call drift- or growth-dominant cohorts.",
    )
    return parser.parse_args()


def load_json(path):
    import json

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


def parse_timepoint_args(raw_items):
    if not raw_items:
        return None
    timepoints = []
    for item in raw_items:
        parts = [part.strip() for part in item.split(",") if part.strip()]
        for part in parts:
            timepoints.append(float(part))
    return sorted(set(timepoints)) or None


def filter_obs_df(obs_df, filter_key=None, filter_values=None):
    if filter_key is None or filter_values is None:
        return obs_df.copy()
    if filter_key not in obs_df.columns:
        raise KeyError(f"{filter_key} is not found in the provided obs dataframe")
    filter_values = [str(value) for value in filter_values]
    mask = obs_df[filter_key].astype(str).isin(filter_values)
    return obs_df.loc[mask].copy()


def resolve_output_dir(run_dir, explicit_output_dir, use_all_conditions, has_explicit_conditions):
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    if use_all_conditions:
        return run_dir / "mechanism_analysis_all"
    if has_explicit_conditions:
        return run_dir / "mechanism_analysis_custom"
    return run_dir / "mechanism_analysis_pilot"


def resolve_prediction_paths(run_dir, args):
    defaults = {
        "full": run_dir / "real_source_eval" / "predictions_real_source_latent.h5ad",
        "transport-only": run_dir / "real_source_eval_transport_only" / "predictions_real_source_latent.h5ad",
        "growth-only": run_dir / "real_source_eval_growth_only" / "predictions_real_source_latent.h5ad",
    }
    paths = {
        "full": args.full_predictions_path or defaults["full"],
        "transport-only": args.transport_predictions_path or defaults["transport-only"],
        "growth-only": args.growth_predictions_path or defaults["growth-only"],
    }
    missing = [mode for mode, path in paths.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing cached prediction files for modes: "
            + ", ".join(missing)
        )
    return paths


def resolve_selected_conditions(summary_conditions, args):
    explicit_conditions = parse_list_args(args.conditions)
    if args.all_conditions:
        return list(summary_conditions)
    if explicit_conditions is not None:
        return explicit_conditions
    return [condition for condition in DEFAULT_PILOT_CONDITIONS if condition in summary_conditions]


def resolve_real_obs_columns(label_key):
    return [
        label_key,
        "cell_type_sub",
        "cell_type_broad",
        "tissue",
        "germ_layer",
        "germ_layer_adapted",
        "major_group",
        "group_cluster",
        "timepoint",
    ]


def load_scaled_real_treated(config, selected_conditions, label_key):
    control_key = config.get("control_key", "is_control")
    first_control_key = config.get("first_control_key", "first_t_control")
    condition_key = config["condition_key"]
    marginal_key = config["marginal_key"]
    sample_rep = config["sample_rep"]
    use_first_control = not config.get("use_all_control", False)
    selected_conditions = [str(condition) for condition in selected_conditions]

    backed = ad.read_h5ad(config["data_path"], backed="r")
    try:
        requested_obs_columns = []
        for column in [
            condition_key,
            marginal_key,
            control_key,
            first_control_key,
            label_key,
            *resolve_real_obs_columns(label_key),
        ]:
            if column in backed.obs.columns and column not in requested_obs_columns:
                requested_obs_columns.append(column)

        obs_df = backed.obs[requested_obs_columns].copy()
        obs_df[condition_key] = obs_df[condition_key].astype(str)
        obs_df["_obs_position"] = np.arange(obs_df.shape[0], dtype=np.int64)

        treated_mask = (
            (~obs_df[control_key].astype(bool))
            & (obs_df[condition_key].isin(selected_conditions))
        )
        treated_positions = np.flatnonzero(treated_mask.to_numpy())

        if use_first_control and first_control_key in obs_df.columns:
            control_mask = obs_df[first_control_key].astype(bool)
        else:
            control_mask = obs_df[control_key].astype(bool)
        control_positions = np.flatnonzero(control_mask.to_numpy())

        adata_control = ad.AnnData(
            X=np.zeros((control_positions.shape[0], 1), dtype=np.float32),
            obs=obs_df.iloc[control_positions][requested_obs_columns].copy(),
        )
        adata_control.var_names = ["placeholder"]
        adata_control.obsm[sample_rep] = np.asarray(
            backed.obsm[sample_rep][control_positions],
            dtype=np.float32,
        )

        adata_treated = ad.AnnData(
            X=np.zeros((treated_positions.shape[0], 1), dtype=np.float32),
            obs=obs_df.iloc[treated_positions][requested_obs_columns].copy(),
        )
        adata_treated.var_names = ["placeholder"]
        adata_treated.obsm[sample_rep] = np.asarray(
            backed.obsm[sample_rep][treated_positions],
            dtype=np.float32,
        )
    finally:
        backed.file.close()

    scale_representation_by_control(
        adata_control=adata_control,
        adata_treated=adata_treated,
        source_rep=config["sample_rep"],
        target_rep=config["scaled_rep"],
    )
    return adata_treated


def filter_condition_rows(adata_obj, condition_name):
    return adata_obj[
        adata_obj.obs["target_condition"].astype(str) == str(condition_name)
    ].copy()


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = load_json(run_dir / "config.json")
    data_summary = load_json(run_dir / "data_summary.json")

    selected_conditions = resolve_selected_conditions(
        summary_conditions=data_summary.get("selected_conditions", []),
        args=args,
    )
    source_filter_values = parse_list_args(args.source_filter_values)
    selected_timepoints = parse_timepoint_args(args.timepoints)
    if not selected_conditions:
        raise RuntimeError("No conditions were selected for mechanism analysis.")

    output_dir = resolve_output_dir(
        run_dir=run_dir,
        explicit_output_dir=args.output_dir,
        use_all_conditions=args.all_conditions,
        has_explicit_conditions=args.conditions is not None,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_paths = resolve_prediction_paths(run_dir, args)
    predictions = {
        rollout_mode: load_prediction_collection(
            path,
            selected_conditions=selected_conditions,
        )
        for rollout_mode, path in prediction_paths.items()
    }

    latent_key = args.latent_key or str(config["scaled_rep"])
    real_treated = load_scaled_real_treated(
        config=config,
        selected_conditions=selected_conditions,
        label_key=args.label_key,
    )
    label_order = infer_label_order(real_treated, args.label_key)

    global_tables = []
    cohort_tables = []
    analysis_adatas = []

    for condition_name in tqdm(selected_conditions):
        full_condition = filter_condition_rows(predictions["full"], condition_name)
        if full_condition.n_obs == 0:
            continue

        source_rows = full_condition[
            full_condition.obs["record_kind"].astype(str) == "source_observed"
        ].obs.copy()
        if source_rows.empty:
            continue
        n_source_total = int(source_rows.shape[0])
        source_rows_filtered = filter_obs_df(
            source_rows,
            filter_key=args.source_filter_key,
            filter_values=source_filter_values,
        )
        if source_rows_filtered.empty:
            continue

        predicted_rows = full_condition[
            full_condition.obs["record_kind"].astype(str) == "predicted_from_real_source"
        ].obs
        target_timepoints = sorted(
            predicted_rows["target_timepoint"].astype(float).unique().tolist()
        )
        if selected_timepoints is not None:
            target_timepoints = [
                float(timepoint) for timepoint in target_timepoints
                if float(timepoint) in selected_timepoints
            ]
        if not target_timepoints:
            continue

        pair_analysis_adatas = []
        for timepoint in target_timepoints:
            real_endpoint = build_real_endpoint_adata(
                real_adata=real_treated,
                condition=condition_name,
                timepoint=timepoint,
                condition_key=config["condition_key"],
                time_key=config["marginal_key"],
                latent_key=latent_key,
                label_key=args.label_key,
            )
            if real_endpoint.n_obs == 0:
                continue

            pred_by_mode = {}
            pred_by_mode_filtered = {}
            for rollout_mode in ROLLOUT_MODE_ORDER:
                pred_condition = filter_condition_rows(predictions[rollout_mode], condition_name)
                pred_subset = pred_condition[
                    (pred_condition.obs["record_kind"].astype(str) == "predicted_from_real_source")
                    & (pred_condition.obs["target_timepoint"].astype(float) == float(timepoint))
                ].copy()
                if pred_subset.n_obs == 0:
                    raise RuntimeError(
                        f"Missing predicted rows for {condition_name} @ {float(timepoint):g} in {rollout_mode}"
                    )
                pred_subset = attach_terminal_labels_knn(
                    pred_adata=pred_subset,
                    real_adata=real_endpoint,
                    latent_key=latent_key,
                    label_key="terminal_label",
                    output_key="terminal_label",
                    k=args.knn_k,
                )
                pred_by_mode[rollout_mode] = pred_subset
                pred_by_mode_filtered[rollout_mode] = pred_subset[
                    filter_obs_df(
                        pred_subset.obs,
                        filter_key=args.source_filter_key,
                        filter_values=source_filter_values,
                    ).index
                ].copy()

            global_pair = summarize_global_validation(
                real_endpoint=real_endpoint,
                pred_by_mode=pred_by_mode,
                condition=condition_name,
                timepoint=timepoint,
                n_source_total=n_source_total,
                label_order=label_order,
                terminal_label_key="terminal_label",
            )
            global_tables.append(global_pair)

            cohort_pair = summarize_cohort_scores(
                pred_by_mode=pred_by_mode_filtered,
                source_obs=source_rows_filtered,
                condition=condition_name,
                timepoint=timepoint,
                source_cohort_key=args.source_cohort_key,
                terminal_label_key="terminal_label",
                label_order=label_order,
                dominance_threshold=args.dominance_threshold,
            )
            if cohort_pair.empty:
                continue
            if args.source_filter_key is not None and source_filter_values is not None:
                cohort_pair["source_filter_key"] = str(args.source_filter_key)
                cohort_pair["source_filter_values"] = ",".join(source_filter_values)
            cohort_tables.append(cohort_pair)

            pair_adata = combine_endpoint_analysis_adata(
                real_endpoint=real_endpoint,
                pred_by_mode=pred_by_mode_filtered,
                source_cohort_key=args.source_cohort_key,
                terminal_label_key="terminal_label",
            )
            pair_analysis_adatas.append(pair_adata)

        if not pair_analysis_adatas:
            continue

        condition_analysis = ad.concat(
            pair_analysis_adatas,
            axis=0,
            join="outer",
            merge="same",
            index_unique=None,
        )
        condition_analysis.uns.setdefault("mechanism_analysis", {})
        condition_analysis.uns["mechanism_analysis"]["condition"] = str(condition_name)
        condition_analysis.uns["mechanism_analysis"]["label_key"] = args.label_key
        condition_analysis.uns["mechanism_analysis"]["source_cohort_key"] = args.source_cohort_key
        analysis_adatas.append(condition_analysis)

    if not global_tables or not cohort_tables or not analysis_adatas:
        raise RuntimeError("No mechanism-analysis tables were generated.")

    global_validation_df = pd.concat(global_tables, axis=0, ignore_index=True)
    cohort_scores_df = pd.concat(cohort_tables, axis=0, ignore_index=True)
    analysis_adata = ad.concat(
        analysis_adatas,
        axis=0,
        join="outer",
        merge="same",
        index_unique=None,
    )
    analysis_adata.uns["mechanism_analysis"] = {
        "run_dir": str(run_dir),
        "label_key": args.label_key,
        "source_cohort_key": args.source_cohort_key,
        "latent_key": latent_key,
        "selected_conditions": [str(condition) for condition in selected_conditions],
        "rollout_modes": list(ROLLOUT_MODE_ORDER),
        "prediction_paths": {mode: str(path) for mode, path in prediction_paths.items()},
        "dominance_threshold": float(args.dominance_threshold),
        "knn_k": int(args.knn_k),
        "source_filter_key": args.source_filter_key,
        "source_filter_values": source_filter_values,
        "selected_timepoints": selected_timepoints,
    }

    analysis_adata.write_h5ad(output_dir / "mechanism_analysis_latent.h5ad")
    global_validation_df.to_csv(output_dir / "global_validation.csv", index=False)
    cohort_scores_df.to_csv(output_dir / "cohort_scores.csv", index=False)

    summary_payload = build_mechanism_summary(global_validation_df, cohort_scores_df)
    summary_payload["selected_conditions"] = [str(condition) for condition in selected_conditions]
    summary_payload["label_key"] = args.label_key
    summary_payload["source_cohort_key"] = args.source_cohort_key
    summary_payload["source_filter_key"] = args.source_filter_key
    summary_payload["source_filter_values"] = source_filter_values
    summary_payload["selected_timepoints"] = selected_timepoints
    save_summary_json(output_dir / "mechanism_summary.json", summary_payload)

    plot_mechanism_overview(
        global_validation_df=global_validation_df,
        cohort_scores_df=cohort_scores_df,
        output_path=output_dir / "mechanism_overview.png",
    )
    plot_mechanism_heatmap_grid(
        cohort_scores_df=cohort_scores_df,
        metric_cols=["drift_support", "growth_support"],
        output_path=output_dir / "mechanism_score_heatmaps.png",
    )

    per_condition_dir = output_dir / "conditions"
    per_condition_dir.mkdir(parents=True, exist_ok=True)
    for condition_name in sorted(cohort_scores_df["target_condition"].astype(str).unique().tolist()):
        plot_condition_mechanism_summary(
            global_validation_df=global_validation_df,
            cohort_scores_df=cohort_scores_df,
            output_path=per_condition_dir / f"{condition_name}_mechanism.png",
            condition=condition_name,
        )

    print(f"Saved mechanism analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
