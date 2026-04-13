import argparse
import anndata as ad
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cosine
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate.inference import (  # noqa: E402
    VALID_CONDITION_MODES,
    build_latent_prediction_adata,
    run_condition_timepoint_trajectory_inference,
)
from src.evaluate.util import draw_loss  # noqa: E402
from src.preprocessing.zebrafish import (  # noqa: E402
    attach_condition_embeddings,
    build_zebrafish_control_condition_embedding,
    load_zebrafish_training_data,
    scale_representation_by_control,
)
from src.training.model import FNet  # noqa: E402


DEFAULT_SOURCE_OBS_COLUMNS = [
    "cell_type_sub",
    "cell_type_broad",
    "tissue",
    "germ_layer",
    "germ_layer_adapted",
    "major_group",
    "group_cluster",
    "timepoint",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Basic post-hoc evaluation for a trained zebrafish WFRFM run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory containing config.json, data_summary.json, final.pt, and checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional explicit checkpoint file used for loss plotting. Defaults to the latest checkpoint_epoch_*.pt in the run directory.",
    )
    parser.add_argument(
        "--final-path",
        type=Path,
        default=None,
        help="Optional explicit model file. Defaults to <run-dir>/final.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory used to save plots and tables. Defaults to <run-dir>/basic_eval.",
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
        "--n-steps",
        type=int,
        default=50,
        help="Number of Euler steps used for each observed time interval during inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device. Falls back to cpu when cuda is unavailable.",
    )
    parser.add_argument(
        "--loss-cut",
        type=int,
        default=0,
        help="Optional prefix of iterations to skip when plotting and summarizing the loss curves.",
    )
    parser.add_argument(
        "--condition-mode",
        type=str,
        choices=VALID_CONDITION_MODES,
        default="full",
        help="How transport and growth heads consume condition embeddings during inference.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


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


def resolve_device(requested_device):
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable. Falling back to cpu for evaluation.")
        return "cpu"
    return requested_device


def _mode_suffix(condition_mode):
    return condition_mode.replace("-", "_")


def resolve_mode_output_dir(run_dir, explicit_output_dir, condition_mode):
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    if condition_mode == "full":
        return run_dir / "basic_eval"
    return run_dir / f"basic_eval_{_mode_suffix(condition_mode)}"


def _checkpoint_epoch(path):
    match = re.search(r"checkpoint_epoch_(\d+)\.pt$", path.name)
    if not match:
        return -1
    return int(match.group(1))


def resolve_checkpoint_path(run_dir, explicit_path=None):
    if explicit_path is not None:
        return explicit_path
    checkpoint_paths = sorted(
        run_dir.glob("checkpoint_epoch_*.pt"),
        key=_checkpoint_epoch,
    )
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoint_epoch_*.pt files found in {run_dir}")
    return checkpoint_paths[-1]


def resolve_final_path(run_dir, explicit_path=None):
    if explicit_path is not None:
        return explicit_path
    final_path = run_dir / "final.pt"
    if not final_path.exists():
        raise FileNotFoundError(f"Missing final model file: {final_path}")
    return final_path


def load_run_artifacts(run_dir):
    config_path = run_dir / "config.json"
    summary_path = run_dir / "data_summary.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing data summary file: {summary_path}")
    return load_json(config_path), load_json(summary_path)


def prepare_zebrafish_data(config):
    control_key = config.get("control_key", "is_control")
    first_control_key = config.get("first_control_key", "first_t_control")
    adata_control, adata_treated, condition_rep_dict, selected_conditions = load_zebrafish_training_data(
        data_path=config["data_path"],
        rep_keys=[config["sample_rep"]],
        condition_key=config["condition_key"],
        marginal_key=config["marginal_key"],
        control_key=control_key,
        first_control_key=first_control_key,
        use_first_control=not config.get("use_all_control", False),
        max_conditions=config.get("max_conditions"),
        control_max_cells=config.get("control_max_cells"),
        treated_max_cells_per_group=config.get("treated_max_cells_per_group"),
        extra_obs_columns=DEFAULT_SOURCE_OBS_COLUMNS,
        seed=config["seed"],
    )

    scale_representation_by_control(
        adata_control=adata_control,
        adata_treated=adata_treated,
        source_rep=config["sample_rep"],
        target_rep=config["scaled_rep"],
    )
    attach_condition_embeddings(
        adata_treated,
        condition_rep_dict=condition_rep_dict,
        condition_key=config["condition_key"],
        embedding_key=config["condition_rep_key"],
    )
    control_condition_name, control_condition_vec = build_zebrafish_control_condition_embedding(
        data_path=config["data_path"],
        adata_control=adata_control,
        condition_key=config["condition_key"],
    )
    return adata_control, adata_treated, selected_conditions, control_condition_name, control_condition_vec


def select_conditions(explicit_conditions, summary_conditions, available_conditions):
    if explicit_conditions is None:
        selected = list(summary_conditions)
    else:
        selected = explicit_conditions

    missing = [condition for condition in selected if condition not in available_conditions]
    if missing:
        raise KeyError(f"Unknown evaluation conditions: {missing}")
    return selected


def build_model(config, in_out_dim, condition_dim, model_path, device):
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model = FNet(
        in_out_dim=in_out_dim,
        hidden_dim_v=config["hidden_dim_v"],
        n_hiddens_v=config["n_hiddens_v"],
        hidden_dim_g=config["hidden_dim_g"],
        n_hiddens_g=config["n_hiddens_g"],
        condition_dim=condition_dim,
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


def summarize_latent_metrics(metrics_df):
    summary = {
        "n_pairs_evaluated": int(len(metrics_df)),
        "n_conditions_evaluated": int(metrics_df["target_condition"].nunique()),
        "conditions": sorted(metrics_df["target_condition"].unique().tolist()),
        "timepoints": sorted(metrics_df["target_timepoint"].astype(float).unique().tolist()),
        "metrics": {},
        "gene_space_evaluation": "skipped_missing_pca_reconstruction_metadata",
    }
    for metric_name in ["latent_mse", "latent_cosine", "latent_delta_cosine"]:
        values = metrics_df[metric_name].astype(float)
        summary["metrics"][metric_name] = {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    ranking_cols = ["target_condition", "target_timepoint"]
    summary["worst_by_latent_mse"] = metrics_df.nlargest(
        min(5, len(metrics_df)),
        "latent_mse",
    )[ranking_cols + ["latent_mse"]].to_dict(orient="records")
    summary["worst_by_latent_delta_cosine"] = metrics_df.nsmallest(
        min(5, len(metrics_df)),
        "latent_delta_cosine",
    )[ranking_cols + ["latent_delta_cosine"]].to_dict(orient="records")
    summary["best_by_latent_delta_cosine"] = metrics_df.nlargest(
        min(5, len(metrics_df)),
        "latent_delta_cosine",
    )[ranking_cols + ["latent_delta_cosine"]].to_dict(orient="records")
    return summary


def build_target_specs(adata_treated, condition_key, time_key, selected_conditions, selected_timepoints=None):
    obs_df = adata_treated.obs[[condition_key, time_key]].copy()
    obs_df[condition_key] = obs_df[condition_key].astype(str)
    obs_df[time_key] = obs_df[time_key].astype(float)
    obs_df = obs_df[obs_df[condition_key].isin(selected_conditions)].copy()
    if selected_timepoints is not None:
        obs_df = obs_df[obs_df[time_key].isin(selected_timepoints)].copy()

    pair_df = obs_df.drop_duplicates().sort_values([condition_key, time_key]).reset_index(drop=True)
    return [tuple(row) for row in pair_df.itertuples(index=False, name=None)]


def group_target_specs_by_condition(target_specs):
    grouped = {}
    for condition, target_timepoint in target_specs:
        grouped.setdefault(str(condition), []).append(float(target_timepoint))
    return grouped


def _normalize_weights(weights):
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    total = weights.sum()
    if total <= 0:
        return np.full(weights.shape[0], 1.0 / weights.shape[0], dtype=np.float64)
    return weights / total


def evaluate_latent_pair(result_dict, true_subset, ctrl_mean_latent):
    z_true = np.asarray(true_subset.obsm["__metric_rep__"], dtype=np.float32)
    z_pred = np.asarray(result_dict["z_pred"], dtype=np.float32)
    weights = _normalize_weights(result_dict["m_pred"])

    z_mean_true = np.mean(z_true, axis=0)
    z_mean_pred = np.average(z_pred, axis=0, weights=weights)

    delta_true = z_mean_true - ctrl_mean_latent
    delta_pred = z_mean_pred - ctrl_mean_latent

    return {
        "latent_mse": float(np.mean((z_mean_true - z_mean_pred) ** 2)),
        "latent_cosine": float(1 - cosine(z_mean_true, z_mean_pred)),
        "latent_delta_cosine": float(1 - cosine(delta_true, delta_pred)),
    }


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = resolve_mode_output_dir(run_dir, args.output_dir, args.condition_mode)
    output_dir.mkdir(parents=True, exist_ok=True)

    config, data_summary = load_run_artifacts(run_dir)
    checkpoint_path = resolve_checkpoint_path(run_dir, args.checkpoint_path)
    final_path = resolve_final_path(run_dir, args.final_path)
    device = resolve_device(args.device)

    loss_summary = draw_loss(
        load_path=str(checkpoint_path),
        eval_interval=config.get("eval_interval", 10),
        cut=args.loss_cut,
        save_path=str(output_dir / "loss_curve.png"),
        show=False,
    )
    loss_summary["checkpoint_path"] = str(checkpoint_path)
    loss_summary["plot_path"] = str(output_dir / "loss_curve.png")
    loss_summary["condition_mode"] = args.condition_mode
    save_json(output_dir / "loss_summary.json", loss_summary)

    adata_control, adata_treated, loaded_conditions, control_condition_name, control_condition_vec = prepare_zebrafish_data(config)
    available_conditions = set(adata_treated.obs[config["condition_key"]].astype(str).unique().tolist())
    summary_conditions = data_summary.get("selected_conditions", loaded_conditions)
    selected_conditions = select_conditions(
        parse_condition_args(args.conditions),
        summary_conditions,
        available_conditions,
    )
    selected_timepoints = parse_timepoint_args(args.timepoints)
    target_specs = build_target_specs(
        adata_treated=adata_treated,
        condition_key=config["condition_key"],
        time_key=config["marginal_key"],
        selected_conditions=selected_conditions,
        selected_timepoints=selected_timepoints,
    )
    if not target_specs:
        raise RuntimeError("No observed (condition, timepoint) pairs were selected for evaluation.")

    in_out_dim = int(adata_control.obsm[config["scaled_rep"]].shape[1])
    condition_dim = int(adata_treated.obsm[config["condition_rep_key"]].shape[1])
    model = build_model(
        config=config,
        in_out_dim=in_out_dim,
        condition_dim=condition_dim,
        model_path=final_path,
        device=device,
    )
    ctrl_mean_latent = np.mean(
        np.asarray(adata_control.obsm[config["scaled_rep"]], dtype=np.float32),
        axis=0,
    )

    latent_rows = []
    mass_rows = []
    prediction_adatas = []
    source_obs_columns = [column for column in DEFAULT_SOURCE_OBS_COLUMNS if column in adata_control.obs.columns]

    adata_treated.obsm["__metric_rep__"] = np.asarray(adata_treated.obsm[config["scaled_rep"]], dtype=np.float32)

    grouped_target_specs = group_target_specs_by_condition(target_specs)

    for condition, target_timepoints in tqdm(
        grouped_target_specs.items(),
        desc="Running cached trajectories",
    ):
        condition_predictions = run_condition_timepoint_trajectory_inference(
            model=model,
            adata_source=adata_control,
            source_rep=config["scaled_rep"],
            adata_conditions=adata_treated,
            condition_name=condition,
            target_timepoints=target_timepoints,
            condition_keys=config["condition_key"],
            time_key=config["marginal_key"],
            embedding_key=config["condition_rep_key"],
            source_timepoint=0.0,
            n_steps=args.n_steps,
            device=device,
            condition_mode=args.condition_mode,
            control_condition_vec=control_condition_vec,
        )

        for target_timepoint in target_timepoints:
            pair_key = (condition, float(target_timepoint))
            if pair_key not in condition_predictions:
                continue
            result_dict = condition_predictions[pair_key]
            true_subset = adata_treated[
                (adata_treated.obs[config["condition_key"]].astype(str) == condition)
                & (adata_treated.obs[config["marginal_key"]].astype(float) == float(target_timepoint))
            ]
            if true_subset.n_obs == 0:
                continue

            metrics = evaluate_latent_pair(
                result_dict=result_dict,
                true_subset=true_subset,
                ctrl_mean_latent=ctrl_mean_latent,
            )
            latent_rows.append(
                {
                    "target_condition": condition,
                    "target_timepoint": float(target_timepoint),
                    "pair_id": f"{condition}|t{float(target_timepoint):g}",
                    "n_true_obs": int(true_subset.n_obs),
                    "n_pred_obs": int(result_dict["z_pred"].shape[0]),
                    **metrics,
                }
            )

            mass = np.asarray(result_dict["m_pred"]).reshape(-1)
            weights = _normalize_weights(mass)
            mass_rows.append(
                {
                    "target_condition": condition,
                    "target_timepoint": float(target_timepoint),
                    "pair_id": f"{condition}|t{float(target_timepoint):g}",
                    "n_particles": int(result_dict["z_pred"].shape[0]),
                    "mass_mean": float(mass.mean()),
                    "mass_std": float(mass.std()),
                    "mass_min": float(mass.min()),
                    "mass_max": float(mass.max()),
                    "mass_sum": float(mass.sum()),
                    "mass_effective_sample_size": float(1.0 / np.sum(weights ** 2)),
                }
            )

            pred_adata = build_latent_prediction_adata(
                source_adata=adata_control,
                z_pred=result_dict["z_pred"],
                m_pred=result_dict["m_pred"],
                target_condition=condition,
                target_timepoint=float(target_timepoint),
                source_timepoint=0.0,
                obs_columns=source_obs_columns,
                latent_key=config["scaled_rep"],
            )
            pred_adata.obs["condition_mode"] = args.condition_mode
            pred_adata.obs["control_condition_name"] = control_condition_name
            pred_adata.uns["prediction_metadata"]["condition_mode"] = args.condition_mode
            pred_adata.uns["prediction_metadata"]["control_condition_name"] = control_condition_name
            prediction_adatas.append(pred_adata)

        del condition_predictions
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if "__metric_rep__" in adata_treated.obsm:
        del adata_treated.obsm["__metric_rep__"]

    if not latent_rows:
        raise RuntimeError("No latent metrics were generated. Please check the selected condition/timepoint pairs.")

    latent_metrics = pd.DataFrame(latent_rows).sort_values(
        ["target_condition", "target_timepoint"]
    ).reset_index(drop=True)
    mass_summary = pd.DataFrame(mass_rows).sort_values(
        ["target_condition", "target_timepoint"]
    ).reset_index(drop=True)
    latent_summary = summarize_latent_metrics(latent_metrics)
    latent_summary.update(
        {
            "run_dir": str(run_dir),
            "output_dir": str(output_dir),
            "final_path": str(final_path),
            "checkpoint_path": str(checkpoint_path),
            "n_steps": int(args.n_steps),
            "device": device,
            "evaluation_mode": "condition_timepoint_cached_trajectory",
            "condition_mode": args.condition_mode,
            "control_condition_name": control_condition_name,
        }
    )

    predictions_adata = ad.concat(
        prediction_adatas,
        axis=0,
        join="outer",
        merge="same",
        index_unique=None,
    )
    predictions_adata.uns["prediction_collection"] = {
        "run_dir": str(run_dir),
        "latent_key": config["scaled_rep"],
        "source_model_timepoint": 0.0,
        "condition_key": config["condition_key"],
        "time_key": config["marginal_key"],
        "n_pairs": int(len(target_specs)),
        "n_conditions": int(len(grouped_target_specs)),
        "inference_mode": "cached_condition_trajectory",
        "n_steps_per_interval": int(args.n_steps),
        "condition_mode": args.condition_mode,
        "control_condition_name": control_condition_name,
    }

    latent_metrics.to_csv(output_dir / "latent_metrics.csv", index=False)
    mass_summary.to_csv(output_dir / "mass_summary.csv", index=False)
    save_json(output_dir / "latent_metrics_summary.json", latent_summary)
    predictions_adata.write_h5ad(output_dir / "predictions_latent.h5ad", compression="gzip")

    print(f"Saved loss summary to {output_dir / 'loss_summary.json'}")
    print(f"Saved latent metrics to {output_dir / 'latent_metrics.csv'}")
    print(f"Saved latent metric summary to {output_dir / 'latent_metrics_summary.json'}")
    print(f"Saved mass summary to {output_dir / 'mass_summary.csv'}")
    print(f"Saved latent predictions to {output_dir / 'predictions_latent.h5ad'}")


if __name__ == "__main__":
    main()
