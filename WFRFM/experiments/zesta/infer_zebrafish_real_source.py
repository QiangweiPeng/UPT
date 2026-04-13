import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate.inference import (  # noqa: E402
    VALID_CONDITION_MODES,
    build_latent_prediction_adata,
    run_condition_timepoint_trajectory_inference,
)
from src.preprocessing.zebrafish import (  # noqa: E402
    attach_condition_embeddings,
    build_zebrafish_control_condition_embedding,
    load_zebrafish_training_data,
    scale_representation_by_control,
)
from src.training.model import FNet  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run zebrafish latent inference from each condition's first real observed "
            "timepoint and save one combined AnnData."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory containing config.json, data_summary.json, and final.pt.",
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
        help="Directory used to save the combined predictions file. Defaults to <run-dir>/real_source_eval.",
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
        help=(
            "Optional observed timepoint list to keep. Source timepoint for each condition "
            "is always retained even if omitted here."
        ),
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=50,
        help="Number of Euler steps used for each observed interval during inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device. Falls back to cpu when cuda is unavailable.",
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
        return run_dir / "real_source_eval"
    return run_dir / f"real_source_eval_{_mode_suffix(condition_mode)}"


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


def resolve_real_obs_columns(data_path):
    backed = ad.read_h5ad(data_path, backed="r")
    try:
        return list(backed.obs.columns)
    finally:
        backed.file.close()


def prepare_zebrafish_data(config, source_obs_columns):
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
        extra_obs_columns=source_obs_columns,
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
    return adata_treated, selected_conditions, control_condition_name, control_condition_vec


def select_conditions(explicit_conditions, summary_conditions, available_conditions):
    if explicit_conditions is None:
        selected = list(summary_conditions)
    else:
        selected = explicit_conditions

    missing = [condition for condition in selected if condition not in available_conditions]
    if missing:
        raise KeyError(f"Unknown inference conditions: {missing}")
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


def build_condition_timepoint_map(adata_treated, condition_key, time_key, selected_conditions):
    obs_df = adata_treated.obs[[condition_key, time_key]].copy()
    obs_df[condition_key] = obs_df[condition_key].astype(str)
    obs_df[time_key] = obs_df[time_key].astype(float)
    obs_df = obs_df[obs_df[condition_key].isin(selected_conditions)].copy()

    mapping = {}
    for condition, group_df in obs_df.groupby(condition_key, sort=True):
        mapping[str(condition)] = sorted(group_df[time_key].astype(float).unique().tolist())
    return mapping


def filter_observed_timepoints(observed_timepoints, selected_timepoints):
    if selected_timepoints is None:
        return list(observed_timepoints)
    keep = [timepoint for timepoint in observed_timepoints if float(timepoint) in selected_timepoints]
    if not keep:
        return []
    return keep


def finalize_prediction_adata(
    pred_adata,
    condition_name,
    source_timepoint,
    record_kind,
    condition_mode,
    control_condition_name,
):
    pred_adata.obs["condition"] = str(condition_name)
    pred_adata.obs["source_condition"] = str(condition_name)
    pred_adata.obs["source_timepoint"] = float(source_timepoint)
    pred_adata.obs["record_kind"] = str(record_kind)
    pred_adata.obs["condition_mode"] = str(condition_mode)
    pred_adata.obs["control_condition_name"] = str(control_condition_name)
    pred_adata.uns.setdefault("prediction_metadata", {})
    pred_adata.uns["prediction_metadata"]["condition"] = str(condition_name)
    pred_adata.uns["prediction_metadata"]["source_condition"] = str(condition_name)
    pred_adata.uns["prediction_metadata"]["source_timepoint"] = float(source_timepoint)
    pred_adata.uns["prediction_metadata"]["condition_mode"] = str(condition_mode)
    pred_adata.uns["prediction_metadata"]["control_condition_name"] = str(control_condition_name)
    return pred_adata


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = resolve_mode_output_dir(run_dir, args.output_dir, args.condition_mode)
    output_dir.mkdir(parents=True, exist_ok=True)

    config, data_summary = load_run_artifacts(run_dir)
    final_path = resolve_final_path(run_dir, args.final_path)
    device = resolve_device(args.device)
    selected_timepoints = parse_timepoint_args(args.timepoints)

    source_obs_columns = resolve_real_obs_columns(config["data_path"])
    adata_treated, loaded_conditions, control_condition_name, control_condition_vec = prepare_zebrafish_data(config, source_obs_columns)
    available_conditions = set(adata_treated.obs[config["condition_key"]].astype(str).unique().tolist())
    summary_conditions = data_summary.get("selected_conditions", loaded_conditions)
    selected_conditions = select_conditions(
        parse_condition_args(args.conditions),
        summary_conditions,
        available_conditions,
    )
    condition_timepoint_map = build_condition_timepoint_map(
        adata_treated=adata_treated,
        condition_key=config["condition_key"],
        time_key=config["marginal_key"],
        selected_conditions=selected_conditions,
    )

    if not condition_timepoint_map:
        raise RuntimeError("No observed condition/timepoint pairs were selected for inference.")

    in_out_dim = int(adata_treated.obsm[config["scaled_rep"]].shape[1])
    condition_dim = int(adata_treated.obsm[config["condition_rep_key"]].shape[1])
    model = build_model(
        config=config,
        in_out_dim=in_out_dim,
        condition_dim=condition_dim,
        model_path=final_path,
        device=device,
    )

    prediction_adatas = []
    collection_rows = []
    source_timepoint_per_condition = {}

    for condition_name in tqdm(sorted(condition_timepoint_map), desc="Real-source trajectories"):
        observed_timepoints = condition_timepoint_map[condition_name]
        if not observed_timepoints:
            continue

        source_timepoint = float(min(observed_timepoints))
        source_timepoint_per_condition[condition_name] = source_timepoint
        kept_timepoints = filter_observed_timepoints(observed_timepoints, selected_timepoints)
        if float(source_timepoint) not in kept_timepoints:
            kept_timepoints = sorted(set(kept_timepoints + [float(source_timepoint)]))

        source_mask = (
            (adata_treated.obs[config["condition_key"]].astype(str) == condition_name)
            & (adata_treated.obs[config["marginal_key"]].astype(float) == source_timepoint)
        )
        source_adata = adata_treated[source_mask].copy()
        if source_adata.n_obs == 0:
            continue

        source_pred_adata = build_latent_prediction_adata(
            source_adata=source_adata,
            z_pred=np.asarray(source_adata.obsm[config["scaled_rep"]], dtype=np.float32),
            m_pred=np.ones(source_adata.n_obs, dtype=np.float32),
            target_condition=condition_name,
            target_timepoint=source_timepoint,
            source_timepoint=source_timepoint,
            obs_columns=source_obs_columns,
            latent_key=config["scaled_rep"],
            record_kind="source_observed",
            is_model_output=False,
        )
        source_pred_adata = finalize_prediction_adata(
            source_pred_adata,
            condition_name=condition_name,
            source_timepoint=source_timepoint,
            record_kind="source_observed",
            condition_mode=args.condition_mode,
            control_condition_name=control_condition_name,
        )
        prediction_adatas.append(source_pred_adata)
        collection_rows.append(
            {
                "condition": condition_name,
                "target_timepoint": source_timepoint,
                "record_kind": "source_observed",
                "n_obs": int(source_adata.n_obs),
                "is_model_output": False,
                "condition_mode": args.condition_mode,
            }
        )

        future_timepoints = [
            float(timepoint)
            for timepoint in kept_timepoints
            if float(timepoint) > source_timepoint
        ]
        if not future_timepoints:
            continue

        condition_predictions = run_condition_timepoint_trajectory_inference(
            model=model,
            adata_source=source_adata,
            source_rep=config["scaled_rep"],
            adata_conditions=adata_treated,
            condition_name=condition_name,
            target_timepoints=future_timepoints,
            condition_keys=config["condition_key"],
            time_key=config["marginal_key"],
            embedding_key=config["condition_rep_key"],
            source_timepoint=source_timepoint,
            n_steps=args.n_steps,
            device=device,
            condition_mode=args.condition_mode,
            control_condition_vec=control_condition_vec,
        )

        for target_timepoint in future_timepoints:
            pair_key = (condition_name, float(target_timepoint))
            if pair_key not in condition_predictions:
                continue
            result_dict = condition_predictions[pair_key]
            pred_adata = build_latent_prediction_adata(
                source_adata=source_adata,
                z_pred=result_dict["z_pred"],
                m_pred=result_dict["m_pred"],
                target_condition=condition_name,
                target_timepoint=float(target_timepoint),
                source_timepoint=source_timepoint,
                obs_columns=source_obs_columns,
                latent_key=config["scaled_rep"],
                record_kind="predicted_from_real_source",
                is_model_output=True,
            )
            pred_adata = finalize_prediction_adata(
                pred_adata,
                condition_name=condition_name,
                source_timepoint=source_timepoint,
                record_kind="predicted_from_real_source",
                condition_mode=args.condition_mode,
                control_condition_name=control_condition_name,
            )
            prediction_adatas.append(pred_adata)
            collection_rows.append(
                {
                    "condition": condition_name,
                    "target_timepoint": float(target_timepoint),
                    "record_kind": "predicted_from_real_source",
                    "n_obs": int(pred_adata.n_obs),
                    "is_model_output": True,
                    "condition_mode": args.condition_mode,
                }
            )

    if not prediction_adatas:
        raise RuntimeError("No predictions were generated for the selected conditions/timepoints.")

    combined_adata = ad.concat(
        prediction_adatas,
        join="outer",
        merge="same",
        uns_merge="first",
        index_unique=None,
    )
    combined_adata.uns["prediction_collection"] = {
        "source_mode": "real_first_timepoint_per_condition",
        "latent_key": config["scaled_rep"],
        "condition_key": config["condition_key"],
        "time_key": config["marginal_key"],
        "n_conditions": int(len(source_timepoint_per_condition)),
        "n_pairs": int(sum(1 for row in collection_rows if row["record_kind"] == "predicted_from_real_source")),
        "include_source_timepoint": True,
        "source_timepoint_per_condition": {
            condition: float(timepoint)
            for condition, timepoint in sorted(source_timepoint_per_condition.items())
        },
        "selected_conditions": sorted(source_timepoint_per_condition.keys()),
        "selected_timepoints_filter": (
            None if selected_timepoints is None else sorted(float(timepoint) for timepoint in selected_timepoints)
        ),
        "condition_mode": args.condition_mode,
        "control_condition_name": control_condition_name,
        "run_dir": str(run_dir),
        "model_path": str(final_path),
    }

    output_path = output_dir / "predictions_real_source_latent.h5ad"
    combined_adata.write_h5ad(output_path)

    summary_payload = {
        "output_path": str(output_path),
        "n_total_obs": int(combined_adata.n_obs),
        "n_latent_dim": int(combined_adata.n_vars),
        "n_conditions": int(len(source_timepoint_per_condition)),
        "n_records": int(len(collection_rows)),
        "condition_mode": args.condition_mode,
        "control_condition_name": control_condition_name,
        "source_timepoint_per_condition": {
            condition: float(timepoint)
            for condition, timepoint in sorted(source_timepoint_per_condition.items())
        },
    }
    save_json(output_dir / "real_source_summary.json", summary_payload)

    pd.DataFrame(collection_rows).to_csv(output_dir / "prediction_records.csv", index=False)
    print(f"Saved real-source latent predictions to {output_path}")


if __name__ == "__main__":
    main()
