import argparse
import json
import pickle
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate.inference import build_latent_prediction_adata, run_batch_inference  # noqa: E402
from src.preprocessing import attach_condition_embeddings, load_norman_training_data  # noqa: E402
from src.preprocessing.pca import project_pca  # noqa: E402
from src.training import FNet, seed_everything  # noqa: E402


DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "results"
    / "checkpoints"
    / "norman"
    / "norman_norman_simulation_1_0.75_X_pca_scaled_delta10_regm1_seed42_iter3000"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Norman test-set inference from a trained run directory. "
            "Each test condition is generated at a strict 1:1 particle count matching "
            "its real test-cell count, while preserving predicted mass in the outputs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Run directory produced by experiments/norman/train_norman_simulation.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run-dir>/test_inference_1to1_<solver>_steps<n-steps>.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=50,
        help="Integration steps used during WFR inference.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="euler",
        choices=["euler", "rk4"],
        help="Latent-space ODE solver used during WFR inference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for per-condition control-cell sampling.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--max-conditions",
        type=int,
        default=None,
        help="Optional cap on the number of test conditions. Useful for debugging.",
    )
    parser.add_argument(
        "--target-conditions",
        type=str,
        nargs="+",
        default=None,
        help="Optional explicit subset of test conditions to infer.",
    )
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to show progress bars during inference.",
    )
    return parser.parse_args()


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _flatten_mean(mean_array):
    mean = np.asarray(mean_array, dtype=np.float32)
    if mean.ndim > 1:
        mean = mean.reshape(-1)
    return mean


def _project_and_prepare_rep(adata_query, pca_reference, train_rep):
    project_pca(
        query_adata=adata_query,
        ref_adata=pca_reference,
        obsm_key_added="X_pca",
    )
    if train_rep == "X_pca":
        return

    if train_rep != "X_pca_scaled":
        raise ValueError(f"Unsupported Norman train_rep during inference: {train_rep}")

    variance = np.asarray(pca_reference.uns["pca"]["variance"], dtype=np.float32)
    adata_query.obsm[train_rep] = (
        np.asarray(adata_query.obsm["X_pca"], dtype=np.float32)
        / np.sqrt(variance).astype(np.float32)
    )


def _instantiate_model(config, checkpoint, device, condition_dim):
    model = FNet(
        in_out_dim=int(config["n_comps"]),
        hidden_dim_v=int(config["hidden_dim_v"]),
        n_hiddens_v=int(config["n_hiddens_v"]),
        hidden_dim_g=int(config["hidden_dim_g"]),
        n_hiddens_g=int(config["n_hiddens_g"]),
        condition_dim=int(condition_dim),
        con_embedding_dim=int(config["con_embedding_dim"]),
        hidden_dim_con=int(config["hidden_dim_con"]),
        time_dim=int(config["time_dim"]),
        time_embedding_dim=int(config["time_embedding_dim"]),
        hidden_dim_time=int(config["hidden_dim_time"]),
        bottle_dim=int(config["bottle_dim"]),
        activation=str(config["activation"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def _resolve_target_conditions(adata_test, requested_conditions=None, max_conditions=None):
    available = adata_test.obs["condition"].astype(str).unique().tolist()
    if requested_conditions is not None:
        requested = [str(condition) for condition in requested_conditions]
        missing = sorted(set(requested).difference(available))
        if missing:
            raise KeyError(
                "Requested target conditions are not present in the test split: "
                + ", ".join(missing[:10])
            )
        selected = requested
    else:
        selected = sorted(available)

    if max_conditions is not None:
        selected = selected[:max_conditions]
    return selected


def _sample_control_subset(adata_control, n_cells, rng):
    replace = n_cells > adata_control.n_obs
    indices = rng.choice(adata_control.n_obs, size=n_cells, replace=replace)
    return adata_control[np.asarray(indices, dtype=np.int64)].copy(), replace


def _reconstruct_pca_block(z_pred, pcs, mean, variance_sqrt, train_rep):
    z_pred = np.asarray(z_pred, dtype=np.float32)
    if train_rep == "X_pca_scaled":
        z_pred = z_pred * variance_sqrt[None, :]
    elif train_rep != "X_pca":
        raise ValueError(f"Unsupported Norman train_rep during reconstruction: {train_rep}")
    return np.asarray(z_pred @ pcs.T + mean[None, :], dtype=np.float32)


def _build_gene_prediction_adata(X_recon, latent_pred_adata, var_df):
    gene_adata = ad.AnnData(
        X=np.asarray(X_recon, dtype=np.float32),
        obs=latent_pred_adata.obs.copy(),
        var=var_df.copy(),
    )
    gene_adata.uns["prediction_metadata"] = dict(latent_pred_adata.uns.get("prediction_metadata", {}))
    gene_adata.uns["prediction_metadata"]["space"] = "gene_expression"
    return gene_adata


def main():
    args = parse_args()
    seed_everything(args.seed)

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    default_output_name = f"test_inference_1to1_{args.solver}_steps{args.n_steps}"
    output_dir = args.output_dir or (run_dir / default_output_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_json(run_dir / "config.json")
    split_conditions = _load_json(run_dir / "split_conditions.json")
    final_checkpoint_path = run_dir / "final.pt"
    pca_reference_path = run_dir / "pca_reference.h5ad"
    condition_embedding_path = run_dir / "condition_embeddings.pkl"

    checkpoint = torch.load(final_checkpoint_path, map_location="cpu")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    with open(condition_embedding_path, "rb") as handle:
        condition_rep_dict = pickle.load(handle)
    first_condition = next(iter(condition_rep_dict))
    condition_dim = int(np.asarray(condition_rep_dict[first_condition]).shape[0])
    model = _instantiate_model(config, checkpoint, device, condition_dim=condition_dim)

    adata_control, _, _, adata_test, _, _ = load_norman_training_data(
        data_path=config["data_path"],
        split_path=config["split_path"],
        condition_rep_path=config["condition_rep_path"],
        condition_key=config["condition_key"],
        control_key=config["control_key"],
        control_condition=config["control_condition"],
        max_train_conditions=0,
        max_val_conditions=0,
        max_test_conditions=None,
        materialize_test=True,
        seed=args.seed,
    )
    adata_control.obs[config["condition_key"]] = adata_control.obs[config["condition_key"]].astype(str)
    adata_test.obs[config["condition_key"]] = adata_test.obs[config["condition_key"]].astype(str)

    pca_reference = ad.read_h5ad(pca_reference_path)
    train_rep = config.get("train_rep", config["scaled_rep"])
    _project_and_prepare_rep(adata_control, pca_reference, train_rep)
    _project_and_prepare_rep(adata_test, pca_reference, train_rep)
    attach_condition_embeddings(
        adata=adata_test,
        condition_rep_dict=condition_rep_dict,
        condition_key=config["condition_key"],
        embedding_key=config["condition_rep_key"],
    )

    target_conditions = _resolve_target_conditions(
        adata_test=adata_test,
        requested_conditions=args.target_conditions,
        max_conditions=args.max_conditions,
    )

    pcs = np.asarray(pca_reference.varm["PCs"], dtype=np.float32)
    mean = _flatten_mean(pca_reference.varm["X_mean"])
    variance = np.asarray(pca_reference.uns["pca"]["variance"], dtype=np.float32)
    variance_sqrt = np.sqrt(variance).astype(np.float32)

    rng = np.random.default_rng(args.seed)
    latent_parts = []
    gene_parts = []
    summary_rows = []

    iterator = target_conditions
    if args.show_progress:
        from tqdm import tqdm

        iterator = tqdm(target_conditions, desc="Norman test inference")

    obs_columns_to_copy = ["cell_type", "dose_val", config["condition_key"], config["control_key"], "condition_name"]
    for condition_name in iterator:
        real_subset = adata_test[adata_test.obs[config["condition_key"]].astype(str) == str(condition_name)].copy()
        n_real = int(real_subset.n_obs)
        if n_real == 0:
            continue

        source_subset, sampled_with_replacement = _sample_control_subset(
            adata_control=adata_control,
            n_cells=n_real,
            rng=rng,
        )
        source_tensor = torch.tensor(
            np.asarray(source_subset.obsm[train_rep], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )

        inference_results = run_batch_inference(
            model=model,
            adata_source=source_tensor,
            adata_conditions=real_subset,
            target_conditions=[condition_name],
            condition_keys=config["condition_key"],
            embedding_key=config["condition_rep_key"],
            source_rep=train_rep,
            n_steps=args.n_steps,
            solver=args.solver,
            device=device,
            random_seed=args.seed,
            show_progress=False,
        )
        result = inference_results[str(condition_name)]

        latent_pred = build_latent_prediction_adata(
            source_adata=source_subset,
            z_pred=result["z_pred"],
            m_pred=result["m_pred"],
            target_condition=str(condition_name),
            target_timepoint=1.0,
            source_timepoint=0.0,
            obs_columns=obs_columns_to_copy,
            latent_key=train_rep,
            record_kind="predicted_test_1to1",
            is_model_output=True,
        )
        latent_pred.obs["real_n_obs"] = n_real
        latent_pred.obs["pred_n_obs"] = int(latent_pred.n_obs)
        latent_pred.obs["source_sampled_with_replacement"] = bool(sampled_with_replacement)
        latent_pred.obs["generation_mode"] = "test_1to1"
        latent_pred.uns["prediction_metadata"]["real_n_obs"] = n_real
        latent_pred.uns["prediction_metadata"]["pred_n_obs"] = int(latent_pred.n_obs)
        latent_pred.uns["prediction_metadata"]["source_sampled_with_replacement"] = bool(sampled_with_replacement)
        latent_pred.uns["prediction_metadata"]["n_steps"] = int(args.n_steps)
        latent_pred.uns["prediction_metadata"]["solver"] = str(args.solver)

        X_recon = _reconstruct_pca_block(
            z_pred=result["z_pred"],
            pcs=pcs,
            mean=mean,
            variance_sqrt=variance_sqrt,
            train_rep=train_rep,
        )
        gene_pred = _build_gene_prediction_adata(
            X_recon=X_recon,
            latent_pred_adata=latent_pred,
            var_df=pca_reference.var.copy(),
        )

        latent_parts.append(latent_pred)
        gene_parts.append(gene_pred)
        mass = np.asarray(latent_pred.obs["mass"], dtype=np.float64)
        summary_rows.append(
            {
                "condition": str(condition_name),
                "real_n_obs": n_real,
                "pred_n_obs": int(latent_pred.n_obs),
                "count_match_1to1": bool(int(latent_pred.n_obs) == n_real),
                "source_sampled_with_replacement": bool(sampled_with_replacement),
                "mass_sum": float(mass.sum()),
                "mass_mean": float(mass.mean()),
                "mass_std": float(mass.std()),
                "mass_min": float(mass.min()),
                "mass_max": float(mass.max()),
            }
        )

    if not latent_parts:
        raise RuntimeError("No Norman test predictions were generated.")

    latent_combined = ad.concat(latent_parts, axis=0, join="outer", merge="same")
    latent_combined.uns["inference_metadata"] = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "mode": "test_1to1",
        "solver": str(args.solver),
        "n_steps": int(args.n_steps),
        "seed": int(args.seed),
        "target_condition_count": int(len(target_conditions)),
        "control_n_obs": int(adata_control.n_obs),
        "test_n_obs": int(adata_test.n_obs),
        "target_conditions": [str(condition) for condition in target_conditions],
    }
    latent_combined.uns["split_metadata"] = split_conditions

    gene_combined = ad.concat(gene_parts, axis=0, join="outer", merge="same")
    gene_combined.uns["inference_metadata"] = dict(latent_combined.uns["inference_metadata"])
    gene_combined.uns["split_metadata"] = split_conditions

    summary_df = pd.DataFrame(summary_rows).sort_values("condition").reset_index(drop=True)
    summary_metrics = {
        "total_conditions": int(summary_df.shape[0]),
        "total_real_n_obs": int(summary_df["real_n_obs"].sum()),
        "total_pred_n_obs": int(summary_df["pred_n_obs"].sum()),
        "all_count_match_1to1": bool(summary_df["count_match_1to1"].all()),
        "total_mass_sum": float(summary_df["mass_sum"].sum()),
    }

    latent_path = output_dir / "predictions_test_latent_1to1.h5ad"
    gene_path = output_dir / "predictions_test_genes_1to1.h5ad"
    summary_csv_path = output_dir / "prediction_summary_1to1.csv"
    summary_json_path = output_dir / "prediction_summary_1to1.json"

    latent_combined.write_h5ad(latent_path)
    gene_combined.write_h5ad(gene_path)
    summary_df.to_csv(summary_csv_path, index=False)
    save_json(
        summary_json_path,
        {
            "run_dir": str(run_dir),
            "output_dir": str(output_dir),
            "config_path": str(run_dir / "config.json"),
            "pca_reference_path": str(pca_reference_path),
            "condition_embedding_path": str(condition_embedding_path),
            "split_path": config["split_path"],
            "data_path": config["data_path"],
            "selected_target_conditions": [str(condition) for condition in target_conditions],
            "metrics": summary_metrics,
            "solver": str(args.solver),
            "n_steps": int(args.n_steps),
        },
    )

    print(f"Saved latent predictions to {latent_path}")
    print(f"Saved gene-space predictions to {gene_path}")
    print(f"Saved per-condition summary to {summary_csv_path}")


if __name__ == "__main__":
    main()
