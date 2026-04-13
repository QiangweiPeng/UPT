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

from experiments.adamson.infer_adamson_test import (  # noqa: E402
    _build_gene_prediction_adata,
    _flatten_mean,
    _instantiate_model,
    _load_json,
    _project_and_scale,
    _reconstruct_pca_block,
    _resolve_target_conditions,
    _sample_control_subset,
    save_json,
)
from src.evaluate.inference import build_latent_prediction_adata, run_batch_inference  # noqa: E402
from src.preprocessing import load_replogle_training_data  # noqa: E402


DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "results"
    / "checkpoints"
    / "replogle_k562_essential"
    / "replogle_k562_essential_replogle_k562_essential_simulation_1_0.75_X_pca_scaled_delta10_regm1_seed42_iter3000"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Replogle test-set inference from a trained run directory. "
            "Each test condition is generated at a strict 1:1 particle count matching "
            "its real test-cell count, while preserving predicted mass in the outputs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Run directory produced by experiments/replogle/train_replogle_simulation.py.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-conditions", type=int, default=None)
    parser.add_argument("--target-conditions", type=str, nargs="+", default=None)
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from src.training import seed_everything  # noqa: E402

    seed_everything(args.seed)

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_dir = args.output_dir or (run_dir / "test_inference_1to1")
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

    adata_control, _, _, adata_test, _, _ = load_replogle_training_data(
        data_path=config["data_path"],
        split_path=config["split_path"],
        condition_rep_path=config["condition_rep_path"],
        go_path=config.get("go_path"),
        condition_key=config["condition_key"],
        control_key=config["control_key"],
        control_condition=config["control_condition"],
        max_train_conditions=0,
        max_val_conditions=0,
        materialize_test=True,
        seed=args.seed,
    )

    pca_reference = ad.read_h5ad(pca_reference_path)
    scaled_rep = config["scaled_rep"]
    condition_key = config["condition_key"]
    embedding_key = config["condition_rep_key"]

    _project_and_scale(adata_control, pca_reference, scaled_rep=scaled_rep)
    _project_and_scale(adata_test, pca_reference, scaled_rep=scaled_rep)
    available_embedding_conditions = set(condition_rep_dict)
    available_test_mask = adata_test.obs[condition_key].astype(str).isin(available_embedding_conditions)
    adata_test = adata_test[available_test_mask].copy()
    if adata_test.n_obs == 0:
        raise RuntimeError("No Replogle test cells remain after filtering to conditions present in condition_embeddings.pkl.")
    adata_test.obsm[embedding_key] = np.stack(
        [condition_rep_dict[condition] for condition in adata_test.obs[condition_key].astype(str)],
        axis=0,
    ).astype(np.float32)

    target_conditions = _resolve_target_conditions(
        adata_test=adata_test,
        requested_conditions=args.target_conditions,
        max_conditions=args.max_conditions,
    )
    rng = np.random.default_rng(args.seed)

    pcs = np.asarray(pca_reference.varm["PCs"], dtype=np.float32)
    mean = _flatten_mean(pca_reference.varm["X_mean"])
    variance_sqrt = np.sqrt(np.asarray(pca_reference.uns["pca"]["variance"], dtype=np.float32))
    var_df = pca_reference.var.copy()

    latent_predictions = []
    gene_predictions = []
    summary_rows = []

    for condition in target_conditions:
        adata_target = adata_test[adata_test.obs[condition_key].astype(str) == condition].copy()
        real_n_obs = int(adata_target.n_obs)
        if real_n_obs <= 0:
            continue

        adata_source, source_sampled_with_replacement = _sample_control_subset(
            adata_control=adata_control,
            n_cells=real_n_obs,
            rng=rng,
        )
        source_tensor = torch.tensor(
            np.asarray(adata_source.obsm[scaled_rep], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        inference_results = run_batch_inference(
            model=model,
            adata_source=source_tensor,
            adata_conditions=adata_target,
            target_conditions=[condition],
            condition_keys=condition_key,
            embedding_key=embedding_key,
            source_rep=scaled_rep,
            n_steps=args.n_steps,
            device=device,
            random_seed=args.seed,
            show_progress=args.show_progress,
        )
        result = inference_results[condition]

        latent_pred_adata = build_latent_prediction_adata(
            source_adata=adata_source,
            z_pred=result["z_pred"],
            m_pred=result["m_pred"],
            target_condition=condition,
            target_timepoint=1.0,
            source_timepoint=0.0,
            obs_columns=list(adata_source.obs.columns),
            latent_key=scaled_rep,
            record_kind="predicted",
            is_model_output=True,
        )
        latent_pred_adata.obs["real_n_obs"] = real_n_obs
        latent_pred_adata.obs["pred_n_obs"] = int(latent_pred_adata.n_obs)
        latent_pred_adata.obs["source_sampled_with_replacement"] = bool(source_sampled_with_replacement)
        latent_pred_adata.obs["target_timepoint"] = "endpoint"

        mass = np.asarray(result["m_pred"], dtype=np.float32).reshape(-1)
        latent_pred_adata.obs["mass"] = mass

        X_recon = _reconstruct_pca_block(
            z_pred_scaled=result["z_pred"],
            pcs=pcs,
            mean=mean,
            variance_sqrt=variance_sqrt,
        )
        gene_pred_adata = _build_gene_prediction_adata(
            X_recon=X_recon,
            latent_pred_adata=latent_pred_adata,
            var_df=var_df,
        )

        latent_predictions.append(latent_pred_adata)
        gene_predictions.append(gene_pred_adata)
        summary_rows.append(
            {
                "condition": condition,
                "real_n_obs": real_n_obs,
                "pred_n_obs": int(latent_pred_adata.n_obs),
                "mass_sum": float(mass.sum()),
                "source_sampled_with_replacement": bool(source_sampled_with_replacement),
            }
        )

    if not latent_predictions:
        raise RuntimeError("No Replogle test predictions were produced.")

    latent_all = ad.concat(latent_predictions, join="outer", merge="same")
    gene_all = ad.concat(gene_predictions, join="outer", merge="same")

    latent_output_path = output_dir / "predictions_test_latent_1to1.h5ad"
    gene_output_path = output_dir / "predictions_test_genes_1to1.h5ad"
    summary_csv_path = output_dir / "prediction_summary_1to1.csv"
    summary_json_path = output_dir / "prediction_summary_1to1.json"

    latent_all.write_h5ad(latent_output_path)
    gene_all.write_h5ad(gene_output_path)

    summary_df = pd.DataFrame(summary_rows).sort_values("condition").reset_index(drop=True)
    summary_df.to_csv(summary_csv_path, index=False)

    summary_payload = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "n_target_conditions": int(summary_df.shape[0]),
        "total_real_n_obs": int(summary_df["real_n_obs"].sum()),
        "total_pred_n_obs": int(summary_df["pred_n_obs"].sum()),
        "all_count_match_1to1": bool((summary_df["real_n_obs"] == summary_df["pred_n_obs"]).all()),
        "any_source_sampled_with_replacement": bool(summary_df["source_sampled_with_replacement"].any()),
        "selected_conditions": summary_df["condition"].tolist(),
        "split_selected_test_conditions": split_conditions["selected_split_conditions"]["test"],
        "latent_output_path": str(latent_output_path),
        "gene_output_path": str(gene_output_path),
        "summary_csv_path": str(summary_csv_path),
    }
    save_json(summary_json_path, summary_payload)
    print(json.dumps(summary_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
