import argparse
import json
import pickle
import sys
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate.inference import batch_reconstruct_pca, run_batch_inference  # noqa: E402
from src.preprocessing import (  # noqa: E402
    DEFAULT_NORMAN_CONTROL_CONDITION,
    build_pca_reference_adata,
    load_norman_training_data,
)
from src.preprocessing import pp  # noqa: E402
from src.training import FNet, pre_compute_wfr_ot, seed_everything  # noqa: E402
from src.training import train as train_lib  # noqa: E402


FORMAL_NORMAN_PRESET = {
    "n_comps": 100,
    "delta": 10.0,
    "reg_m": 1.0,
    "group_number": 3,
    "n_iterations": 3000,
    "batch_size_per_condition": 256,
    "batch_size_condition": 10,
    "lr": 5e-5,
    "weight_decay": 1e-4,
    "eta_min": 5e-6,
    "save_interval": 1000,
    "eval_interval": 10,
    "hidden_dim_v": 4096,
    "hidden_dim_g": 2048,
    "n_hiddens_v": 4,
    "n_hiddens_g": 2,
    "con_embedding_dim": 512,
    "hidden_dim_con": 1024,
    "time_dim": 1024,
    "time_embedding_dim": 256,
    "hidden_dim_time": 512,
    "bottle_dim": 1024,
    "activation": "SiLU",
    "save_only_last": True,
    "use_mini_batch_uot": False,
    "run_gene_space_smoke": True,
    "smoke_source_cells": 128,
    "smoke_n_steps": 10,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Norman WFRFM using the simulation split and save PCA reconstruction artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "norman" / "perturb_processed.h5ad",
        help="Processed Norman AnnData. Training uses adata.X as the expression input.",
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "norman" / "splits" / "norman_simulation_1_0.75.pkl",
        help="Pickle containing the Norman train/val/test condition split.",
    )
    parser.add_argument(
        "--split-summary-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "norman" / "norman_simulation_1_0.75_summary.json",
        help="Optional summary JSON copied into the run metadata when present.",
    )
    parser.add_argument(
        "--condition-rep-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "norman_embeddings_filtered.pkl",
        help="Pickle mapping condition name to the Norman condition embedding vector.",
    )
    parser.add_argument(
        "--ot-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "norman",
        help="Directory where precomputed OT plans are saved.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "checkpoints" / "norman",
        help="Directory where the training run artifacts are saved.",
    )
    parser.add_argument(
        "--sample-rep",
        type=str,
        default="X_pca",
        help="Latent representation used during training. Norman currently defaults to PCA.",
    )
    parser.add_argument(
        "--scaled-rep",
        type=str,
        default="X_pca",
        help=(
            "Actual latent representation key used for OT/training. "
            "Default is 'X_pca' for unscaled PCA space. Set to 'X_pca_scaled' "
            "to recover the previous scaled-PCA training path."
        ),
    )
    parser.add_argument(
        "--condition-key",
        type=str,
        default="condition",
        help="obs column containing the perturbation condition label.",
    )
    parser.add_argument(
        "--condition-rep-key",
        type=str,
        default="gene_embeddings",
        help="obsm key used to store the condition embedding.",
    )
    parser.add_argument(
        "--control-key",
        type=str,
        default="control",
        help="obs column used to identify control cells in the source Norman h5ad.",
    )
    parser.add_argument(
        "--control-condition",
        type=str,
        default=DEFAULT_NORMAN_CONTROL_CONDITION,
        help="Condition string representing the control perturbation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--n-comps", type=int, default=FORMAL_NORMAN_PRESET["n_comps"], help="Number of PCA components.")
    parser.add_argument("--delta", type=float, default=FORMAL_NORMAN_PRESET["delta"], help="WFR geometry scale.")
    parser.add_argument("--reg-m", type=float, default=FORMAL_NORMAN_PRESET["reg_m"], help="Unbalanced OT mass regularization.")
    parser.add_argument("--group-number", type=int, default=FORMAL_NORMAN_PRESET["group_number"], help="Grouped UOT source/target partition count.")
    parser.add_argument("--n-iterations", type=int, default=FORMAL_NORMAN_PRESET["n_iterations"], help="Number of training iterations.")
    parser.add_argument("--batch-size-per-condition", type=int, default=FORMAL_NORMAN_PRESET["batch_size_per_condition"], help="Samples drawn per condition in one training step.")
    parser.add_argument("--batch-size-condition", type=int, default=FORMAL_NORMAN_PRESET["batch_size_condition"], help="How many conditions are sampled in one training step.")
    parser.add_argument("--lr", type=float, default=FORMAL_NORMAN_PRESET["lr"], help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=FORMAL_NORMAN_PRESET["weight_decay"], help="AdamW weight decay.")
    parser.add_argument("--eta-min", type=float, default=FORMAL_NORMAN_PRESET["eta_min"], help="CosineAnnealingLR minimum learning rate.")
    parser.add_argument("--save-interval", type=int, default=FORMAL_NORMAN_PRESET["save_interval"], help="Checkpoint save interval in iterations.")
    parser.add_argument("--eval-interval", type=int, default=FORMAL_NORMAN_PRESET["eval_interval"], help="Validation interval in iterations.")
    parser.add_argument("--hidden-dim-v", type=int, default=FORMAL_NORMAN_PRESET["hidden_dim_v"], help="Velocity network hidden width.")
    parser.add_argument("--hidden-dim-g", type=int, default=FORMAL_NORMAN_PRESET["hidden_dim_g"], help="Growth network hidden width.")
    parser.add_argument("--n-hiddens-v", type=int, default=FORMAL_NORMAN_PRESET["n_hiddens_v"], help="Velocity FiLM block count.")
    parser.add_argument("--n-hiddens-g", type=int, default=FORMAL_NORMAN_PRESET["n_hiddens_g"], help="Growth FiLM block count.")
    parser.add_argument("--con-embedding-dim", type=int, default=FORMAL_NORMAN_PRESET["con_embedding_dim"], help="Condition encoder output dimension.")
    parser.add_argument("--hidden-dim-con", type=int, default=FORMAL_NORMAN_PRESET["hidden_dim_con"], help="Condition encoder hidden width.")
    parser.add_argument("--time-dim", type=int, default=FORMAL_NORMAN_PRESET["time_dim"], help="Sinusoidal time embedding width.")
    parser.add_argument("--time-embedding-dim", type=int, default=FORMAL_NORMAN_PRESET["time_embedding_dim"], help="Learned time embedding width.")
    parser.add_argument("--hidden-dim-time", type=int, default=FORMAL_NORMAN_PRESET["hidden_dim_time"], help="Time MLP hidden width.")
    parser.add_argument("--bottle-dim", type=int, default=FORMAL_NORMAN_PRESET["bottle_dim"], help="FiLM bottleneck width.")
    parser.add_argument("--activation", type=str, default=FORMAL_NORMAN_PRESET["activation"], help="Model activation.")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional explicit run name. Defaults to a deterministic name derived from the split and key hyperparameters.",
    )
    parser.add_argument(
        "--recompute-ot",
        action="store_true",
        help="Force OT recomputation even when saved OT files already exist.",
    )
    parser.add_argument(
        "--use-mini-batch-uot",
        action="store_true",
        help="Use grouped mini-batch UOT instead of the notebook-style full UOT.",
    )
    parser.add_argument(
        "--save-only-last",
        action=argparse.BooleanOptionalAction,
        help="Delete older periodic checkpoints and keep only the latest rolling checkpoint.",
    )
    parser.add_argument(
        "--run-gene-space-smoke",
        action=argparse.BooleanOptionalAction,
        help="After training, run a tiny validation-condition inference and reconstruct it back to gene space.",
    )
    parser.add_argument(
        "--smoke-condition",
        type=str,
        default=None,
        help="Optional validation condition used for the gene-space smoke check. Defaults to the first val condition.",
    )
    parser.add_argument(
        "--smoke-source-cells",
        type=int,
        default=FORMAL_NORMAN_PRESET["smoke_source_cells"],
        help="Number of control source cells used in the gene-space smoke check.",
    )
    parser.add_argument(
        "--smoke-n-steps",
        type=int,
        default=FORMAL_NORMAN_PRESET["smoke_n_steps"],
        help="Euler steps used in the gene-space smoke check.",
    )
    parser.add_argument(
        "--control-max-cells",
        type=int,
        default=None,
        help="Optional cap on how many control cells are materialized. Useful for debugging or smoke tests.",
    )
    parser.add_argument(
        "--treated-max-cells-per-condition",
        type=int,
        default=None,
        help="Optional cap on how many cells are materialized for each treated condition.",
    )
    parser.add_argument(
        "--max-train-conditions",
        type=int,
        default=None,
        help="Optional cap on how many train conditions are loaded.",
    )
    parser.add_argument(
        "--max-val-conditions",
        type=int,
        default=None,
        help="Optional cap on how many val conditions are loaded.",
    )
    parser.add_argument(
        "--max-test-conditions",
        type=int,
        default=None,
        help="Optional cap on how many test conditions are kept in split metadata.",
    )
    parser.set_defaults(
        save_only_last=FORMAL_NORMAN_PRESET["save_only_last"],
        run_gene_space_smoke=FORMAL_NORMAN_PRESET["run_gene_space_smoke"],
    )
    return parser.parse_args()


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def build_run_name(args, train_rep):
    if args.run_name:
        return args.run_name

    split_name = args.split_path.stem
    return (
        f"norman_{split_name}_{train_rep}"
        f"_delta{args.delta:g}_regm{args.reg_m:g}_seed{args.seed}"
        f"_iter{args.n_iterations}"
    )


def _stringify_config_value(value):
    if isinstance(value, Path):
        return str(value)
    return value


def _sanitize_condition_name(condition_name):
    keep = []
    for char in str(condition_name):
        if char.isalnum() or char in ("-", "_"):
            keep.append(char)
        elif char == "+":
            keep.append("__")
        else:
            keep.append("_")
    return "".join(keep)


def _build_loss_summary(train_history):
    payload = {
        "n_train_steps": len(train_history["loss_list"]),
        "n_val_steps": len(train_history["test_loss_list"]),
    }
    for key in ("loss_list", "vloss_list", "gloss_list", "test_loss_list", "test_vloss_list", "test_gloss_list"):
        values = train_history.get(key, [])
        if not values:
            continue
        payload[f"{key}_last"] = float(values[-1])
        payload[f"{key}_min"] = float(min(values))
    return payload


def run_gene_space_smoke_check(
    model,
    adata_control,
    adata_val,
    pca_reference,
    run_dir,
    condition_key,
    embedding_key,
    scaled_rep,
    args,
    device,
):
    smoke_summary_path = run_dir / "gene_space_smoke_summary.json"
    if adata_val is None or adata_val.n_obs == 0:
        summary = {"ran": False, "reason": "No validation cells available for smoke reconstruction."}
        save_json(smoke_summary_path, summary)
        return summary

    available_conditions = sorted(adata_val.obs[condition_key].astype(str).unique().tolist())
    if not available_conditions:
        summary = {"ran": False, "reason": "Validation AnnData has no available conditions."}
        save_json(smoke_summary_path, summary)
        return summary

    target_condition = args.smoke_condition or available_conditions[0]
    if target_condition not in available_conditions:
        raise KeyError(
            f"Requested smoke condition '{target_condition}' is not in the validation split. "
            f"Available: {available_conditions[:10]}"
        )

    source_n = min(args.smoke_source_cells, adata_control.n_obs)
    if source_n <= 0:
        raise ValueError("No control cells are available for the smoke reconstruction.")

    rng = torch.Generator(device="cpu")
    rng.manual_seed(args.seed)
    all_indices = torch.arange(adata_control.n_obs)
    if source_n < adata_control.n_obs:
        perm = all_indices[torch.randperm(adata_control.n_obs, generator=rng)[:source_n]]
        source_indices = perm.numpy()
    else:
        source_indices = all_indices.numpy()

    adata_source = torch.tensor(
        adata_control.obsm[scaled_rep][source_indices],
        dtype=torch.float32,
        device=device,
    )
    inference_results = run_batch_inference(
        model=model,
        adata_source=adata_source,
        adata_conditions=adata_val,
        target_conditions=[target_condition],
        condition_keys=condition_key,
        embedding_key=embedding_key,
        source_rep=scaled_rep,
        n_steps=args.smoke_n_steps,
        device=device,
        random_seed=args.seed,
        show_progress=False,
    )
    reconstructed = batch_reconstruct_pca(
        inference_results=inference_results,
        ref_adata=pca_reference,
        store_as_anndata=True,
    )
    smoke_adata = reconstructed[target_condition]

    smoke_slug = _sanitize_condition_name(target_condition)
    smoke_h5ad_path = run_dir / f"gene_space_smoke_{smoke_slug}.h5ad"
    smoke_adata.write_h5ad(smoke_h5ad_path)

    latent_shape = list(inference_results[target_condition]["z_pred"].shape)
    gene_shape = list(smoke_adata.X.shape)
    summary = {
        "ran": True,
        "condition": target_condition,
        "source_n_obs": int(source_n),
        "n_steps": int(args.smoke_n_steps),
        "latent_shape": latent_shape,
        "gene_shape": gene_shape,
        "n_genes": int(smoke_adata.n_vars),
        "var_names_match_reference": bool(smoke_adata.var_names.equals(pca_reference.var_names)),
        "output_path": str(smoke_h5ad_path),
    }
    save_json(smoke_summary_path, summary)
    return summary


def main():
    args = parse_args()
    seed_everything(args.seed)

    if args.sample_rep != "X_pca":
        raise ValueError(
            "train_norman_simulation.py currently supports only --sample-rep X_pca "
            "because later gene-space reconstruction depends on saved PCA reference statistics."
        )

    train_rep = args.scaled_rep or f"{args.sample_rep}_scaled"
    run_name = build_run_name(args, train_rep)
    run_dir = args.checkpoint_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    args.ot_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_prefix = run_dir / "checkpoint"
    final_checkpoint_path = run_dir / "final.pt"
    config_path = run_dir / "config.json"
    summary_path = run_dir / "data_summary.json"
    split_conditions_path = run_dir / "split_conditions.json"
    condition_embedding_path = run_dir / "condition_embeddings.pkl"
    pca_reference_path = run_dir / "pca_reference.h5ad"
    loss_summary_path = run_dir / "loss_summary.json"
    train_ot_path = args.ot_dir / f"{run_name}_train_ot.pkl"
    val_ot_path = args.ot_dir / f"{run_name}_val_ot.pkl"

    adata_control, adata_train, adata_val, _, condition_rep_dict, split_metadata = load_norman_training_data(
        data_path=args.data_path,
        split_path=args.split_path,
        condition_rep_path=args.condition_rep_path,
        condition_key=args.condition_key,
        control_key=args.control_key,
        control_condition=args.control_condition,
        control_max_cells=args.control_max_cells,
        treated_max_cells_per_condition=args.treated_max_cells_per_condition,
        max_train_conditions=args.max_train_conditions,
        max_val_conditions=args.max_val_conditions,
        max_test_conditions=args.max_test_conditions,
        materialize_test=False,
        seed=args.seed,
    )

    adata_control.obs[args.condition_key] = adata_control.obs[args.condition_key].astype(str)
    adata_train.obs[args.condition_key] = adata_train.obs[args.condition_key].astype(str)
    adata_val.obs[args.condition_key] = adata_val.obs[args.condition_key].astype(str)

    adata_control, adata_train, adata_val, _, _, _ = pp.process_to_embedding(
        adata_control=adata_control,
        adata_train=adata_train,
        adata_test=adata_val,
        sample_rep=args.sample_rep,
        n_comps=args.n_comps,
        condition_rep_dict=condition_rep_dict,
        condition_keys=args.condition_key,
        condition_rep_keys=args.condition_rep_key,
    )

    pca_reference = build_pca_reference_adata(adata_control)
    pca_reference.write_h5ad(pca_reference_path)

    with open(condition_embedding_path, "wb") as handle:
        pickle.dump(condition_rep_dict, handle)

    config_payload = {
        key: _stringify_config_value(value)
        for key, value in vars(args).items()
    }
    config_payload.update(
        {
            "run_name": run_name,
            "scaled_rep": train_rep,
            "train_rep": train_rep,
            "train_ot_path": str(train_ot_path),
            "val_ot_path": str(val_ot_path),
            "final_checkpoint_path": str(final_checkpoint_path),
            "pca_reference_path": str(pca_reference_path),
            "condition_embedding_path": str(condition_embedding_path),
            "split_conditions_path": str(split_conditions_path),
        }
    )
    save_json(config_path, config_payload)

    split_payload = {
        "run_name": run_name,
        "control_condition": args.control_condition,
        "raw_split_conditions": split_metadata["raw_split_conditions"],
        "selected_split_conditions": split_metadata["selected_split_conditions"],
        "unassigned_treated_conditions": split_metadata["unassigned_treated_conditions"],
    }
    save_json(split_conditions_path, split_payload)

    summary_payload = {
        "run_name": run_name,
        "sample_rep": args.sample_rep,
        "scaled_rep": train_rep,
        "train_rep": train_rep,
        "n_comps": args.n_comps,
        "control_n_obs": int(adata_control.n_obs),
        "train_n_obs": int(adata_train.n_obs),
        "val_n_obs": int(adata_val.n_obs),
        "pca_reference_path": str(pca_reference_path),
        "split_cell_counts": split_metadata["split_cell_counts"],
        "materialized_n_obs": split_metadata["materialized_n_obs"],
        "selected_split_condition_counts": {
            split_name: len(conditions)
            for split_name, conditions in split_metadata["selected_split_conditions"].items()
        },
        "split_summary_path": str(args.split_summary_path),
    }
    if args.split_summary_path.exists():
        with open(args.split_summary_path, "r", encoding="utf-8") as handle:
            summary_payload["reference_split_summary"] = json.load(handle)
    save_json(summary_path, summary_payload)

    if train_ot_path.exists() and not args.recompute_ot:
        with open(train_ot_path, "rb") as handle:
            ot_results_train = pickle.load(handle)
        print(f"Loaded train OT results from {train_ot_path}")
    else:
        ot_results_train = pre_compute_wfr_ot(
            adata_control=adata_control,
            adata_treated=adata_train,
            save_path=str(train_ot_path),
            sample_rep=train_rep,
            condition_keys=args.condition_key,
            delta=args.delta,
            reg_m=args.reg_m,
            use_mini_batch_uot=args.use_mini_batch_uot,
            group_number=args.group_number,
            draw=False,
        )

    if adata_val.n_obs > 0:
        if val_ot_path.exists() and not args.recompute_ot:
            with open(val_ot_path, "rb") as handle:
                ot_results_val = pickle.load(handle)
            print(f"Loaded val OT results from {val_ot_path}")
        else:
            ot_results_val = pre_compute_wfr_ot(
                adata_control=adata_control,
                adata_treated=adata_val,
                save_path=str(val_ot_path),
                sample_rep=train_rep,
                condition_keys=args.condition_key,
                delta=args.delta,
                reg_m=args.reg_m,
                use_mini_batch_uot=args.use_mini_batch_uot,
                group_number=args.group_number,
                draw=False,
            )
    else:
        ot_results_val = None

    in_out_dim = int(adata_control.obsm[train_rep].shape[1])
    condition_dim = int(adata_train.obsm[args.condition_rep_key].shape[1])
    model = FNet(
        in_out_dim=in_out_dim,
        hidden_dim_v=args.hidden_dim_v,
        n_hiddens_v=args.n_hiddens_v,
        hidden_dim_g=args.hidden_dim_g,
        n_hiddens_g=args.n_hiddens_g,
        condition_dim=condition_dim,
        con_embedding_dim=args.con_embedding_dim,
        hidden_dim_con=args.hidden_dim_con,
        time_dim=args.time_dim,
        time_embedding_dim=args.time_embedding_dim,
        hidden_dim_time=args.hidden_dim_time,
        bottle_dim=args.bottle_dim,
        activation=args.activation,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.n_iterations,
        eta_min=args.eta_min,
    )

    train_history = train_lib.train_model(
        adata_control=adata_control,
        adata_treated=adata_train,
        adata_test=adata_val if ot_results_val is not None else None,
        ot_results_train=ot_results_train,
        ot_results_test=ot_results_val,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        n_iterations=args.n_iterations,
        batch_size_per_condition=args.batch_size_per_condition,
        batch_size_condition=args.batch_size_condition,
        sample_rep=train_rep,
        condition_keys=args.condition_key,
        condition_rep_keys=args.condition_rep_key,
        device=device,
        save_path=str(checkpoint_prefix),
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        save_only_last=args.save_only_last,
    )

    loss_summary = _build_loss_summary(train_history)
    save_json(loss_summary_path, loss_summary)

    smoke_summary = {"ran": False, "reason": "Gene-space smoke check disabled."}
    if args.run_gene_space_smoke:
        smoke_summary = run_gene_space_smoke_check(
            model=model,
            adata_control=adata_control,
            adata_val=adata_val,
            pca_reference=pca_reference,
            run_dir=run_dir,
            condition_key=args.condition_key,
            embedding_key=args.condition_rep_key,
            scaled_rep=train_rep,
            args=args,
            device=device,
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "run_name": run_name,
            "condition_key": args.condition_key,
            "condition_rep_key": args.condition_rep_key,
            "sample_rep": args.sample_rep,
            "scaled_rep": train_rep,
            "train_rep": train_rep,
            "n_comps": args.n_comps,
            "delta": args.delta,
            "reg_m": args.reg_m,
            "config_path": str(config_path),
            "summary_path": str(summary_path),
            "split_conditions_path": str(split_conditions_path),
            "condition_embedding_path": str(condition_embedding_path),
            "pca_reference_path": str(pca_reference_path),
            "train_ot_path": str(train_ot_path),
            "val_ot_path": str(val_ot_path) if ot_results_val is not None else None,
            "loss_summary_path": str(loss_summary_path),
            "gene_space_smoke_summary": smoke_summary,
            "train_history": train_history,
        },
        final_checkpoint_path,
    )
    print(f"Saved final checkpoint to {final_checkpoint_path}")


if __name__ == "__main__":
    main()
