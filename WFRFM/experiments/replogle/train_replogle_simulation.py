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

from experiments.adamson.train_adamson_simulation import (  # noqa: E402
    _build_loss_summary,
    _plot_loss_curves,
    _stringify_config_value,
    build_pca_reference_adata,
    run_gene_space_smoke_check,
    save_json,
)
from src.preprocessing import pp  # noqa: E402
from src.preprocessing import (  # noqa: E402
    DEFAULT_REPLOGLE_CONTROL_CONDITION,
    load_replogle_training_data,
)
from src.training import FNet, pre_compute_wfr_ot, seed_everything  # noqa: E402
from src.training import train as train_lib  # noqa: E402


FORMAL_REPLOGLE_PRESET = {
    "n_comps": 100,
    "delta": 10.0,
    "reg_m": 1.0,
    "group_number": 3,
    "n_iterations": 6000,
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
    "run_gene_space_smoke": True,
    "smoke_source_cells": 128,
    "smoke_n_steps": 10,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Replogle K562 essential WFRFM using the simulation split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "replogle_k562_essential" / "perturb_processed.h5ad",
        help="Processed Replogle AnnData. Training uses adata.X as the expression input.",
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "replogle_k562_essential" / "splits" / "replogle_k562_essential_simulation_1_0.75.pkl",
        help="Pickle containing the Replogle train/val/test condition split.",
    )
    parser.add_argument(
        "--split-summary-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "replogle_k562_essential" / "replogle_k562_essential_simulation_1_0.75_summary.json",
        help="Optional summary JSON copied into the run metadata when present.",
    )
    parser.add_argument(
        "--condition-rep-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "replogle_k562_essential" / "perturbation_genes_emb.csv",
        help=(
            "Replogle gene embedding file (.csv or .pkl). "
            "This dataset currently expects an explicit embedding file rather than rebuilding from GO."
        ),
    )
    parser.add_argument(
        "--ot-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "replogle_k562_essential",
        help="Directory where precomputed OT plans are saved.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "checkpoints" / "replogle_k562_essential",
        help="Directory where the training run artifacts are saved.",
    )
    parser.add_argument("--sample-rep", type=str, default="X_pca")
    parser.add_argument("--scaled-rep", type=str, default=None)
    parser.add_argument("--condition-key", type=str, default="condition")
    parser.add_argument("--condition-rep-key", type=str, default="gene_embeddings")
    parser.add_argument("--control-key", type=str, default="control")
    parser.add_argument("--control-condition", type=str, default=DEFAULT_REPLOGLE_CONTROL_CONDITION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-comps", type=int, default=FORMAL_REPLOGLE_PRESET["n_comps"])
    parser.add_argument("--delta", type=float, default=FORMAL_REPLOGLE_PRESET["delta"])
    parser.add_argument("--reg-m", type=float, default=FORMAL_REPLOGLE_PRESET["reg_m"])
    parser.add_argument("--group-number", type=int, default=FORMAL_REPLOGLE_PRESET["group_number"])
    parser.add_argument("--n-iterations", type=int, default=FORMAL_REPLOGLE_PRESET["n_iterations"])
    parser.add_argument("--batch-size-per-condition", type=int, default=FORMAL_REPLOGLE_PRESET["batch_size_per_condition"])
    parser.add_argument("--batch-size-condition", type=int, default=FORMAL_REPLOGLE_PRESET["batch_size_condition"])
    parser.add_argument("--lr", type=float, default=FORMAL_REPLOGLE_PRESET["lr"])
    parser.add_argument("--weight-decay", type=float, default=FORMAL_REPLOGLE_PRESET["weight_decay"])
    parser.add_argument("--eta-min", type=float, default=FORMAL_REPLOGLE_PRESET["eta_min"])
    parser.add_argument("--save-interval", type=int, default=FORMAL_REPLOGLE_PRESET["save_interval"])
    parser.add_argument("--eval-interval", type=int, default=FORMAL_REPLOGLE_PRESET["eval_interval"])
    parser.add_argument("--hidden-dim-v", type=int, default=FORMAL_REPLOGLE_PRESET["hidden_dim_v"])
    parser.add_argument("--hidden-dim-g", type=int, default=FORMAL_REPLOGLE_PRESET["hidden_dim_g"])
    parser.add_argument("--n-hiddens-v", type=int, default=FORMAL_REPLOGLE_PRESET["n_hiddens_v"])
    parser.add_argument("--n-hiddens-g", type=int, default=FORMAL_REPLOGLE_PRESET["n_hiddens_g"])
    parser.add_argument("--con-embedding-dim", type=int, default=FORMAL_REPLOGLE_PRESET["con_embedding_dim"])
    parser.add_argument("--hidden-dim-con", type=int, default=FORMAL_REPLOGLE_PRESET["hidden_dim_con"])
    parser.add_argument("--time-dim", type=int, default=FORMAL_REPLOGLE_PRESET["time_dim"])
    parser.add_argument("--time-embedding-dim", type=int, default=FORMAL_REPLOGLE_PRESET["time_embedding_dim"])
    parser.add_argument("--hidden-dim-time", type=int, default=FORMAL_REPLOGLE_PRESET["hidden_dim_time"])
    parser.add_argument("--bottle-dim", type=int, default=FORMAL_REPLOGLE_PRESET["bottle_dim"])
    parser.add_argument("--activation", type=str, default=FORMAL_REPLOGLE_PRESET["activation"])
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--recompute-ot", action="store_true")
    parser.add_argument("--save-only-last", action=argparse.BooleanOptionalAction)
    parser.add_argument("--run-gene-space-smoke", action=argparse.BooleanOptionalAction)
    parser.add_argument("--smoke-condition", type=str, default=None)
    parser.add_argument("--smoke-source-cells", type=int, default=FORMAL_REPLOGLE_PRESET["smoke_source_cells"])
    parser.add_argument("--smoke-n-steps", type=int, default=FORMAL_REPLOGLE_PRESET["smoke_n_steps"])
    parser.add_argument("--control-max-cells", type=int, default=None)
    parser.add_argument("--treated-max-cells-per-condition", type=int, default=None)
    parser.add_argument("--max-train-conditions", type=int, default=None)
    parser.add_argument("--max-val-conditions", type=int, default=None)
    parser.add_argument("--max-test-conditions", type=int, default=None)
    parser.set_defaults(
        save_only_last=FORMAL_REPLOGLE_PRESET["save_only_last"],
        run_gene_space_smoke=FORMAL_REPLOGLE_PRESET["run_gene_space_smoke"],
    )
    return parser.parse_args()


def build_run_name(args, scaled_rep):
    if args.run_name:
        return args.run_name
    split_name = args.split_path.stem
    return (
        f"replogle_k562_essential_{split_name}_{scaled_rep}"
        f"_delta{args.delta:g}_regm{args.reg_m:g}_seed{args.seed}"
        f"_iter{args.n_iterations}"
    )


def main():
    args = parse_args()
    seed_everything(args.seed)

    if not args.condition_rep_path.exists():
        raise FileNotFoundError(
            f"Replogle condition embedding file not found: {args.condition_rep_path}. "
            "Generate it first from the exported FASTA / embedding workflow."
        )
    if args.sample_rep != "X_pca":
        raise ValueError(
            "train_replogle_simulation.py currently supports only --sample-rep X_pca "
            "because later gene-space reconstruction depends on saved PCA reference statistics."
        )

    scaled_rep = args.scaled_rep or f"{args.sample_rep}_scaled"
    run_name = build_run_name(args, scaled_rep)
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
    loss_curve_path = run_dir / "loss_curve.png"
    loss_curve_summary_path = run_dir / "loss_curve_summary.json"
    train_ot_path = args.ot_dir / f"{run_name}_train_ot.pkl"
    val_ot_path = args.ot_dir / f"{run_name}_val_ot.pkl"

    adata_control, adata_train, adata_val, _, condition_rep_dict, split_metadata = load_replogle_training_data(
        data_path=args.data_path,
        split_path=args.split_path,
        condition_rep_path=args.condition_rep_path,
        go_path=None,
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

    config_payload = {key: _stringify_config_value(value) for key, value in vars(args).items()}
    config_payload.update(
        {
            "run_name": run_name,
            "scaled_rep": scaled_rep,
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
        "scaled_rep": scaled_rep,
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
        "embedding_metadata": split_metadata["embedding_metadata"],
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
            sample_rep=scaled_rep,
            condition_keys=args.condition_key,
            delta=args.delta,
            reg_m=args.reg_m,
            use_mini_batch_uot=False,
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
                sample_rep=scaled_rep,
                condition_keys=args.condition_key,
                delta=args.delta,
                reg_m=args.reg_m,
                use_mini_batch_uot=False,
                group_number=args.group_number,
                draw=False,
            )
    else:
        ot_results_val = None

    in_out_dim = int(adata_control.obsm[scaled_rep].shape[1])
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_iterations, eta_min=args.eta_min)

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
        sample_rep=scaled_rep,
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
    _plot_loss_curves(train_history, loss_curve_path, eval_interval=args.eval_interval)
    save_json(
        loss_curve_summary_path,
        {
            "loss_curve_path": str(loss_curve_path),
            "train_steps": len(train_history["loss_list"]),
            "val_points": len(train_history["test_loss_list"]),
        },
    )

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
            scaled_rep=scaled_rep,
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
            "scaled_rep": scaled_rep,
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
            "loss_curve_path": str(loss_curve_path),
            "gene_space_smoke_summary": smoke_summary,
            "train_history": train_history,
        },
        final_checkpoint_path,
    )
    print(f"Saved final checkpoint to {final_checkpoint_path}")


if __name__ == "__main__":
    main()
