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

from src.preprocessing.zebrafish import (  # noqa: E402
    DEFAULT_ZEBRAFISH_TIMEPOINTS,
    attach_condition_embeddings,
    load_zebrafish_training_data,
    scale_representation_by_control,
)
from src.training import FNet, pre_compute_wfr_ot_multi_marginal, seed_everything  # noqa: E402
from src.training import train as train_lib  # noqa: E402


FORMAL_ZEBRAFISH_PRESET = {
    "delta": 10.0,
    "reg_m": 1.0,
    "group_number": 16,
    "batch_save_size": 10,
    "n_iterations": 30000,
    "batch_size_per_condition": 256,
    "batch_size_condition": 4,
    "lr": 5e-5,
    "weight_decay": 1e-4,
    "save_interval": 5000,
    "eval_interval": 200,
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
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train zebrafish WFRFM with multi-marginal OT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["formal_zebrafish"],
        default="formal_zebrafish",
        help="Named parameter preset. Manual CLI arguments still override preset values.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "zesta" / "zebrafish_processed.h5ad",
        help="Input AnnData path for the processed zebrafish dataset.",
    )
    parser.add_argument(
        "--ot-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "zesta",
        help="Directory used to save the precomputed OT plans.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "checkpoints" / "zesta",
        help="Directory used to save checkpoints, config files, and the final model.",
    )
    parser.add_argument(
        "--sample-rep",
        type=str,
        default="X_pca",
        help="Embedding representation loaded from adata.obsm and used as the model state space.",
    )
    parser.add_argument(
        "--scaled-rep",
        type=str,
        default=None,
        help="Name of the scaled representation written back to adata.obsm. Defaults to '<sample_rep>_scaled'.",
    )
    parser.add_argument(
        "--condition-key",
        type=str,
        default="gene_target",
        help="obs column used as the perturbation condition identifier inside each temporal trajectory.",
    )
    parser.add_argument(
        "--condition-rep-key",
        type=str,
        default="gene_embeddings",
        help="obsm key where per-cell condition embeddings will be stored.",
    )
    parser.add_argument(
        "--marginal-key",
        type=str,
        default="timepoint",
        help="obs column that defines the ordered temporal marginals for multi-marginal OT.",
    )
    parser.add_argument(
        "--control-key",
        type=str,
        default="is_control",
        help="obs boolean column marking control cells.",
    )
    parser.add_argument(
        "--first-control-key",
        type=str,
        default="first_t_control",
        help="obs boolean column marking the earliest control subset used as the default trajectory start.",
    )
    parser.add_argument(
        "--timepoints",
        type=float,
        nargs="+",
        default=DEFAULT_ZEBRAFISH_TIMEPOINTS,
        help="Ordered timepoints used when chaining multi-marginal OT inside each condition.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for OT sampling and training.")
    parser.add_argument(
        "--delta",
        type=float,
        default=FORMAL_ZEBRAFISH_PRESET["delta"],
        help="WFR geometry scale used in OT cost and in the training interpolation formula.",
    )
    parser.add_argument(
        "--reg-m",
        type=float,
        default=FORMAL_ZEBRAFISH_PRESET["reg_m"],
        help="Unbalanced OT mass regularization strength.",
    )
    parser.add_argument(
        "--group-number",
        type=int,
        default=FORMAL_ZEBRAFISH_PRESET["group_number"],
        help="Number of source/target groups used by mini-batch UOT.",
    )
    parser.add_argument(
        "--batch-save-size",
        type=int,
        default=FORMAL_ZEBRAFISH_PRESET["batch_save_size"],
        help="How many conditions are precomputed before writing an intermediate OT batch file.",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=FORMAL_ZEBRAFISH_PRESET["n_iterations"],
        help="Number of training iterations for flow/growth matching.",
    )
    parser.add_argument(
        "--batch-size-per-condition",
        type=int,
        default=FORMAL_ZEBRAFISH_PRESET["batch_size_per_condition"],
        help="Number of OT-matched samples drawn from each condition in one training step.",
    )
    parser.add_argument(
        "--batch-size-condition",
        type=int,
        default=FORMAL_ZEBRAFISH_PRESET["batch_size_condition"],
        help="Number of conditions sampled per training step.",
    )
    parser.add_argument("--lr", type=float, default=FORMAL_ZEBRAFISH_PRESET["lr"], help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=FORMAL_ZEBRAFISH_PRESET["weight_decay"], help="AdamW weight decay.")
    parser.add_argument(
        "--save-interval",
        type=int,
        default=FORMAL_ZEBRAFISH_PRESET["save_interval"],
        help="Checkpoint save interval in training iterations.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=FORMAL_ZEBRAFISH_PRESET["eval_interval"],
        help="Validation interval in training iterations. With no test set this mostly affects logging cadence.",
    )
    parser.add_argument(
        "--max-conditions",
        type=int,
        default=None,
        help="Optional cap on the number of perturbation conditions. Useful for debugging or staged runs.",
    )
    parser.add_argument(
        "--control-max-cells",
        type=int,
        default=None,
        help="Optional cap on how many control cells are loaded into memory.",
    )
    parser.add_argument(
        "--treated-max-cells-per-group",
        type=int,
        default=None,
        help="Optional cap on cells loaded for each (condition, timepoint) group.",
    )
    parser.add_argument(
        "--hidden-dim-v",
        type=int,
        default=FORMAL_ZEBRAFISH_PRESET["hidden_dim_v"],
        help="Hidden width of the velocity network.",
    )
    parser.add_argument(
        "--hidden-dim-g",
        type=int,
        default=FORMAL_ZEBRAFISH_PRESET["hidden_dim_g"],
        help="Hidden width of the growth network.",
    )
    parser.add_argument(
        "--n-hiddens-v",
        type=int,
        default=FORMAL_ZEBRAFISH_PRESET["n_hiddens_v"],
        help="Number of FiLM residual blocks in the velocity network.",
    )
    parser.add_argument(
        "--n-hiddens-g",
        type=int,
        default=FORMAL_ZEBRAFISH_PRESET["n_hiddens_g"],
        help="Number of FiLM residual blocks in the growth network.",
    )
    parser.add_argument(
        "--con-embedding-dim",
        type=int,
        default=FORMAL_ZEBRAFISH_PRESET["con_embedding_dim"],
        help="Condition encoder output dimension fed into FiLM modulation.",
    )
    parser.add_argument(
        "--hidden-dim-con",
        type=int,
        default=FORMAL_ZEBRAFISH_PRESET["hidden_dim_con"],
        help="Hidden width of the condition encoder MLP.",
    )
    parser.add_argument(
        "--time-dim",
        type=int,
        default=FORMAL_ZEBRAFISH_PRESET["time_dim"],
        help="Raw sinusoidal time encoding dimension before the learned time MLP.",
    )
    parser.add_argument(
        "--time-embedding-dim",
        type=int,
        default=FORMAL_ZEBRAFISH_PRESET["time_embedding_dim"],
        help="Final learned time embedding dimension fed into FiLM modulation.",
    )
    parser.add_argument(
        "--hidden-dim-time",
        type=int,
        default=FORMAL_ZEBRAFISH_PRESET["hidden_dim_time"],
        help="Hidden width of the time encoder MLP.",
    )
    parser.add_argument(
        "--bottle-dim",
        type=int,
        default=FORMAL_ZEBRAFISH_PRESET["bottle_dim"],
        help="Bottleneck width inside each FiLM modulation block.",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=FORMAL_ZEBRAFISH_PRESET["activation"],
        help="Nonlinearity used across the model.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional explicit run name. If omitted, the script builds one from the key hyperparameters.",
    )
    parser.add_argument(
        "--recompute-ot",
        action="store_true",
        help="Force OT recomputation even if a saved OT file already exists.",
    )
    parser.add_argument(
        "--disable-mini-batch-uot",
        action="store_true",
        help="Use full UOT instead of grouped mini-batch UOT during OT precomputation.",
    )
    parser.add_argument(
        "--use-all-control",
        action="store_true",
        help="Use all control cells instead of the earliest control subset marked by first_t_control.",
    )
    parser.add_argument(
        "--save-only-last",
        action=argparse.BooleanOptionalAction,
        help="Whether to delete older periodic checkpoints and keep only the latest rolling checkpoint.",
    )
    parser.set_defaults(save_only_last=FORMAL_ZEBRAFISH_PRESET["save_only_last"])
    args = parser.parse_args()
    apply_preset(args, parser)
    return args


def apply_preset(args, parser):
    preset_name = getattr(args, "preset", None)
    if preset_name is None:
        return args

    if preset_name != "formal_zebrafish":
        parser.error(f"Unknown preset: {preset_name}")

    for key, value in FORMAL_ZEBRAFISH_PRESET.items():
        if parser.get_default(key) == getattr(args, key):
            setattr(args, key, value)
    return args


def build_run_name(args):
    if args.run_name:
        return args.run_name

    rep_name = args.scaled_rep or f"{args.sample_rep}_scaled"
    control_name = "all_control" if args.use_all_control else "first_t_control"
    suffix = f"_cond{args.max_conditions}" if args.max_conditions is not None else ""
    return (
        f"zesta_multimarginal_{control_name}_{rep_name}"
        f"_delta{args.delta:g}_regm{args.reg_m:g}_seed{args.seed}{suffix}"
    )


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main():
    args = parse_args()
    seed_everything(args.seed)

    run_name = build_run_name(args)
    run_dir = args.checkpoint_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    args.ot_dir.mkdir(parents=True, exist_ok=True)

    scaled_rep = args.scaled_rep or f"{args.sample_rep}_scaled"
    checkpoint_prefix = run_dir / "checkpoint"
    ot_save_path = args.ot_dir / f"{run_name}_ot.pkl"
    condition_embedding_path = run_dir / "condition_embeddings.pkl"
    final_checkpoint_path = run_dir / "final.pt"
    config_path = run_dir / "config.json"
    summary_path = run_dir / "data_summary.json"

    adata_control, adata_treated, condition_rep_dict, selected_conditions = load_zebrafish_training_data(
        data_path=args.data_path,
        rep_keys=[args.sample_rep],
        condition_key=args.condition_key,
        marginal_key=args.marginal_key,
        control_key=args.control_key,
        first_control_key=args.first_control_key,
        use_first_control=not args.use_all_control,
        max_conditions=args.max_conditions,
        control_max_cells=args.control_max_cells,
        treated_max_cells_per_group=args.treated_max_cells_per_group,
        seed=args.seed,
    )

    scale_representation_by_control(
        adata_control=adata_control,
        adata_treated=adata_treated,
        source_rep=args.sample_rep,
        target_rep=scaled_rep,
    )
    attach_condition_embeddings(
        adata_treated,
        condition_rep_dict=condition_rep_dict,
        condition_key=args.condition_key,
        embedding_key=args.condition_rep_key,
    )

    with open(condition_embedding_path, "wb") as handle:
        pickle.dump(condition_rep_dict, handle)

    config_payload = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    config_payload["run_name"] = run_name
    config_payload["scaled_rep"] = scaled_rep
    config_payload["ot_save_path"] = str(ot_save_path)
    config_payload["final_checkpoint_path"] = str(final_checkpoint_path)
    save_json(config_path, config_payload)

    summary_payload = {
        "run_name": run_name,
        "selected_conditions": selected_conditions,
        "n_selected_conditions": len(selected_conditions),
        "control_n_obs": int(adata_control.n_obs),
        "treated_n_obs": int(adata_treated.n_obs),
        "sample_rep": args.sample_rep,
        "scaled_rep": scaled_rep,
        "timepoints": list(args.timepoints),
    }
    save_json(summary_path, summary_payload)

    if ot_save_path.exists() and not args.recompute_ot:
        with open(ot_save_path, "rb") as handle:
            ot_results_train = pickle.load(handle)
        print(f"Loaded OT results from {ot_save_path}")
    else:
        ot_results_train = pre_compute_wfr_ot_multi_marginal(
            adata_control=adata_control,
            adata_treated=adata_treated,
            save_path=str(ot_save_path),
            marginal_key=args.marginal_key,
            sample_rep=scaled_rep,
            condition_keys=args.condition_key,
            marginal_order=list(args.timepoints),
            control_time=0.0,
            delta=args.delta,
            reg_m=args.reg_m,
            use_mini_batch_uot=not args.disable_mini_batch_uot,
            group_number=args.group_number,
            batch_save_size=args.batch_save_size,
            draw=False,
        )

    in_out_dim = int(adata_control.obsm[scaled_rep].shape[1])
    condition_dim = int(adata_treated.obsm[args.condition_rep_key].shape[1])
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
        eta_min=max(args.lr * 0.1, 1e-6),
    )

    train_lib.train_model(
        adata_control=adata_control,
        adata_treated=adata_treated,
        adata_test=None,
        ot_results_train=ot_results_train,
        ot_results_test=None,
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

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "run_name": run_name,
            "condition_key": args.condition_key,
            "condition_rep_key": args.condition_rep_key,
            "marginal_key": args.marginal_key,
            "sample_rep": args.sample_rep,
            "scaled_rep": scaled_rep,
            "timepoints": list(args.timepoints),
            "ot_save_path": str(ot_save_path),
            "config_path": str(config_path),
            "summary_path": str(summary_path),
        },
        final_checkpoint_path,
    )
    print(f"Saved final checkpoint to {final_checkpoint_path}")


if __name__ == "__main__":
    main()
