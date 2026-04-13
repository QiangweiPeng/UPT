import argparse
import pickle
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (  # noqa: E402
    DEFAULT_REPLOGLE_CONTROL_CONDITION,
    build_replogle_condition_rep_dict,
    load_replogle_gene_rep_dict,
    load_replogle_split_conditions,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Replogle gene embedding CSV into repository pickle formats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "replogle_k562_essential" / "perturbation_genes_emb.csv",
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "replogle_k562_essential" / "splits" / "replogle_k562_essential_simulation_1_0.75.pkl",
    )
    parser.add_argument(
        "--gene-output-pkl",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "replogle_k562_essential_gene_embeddings.pkl",
    )
    parser.add_argument(
        "--condition-output-pkl",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "replogle_k562_essential_embeddings.pkl",
    )
    parser.add_argument(
        "--control-condition",
        type=str,
        default=DEFAULT_REPLOGLE_CONTROL_CONDITION,
    )
    return parser.parse_args()


def save_pickle(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def main():
    args = parse_args()
    gene_rep_dict = load_replogle_gene_rep_dict(args.input_csv)
    split_conditions = load_replogle_split_conditions(args.split_path)
    condition_names = [args.control_condition]
    for split_name in ("train", "val", "test"):
        condition_names.extend(split_conditions[split_name])

    condition_rep_dict, metadata = build_replogle_condition_rep_dict(
        condition_names=condition_names,
        gene_rep_dict=gene_rep_dict,
        control_condition=args.control_condition,
    )

    save_pickle(args.gene_output_pkl, gene_rep_dict)
    save_pickle(args.condition_output_pkl, condition_rep_dict)
    print(
        {
            "n_gene_embeddings": len(gene_rep_dict),
            "gene_embedding_dim": int(next(iter(gene_rep_dict.values())).shape[0]),
            "n_condition_embeddings": len(condition_rep_dict),
            "condition_embedding_dim": int(next(iter(condition_rep_dict.values())).shape[0]),
            "gene_output_pkl": str(args.gene_output_pkl),
            "condition_output_pkl": str(args.condition_output_pkl),
            "metadata": metadata,
        }
    )


if __name__ == "__main__":
    main()
