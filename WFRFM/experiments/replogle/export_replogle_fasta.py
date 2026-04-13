import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.adamson.export_adamson_fasta import (  # noqa: E402
    fetch_gene_record,
    load_gene_list,
    write_fasta,
    ensure_writable,
)
import json  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import time  # noqa: E402
from tqdm import tqdm  # noqa: E402

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export a merged FASTA for Replogle perturbation genes by mapping "
            "gene symbols to reviewed human UniProt entries."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--genes-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "replogle_k562_essential" / "perturbation_genes.csv",
    )
    parser.add_argument(
        "--output-fasta",
        type=Path,
        default=PROJECT_ROOT / "data" / "replogle_k562_essential" / "perturbation_genes.fasta",
    )
    parser.add_argument(
        "--mapping-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "replogle_k562_essential" / "perturbation_gene_uniprot_mapping.csv",
    )
    parser.add_argument(
        "--failed-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "replogle_k562_essential" / "perturbation_gene_uniprot_failed.json",
    )
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_writable(args.output_fasta, args.overwrite)
    ensure_writable(args.mapping_output, args.overwrite)
    ensure_writable(args.failed_output, args.overwrite)

    genes = load_gene_list(args.genes_path)
    session = requests.Session()
    records = []
    failures = []
    for gene in tqdm(genes, desc="Fetching gene records"):
        record, failure = fetch_gene_record(session=session, gene=gene, timeout=args.request_timeout)
        if record is not None:
            records.append(record)
        else:
            failures.append(failure)
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    write_fasta(records, args.output_fasta)
    mapping_df = pd.DataFrame(
        [
            {
                "gene": record["gene"],
                "entry": record["entry"],
                "entry_name": record["entry_name"],
                "protein_names": record["protein_names"],
                "gene_names": record["gene_names"],
                "length": record["length"],
                "organism": record["organism"],
                "reviewed": record["reviewed"],
                "fasta_header": record["fasta_header"],
            }
            for record in records
        ]
    )
    mapping_df.to_csv(args.mapping_output, index=False)

    summary = {
        "genes_path": str(args.genes_path),
        "output_fasta": str(args.output_fasta),
        "mapping_output": str(args.mapping_output),
        "n_input_genes": len(genes),
        "n_fasta_records": len(records),
        "n_failures": len(failures),
        "failed_genes": [failure["gene"] for failure in failures],
        "failures": failures,
    }
    with open(args.failed_output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(
        json.dumps(
            {
                "n_input_genes": len(genes),
                "n_fasta_records": len(records),
                "n_failures": len(failures),
                "output_fasta": str(args.output_fasta),
                "mapping_output": str(args.mapping_output),
                "failed_output": str(args.failed_output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
