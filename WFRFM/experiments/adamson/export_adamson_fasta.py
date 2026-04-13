import argparse
import json
import sys
import time
from urllib.parse import quote
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export a merged FASTA for Adamson perturbation genes by mapping "
            "gene symbols to reviewed human UniProt entries."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--genes-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "adamson" / "perturbation_genes.csv",
        help="CSV with a 'gene' column.",
    )
    parser.add_argument(
        "--output-fasta",
        type=Path,
        default=PROJECT_ROOT / "data" / "adamson" / "perturbation_genes.fasta",
        help="Merged FASTA output path.",
    )
    parser.add_argument(
        "--mapping-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "adamson" / "perturbation_gene_uniprot_mapping.csv",
        help="CSV mapping genes to UniProt accessions and sequence metadata.",
    )
    parser.add_argument(
        "--failed-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "adamson" / "perturbation_gene_uniprot_failed.json",
        help="JSON containing lookup failures and reasons.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for each UniProt FASTA request.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Sleep between FASTA requests to avoid hammering UniProt.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    return parser.parse_args()


def ensure_writable(path, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}. Pass --overwrite to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def load_gene_list(path):
    genes_df = pd.read_csv(path)
    if "gene" not in genes_df.columns:
        raise KeyError(f"{path} must contain a 'gene' column.")
    genes = []
    seen = set()
    for gene in genes_df["gene"].astype(str):
        gene = gene.strip()
        if not gene or gene in seen:
            continue
        genes.append(gene)
        seen.add(gene)
    if not genes:
        raise ValueError(f"No genes found in {path}.")
    return genes


def choose_reviewed_human_entry(result_df):
    if result_df is None or result_df.empty:
        return None

    df = result_df.copy()
    if "Organism" in df.columns:
        df = df.loc[df["Organism"].astype(str) == "Homo sapiens (Human)"]
    if df.empty:
        return None
    if "Reviewed" in df.columns:
        reviewed_df = df.loc[df["Reviewed"].astype(str).str.lower() == "reviewed"]
        if not reviewed_df.empty:
            df = reviewed_df
    df = df.sort_values(
        by=[col for col in ["Length", "Entry"] if col in df.columns],
        ascending=[False, True][: len([col for col in ["Length", "Entry"] if col in df.columns])],
    )
    return df.iloc[0]


def search_uniprot_by_gene(session, gene, timeout):
    query = (
        f'(gene_exact:{quote(gene)} OR gene:{quote(gene)}) '
        'AND organism_id:9606 AND reviewed:true'
    )
    response = session.get(
        UNIPROT_SEARCH_URL,
        params={
            "query": query,
            "format": "json",
            "fields": "accession,id,protein_name,gene_names,organism_name,length,reviewed",
            "size": 10,
        },
        timeout=timeout,
    )
    if not response.ok:
        return None, {
            "gene": gene,
            "stage": "search",
            "status_code": response.status_code,
            "response_text": response.text[:500],
        }
    payload = response.json()
    return payload.get("results", []), None


def choose_search_result(results):
    if not results:
        return None

    normalized = []
    for result in results:
        genes = result.get("genes", [])
        gene_names = []
        for gene_obj in genes:
            if "geneName" in gene_obj and "value" in gene_obj["geneName"]:
                gene_names.append(str(gene_obj["geneName"]["value"]))
            for synonym in gene_obj.get("synonyms", []):
                if "value" in synonym:
                    gene_names.append(str(synonym["value"]))
        normalized.append(
            {
                "raw": result,
                "primaryAccession": str(result.get("primaryAccession", "")),
                "entryId": str(result.get("uniProtkbId", "")),
                "proteinName": str(
                    result.get("proteinDescription", {})
                    .get("recommendedName", {})
                    .get("fullName", {})
                    .get("value", "")
                ),
                "gene_names": gene_names,
                "length": int(result.get("sequence", {}).get("length", 0) or 0),
                "organism": str(result.get("organism", {}).get("scientificName", "")),
                "reviewed": "reviewed" if result.get("entryType", "").lower().startswith("swiss-prot") else "",
            }
        )

    normalized.sort(key=lambda item: (item["length"], item["primaryAccession"]), reverse=True)
    return normalized[0]


def fetch_gene_record(session, gene, timeout):
    results, failure = search_uniprot_by_gene(session=session, gene=gene, timeout=timeout)
    if failure is not None:
        return None, failure
    chosen = choose_search_result(results)
    if chosen is None:
        return None, {
            "gene": gene,
            "stage": "mapping",
            "n_candidates": 0 if results is None else len(results),
        }

    accession = chosen["primaryAccession"]
    fasta_response = session.get(
        UNIPROT_FASTA_URL.format(accession=accession),
        timeout=timeout,
    )
    if not fasta_response.ok:
        return None, {
            "gene": gene,
            "stage": "fasta",
            "entry": accession,
            "status_code": fasta_response.status_code,
            "response_text": fasta_response.text[:500],
        }

    lines = [line.strip() for line in fasta_response.text.splitlines() if line.strip()]
    if not lines or not lines[0].startswith(">"):
        return None, {
            "gene": gene,
            "stage": "parse_fasta",
            "entry": accession,
            "response_text": fasta_response.text[:500],
        }

    header = lines[0]
    sequence = "".join(lines[1:])
    if not sequence:
        return None, {
            "gene": gene,
            "stage": "empty_sequence",
            "entry": accession,
        }

    record = {
        "gene": gene,
        "entry": accession,
        "entry_name": chosen["entryId"],
        "protein_names": chosen["proteinName"],
        "gene_names": " ".join(chosen["gene_names"]),
        "length": chosen["length"] or len(sequence),
        "organism": chosen["organism"],
        "reviewed": chosen["reviewed"],
        "fasta_header": header,
        "sequence": sequence,
    }
    return record, None


def write_fasta(records, output_path):
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            header = (
                f">{record['gene']}|{record['entry']}"
                f"|{record['entry_name']}"
            )
            handle.write(header + "\n")
            sequence = record["sequence"]
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def main():
    args = parse_args()
    ensure_writable(args.output_fasta, args.overwrite)
    ensure_writable(args.mapping_output, args.overwrite)
    ensure_writable(args.failed_output, args.overwrite)

    genes = load_gene_list(args.genes_path)
    session = requests.Session()

    records = []
    failures = []
    for gene in genes:
        record, failure = fetch_gene_record(
            session=session,
            gene=gene,
            timeout=args.request_timeout,
        )
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
