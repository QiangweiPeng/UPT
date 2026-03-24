import argparse
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


"""
输入smiles输出embedding

python make_embedding_drug_chembert.py \
  --input_csv target_drugs_sciplex3.csv \
  --smiles_json resolved_smiles_sciplex3.json \
  --output_pkl processed/condition_embedding_drug_sciplex3_chembert.pkl \
  --drug_col drug \
  --device cuda \
  --batch_size 16
"""


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def atomic_write_pickle(obj, path: Path) -> None:
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp_path, path)


def load_pickle_if_exists(path: Path, default):
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return default


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_and_tokenizer(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def encode_smiles_batch(
    smiles_list: List[str],
    tokenizer,
    model,
    device: str,
    max_length: int = 128,
) -> np.ndarray:
    inputs = tokenizer(
        smiles_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)

    # 沿用你前面讨论过的 CLS / 首 token 表征
    emb = outputs.last_hidden_state[:, 0, :]
    return emb.detach().cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Build drug embeddings from pre-resolved SMILES.")
    parser.add_argument("--input_csv", required=True, help="Input CSV containing drug names")
    parser.add_argument("--smiles_json", required=True, help="Resolved drug->SMILES json")
    parser.add_argument("--output_pkl", required=True, help="Output pickle path")
    parser.add_argument("--drug_col", default=None, help="Drug column in CSV. Default: first column")
    parser.add_argument("--model_name", default="seyonec/ChemBERTa-zinc-base-v1")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--work_dir", default="embedding_cache", help="Progress cache directory")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    smiles_json = Path(args.smiles_json)
    output_pkl = Path(args.output_pkl)
    work_dir = Path(args.work_dir)

    work_dir.mkdir(parents=True, exist_ok=True)
    output_pkl.parent.mkdir(parents=True, exist_ok=True)

    progress_pkl_path = work_dir / "embedding_progress.pkl"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError("Input CSV is empty.")

    drug_col = args.drug_col if args.drug_col else df.columns[0]
    if drug_col not in df.columns:
        raise ValueError(f"Column {drug_col!r} not found. Available columns: {list(df.columns)}")

    drug_names = df[drug_col].astype(str).str.strip().tolist()
    drug_names = [x for x in drug_names if x and x.lower() != "nan"]

    logger.info("Total rows: %d", len(drug_names))
    logger.info("Unique drug names in CSV: %d", len(set(drug_names)))

    drug_to_smiles: Dict[str, str] = load_json(smiles_json)
    if not drug_to_smiles:
        raise RuntimeError("SMILES json is empty.")

    logger.info("Resolved drug->SMILES entries: %d", len(drug_to_smiles))

    progress = load_pickle_if_exists(
        progress_pkl_path,
        {
            "model_name": args.model_name,
            "embedding_dim": None,
            "drug_to_embedding": {},
        },
    )

    # 防止换模型后混用旧 embedding
    if progress.get("model_name") != args.model_name:
        logger.warning(
            "Cached model_name=%s != current model_name=%s, resetting embedding cache.",
            progress.get("model_name"),
            args.model_name,
        )
        progress = {
            "model_name": args.model_name,
            "embedding_dim": None,
            "drug_to_embedding": {},
        }

    embedding_dim = progress.get("embedding_dim", None)
    drug_to_embedding: Dict[str, np.ndarray] = progress.get("drug_to_embedding", {})

    unique_drugs = list(dict.fromkeys(drug_names))
    valid_unique_drugs = [d for d in unique_drugs if d in drug_to_smiles]
    missing_smiles_drugs = [d for d in unique_drugs if d not in drug_to_smiles]

    logger.info("Unique drugs with SMILES: %d", len(valid_unique_drugs))
    logger.info("Unique drugs missing SMILES: %d", len(missing_smiles_drugs))
    if missing_smiles_drugs:
        logger.warning("Missing SMILES examples: %s", missing_smiles_drugs[:20])

    pending_drugs = [d for d in valid_unique_drugs if d not in drug_to_embedding]
    logger.info("Unique drugs pending embedding: %d", len(pending_drugs))

    if pending_drugs:
        tokenizer, model = load_model_and_tokenizer(args.model_name, device)
        last_save_time = time.time()

        for start in tqdm(range(0, len(pending_drugs), args.batch_size), desc="Encoding batches"):
            batch_drugs = pending_drugs[start:start + args.batch_size]
            batch_smiles = [drug_to_smiles[d] for d in batch_drugs]

            try:
                batch_embeddings = encode_smiles_batch(
                    batch_smiles,
                    tokenizer,
                    model,
                    device,
                    max_length=args.max_length,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error("CUDA OOM. Try smaller --batch_size.")
                    torch.cuda.empty_cache()
                    raise
                raise

            if embedding_dim is None:
                embedding_dim = int(batch_embeddings.shape[1])

            for drug, emb in zip(batch_drugs, batch_embeddings):
                drug_to_embedding[drug] = emb.astype(np.float32)

            now = time.time()
            if (start + args.batch_size) >= len(pending_drugs) or (now - last_save_time > 15):
                progress_obj = {
                    "model_name": args.model_name,
                    "embedding_dim": embedding_dim,
                    "drug_to_embedding": drug_to_embedding,
                }
                atomic_write_pickle(progress_obj, progress_pkl_path)
                last_save_time = now

    final_unique_drugs = [d for d in valid_unique_drugs if d in drug_to_embedding]
    if not final_unique_drugs:
        raise RuntimeError("No valid embeddings were generated.")

    embedding_matrix = np.stack([drug_to_embedding[d] for d in final_unique_drugs], axis=0)
    control_embedding = embedding_matrix.mean(axis=0).astype(np.float32)

    # 行级展开：保留和原 CSV 对齐的 drug 顺序
    final_row_drugs = [d for d in drug_names if d in drug_to_embedding]

    final_payload = {
        "model_name": args.model_name,
        "embedding_dim": int(embedding_matrix.shape[1]),
        "drug_column": drug_col,
        "num_input_rows": len(drug_names),
        "num_unique_drugs_in_csv": len(unique_drugs),
        "num_unique_drugs_with_smiles": len(valid_unique_drugs),
        "num_unique_drugs_embedded": len(final_unique_drugs),
        "num_row_drugs_embedded": len(final_row_drugs),
        "missing_smiles_drugs": missing_smiles_drugs,
        "drug_to_smiles": {d: drug_to_smiles[d] for d in final_unique_drugs},
        "drug_to_embedding": {d: drug_to_embedding[d] for d in final_unique_drugs},
        "control_embedding": control_embedding,
    }

    atomic_write_pickle(final_payload, output_pkl)

    logger.info("Done.")
    logger.info("Saved to: %s", output_pkl)
    logger.info("Embedded unique drugs: %d", len(final_unique_drugs))
    logger.info("Embedding dim: %d", int(embedding_matrix.shape[1]))


if __name__ == "__main__":
    main()
