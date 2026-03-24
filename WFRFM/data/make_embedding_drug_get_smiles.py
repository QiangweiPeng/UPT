import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pubchempy as pcp
from tqdm import tqdm


"""
get smiles from drug names


python make_embedding_drug_get_smiles.py \
  --input_csv target_drugs_sciplex3.csv \
  --output_json resolved_smiles_sciplex3.json \
  --drug_col drug

注意，有时候因为网络等原因没法找到全部的smiles，可以多跑几次，对于实在找不出的可以手动查一下补上
"""


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PUBCHEM_RETRY = 3
NOT_FOUND_FLAG = "NOT_FOUND_IN_PUBCHEM"


ALIAS_MAP = {
    "Glesatinib?(MGCD265)": "Glesatinib",
}



def atomic_write_json(obj, path: Path) -> None:
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def load_json_if_exists(path: Path, default):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def normalize_drug_name(name: str) -> str:
    return str(name).strip()


def candidate_names(raw_name: str) -> List[str]:
    """
    生成一组候选名字，从保守到稍微激进一些。
    先保留原名，再尝试去尾部括号、压缩空格。
    """
    name = ALIAS_MAP.get(raw_name,raw_name)
    name = normalize_drug_name(raw_name)
    cands: List[str] = []

    if not name or name.lower() == "nan":
        return cands

    cands.append(name)

    # 去掉尾部括号，例如 "Alisertib (MLN8237)" -> "Alisertib"
    cleaned = re.sub(r"\s*\(.*?\)$", "", name).strip()
    if cleaned and cleaned not in cands:
        cands.append(cleaned)

    # 压缩多余空格
    squeezed = re.sub(r"\s+", " ", cleaned).strip()
    if squeezed and squeezed not in cands:
        cands.append(squeezed)

    return cands


def pick_smiles(compound) -> Optional[str]:
    """
    尽量兼容不同版本字段名。
    PubChemPy 文档里常见 smiles / canonical_smiles / isomeric_smiles。
    """
    for attr in [
        "smiles",
        "canonical_smiles",
        "isomeric_smiles",
        "connectivity_smiles",
    ]:
        value = getattr(compound, attr, None)
        if value:
            return str(value).strip()
    return None


def query_pubchem(name: str) -> Tuple[str, Optional[str]]:
    """
    返回:
      ("FOUND", smiles)
      ("NOT_FOUND", None)
      ("TEMP_FAIL", None)
    """
    for attempt in range(PUBCHEM_RETRY):
        try:
            compounds = pcp.get_compounds(name, "name")
            if not compounds:
                return "NOT_FOUND", None

            # 取第一个命中
            smiles = pick_smiles(compounds[0])
            if smiles:
                return "FOUND", smiles

            return "NOT_FOUND", None

        except Exception as e:
            logger.warning(
                "PubChemPy temporary failure for %r on attempt %d/%d: %s",
                name,
                attempt + 1,
                PUBCHEM_RETRY,
                e,
            )
            if attempt < PUBCHEM_RETRY - 1:
                time.sleep(2 ** attempt)
                continue
            return "TEMP_FAIL", None

    return "TEMP_FAIL", None


def resolve_smiles_from_name(raw_drug_name: str) -> Tuple[str, str, Optional[str]]:
    """
    返回:
      status: FOUND / NOT_FOUND / TEMP_FAIL
      smiles_or_flag: smiles 或 NOT_FOUND_FLAG
      matched_name: 实际命中的候选名字（FOUND 时返回）
    """
    for cand in candidate_names(raw_drug_name):
        status, smiles = query_pubchem(cand)
        if status == "FOUND" and smiles:
            return "FOUND", smiles, cand
        if status == "TEMP_FAIL":
            return "TEMP_FAIL", NOT_FOUND_FLAG, None

    return "NOT_FOUND", NOT_FOUND_FLAG, None


def main():
    parser = argparse.ArgumentParser(description="Resolve drug names to SMILES using PubChemPy.")
    parser.add_argument("--input_csv", required=True, help="Input CSV path")
    parser.add_argument("--output_json", required=True, help="Output resolved drug->SMILES JSON path")
    parser.add_argument("--drug_col", default=None, help="Drug name column. Default: first column")
    parser.add_argument("--work_dir", default="smiles_cache_only", help="Working directory")
    parser.add_argument("--deduplicate", action="store_true", help="Process unique drug names only")
    parser.add_argument("--sleep_sec", type=float, default=0.25, help="Sleep between queries")
    parser.add_argument("--save_every", type=int, default=50, help="Save cache every N queries")
    parser.add_argument("--max_names", type=int, default=None, help="Only process first N names for debugging")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_json = Path(args.output_json)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    smiles_cache_path = work_dir / "smiles_cache.json"
    failed_names_path = work_dir / "failed_names.json"
    temp_failed_names_path = work_dir / "temp_failed_names.json"
    matched_name_map_path = work_dir / "matched_name_map.json"

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError("Input CSV is empty.")

    drug_col = args.drug_col if args.drug_col else df.columns[0]
    if drug_col not in df.columns:
        raise ValueError(f"Column {drug_col!r} not found. Available columns: {list(df.columns)}")

    drug_names = df[drug_col].astype(str).map(normalize_drug_name).tolist()
    drug_names = [x for x in drug_names if x and x.lower() != "nan"]

    if args.deduplicate:
        drug_names = list(dict.fromkeys(drug_names))

    if args.max_names is not None:
        drug_names = drug_names[: args.max_names]

    unique_names = list(dict.fromkeys(drug_names))

    logger.info("Input column: %s", drug_col)
    logger.info("Total rows considered: %d", len(drug_names))
    logger.info("Unique drug names: %d", len(unique_names))
    logger.info("First 10 names: %s", unique_names[:10])

    smiles_cache: Dict[str, str] = load_json_if_exists(smiles_cache_path, {})
    matched_name_map: Dict[str, str] = load_json_if_exists(matched_name_map_path, {})

    names_need_query = [n for n in unique_names if n not in smiles_cache]
    logger.info("Need query: %d", len(names_need_query))

    temp_failed_names: List[str] = []
    last_save_time = time.time()

    for i, name in enumerate(tqdm(names_need_query, desc="Resolving SMILES"), 1):
        status, smiles_or_flag, matched_name = resolve_smiles_from_name(name)

        if status == "FOUND":
            smiles_cache[name] = smiles_or_flag
            if matched_name:
                matched_name_map[name] = matched_name
        elif status == "NOT_FOUND":
            smiles_cache[name] = NOT_FOUND_FLAG
        else:
            # TEMP_FAIL: 不写死到 cache，下次还能重试
            temp_failed_names.append(name)

        now = time.time()
        if i % args.save_every == 0 or now - last_save_time > 15 or i == len(names_need_query):
            atomic_write_json(smiles_cache, smiles_cache_path)
            atomic_write_json(matched_name_map, matched_name_map_path)
            atomic_write_json(sorted(set(temp_failed_names)), temp_failed_names_path)
            last_save_time = now

        time.sleep(args.sleep_sec)

    resolved_map = {k: v for k, v in smiles_cache.items() if v != NOT_FOUND_FLAG}
    failed_names = sorted([k for k, v in smiles_cache.items() if v == NOT_FOUND_FLAG])
    temp_failed_names = sorted(set(temp_failed_names))

    atomic_write_json(resolved_map, output_json)
    atomic_write_json(failed_names, failed_names_path)
    atomic_write_json(temp_failed_names, temp_failed_names_path)
    atomic_write_json(matched_name_map, matched_name_map_path)

    logger.info("Resolved unique names: %d", len(resolved_map))
    logger.info("Not found unique names: %d", len(failed_names))
    logger.info("Temporary failed unique names: %d", len(temp_failed_names))

    if failed_names:
        logger.warning("Failed examples: %s", failed_names[:20])
    if temp_failed_names:
        logger.warning("Temp-failed examples: %s", temp_failed_names[:20])

    logger.info("Done.")


if __name__ == "__main__":
    main()
