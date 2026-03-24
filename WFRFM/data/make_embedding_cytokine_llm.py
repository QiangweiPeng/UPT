import os
import re
import json
import time
import unicodedata
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


"""
Cytokine function-text embedding pipeline (UniProt -> rich fields -> canonical English text -> embedding)
- Robust UniProt access (search candidates + entry JSON)
- Deterministic canonical text for embedding (reproducible)
- Also emits an English LLM prompt (instruction + JSON) for optional LLM formatting step

python make_embedding_cytokine_llm.py \
  --cytokines target_cytokine_PBMC.csv \
  --out processed/condition_embedding_cytokine_PBMC_e5test6.pkl \
  --batch_size 1  
"""

# ================= Config =================
#MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
MODEL_NAME = 'intfloat/e5-mistral-7b-instruct'
#MODEL_NAME = 'BAAI/bge-m3'
#MODEL_NAME = 'FremyCompany/BioLORD-2023'

MANUAL_MAPPING = {
    "LT-alpha2-beta1": "LT-alpha",
    "LT-alpha1-beta2": "LT-beta",
    "IFN-lambda2": "IFN-lambda-2",
    "IFN-lambda3": "IFN-lambda-3",
    "IL-18Ra": "IL-18R-1",
}

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{}"  # /{accession}.json
HUMAN_TAXID = "9606"
# =========================================


# ---------- HTTP session with retries ----------
def get_retry_session(
    retries: int = 6,
    backoff_factor: float = 0.8,
    status_forcelist: List[int] = [429, 500, 502, 503, 504],
) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": "cytokine-embedding/1.0 (research; contact: none)",
            "Accept": "application/json",
        }
    )
    return session


http_session = get_retry_session()


# ---------- Embedding model ----------
class TextEmbeddingConverter:
    def __init__(self, model_name: str = "intfloat/e5-mistral-7b-instruct"):
        print(f"Loading embedding model: {model_name} ...")
        
        # 显存优化：FP16
        model_kwargs = {"torch_dtype": torch.float16} if torch.cuda.is_available() else {}
        
        self.model = SentenceTransformer(model_name, model_kwargs=model_kwargs)
        
        # Mistral 支持 4096 甚至更长
        self.model.max_seq_length = 4096
        
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            print("Model moved to GPU (FP16 mode).")
        else:
            print("Model running on CPU (Warning: This will be extremely slow).")

    def encode_batch(self, 
                     texts: List[str], 
                     prefixes: List[str],  
                     device_batch_size: int = 1  # 修改：控制显卡推理时的 batch size
                     ) -> np.ndarray:
        """
        Args:
            texts: 文本列表 (Queries/Inputs)
            prefixes: 对应每条文本的 Instruction 列表
            device_batch_size: 喂给显卡的 Batch 大小 (7B 模型建议 1-4)
        """
        
        # 构造 E5-Mistral 格式: "Task: <instruction>\n\n<input>"
        # 使用 zip 确保一一对应
        formatted_inputs = []
        for p, t in zip(prefixes, texts):
            full_text = f"Instruct: {p}\nQuery: {t}"
            formatted_inputs.append(full_text)

        # 这里实际上不需要 print 太多，SentenceTransformer 自带进度条
        # print(f"Encoding {len(formatted_inputs)} items...")
        print(formatted_inputs)
        
        with torch.no_grad():
            emb = self.model.encode(
                formatted_inputs,
                batch_size=device_batch_size, # 这里控制显存占用
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False, 
            )
            
        return emb.astype(np.float32)


# ---------- Name normalization ----------
def normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKC", str(name).strip())
    # normalize dash variants
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    # collapse spaces
    s = re.sub(r"\s+", " ", s)
    return s


def apply_manual_mapping(clean_name: str) -> str:
    return MANUAL_MAPPING.get(clean_name, clean_name)


# ---------- UniProt helpers ----------
def uniprot_search_candidates(search_term: str, size: int = 5) -> List[Dict[str, Any]]:
    """
    Returns search results list (each contains at least primaryAccession and some fields if requested).
    """
    query = f'({search_term}) AND organism_id:{HUMAN_TAXID} AND reviewed:true'
    params = {
        "query": query,
        "size": size,
        "format": "json",
        # request a few lightweight fields helpful for ranking
        "fields": "accession,protein_name,gene_names,organism_id,reviewed",
    }

    resp = http_session.get(UNIPROT_SEARCH_URL, params=params, timeout=30)
    if resp.status_code != 200:
        return []
    data = resp.json()
    return data.get("results", []) or []


def pick_best_candidate(results: List[Dict[str, Any]], original_term: str) -> Optional[str]:
    """
    Pick the best accession from candidate results by simple heuristics.
    """
    if not results:
        return None

    term = original_term.lower()

    def get_gene_blob(r: Dict[str, Any]) -> str:
        genes = r.get("genes") or []
        parts = []
        for g in genes:
            gn = (g.get("geneName") or {}).get("value")
            if gn:
                parts.append(gn)
            for syn in g.get("synonyms") or []:
                v = syn.get("value")
                if v:
                    parts.append(v)
        return " ".join(parts).lower()

    def get_protein_blob(r: Dict[str, Any]) -> str:
        pdict = r.get("proteinDescription") or {}
        rec = pdict.get("recommendedName") or {}
        full = (rec.get("fullName") or {}).get("value") or ""
        # also include alternative names if any
        alts = []
        for an in pdict.get("alternativeNames") or []:
            alts.append(((an.get("fullName") or {}).get("value")) or "")
        return (" ".join([full] + alts)).lower()

    best = None
    best_score = -1

    for r in results:
        acc = r.get("primaryAccession")
        if not acc:
            continue

        gene_blob = get_gene_blob(r)
        protein_blob = get_protein_blob(r)

        score = 0
        # exact-ish match in gene names is strong
        if term and term in gene_blob:
            score += 8
        # protein name contains term
        if term and term in protein_blob:
            score += 5
        # fallback: any overlap on token
        for tok in re.split(r"[\s\-_/]+", term):
            if tok and tok in gene_blob:
                score += 2
            if tok and tok in protein_blob:
                score += 1

        if score > best_score:
            best_score = score
            best = acc

    return best or results[0].get("primaryAccession")


def fetch_uniprot_entry_json(accession: str) -> Optional[Dict[str, Any]]:
    url = UNIPROT_ENTRY_URL.format(accession)
    if not url.endswith(".json"):
        url += ".json"
    resp = http_session.get(url, timeout=30)
    if resp.status_code != 200:
        return None
    return resp.json()


# ---------- Field extraction from entry JSON ----------
def extract_entry_fields(entry: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    out["accession"] = entry.get("primaryAccession")
    out["organism_name"] = ((entry.get("organism") or {}).get("scientificName")) or "Homo sapiens"
    out["organism_id"] = ((entry.get("organism") or {}).get("taxonId")) or int(HUMAN_TAXID)

    # Protein name(s)
    pdict = entry.get("proteinDescription") or {}
    rec = pdict.get("recommendedName") or {}
    out["protein_name"] = ((rec.get("fullName") or {}).get("value")) or ""
    alt_names = []
    for an in pdict.get("alternativeNames") or []:
        v = ((an.get("fullName") or {}).get("value")) or ""
        if v:
            alt_names.append(v)
    out["protein_alt_names"] = alt_names

    # Genes
    genes = entry.get("genes") or []
    primary_gene = ""
    gene_synonyms = []
    for g in genes:
        gn = (g.get("geneName") or {}).get("value")
        if gn and not primary_gene:
            primary_gene = gn
        for syn in g.get("synonyms") or []:
            v = syn.get("value")
            if v:
                gene_synonyms.append(v)
    out["gene_primary"] = primary_gene
    out["gene_synonyms"] = sorted(list(set(gene_synonyms)))

    # Comments: FUNCTION, SUBCELLULAR LOCATION, DISEASE, etc.
    comments = entry.get("comments") or []
    functions = []
    subcell = []
    disease = []
    pathway = []
    for c in comments:
        ctype = c.get("commentType")
        if ctype == "FUNCTION":
            for t in c.get("texts") or []:
                v = t.get("value")
                if v:
                    functions.append(v.strip())
        elif ctype == "SUBCELLULAR_LOCATION":
            # locations list
            for loc in c.get("subcellularLocations") or []:
                l = ((loc.get("location") or {}).get("value")) or ""
                if l:
                    subcell.append(l.strip())
        elif ctype == "DISEASE":
            # disease might be structured
            dis = c.get("disease") or {}
            name = dis.get("diseaseId") or dis.get("diseaseAccession") or dis.get("name")
            desc = dis.get("description") or ""
            blob = " ".join([str(name or "").strip(), str(desc or "").strip()]).strip()
            if blob:
                disease.append(blob)
        elif ctype == "PATHWAY":
            for t in c.get("texts") or []:
                v = t.get("value")
                if v:
                    pathway.append(v.strip())

    out["function_texts"] = functions
    out["subcellular_locations"] = sorted(list(set(subcell)))
    out["disease_texts"] = disease
    out["pathway_texts"] = pathway

    # Keywords
    kws = []
    for k in entry.get("keywords") or []:
        v = k.get("value")
        if v:
            kws.append(v.strip())
    out["keywords"] = sorted(list(set(kws)))

    # GO terms: UniProt JSON often stores as uniProtKBCrossReferences with database "GO"
    go_bp, go_mf, go_cc = [], [], []
    for xref in entry.get("uniProtKBCrossReferences") or []:
        if xref.get("database") != "GO":
            continue
        props = {p.get("key"): p.get("value") for p in (xref.get("properties") or [])}
        term = props.get("GoTerm") or ""
        # Examples: "P:immune response", "F:cytokine activity", "C:extracellular space"
        if term.startswith("P:"):
            go_bp.append(term[2:].strip())
        elif term.startswith("F:"):
            go_mf.append(term[2:].strip())
        elif term.startswith("C:"):
            go_cc.append(term[2:].strip())

    out["go_biological_process"] = sorted(list(set(go_bp)))
    out["go_molecular_function"] = sorted(list(set(go_mf)))
    out["go_cellular_component"] = sorted(list(set(go_cc)))

    return out

from typing import Optional

PRIORITY = ["signal", "transcription", "regulation", "immune", "cytokine",
            "receptor", "inflamm", "nf-kappa", "jak", "stat", "mapk"]

RECEPTOR_PATTERNS = [
    r"\bTNFRSF\d+[A-Z0-9]*\b",
    r"\bIL\d+[A-Z]*R[A-Z0-9-]*\b",
    r"\bIFNAR[12]\b|\bIFNGR[12]\b",
    r"\bTGFBR[123]\b",
    r"\bCCR\d+\b|\bCXCR\d+\b",
]

def prioritize_terms(terms: List[str], k: int) -> Optional[str]:
    terms = [t.strip() for t in terms if t and t.strip()]
    if not terms:
        return None
    dedup = list(dict.fromkeys(terms))  # keep order
    hi = [t for t in dedup if any(p in t.lower() for p in PRIORITY)]
    lo = [t for t in dedup if t not in hi]
    return "; ".join((hi + lo)[:k])


def canonicalize_deterministic(fields: Dict[str, Any], query_name: str) -> Tuple[str, str]:
    def fmt_list(xs: List[str], max_items: int = 12) -> Optional[str]:
        xs = [x.strip() for x in xs if x and str(x).strip()]
        if not xs:
            return None
        return "; ".join(xs[:max_items])

    protein_name = (fields.get("protein_name") or "").strip() or None
    gene_primary = (fields.get("gene_primary") or "").strip() or None

    func_texts = fields.get("function_texts") or []
    func_blob = " ".join(func_texts).strip() if func_texts else ""
    if func_blob:
        func_blob = re.sub(r"\(By similarity\).*?(\.|$)", "", func_blob, flags=re.I)
        func_blob = re.sub(r"\b(May|Probably|Potentially)\b.*?(\.|$)", "", func_blob, flags=re.I)
        func_blob = re.sub(r"\s+", " ", func_blob).strip()
        sentences = re.split(r"(?<=[.!?])\s+", func_blob)
        func_blob = " ".join(sentences[:3]).strip() or None
    else:
        func_blob = None

    subcell = fmt_list(fields.get("subcellular_locations") or [], max_items=5)
    pathway = fmt_list(fields.get("pathway_texts") or [], max_items=4)
    disease = fmt_list(fields.get("disease_texts") or [], max_items=4)

    go_bp = prioritize_terms(fields.get("go_biological_process") or [], 8)
    go_mf = prioritize_terms(fields.get("go_molecular_function") or [], 6)
    keywords = prioritize_terms(fields.get("keywords") or [], 10)

    prefix = "Encode this perturbation condition into a vector for single-cell transcriptomic response prediction."

    lines = ["Condition: cytokine stimulation in PBMC"]

    if gene_primary or protein_name:
        lines.append(f"Perturbation cytokine: {gene_primary or protein_name}")
    if protein_name and gene_primary:
        lines.append(f"Protein name: {protein_name}")

    receptors = []
    if func_blob:
        for pat in RECEPTOR_PATTERNS:
            receptors += re.findall(pat, func_blob)
        receptors = sorted(set(receptors))
    if receptors:
        lines.append(f"Receptors: {'; '.join(receptors[:8])}")

    if func_blob: lines.append(f"Function: {func_blob}")
    if pathway: lines.append(f"Pathways: {pathway}")
    if subcell: lines.append(f"Localization: {subcell}")
    if go_bp: lines.append(f"GO BP: {go_bp}")
    if go_mf: lines.append(f"GO MF: {go_mf}")
    if disease:
        # keep only the first sentence / first 160 chars
        disease_short = re.split(r"(?<=[.!?])\s+", disease)[0]
        disease_short = disease_short[:160].strip()
        lines.append(f"Disease: {disease_short}")

    if keywords: lines.append(f"Keywords: {keywords}")

    return prefix, "\n".join(lines)


# ---------- English LLM input (instruction + JSON) ----------
def build_llm_input_english(fields: Dict[str, Any], query_name: str) -> str:
    """
    This string is meant to be sent to an LLM. It starts with an English instruction prompt
    and includes JSON payload. The LLM should only reformat/compact using provided facts.
    """
    instruction = (
        "You are a biomedical text formatter.\n"
        "TASK: Convert the provided JSON fields into a concise, factual, and standardized protein profile in English.\n"
        "STRICT RULES:\n"
        "1) Use ONLY the information present in the JSON. Do NOT add or infer new facts.\n"
        "2) If a field is missing/empty, write 'Unknown.' for that field.\n"
        "3) Output MUST follow the template exactly (one line per field), no extra commentary.\n"
        "4) Keep each line <= 25 words. Deduplicate repeated phrases.\n"
        "\n"
        "OUTPUT TEMPLATE:\n"
        "Entity: Cytokine protein\n"
        "Query name: <...>\n"
        "UniProt accession: <...>\n"
        "Organism: <...>\n"
        "Protein name: <...>\n"
        "Alternative protein names: <...>\n"
        "Gene (primary): <...>\n"
        "Gene synonyms: <...>\n"
        "Function: <...>\n"
        "Subcellular location: <...>\n"
        "GO Biological process: <...>\n"
        "GO Molecular function: <...>\n"
        "GO Cellular component: <...>\n"
        "Pathway: <...>\n"
        "Keywords: <...>\n"
        "Disease association: <...>\n"
    )

    payload = {
        "query_name": query_name,
        "fields": fields,
    }
    return instruction + "\n\nJSON INPUT:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


# ---------- Main pipeline per name ----------
def build_profile_texts_for_name(name: str, candidate_size: int = 5) -> Tuple[Optional[str], Dict[str, Any], str, str, str]:
    """
    Returns:
      accession (or None),
      extracted_fields (dict),
      canonical_text (deterministic),
      llm_input_en (string)
    """
    clean_name = normalize_name(name)
    search_term = apply_manual_mapping(clean_name)

    results = uniprot_search_candidates(search_term, size=candidate_size)
    accession = pick_best_candidate(results, search_term) if results else None

    fields: Dict[str, Any] = {}
    if accession:
        entry = fetch_uniprot_entry_json(accession)
        if entry:
            fields = extract_entry_fields(entry)

    # Even if UniProt failed, keep a minimal fields dict
    if not fields:
        fields = {
            "accession": None,
            "organism_name": "Homo sapiens",
            "organism_id": int(HUMAN_TAXID),
            "protein_name": "",
            "protein_alt_names": [],
            "gene_primary": "",
            "gene_synonyms": [],
            "function_texts": [],
            "subcellular_locations": [],
            "keywords": [],
            "go_biological_process": [],
            "go_molecular_function": [],
            "go_cellular_component": [],
            "pathway_texts": [],
            "disease_texts": [],
        }

    prefix,canonical_text = canonicalize_deterministic(fields, clean_name)
    llm_input_en = build_llm_input_english(fields, clean_name)
    return accession, fields, prefix,canonical_text, llm_input_en


# ---------- IO helpers ----------
def load_targets(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)
    if "cytokine" in df.columns:
        targets = df["cytokine"].dropna().unique().tolist()
    else:
        targets = pd.read_csv(csv_path, header=None)[0].dropna().unique().tolist()
    return [normalize_name(t) for t in targets]


def load_checkpoint(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        obj = pd.read_pickle(path)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


def save_checkpoint(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    pd.to_pickle(obj, path)


# ===================== CLI =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cytokines", type=str, default="target_cytokine_PBMC.csv")
    parser.add_argument("--out", type=str, default="processed/cytokine_embeddings.pkl")
    parser.add_argument("--candidate_size", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--embed_source", type=str, default="canonical")
    parser.add_argument("--batch_size", type=int, default=1, help="How many items to buffer before saving")
    args = parser.parse_args()

    # 1) Load targets
    try:
        all_targets = load_targets(args.cytokines)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        raise SystemExit(1)

    # 2) Load checkpoint
    result_dict: Dict[str, Any] = load_checkpoint(args.out)
    targets_to_process = [t for t in all_targets if t not in result_dict]

    print(f"Total: {len(all_targets)} | Processed: {len(result_dict)} | Remaining: {len(targets_to_process)}")
    if not targets_to_process:
        print("All tasks completed.")
        raise SystemExit(0)

    # 3) Load embedding model
    converter = TextEmbeddingConverter(MODEL_NAME)

    # 4) Process in mini-batches for faster embedding
    pending_names: List[str] = []
    pending_prefix: List[str] = []
    pending_texts: List[str] = []
    pending_payloads: List[Dict[str, Any]] = []

    processed_state = {"count": 0}
    
    def flush_batch():
        if not pending_names:
            return
    
        embs = converter.encode_batch(pending_texts, pending_prefix, device_batch_size=args.batch_size)
    
        for i, name in enumerate(pending_names):
            payload = pending_payloads[i]
            payload["embedding"] = embs[i]  # numpy float32 vector
            result_dict[name] = payload
            processed_state["count"] += 1
    
        # clear buffers in-place
        pending_names.clear()
        pending_prefix.clear()
        pending_texts.clear()
        pending_payloads.clear()
    
        if processed_state["count"] % args.save_interval == 0:
            save_checkpoint(result_dict, args.out)


    print("Start processing...")

    for name in tqdm(targets_to_process, desc="Embedding"):
        accession, fields, prefix, canonical_text, llm_input_en = build_profile_texts_for_name(
            name, candidate_size=args.candidate_size
        )

        # Decide what text to embed
        if args.embed_source == "canonical":
            text_to_embed = canonical_text
        elif args.embed_source == "canonical_plus_llm_input":
            text_to_embed = canonical_text + "\n\n" + llm_input_en
        else:
            # Placeholder: you can later populate llm_output by running LLM externally.
            # For now we fallback to canonical to avoid empty embeddings.
            text_to_embed = canonical_text

        payload = {
            "query_name": name,
            "mapped_query": apply_manual_mapping(normalize_name(name)),
            "accession": accession,
            "fields": fields,              # extracted structured fields
            "canonical_text": canonical_text,  # deterministic embedding input
            "llm_input_en": llm_input_en,  # English instruction+JSON for your LLM formatter
            "llm_output": None,            # fill externally if you want
            "embedded_text_source": args.embed_source,
            "embedded_text_used": text_to_embed,
        }

        pending_names.append(name)
        pending_prefix.append(prefix)
        pending_texts.append(text_to_embed)
        pending_payloads.append(payload)

        # light jitter to be nicer to UniProt if you have many entries
        time.sleep(0.05)

        # flush when enough
        if len(pending_names) >= args.batch_size:
            flush_batch()

    flush_batch()
    save_checkpoint(result_dict, args.out)
    print(f"Finished! Saved {len(result_dict)} records to {args.out}")