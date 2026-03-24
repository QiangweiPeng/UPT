import os
import re
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
Gene embedding pipeline
NCBI Gene summary (main text) + UniProt protein/function/GO (supplement)
Designed for gene perturbation embedding (GenePT-like), especially for human genes.

Main flow:
gene symbol
-> NCBI Gene: official symbol / aliases / summary
-> UniProt: protein name / function / GO / localization / pathway
-> deterministic canonical English text
-> embedding

python gene_embedding.py \
  --genes target_genes.csv \
  --out processed/gene_embeddings.pkl \
  --candidate_size 5 \
  --save_interval 10 \
  --batch_size 1


"""


# ================= Config =================
MODEL_NAME = "intfloat/e5-mistral-7b-instruct"

HUMAN_TAXID = "9606"

# NCBI E-utilities
NCBI_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

# UniProt
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{}"  # accession + ".json"

# Optional: set these env vars if you have them
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
NCBI_TOOL = os.getenv("NCBI_TOOL", "gene-embedding")
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")

# Optional manual alias -> preferred symbol mapping
MANUAL_GENE_MAPPING: Dict[str, str] = {
    # "PD-1": "PDCD1",
    # "4-1BB": "TNFRSF9",
    # "ICOSL": "ICOSLG",
}
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
            "User-Agent": "gene-embedding/1.0 (research)",
            "Accept": "application/json",
        }
    )
    return session


http_session = get_retry_session()


# ---------- Embedding model ----------
class TextEmbeddingConverter:
    def __init__(self, model_name: str = MODEL_NAME):
        print(f"Loading embedding model: {model_name} ...")

        model_kwargs = {"torch_dtype": torch.float16} if torch.cuda.is_available() else {}
        self.model = SentenceTransformer(model_name, model_kwargs=model_kwargs)
        self.model.max_seq_length = 4096

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            print("Model moved to GPU (FP16 mode).")
        else:
            print("Model running on CPU (Warning: This may be slow).")

    def encode_batch(
        self,
        texts: List[str],
        prefixes: List[str],
        device_batch_size: int = 1,
    ) -> np.ndarray:
        """
        Encode texts using E5-Mistral style formatting.

        Args:
            texts: canonical text list
            prefixes: instruction list
            device_batch_size: actual GPU/CPU batch size for encoding
        """
        formatted_inputs = [
            f"Instruct: {p}\nQuery: {t}"
            for p, t in zip(prefixes, texts)
        ]
        print(formatted_inputs)

        with torch.no_grad():
            emb = self.model.encode(
                formatted_inputs,
                batch_size=device_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

        return emb.astype(np.float32)


# ---------- Name normalization ----------
def normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKC", str(name).strip())
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s = re.sub(r"\s+", " ", s)
    return s


def apply_manual_mapping(clean_name: str) -> str:
    return MANUAL_GENE_MAPPING.get(clean_name, clean_name)


# ---------- Generic helpers ----------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def dedup_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        x = str(x).strip()
        if not x or x in seen:
            continue
        out.append(x)
        seen.add(x)
    return out


def fmt_list(xs: List[str], max_items: int = 15) -> Optional[str]:
    xs = dedup_keep_order(xs)
    if not xs:
        return None
    return ", ".join(xs[:max_items])


def sentence_truncate(text: str, max_sentences: int = 5) -> str:
    text = clean_text(text)
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentences[:max_sentences]).strip()


def safe_get(d: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


# ---------- NCBI helpers ----------
def _ncbi_common_params() -> Dict[str, str]:
    params = {"retmode": "json", "tool": NCBI_TOOL}
    if NCBI_EMAIL:
        params["email"] = NCBI_EMAIL
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    return params


def ncbi_search_gene_id(symbol: str, taxid: str = HUMAN_TAXID) -> Optional[str]:
    """
    Search NCBI Gene ID by human gene symbol.
    """
    symbol = normalize_name(symbol)

    # Exact-ish gene symbol search within species
    term_candidates = [
        f'{symbol}[Gene Name] AND txid{taxid}[Organism]',
        f'{symbol}[Symbol] AND txid{taxid}[Organism]',
        f'{symbol} AND txid{taxid}[Organism]',
    ]

    for term in term_candidates:
        params = {
            **_ncbi_common_params(),
            "db": "gene",
            "term": term,
            "retmax": 3,
        }
        try:
            resp = http_session.get(NCBI_ESEARCH_URL, params=params, timeout=30)
            if resp.status_code != 200:
                continue
            data = resp.json()
            idlist = safe_get(data, "esearchresult", "idlist", default=[])
            if idlist:
                return str(idlist[0])
        except Exception:
            continue

    return None


def ncbi_fetch_gene_summary(gene_id: str) -> Dict[str, Any]:
    """
    Fetch NCBI Gene summary and basic metadata from ESummary.
    """
    if not gene_id:
        return {}

    params = {
        **_ncbi_common_params(),
        "db": "gene",
        "id": str(gene_id),
    }

    try:
        resp = http_session.get(NCBI_ESUMMARY_URL, params=params, timeout=30)
        if resp.status_code != 200:
            return {}
        data = resp.json()
    except Exception:
        return {}

    result = (data or {}).get("result") or {}
    uid_block = result.get(str(gene_id)) or {}

    # ESummary field names can vary a bit, so make parsing tolerant
    symbol = (
        uid_block.get("name")
        or uid_block.get("nomenclaturesymbol")
        or uid_block.get("symbol")
        or ""
    )
    description = (
        uid_block.get("description")
        or uid_block.get("nomenclaturename")
        or ""
    )
    summary = uid_block.get("summary") or ""

    organism_name = safe_get(uid_block, "organism", "scientificname", default="Homo sapiens")

    alias_candidates: List[str] = []
    for key in ["otheraliases", "otherdesignations"]:
        v = uid_block.get(key)
        if isinstance(v, str) and v.strip():
            # NCBI often uses comma-separated or pipe/semicolon-ish formats
            alias_candidates.extend(re.split(r"[;,|]", v))

    aliases = dedup_keep_order([x.strip() for x in alias_candidates if x.strip()])

    out = {
        "gene_id": str(gene_id),
        "symbol": symbol,
        "description": description,
        "summary": summary,
        "organism": organism_name,
        "aliases": aliases,
    }
    return out


# ---------- UniProt helpers ----------
def _run_uniprot_query(query: str, size: int = 5) -> List[Dict[str, Any]]:
    params = {
        "query": query,
        "size": size,
        "format": "json",
        "fields": "accession,protein_name,gene_names,organism_name,reviewed",
    }

    try:
        resp = http_session.get(UNIPROT_SEARCH_URL, params=params, timeout=30)
        if resp.status_code != 200:
            return []
        data = resp.json()
        return data.get("results", []) or []
    except Exception:
        return []


def uniprot_search_candidates_for_gene(search_term: str, size: int = 5) -> List[Dict[str, Any]]:
    """
    Prefer gene-specific queries over free-text.
    """
    search_term = normalize_name(search_term)

    queries = [
        f"gene_exact:{search_term} AND organism_id:{HUMAN_TAXID} AND reviewed:true",
        f"gene:{search_term} AND organism_id:{HUMAN_TAXID} AND reviewed:true",
        f"({search_term}) AND organism_id:{HUMAN_TAXID} AND reviewed:true",
    ]

    for query in queries:
        results = _run_uniprot_query(query, size=size)
        if results:
            return results

    return []


def pick_best_candidate(results: List[Dict[str, Any]], original_symbol: str) -> Optional[str]:
    """
    Pick the best UniProt accession based on gene name matching heuristics.
    """
    if not results:
        return None

    symbol = original_symbol.lower()

    def get_gene_names(r: Dict[str, Any]) -> Tuple[str, List[str]]:
        genes = r.get("genes") or []
        primary = ""
        syns = []
        for g in genes:
            gn = safe_get(g, "geneName", "value", default="")
            if gn and not primary:
                primary = gn
            for syn in g.get("synonyms") or []:
                v = syn.get("value")
                if v:
                    syns.append(v)
        return primary.lower(), [x.lower() for x in syns]

    def get_protein_blob(r: Dict[str, Any]) -> str:
        pdict = r.get("proteinDescription") or {}
        rec = pdict.get("recommendedName") or {}
        full = safe_get(rec, "fullName", "value", default="")
        alts = []
        for an in pdict.get("alternativeNames") or []:
            v = safe_get(an, "fullName", "value", default="")
            if v:
                alts.append(v)
        return " ".join([full] + alts).lower()

    best_acc = None
    best_score = -1

    for r in results:
        acc = r.get("primaryAccession")
        if not acc:
            continue

        primary_gene, gene_syns = get_gene_names(r)
        protein_blob = get_protein_blob(r)

        score = 0
        if primary_gene == symbol:
            score += 20
        if symbol in gene_syns:
            score += 12
        if symbol in protein_blob:
            score += 3

        for tok in re.split(r"[\s\-_/]+", symbol):
            if tok and tok in primary_gene:
                score += 2
            if tok and any(tok in s for s in gene_syns):
                score += 1

        if score > best_score:
            best_score = score
            best_acc = acc

    return best_acc or results[0].get("primaryAccession")


def fetch_uniprot_entry_json(accession: str) -> Optional[Dict[str, Any]]:
    if not accession:
        return None

    url = UNIPROT_ENTRY_URL.format(accession)
    if not url.endswith(".json"):
        url += ".json"

    try:
        resp = http_session.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


# ---------- Field extraction from UniProt entry ----------
def extract_uniprot_entry_fields(entry: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    out["accession"] = entry.get("primaryAccession")
    out["organism_name"] = safe_get(entry, "organism", "scientificName", default="Homo sapiens")
    out["organism_id"] = safe_get(entry, "organism", "taxonId", default=int(HUMAN_TAXID))

    # Protein names
    pdict = entry.get("proteinDescription") or {}
    rec = pdict.get("recommendedName") or {}
    out["protein_name"] = safe_get(rec, "fullName", "value", default="")

    alt_names = []
    for an in pdict.get("alternativeNames") or []:
        v = safe_get(an, "fullName", "value", default="")
        if v:
            alt_names.append(v)
    out["protein_alt_names"] = dedup_keep_order(alt_names)

    # Gene names
    genes = entry.get("genes") or []
    primary_gene = ""
    gene_synonyms = []
    for g in genes:
        gn = safe_get(g, "geneName", "value", default="")
        if gn and not primary_gene:
            primary_gene = gn
        for syn in g.get("synonyms") or []:
            v = syn.get("value")
            if v:
                gene_synonyms.append(v)
    out["gene_primary"] = primary_gene
    out["gene_synonyms"] = dedup_keep_order(gene_synonyms)

    # Comments
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
            for loc in c.get("subcellularLocations") or []:
                l = safe_get(loc, "location", "value", default="")
                if l:
                    subcell.append(l.strip())

        elif ctype == "DISEASE":
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

    out["function_texts"] = dedup_keep_order(functions)
    out["subcellular_locations"] = dedup_keep_order(subcell)
    out["disease_texts"] = dedup_keep_order(disease)
    out["pathway_texts"] = dedup_keep_order(pathway)

    # Keywords
    kws = []
    for k in entry.get("keywords") or []:
        v = k.get("value")
        if v:
            kws.append(v.strip())
    out["keywords"] = dedup_keep_order(kws)

    # GO terms
    go_bp, go_mf, go_cc = [], [], []
    for xref in entry.get("uniProtKBCrossReferences") or []:
        if xref.get("database") != "GO":
            continue
        props = {p.get("key"): p.get("value") for p in (xref.get("properties") or [])}
        term = props.get("GoTerm") or ""
        if term.startswith("P:"):
            go_bp.append(term[2:].strip())
        elif term.startswith("F:"):
            go_mf.append(term[2:].strip())
        elif term.startswith("C:"):
            go_cc.append(term[2:].strip())

    out["go_biological_process"] = dedup_keep_order(go_bp)
    out["go_molecular_function"] = dedup_keep_order(go_mf)
    out["go_cellular_component"] = dedup_keep_order(go_cc)

    return out


# ---------- Canonical text ----------
def canonicalize_gene_deterministic(fields: Dict[str, Any], query_name: str) -> Tuple[str, str]:
    """
    Main text priority:
    1) NCBI Gene summary
    2) UniProt function
    3) GO / pathway / localization
    """
    gene_symbol = (fields.get("gene_symbol") or query_name).strip()
    gene_description = (fields.get("gene_description") or "").strip()
    gene_summary = clean_text(fields.get("gene_summary") or "")
    gene_aliases = fmt_list(fields.get("gene_aliases") or [], max_items=10)

    protein_name = (fields.get("protein_name") or "").strip()
    protein_alt = fmt_list(fields.get("protein_alt_names") or [], max_items=5)

    func_blob = sentence_truncate(" ".join(fields.get("function_texts") or []), max_sentences=5)

    pathway = fmt_list(fields.get("pathway_texts") or [], max_items=10)
    go_mf = fmt_list(fields.get("go_molecular_function") or [], max_items=10)
    go_bp = fmt_list(fields.get("go_biological_process") or [], max_items=15)
    go_cc = fmt_list(fields.get("go_cellular_component") or [], max_items=10)
    subcell = fmt_list(fields.get("subcellular_locations") or [], max_items=8)
    keywords = fmt_list(fields.get("keywords") or [], max_items=10)

    lines: List[str] = []

    identity_parts = [f"Gene Symbol: {gene_symbol}"]
    if gene_description:
        identity_parts.append(f"Gene Name: {gene_description}")
    if gene_aliases:
        identity_parts.append(f"Aliases: {gene_aliases}")
    if protein_name:
        identity_parts.append(f"Protein Name: {protein_name}")
    if protein_alt:
        identity_parts.append(f"Protein Alternative Names: {protein_alt}")
    lines.append(" | ".join(identity_parts))

    # Main text: NCBI gene summary
    if gene_summary:
        lines.append(f"Gene Summary: {gene_summary}")

    # Supplementary text: UniProt function
    if func_blob:
        lines.append(f"Protein Function: {func_blob}")

    if pathway:
        lines.append(f"Pathways: {pathway}")
    if go_mf:
        lines.append(f"Molecular Function (GO MF): {go_mf}")
    if go_bp:
        lines.append(f"Biological Process (GO BP): {go_bp}")
    if go_cc:
        lines.append(f"Cellular Component (GO CC): {go_cc}")
    if subcell:
        lines.append(f"Subcellular Localization: {subcell}")
    if keywords:
        lines.append(f"Keywords: {keywords}")

    doc_content = "\n".join(lines).strip()

    prefix = (
        "Represent the function, pathway context, and likely transcriptomic "
        "consequences of perturbing this human gene."
    )

    return prefix, doc_content


# ---------- Main pipeline per gene ----------
def build_profile_texts_for_gene(
    name: str,
    candidate_size: int = 5,
) -> Tuple[Optional[str], Dict[str, Any], str, str]:
    clean_name = normalize_name(name)
    search_term = apply_manual_mapping(clean_name)

    # ---------- 1) NCBI Gene ----------
    gene_id = ncbi_search_gene_id(search_term, taxid=HUMAN_TAXID)
    ncbi_fields = ncbi_fetch_gene_summary(gene_id) if gene_id else {
        "gene_id": None,
        "symbol": clean_name,
        "description": "",
        "summary": "",
        "organism": "Homo sapiens",
        "aliases": [],
    }

    # ---------- 2) UniProt ----------
    results = uniprot_search_candidates_for_gene(search_term, size=candidate_size)
    accession = pick_best_candidate(results, search_term) if results else None

    uniprot_fields: Dict[str, Any] = {}
    if accession:
        entry = fetch_uniprot_entry_json(accession)
        if entry:
            uniprot_fields = extract_uniprot_entry_fields(entry)

    if not uniprot_fields:
        uniprot_fields = {
            "accession": None,
            "organism_name": "Homo sapiens",
            "organism_id": int(HUMAN_TAXID),
            "protein_name": "",
            "protein_alt_names": [],
            "gene_primary": ncbi_fields.get("symbol") or clean_name,
            "gene_synonyms": ncbi_fields.get("aliases") or [],
            "function_texts": [],
            "subcellular_locations": [],
            "keywords": [],
            "go_biological_process": [],
            "go_molecular_function": [],
            "go_cellular_component": [],
            "pathway_texts": [],
            "disease_texts": [],
        }

    merged_fields = {
        "query_name": clean_name,
        "mapped_query": search_term,

        "ncbi_gene_id": ncbi_fields.get("gene_id"),
        "gene_symbol": ncbi_fields.get("symbol") or uniprot_fields.get("gene_primary") or clean_name,
        "gene_description": ncbi_fields.get("description") or "",
        "gene_summary": ncbi_fields.get("summary") or "",
        "gene_aliases": dedup_keep_order(
            (ncbi_fields.get("aliases") or []) + (uniprot_fields.get("gene_synonyms") or [])
        ),

        "accession": accession,
        "protein_name": uniprot_fields.get("protein_name") or "",
        "protein_alt_names": uniprot_fields.get("protein_alt_names") or [],

        "function_texts": uniprot_fields.get("function_texts") or [],
        "subcellular_locations": uniprot_fields.get("subcellular_locations") or [],
        "keywords": uniprot_fields.get("keywords") or [],
        "go_biological_process": uniprot_fields.get("go_biological_process") or [],
        "go_molecular_function": uniprot_fields.get("go_molecular_function") or [],
        "go_cellular_component": uniprot_fields.get("go_cellular_component") or [],
        "pathway_texts": uniprot_fields.get("pathway_texts") or [],
        "disease_texts": uniprot_fields.get("disease_texts") or [],
    }

    prefix, canonical_text = canonicalize_gene_deterministic(merged_fields, clean_name)
    return accession, merged_fields, prefix, canonical_text


# ---------- IO helpers ----------
def load_targets(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)

    candidate_cols = ["gene", "symbol", "gene_symbol", "Gene", "Symbol", "GeneSymbol"]
    target_col = None
    for c in candidate_cols:
        if c in df.columns:
            target_col = c
            break

    if target_col is not None:
        targets = df[target_col].dropna().astype(str).unique().tolist()
    else:
        targets = pd.read_csv(csv_path, header=None)[0].dropna().astype(str).unique().tolist()

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


# ---------- Main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--genes", type=str, default="target_genes.csv")
    parser.add_argument("--out", type=str, default="processed/gene_embeddings.pkl")
    parser.add_argument("--candidate_size", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Embedding batch size fed to the model/device.",
    )
    args = parser.parse_args()

    # 1) Load targets
    try:
        all_targets = load_targets(args.genes)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        raise SystemExit(1)

    # 2) Load checkpoint
    result_dict: Dict[str, Any] = load_checkpoint(args.out)
    targets_to_process = [t for t in all_targets if t not in result_dict]

    print(
        f"Total: {len(all_targets)} | "
        f"Processed: {len(result_dict)} | "
        f"Remaining: {len(targets_to_process)}"
    )
    if not targets_to_process:
        print("All tasks completed.")
        raise SystemExit(0)

    # 3) Load embedding model
    converter = TextEmbeddingConverter(MODEL_NAME)

    # 4) Mini-batch buffers
    pending_names: List[str] = []
    pending_prefix: List[str] = []
    pending_texts: List[str] = []
    pending_payloads: List[Dict[str, Any]] = []

    processed_state = {"count": 0}

    def flush_batch() -> None:
        if not pending_names:
            return

        embs = converter.encode_batch(
            pending_texts,
            pending_prefix,
            device_batch_size=args.batch_size,
        )

        for i, name in enumerate(pending_names):
            payload = pending_payloads[i]
            payload["embedding"] = embs[i]
            result_dict[name] = payload
            processed_state["count"] += 1

        pending_names.clear()
        pending_prefix.clear()
        pending_texts.clear()
        pending_payloads.clear()

        if processed_state["count"] % args.save_interval == 0:
            save_checkpoint(result_dict, args.out)

    print("Start processing...")

    for name in tqdm(targets_to_process, desc="Embedding genes"):
        accession, fields, prefix, canonical_text = build_profile_texts_for_gene(
            name,
            candidate_size=args.candidate_size,
        )

        payload = {
            "query_name": name,
            "mapped_query": apply_manual_mapping(normalize_name(name)),
            "ncbi_gene_id": fields.get("ncbi_gene_id"),
            "accession": accession,
            "fields": fields,
            "canonical_text": canonical_text,
            "embedded_text_used": canonical_text,
        }

        pending_names.append(name)
        pending_prefix.append(prefix)
        pending_texts.append(canonical_text)
        pending_payloads.append(payload)

        # be polite to remote APIs
        time.sleep(0.05)

        if len(pending_names) >= args.batch_size:
            flush_batch()

    flush_batch()
    save_checkpoint(result_dict, args.out)
    print(f"Finished! Saved {len(result_dict)} records to {args.out}")
