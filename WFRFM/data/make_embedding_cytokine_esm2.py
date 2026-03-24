import requests
import pandas as pd
import torch
from esm import pretrained
from tqdm import tqdm
import argparse
import os
import time

'''
Generate Cytokine embedding by ESM2 with Resume Capability
Fixed: Added manual mapping and .strip() to handle whitespace.
python make_embedding_cytokine_esm2.py --cytokines target_cytokine_PBMC.csv --out processed/condition_embedding_cytokines_PBMC_esm2.pkl
'''

MANUAL_MAPPING = {
    "LT-alpha2-beta1": "LT-alpha",  
    "LT-alpha1-beta2": "LT-beta",  
    "IFN-lambda2": "IFN-lambda-2",
    "IFN-lambda3": "IFN-lambda-3",
    "IL-18Ra": "IL-18R-1",
}
# ==============================================================================

class ESMConverter:
    def __init__(self, model: str):
        # 只有在需要计算时才加载模型，避免还没跑就占用显存
        print(f"Loading model {model}...")
        self.model, self.alphabet = pretrained.load_model_and_alphabet(model)
        self.batch_converter = self.alphabet.get_batch_converter()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("Model moved to GPU.")

    def convert(self, sequences):
        batch_labels, batch_strs, batch_tokens = self.batch_converter(sequences)
        if torch.cuda.is_available():
            batch_tokens = batch_tokens.cuda()
        with torch.no_grad():
            token_embeddings = self.model(batch_tokens, repr_layers=[33])
            embeddings = token_embeddings['representations'][33]
            average_embeddings = embeddings.mean(dim=1)
        return average_embeddings

import unicodedata

def get_sequence_direct_search(input_name):
    """
    Search UniProt with manual mapping support + DEBUG PRINTS.
    """
    # 1. 强力清洗：去除首尾空格，并将非断行空格(\xa0)转为普通空格
    clean_name = str(input_name).strip()
    clean_name = unicodedata.normalize("NFKC", clean_name) # 修复特殊字符和空格
    
    # 2. 查字典修正
    # 增加 Debug 打印：看看清洗后的名字到底是什么
    search_term = MANUAL_MAPPING.get(clean_name, clean_name)
    
    # === DEBUG BLOCK START ===
    if clean_name in MANUAL_MAPPING:
        print(f"\n[DEBUG] ✅ Mapping Triggered: '{clean_name}' -> '{search_term}'")
    elif clean_name in ["IL-18Ra", "IFN-lambda2", "LT-alpha2-beta1", "IFN-lambda3", "LT-alpha1-beta2"]:
        # 如果是这几个目标名字，但没触发映射，说明 Key 不匹配
        print(f"\n[DEBUG] ❌ Mapping FAILED for: '{clean_name}'")
        print(f"        CSV Input (repr): {repr(clean_name)}")
        print(f"        Dict Keys (sample): {list(MANUAL_MAPPING.keys())}")
        # 强制修正（防止字典 Key 写错）
        if "IL-18Ra" in clean_name: search_term = "IL-18R-1"
        if "IFN-lambda2" in clean_name: search_term = "IFN-lambda-2"
        if "IFN-lambda3" in clean_name: search_term = "IFN-lambda-3"
        if "LT-alpha2" in clean_name: search_term = "LT-alpha"
        if "LT-alpha1" in clean_name: search_term = "LT-beta"
        print(f"        [Force Fix] New search term: '{search_term}'")
    # === DEBUG BLOCK END ===

    # 3. 构造查询
    query = f'({search_term}) AND organism_id:9606 AND reviewed:true'
    
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": query,
        "fields": "accession,gene_names,sequence",
        "size": 1, 
        "format": "json"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data['results']:
                return data['results'][0]['sequence']['value']
            else:
                # 打印出失败的具体 URL，方便你手动点开看
                print(f"[DEBUG] Empty result for query: {query}")
        else:
            tqdm.write(f"[UniProt] HTTP {response.status_code} for {repr(clean_name)}")
            tqdm.write(f"[UniProt] URL: {response.url}")
            tqdm.write(f"[UniProt] Body: {response.text[:200]}")
            return "NONE"
    except Exception as e:
        tqdm.write(f"Error searching {clean_name}: {e}")
    
    return "NONE"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cytokines", type=str, default="target_cytokines.csv")
    parser.add_argument("--out", type=str, default="cytokine_embeddings.pkl")
    args = parser.parse_args()

    # 读取输入
    try:
        df = pd.read_csv(args.cytokines)
        if 'cytokine' in df.columns:
            # 这里的 dropna 和 unique 能去除空行
            all_targets = df['cytokine'].dropna().unique().tolist()
        else:
            all_targets = pd.read_csv(args.cytokines, header=None)[0].dropna().unique().tolist()
        
        # 预处理：把所有名字都去个空格，防止后续匹配出错
        all_targets = [str(t).strip() for t in all_targets]
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        exit()

    # 断点续传
    result_dict = {}
    if os.path.exists(args.out):
        print(f"Loading existing checkpoint: {args.out}")
        try:
            loaded_data = pd.read_pickle(args.out)
            if isinstance(loaded_data, dict):
                result_dict = loaded_data
        except:
            print("Checkpoint load failed, starting fresh.")
            result_dict = {}
    
    # 过滤掉已经跑过的
    # 注意：这里的 t 也必须 strip 过，确保 key 一致
    targets_to_process = [t for t in all_targets if t not in result_dict]
    
    print(f"Total input: {len(all_targets)}")
    print(f"Already processed: {len(result_dict)}")
    print(f"Remaining: {len(targets_to_process)}")

    if len(targets_to_process) == 0:
        print("All tasks completed.")
        exit()

    converter = ESMConverter("esm2_t33_650M_UR50D")
    save_interval = 5
    processed_count = 0

    print("Start processing...")
    for name in tqdm(targets_to_process, desc="Embedding"):
        
        seq = get_sequence_direct_search(name)
        
        if seq == "NONE":
            # 这里的 repr(name) 会把名字带引号打印出来，比如 'IL-18Ra '，让你看到空格
            tqdm.write(f"Warning: Sequence NOT FOUND for {repr(name)}. Check MANUAL_MAPPING.")
            continue
        
        try:
            emb_tensor = converter.convert([(name, seq)])
            # 存入字典
            result_dict[name] = emb_tensor.cpu().view(-1)
            processed_count += 1
            
            if processed_count % save_interval == 0:
                pd.to_pickle(result_dict, args.out)
                
        except Exception as e:
            tqdm.write(f"Error embedding {name}: {e}")

    pd.to_pickle(result_dict, args.out)
    print(f"Finished. Saved {len(result_dict)} embeddings to {args.out}")
