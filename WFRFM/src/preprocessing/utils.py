import torch
import numpy as np

import time
import pickle
import requests
from openai import OpenAI
import google.generativeai as genai
import mygene
from bs4 import BeautifulSoup
import re

def convert_mixed_array_to_2d(arr):
    """
    将 dtype=object 的 array (包含 tensor / numpy / nan) 转换为纯二维 numpy 数组。
    对于 nan 行，会用 np.nan 填充。
    """
    # 把每个元素都转成 numpy 或 nan
    processed = []
    for a in arr:
        if isinstance(a, torch.Tensor):
            processed.append(a.detach().cpu().numpy())
        elif isinstance(a, np.ndarray):
            processed.append(a)
        elif a is None or (isinstance(a, float) and np.isnan(a)):
            processed.append(np.nan)
        else:
            raise TypeError(f"Unsupported element type: {type(a)}")
    
    # 找出有效的非 nan 元素
    valid = [a for a in processed if isinstance(a, np.ndarray)]
    if not valid:
        raise ValueError("No valid tensor/array elements found.")
    
    # 检查维度一致性
    shapes = {a.shape for a in valid}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent tensor shapes found: {shapes}")
    
    d = valid[0].shape[0]  # 向量维度
    result = np.full((len(processed), d), np.nan, dtype=float)
    
    # 填充结果
    for i, a in enumerate(processed):
        if isinstance(a, np.ndarray):
            result[i] = a
    
    return result


def generate_gene_embeddings_gemini_v1(gene_list, save_path, api_key, model="models/gemini-embedding-001"):
    
    genai.configure(api_key=api_key)
    mg = mygene.MyGeneInfo()
    
    # 【修改点2】：新模型默认维度为 3072
    embedding_dim = 3072 
    gene_embeddings_dict = {}

    print(f"🚀 开始处理，共 {len(gene_list)} 个基因，使用模型: {model}...")

    for i, gene in enumerate(gene_list):
        print(f"[{i+1}/{len(gene_list)}] 正在处理: {gene}")
        
        # --- 步骤 A: 获取 Entrez ID ---
        entrez_id = ""
        try:
            mg_result = mg.query(gene, scopes='symbol', fields='entrezgene', species='human')
            if mg_result.get('hits') and len(mg_result['hits']) > 0:
                entrez_id = str(mg_result['hits'][0].get('entrezgene', ''))
        except Exception as e:
            pass # 忽略报错打印，保持界面干净
        
        # --- 步骤 B: 爬取并清洗 NCBI 摘要 ---
        summary_text = ""
        if entrez_id:
            try:
                time.sleep(0.5)
                url = f"https://www.ncbi.nlm.nih.gov/gene/{entrez_id}"
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                summary_dt = soup.find('dt', string='Summary')
                if summary_dt:
                    summary_dd = summary_dt.find_next_sibling('dd')
                    if summary_dd:
                        raw_text = summary_dd.get_text(strip=True)
                        summary_text = re.sub(r'\[provided by.*?\]', '', raw_text).strip()
            except Exception as e:
                print(f"  -> 爬取 NCBI 摘要失败")

        # --- 步骤 C: 调用 Gemini 生成向量 ---
        embedding = np.zeros(embedding_dim) 
        if summary_text:
            text_clean = summary_text.replace("\n", " ")
            try:
                result = genai.embed_content(
                    model=model,
                    content=text_clean,
                    task_type="retrieval_document", 
                )
                embedding = np.array(result['embedding'])
            except Exception as e:
                print(f"  -> Gemini API 调用失败: {e}")
        else:
            print(f"  -> {gene} 没有找到摘要文本，将使用全 0 向量。")

        gene_embeddings_dict[gene] = embedding

    # 3. 保存到本地文件
    print(f"\n✅ 所有基因处理完成！正在保存到: {save_path}")
    with open(save_path, "wb") as f:
        pickle.dump(gene_embeddings_dict, f)
    
    print("🎉 保存成功！")
    return gene_embeddings_dict



def generate_gene_embeddings_gemini_v2(gene_list, save_path, api_key):
    # 1. 初始化 Gemini API
    genai.configure(api_key=api_key)
    mg = mygene.MyGeneInfo()
    
    # 实例化文本生成模型 (用于提纯) 和 向量模型 (用于编码)
    generation_model = genai.GenerativeModel('gemini-2.5-flash')
    embedding_model_name = "models/gemini-embedding-001"
    embedding_dim = 3072
    
    gene_embeddings_dict = {}
    purified_texts_dict = {} # 用于记录提纯后的文本，方便你观察结果

    print(f"🚀 开始处理，共 {len(gene_list)} 个极端基因...")

    for i, gene in enumerate(gene_list):
        print(f"\n[{i+1}/{len(gene_list)}] 正在处理: {gene}")
        
        # --- 步骤 A: 获取 Entrez ID ---
        entrez_id = ""
        try:
            mg_result = mg.query(gene, scopes='symbol', fields='entrezgene', species='human')
            if mg_result.get('hits') and len(mg_result['hits']) > 0:
                entrez_id = str(mg_result['hits'][0].get('entrezgene', ''))
        except Exception:
            pass
        
        # --- 步骤 B: 爬取 NCBI 原始摘要 ---
        raw_summary = ""
        if entrez_id:
            try:
                time.sleep(0.5)
                url = f"https://www.ncbi.nlm.nih.gov/gene/{entrez_id}"
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                summary_dt = soup.find('dt', string='Summary')
                if summary_dt:
                    summary_dd = summary_dt.find_next_sibling('dd')
                    if summary_dd:
                        raw_text = summary_dd.get_text(strip=True)
                        raw_summary = re.sub(r'\[provided by.*?\]', '', raw_text).strip()
            except Exception as e:
                print(f"  -> 爬取 NCBI 摘要失败")

        # --- 步骤 C: 【核心改进】使用 Gemini 提纯文本 ---
        purified_text = ""
        if raw_summary:
            prompt = f"""
            You are an expert bioinformatician. Extract the core biological features from the following NCBI gene summary.
            
            Strict requirements:
            1. Output ONLY comma-separated keywords or short phrases. Do not use bullet points or newlines.
            2. You must extract information related to: Molecular Function, Biological Pathway, and Associated Disease.
            3. ABSOLUTELY NO complete sentences or boilerplate text (e.g., NEVER output "This gene encodes...", "The protein is...", etc.).
            
            Original Summary: {raw_summary}         
            """
            try:
                # 调用 Gemini 大模型进行信息脱水
                gen_res = generation_model.generate_content(prompt)
                purified_text = gen_res.text.strip()
                print(f"  -> 提纯结果: {purified_text[:80]}...") # 打印前80个字符看看效果
            except Exception as e:
                print(f"  -> 文本提纯失败: {e}")
                purified_text = raw_summary # 如果失败，退回使用原始文本
        else:
            print(f"  -> 未找到 {gene} 的摘要。")

        purified_texts_dict[gene] = purified_text

        # --- 步骤 D: 调用 Gemini 生成提纯后的向量 ---
        embedding = np.zeros(embedding_dim) 
        if purified_text:
            try:
                res = genai.embed_content(
                    model=embedding_model_name,
                    content=purified_text,
                    task_type="retrieval_document", 
                )
                embedding = np.array(res['embedding'])
            except Exception as e:
                print(f"  -> 向量生成失败: {e}")
                
        gene_embeddings_dict[gene] = embedding

    # 2. 保存文件
    with open(save_path, "wb") as f:
        pickle.dump(gene_embeddings_dict, f)
    print(f"\n✅ 处理完成！向量已保存至 {save_path}")
    
    return gene_embeddings_dict, purified_texts_dict