from UniProtMapper import ProtMapper
import requests
import pandas as pd
import torch
from esm import pretrained
from tqdm import tqdm
import argparse

'''
Generate gene embedding by ESM2
Author: Zhixuan Jing
Last Modify: Aug 18 2025
To use, run: python make_embedding.py --genes <Your gene list csv file> --out <your output directory>
'''
class ESMConverter:
  def __init__(self, model:str):
    self.model, self.alphabet = pretrained.load_model_and_alphabet(model)
    self.batch_converter = self.alphabet.get_batch_converter()

  def convert(self, sequences):
    batch_labels, batch_strs, batch_tokens = self.batch_converter(sequences)
    with torch.no_grad():
      token_embeddings = self.model(batch_tokens, repr_layers=[33])
      embeddings = token_embeddings['representations'][33]
      average_embeddings = embeddings.mean(dim=1)
    return average_embeddings
  
def get_protein_sequence_by_gene(gene_name):
    mapper = ProtMapper()
    result, failed = mapper.get(
        ids=gene_name, from_db="Gene_Name", to_db="UniProtKB"
    )
    result = result[(result['Organism'] == "Homo sapiens (Human)")&(result['Reviewed'] == "reviewed")]
    protein = result.iloc[0]["Entry"]
    # print(protein)
    # Define the UniProt API endpoint
    sequence_url = f"https://www.uniprot.org/uniprot/{protein}.fasta"
    sequence_response = requests.get(sequence_url)
        
    if sequence_response.status_code == 200:
        # Extract and return the protein sequence
        sequence = ''.join(sequence_response.text.splitlines()[1:])
        return sequence
    else:
        return "NONE"
      
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
    
  parser.add_argument(
      "--genes", 
      type = str, 
      default = "/root/autodl-tmp/dvc/data/vcc_data/target_genes.csv", # 默认值
      help = "candidate gene list"
  )
  parser.add_argument(
      "--out", 
      type = str, 
      default = "/root/autodl-tmp/dvc/data/vcc_data/vcc_data_target_genes_embedding.pkl", 
      help = "output file with .pkl format"
  )

  args = parser.parse_args()

  # genes = pd.read_csv(args.genes, header = None)[0].to_list()
  genes = pd.read_csv(args.genes)['gene'].to_list()

  embedding = pd.DataFrame(columns=["gene", "protein", "embedding"])
  embedding["gene"] = genes
  embedding.set_index(embedding["gene"], drop = False, inplace = True)
  embedding["protein"] = embedding['gene'].apply(get_protein_sequence_by_gene)
  converter = ESMConverter("esm2_t33_650M_UR50D")
  sequences = list(zip(embedding['gene'], embedding['protein']))

  # em = []
  # print("Calculate embedding...")
  # for s in tqdm(sequences):
  #   em.append(converter.convert([s]))
  #   embedding['embedding'] = em
  #   embedding['embedding'] = embedding['embedding'].apply(torch.flatten)

  em = []
  print("Calculate embedding...")
  for s in tqdm(sequences):
      em.append(converter.convert([s]))
  embedding['embedding'] = em
  embedding['embedding'] = embedding['embedding'].apply(torch.flatten)

  uns = {}
  for g in embedding['gene']:
    uns[g] = embedding.loc[g]['embedding']
  print(uns)
  print("Save embedding to file...")
  pd.to_pickle(uns,args.out)
