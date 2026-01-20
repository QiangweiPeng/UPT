import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc

from .utils import convert_mixed_array_to_2d
from .pca import centered_pca, project_pca, reconstruct_pca

def process_to_embedding(adata_control, adata_train, 
                         adata_test = None,
                         sample_rep = "X_pca", # X_ae; X_state
                         n_comps = 100,
                         control_key = "is_control",
                         condition_keys = "target_gene",
                         condition_rep_keys = "gene_embeddings",
                         condition_rep_dict = None):
    """ Process data into embedding.
        centered_pca control group as standard.
    Args:
        adata_control: AnnData object with control data.
        adata_train: AnnData object with train data.
        adata_test: AnnData object with test data.
        sample_rep: Representation to use for sample embeddings.
        n_comps: pca n_comps
        ...
    Returns:
        adata_control: Control.
        adata_train: Train.
        adata_test: Test.
    """
    if sample_rep == "X_pca":
        centered_pca(adata_control, n_comps=n_comps, method="scanpy")
        var = np.asarray(adata_control.uns["pca"]["variance"])
        adata_control.obsm[sample_rep+"_scaled"] = adata_control.obsm[sample_rep] / np.sqrt(var)
        print(np.std(np.array(adata_control.obsm[sample_rep + "_scaled"]), axis=0))
        
        project_pca(adata_train, ref_adata = adata_control, obsm_key_added="X_pca")
        adata_train.obsm[sample_rep+"_scaled"] = adata_train.obsm[sample_rep] / np.sqrt(var)
        print(np.std(np.array(adata_train.obsm[sample_rep + "_scaled"]), axis=0))

        if adata_test is not None:
            project_pca(adata_test, ref_adata = adata_control, obsm_key_added="X_pca")
            adata_test.obsm[sample_rep+"_scaled"] = adata_test.obsm[sample_rep] / np.sqrt(var)
            print(np.std(np.array(adata_test.obsm[sample_rep + "_scaled"]), axis=0))

    elif sample_rep == "??":
        raise ValueError("TODO")
    else:
        raise ValueError("Unimplemented sample_rep")
        

    if condition_rep_dict is not None: 
        temp = adata_train.obs[condition_keys].map(condition_rep_dict).values
        adata_train.obsm[condition_rep_keys] = convert_mixed_array_to_2d(temp) # 放到obsm中
        if adata_test is not None:
            temp = adata_test.obs[condition_keys].map(condition_rep_dict).values
            adata_test.obsm[condition_rep_keys] = convert_mixed_array_to_2d(temp)
    else:
        raise ValueError("need condition_rep_dict")
    
    
    return adata_control, adata_train, adata_test

