import numpy as np
import pandas as pd
import seaborn as sns
import jax
import functools
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
import rapids_singlecell as rsc
import flax.linen as nn
import optax


import cellflow
from cellflow.model import CellFlow
import cellflow.preprocessing as cfpp
from cellflow.utils import match_linear
from cellflow.plotting import plot_condition_embedding
from cellflow.preprocessing import transfer_labels, compute_wknn, centered_pca, project_pca, reconstruct_pca
from cellflow.metrics import compute_r_squared, compute_e_distance


import torch

from .utils import convert_mixed_array_to_2d

def pp_with_cellflow(adata,
                    adata_test = None,
                    sample_rep = "X_pca", # X_ae; X_state
                    control_key = "is_control",
                    condition_keys = "target_gene",
                    condition_rep_keys = "gene_embeddings",
                    condition_rep_method = "one-hot",
                    condition_rep_dict = None):
    """ Preprocess data with CellFlow.
    Args:
        adata: AnnData object with training data.
        adata_test: AnnData object with test data.
        sample_rep: Representation to use for sample embeddings.
        control_key: Key in adata.obs indicating control samples.
    Returns:
        adata: Preprocessed AnnData object.
        adata_test: Preprocessed test AnnData object.
    """

    if sample_rep not in adata.obsm.keys():
        if sample_rep == "X_pca":
            cfpp.centered_pca(adata, n_comps=30, method="scanpy")
            adata.obsm[sample_rep+"_scaled"] = adata.obsm[sample_rep] / adata.uns['pca']['variance']**0.5

            print(np.std(np.array(adata.obsm[sample_rep + "_scaled"]), axis=1))
            if adata_test is not None:
                cfpp.project_pca(adata_test, adata, obsm_key_added="X_pca_reproj")
                adata_test.obsm["X_pca_reproj"+"_scaled"] = adata_test.obsm["X_pca_reproj"] / adata.uns['pca']['variance']**0.5
        if sample_rep == "X_ae":
            # TODO
            pass
        if sample_rep == "X_state":
            # TODO
             pass
    
    adata_control = adata[adata.obs[control_key]==True]
    adata_treated = adata[adata.obs[control_key]==False]

    if condition_rep_keys not in adata.obsm.keys():
        if condition_keys not in adata.obs.keys():
            raise ValueError(f"{condition_keys} not in adata.obs")
        if condition_rep_dict is not None:
            temp = adata_treated.obs[condition_keys].map(condition_rep_dict).values
            adata_treated.obsm[condition_rep_keys] = convert_mixed_array_to_2d(temp)
            if adata_test is not None:
                temp = adata_test.obs[condition_keys].map(condition_rep_dict).values
                adata_test.obsm[condition_rep_keys] = convert_mixed_array_to_2d(temp)
        else:
            if condition_rep_method == "one-hot":
                all_conditions = adata_treated.obs[condition_keys].unique().tolist()
                if adata_test is not None:
                    all_conditions += adata_test.obs[condition_keys].unique().tolist()
                all_conditions = list(set(all_conditions))

                # 2. 建立类别到 one-hot 向量的映射
                condition_rep_dict = {
                    cond: np.eye(len(all_conditions))[i] for i, cond in enumerate(all_conditions)
                }
                # 3. 对 adata 编码
                adata_treated.obsm[condition_rep_keys] = np.stack(
                    adata_treated.obs[condition_keys].map(condition_rep_dict)
                )

                # 4. 对 adata_test 编码（如果存在）
                if adata_test is not None:
                    adata_test.obsm[condition_rep_keys] = np.stack(
                        adata_test.obs[condition_keys].map(condition_rep_dict)
                    )
            elif condition_rep_method == "ESM2":
                # TODO
                pass
            else:
                raise ValueError(f"Unknown condition_rep_method: {condition_rep_method}")
    
    
    if adata_test is not None:
        return adata_control, adata_treated, adata_test
    else:
        return adata_control, adata_treated