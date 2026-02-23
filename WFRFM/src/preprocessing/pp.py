import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
import scvi
import os
import scipy.sparse as sp

from .utils import convert_mixed_array_to_2d
from .pca import centered_pca, project_pca, reconstruct_pca
from .flatvi_wrapper import FlatVIEmbedding



def combined_view(adata_a,adata_b,batch_key=None):

    assert adata_a.n_vars == adata_b.n_vars
    assert (adata_a.var_names == adata_b.var_names).all()

    if sp.issparse(adata_a.X) or sp.issparse(adata_b.X):
        X = sp.vstack([adata_a.X, adata_b.X])
    else:
        X = np.vstack([adata_a.X, adata_b.X])

    if batch_key is None:
        return ad.AnnData(X=X, var=adata_a.var.copy())
    else:
        obs = pd.concat([adata_a.obs, adata_b.obs], axis=0) 
        return ad.AnnData(X=X, obs=obs.copy(), var=adata_a.var.copy())

def process_to_embedding(adata_control, adata_train, 
                         adata_ref = None, #可以传入一个ref来直接project pca
                         adata_test = None,
                         sample_rep = "X_pca", # X_ae; X_state
                         n_comps = 100,
                         n_hidden = 1024,
                         n_layers = 2,
                         model_ref = None,
                         model_train = None,
                         model_test = None,
                         model_save_path = None,
                         control_key = "is_control",
                         condition_keys = "target_gene",
                         condition_rep_keys = "gene_embeddings",
                         batch_key = "batch",
                         condition_rep_dict = None,
                         flatvi_kwargs = None,
                         pca_method = "scanpy",
                         device = "cuda"):
    """ Process data into embedding.
        centered_pca control group as standard.
    Args:
        adata_control: AnnData object with control data.
        adata_train: AnnData object with train data.
        adata_test: AnnData object with test data.
        sample_rep: Representation to use for sample embeddings.
        n_comps: n_comps
        ...
    Returns:
        adata_control: Control.
        adata_train: Train.
        adata_test: Test.
    """
    if sample_rep == "X_pca":
        if adata_ref is not None:
            var = np.asarray(adata_ref.uns["pca"]["variance"])

            project_pca(adata_control, ref_adata = adata_ref, obsm_key_added="X_pca")
            adata_control.obsm["X_pca_scaled"] = adata_control.obsm["X_pca"] / np.sqrt(var)
            print(np.std(np.array(adata_control.obsm[sample_rep + "_scaled"]), axis=0))
            
            adata_control.uns["pca"] = adata_ref.uns["pca"].copy()
            adata_control.varm["PCs"] = adata_ref.varm["PCs"].copy()
            adata_control.varm["X_mean"] = adata_ref.varm["X_mean"].copy()
            
            project_pca(adata_train, ref_adata = adata_ref, obsm_key_added="X_pca")
            adata_train.obsm["X_pca_scaled"] = adata_train.obsm["X_pca"] / np.sqrt(var)
            print(np.std(np.array(adata_train.obsm[sample_rep + "_scaled"]), axis=0))

            if adata_test is not None:
                project_pca(adata_test, ref_adata = adata_ref, obsm_key_added="X_pca")
                adata_test.obsm["X_pca_scaled"] = adata_test.obsm["X_pca"] / np.sqrt(var)
                print(np.std(np.array(adata_test.obsm[sample_rep + "_scaled"]), axis=0))

        else:
            
            # adata_combined = combined_view(adata_control,adata_train) # 改为在control和train上embedding
            adata_combined = ad.concat([adata_control, adata_train], join='inner')
            
            centered_pca(adata_combined, n_comps=n_comps, method=pca_method)
            var = np.asarray(adata_combined.uns["pca"]["variance"])
            # 需要这些参数满足后续构建
            adata_control.uns["pca"] = adata_combined.uns["pca"].copy()
            adata_control.varm["PCs"] = adata_combined.varm["PCs"].copy()
            adata_control.varm["X_mean"] = adata_combined.varm["X_mean"].copy()
    
            n_c = adata_control.n_obs
            adata_control.obsm[sample_rep] = adata_combined.obsm["X_pca"][:n_c]
            adata_control.obsm[sample_rep+"_scaled"] = adata_control.obsm[sample_rep] / np.sqrt(var)
            print(np.std(np.array(adata_control.obsm[sample_rep + "_scaled"]), axis=0))
    
            adata_train.obsm[sample_rep] = adata_combined.obsm["X_pca"][n_c:]
            adata_train.obsm[sample_rep+"_scaled"] = adata_train.obsm[sample_rep] / np.sqrt(var)
            print(np.std(np.array(adata_train.obsm[sample_rep + "_scaled"]), axis=0))
    
            if adata_test is not None:
                project_pca(adata_test, ref_adata = adata_combined, obsm_key_added="X_pca")
                adata_test.obsm[sample_rep+"_scaled"] = adata_test.obsm[sample_rep] / np.sqrt(var)
                print(np.std(np.array(adata_test.obsm[sample_rep + "_scaled"]), axis=0))
            del adata_combined

    elif sample_rep == "X_scVI":

        scvi.settings.dl_num_workers = 12
        if model_ref is None:
            adata_combined = ad.concat([adata_control, adata_train], join='inner') 
            print("training scvi model_ref")
            scvi.model.SCVI.setup_anndata(
                adata_combined, 
                layer="counts", 
                batch_key=batch_key,  # 自动去批次
            )
            model_ref = scvi.model.SCVI(
                adata_combined, 
                n_latent=n_comps, 
                n_hidden = n_hidden,
                n_layers = n_layers,
                gene_likelihood="nb",
            )
            model_ref.train(
                max_epochs=500,        
                early_stopping=True,  
                early_stopping_patience=50, 
                check_val_every_n_epoch=5,
                plan_kwargs={"lr": 1e-4}
            )
            model_ref.save(f"{model_save_path}_ref", overwrite=True)
            
            full_latent = model_ref.get_latent_representation()
            n_c = adata_control.n_obs
            
            adata_control.obsm[sample_rep] = full_latent[:n_c]
            adata_train.obsm[sample_rep] = full_latent[n_c:]
            
            del adata_combined
        else:
            print("Using loaded model_ref for Control and Train inference...")
            adata_control.obsm[sample_rep] = model_ref.get_latent_representation(adata_control)
            adata_train.obsm[sample_rep] = model_ref.get_latent_representation(adata_train)

        # 为了兼容后续代码
        if model_train is None:
            model_train = model_ref
            model_train.save(f"{model_save_path}_train", overwrite=True)

        if adata_test is not None:
            if model_test is None:
                model_test = scvi.model.SCVI.load_query_data(
                    adata_test, 
                    model_ref
                )
                model_test.train(max_epochs=100, plan_kwargs=dict(weight_decay=0.0))
                model_test.save(f"{model_save_path}_test", overwrite=True)
            adata_test.obsm[sample_rep] = model_test.get_latent_representation()
            
        print("Control Std:", np.std(adata_control.obsm[sample_rep], axis=0))
        print("Train Std:", np.std(adata_train.obsm[sample_rep], axis=0))
        if adata_test is not None:
            print("Test Std:", np.std(adata_test.obsm[sample_rep], axis=0))

    elif sample_rep == "X_flatvi":
        adata_combined = ad.concat([adata_control, adata_train], join='inner') 
        if flatvi_kwargs is None:
            flatvi_kwargs = {
                'n_latent': n_comps,
                'hidden_dims': [n_hidden] * n_layers + [n_comps],
                'fl_weight': 1.0,
                'learning_rate': 1e-4,
                'device': 'cuda'
            }
        flatvi_model = FlatVIEmbedding(**flatvi_kwargs)
        model_path = f"{model_save_path}/model.pt"
        if os.path.exists(model_path):
            flatvi_model.load_model_weights(model_path, in_dim=adata_combined.n_vars)
        else:
            flatvi_model.train_model(
                adata_combined,
                max_epochs=150,
                batch_size=256,
                save_path=model_save_path 
            )
            if model_save_path is not None:
                flatvi_model.save_model(model_path)

        flatvi_model.model.to(device)
        adata_control.obsm[sample_rep] = flatvi_model.get_latent_representation(adata_control)
        adata_train.obsm[sample_rep] = flatvi_model.get_latent_representation(adata_train)
        if adata_test is not None:
            adata_test.obsm[sample_rep] = flatvi_model.get_latent_representation(adata_test)
            
        print("Control Std:", np.std(adata_control.obsm[sample_rep], axis=0)[:5])
        print("Train Std:", np.std(adata_train.obsm[sample_rep], axis=0)[:5])
        if adata_test is not None:
            print("Test Std:", np.std(adata_test.obsm[sample_rep], axis=0)[:5])
        model_ref = flatvi_model
        del adata_combined

    elif sample_rep == "X_state":
        print("Control Std:", np.std(adata_control.obsm[sample_rep], axis=0))
        print("Train Std:", np.std(adata_train.obsm[sample_rep], axis=0))
        if adata_test is not None:
            print("Test Std:", np.std(adata_test.obsm[sample_rep], axis=0))
    
    elif sample_rep == "??":
        raise ValueError("TODO")
    else:
        raise ValueError("Unimplemented sample_rep")
        

    if condition_rep_dict is not None: 
        unique_keys = set(adata_train.obs[condition_keys].unique())
        if adata_test is not None:
            unique_keys.update(adata_test.obs[condition_keys].unique())

        extended_rep_dict = {}
        for key in unique_keys:
            if key in condition_rep_dict:
                extended_rep_dict[key] = condition_rep_dict[key]
            elif '+' in key: # 双扰动
                genes = key.split('+')
                if len(genes) == 2: #认为只有双敲
                    g1, g2 = genes[0].strip(), genes[1].strip()
                    if g1 == 'ctrl' and g2 in condition_rep_dict:
                        extended_rep_dict[key] = condition_rep_dict[g2]
                    elif g2 == 'ctrl' and g1 in condition_rep_dict:
                        extended_rep_dict[key] = condition_rep_dict[g1]
                    
                    elif g1 in condition_rep_dict and g2 in condition_rep_dict:
                        vec1 = np.array(condition_rep_dict[g1])
                        vec2 = np.array(condition_rep_dict[g2])
                        extended_rep_dict[key] = (vec1 + vec2)/2           
                    else:
                        raise KeyError(f"Components of '{key}' not found in condition_rep_dict")
                else:
                     pass 
            else:
                raise KeyError(f"Key '{key}' not found in condition_rep_dict")

        temp = adata_train.obs[condition_keys].astype(str).map(extended_rep_dict).values
        adata_train.obsm[condition_rep_keys] = convert_mixed_array_to_2d(temp) 
        
        if adata_test is not None:
            temp = adata_test.obs[condition_keys].astype(str).map(extended_rep_dict).values
            adata_test.obsm[condition_rep_keys] = convert_mixed_array_to_2d(temp)

    else:
        raise ValueError("need condition_rep_dict")
    
    
    return adata_control, adata_train, adata_test, model_ref, model_train, model_test

