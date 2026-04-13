import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
import os

from .utils import convert_mixed_array_to_2d
from .pca import centered_pca, project_pca, reconstruct_pca

def process_to_embedding(adata_control, adata_train, 
                         adata_test = None,
                         sample_rep = "X_pca", # X_ae; X_state
                         n_comps = 100,
                         n_hidden = 1024,
                         model_ref = None,
                         model_train = None,
                         model_test = None,
                         scvi_save_path = None,
                         flatvi_save_path = None,
                         control_key = "is_control",
                         condition_keys = "target_gene",
                         condition_rep_keys = "gene_embeddings",
                         condition_rep_dict = None,
                         flatvi_kwargs = None,
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
        #sc.pp.normalize_total(adata_control, target_sum=1e4)
        #sc.pp.log1p(adata_control)
        centered_pca(adata_control, n_comps=n_comps, method="scanpy")
        var = np.asarray(adata_control.uns["pca"]["variance"])
        adata_control.obsm[sample_rep+"_scaled"] = adata_control.obsm[sample_rep] / np.sqrt(var)
        print(np.std(np.array(adata_control.obsm[sample_rep + "_scaled"]), axis=0))

        #sc.pp.normalize_total(adata_train, target_sum=1e4)
        #sc.pp.log1p(adata_train)
        project_pca(adata_train, ref_adata = adata_control, obsm_key_added="X_pca")
        adata_train.obsm[sample_rep+"_scaled"] = adata_train.obsm[sample_rep] / np.sqrt(var)
        print(np.std(np.array(adata_train.obsm[sample_rep + "_scaled"]), axis=0))

        if adata_test is not None:
            #sc.pp.normalize_total(adata_test, target_sum=1e4)
            #sc.pp.log1p(adata_test)
            project_pca(adata_test, ref_adata = adata_control, obsm_key_added="X_pca")
            adata_test.obsm[sample_rep+"_scaled"] = adata_test.obsm[sample_rep] / np.sqrt(var)
            print(np.std(np.array(adata_test.obsm[sample_rep + "_scaled"]), axis=0))

    elif sample_rep == "X_scVI":
        import scvi

        scvi.settings.dl_num_workers = 12
        if model_ref is None:
            # 依然是在control上训练
            print("training scvi model_ref")
            scvi.model.SCVI.setup_anndata(
                adata_control, 
                layer="counts", 
                batch_key="batch",  # 自动去批次
            )
            model_ref = scvi.model.SCVI(
                adata_control, 
                n_latent=n_comps, 
                n_hidden = n_hidden,
                n_layers = 2,
                gene_likelihood="nb",
            )
            model_ref.train(
                max_epochs=500,        
                early_stopping=True,  
                early_stopping_patience=20, 
                check_val_every_n_epoch=1,
                plan_kwargs={"lr": 1e-4}
            )
            model_ref.save(f"{scvi_save_path}_ref", overwrite=True)
        adata_control.obsm[sample_rep] = model_ref.get_latent_representation()

        print("projecting")
        # train和test project到上面的空间
        if model_train is None:
            model_train = scvi.model.SCVI.load_query_data(
                adata_train, 
                model_ref
            )
            model_train.train(max_epochs=10, plan_kwargs=dict(weight_decay=0.0))
            model_train.save(f"{scvi_save_path}_train", overwrite=True)
        adata_train.obsm[sample_rep] = model_train.get_latent_representation()

        if adata_test is not None:
            if model_test is None:
                model_test = scvi.model.SCVI.load_query_data(
                    adata_test, 
                    model_ref
                )
                model_test.train(max_epochs=10, plan_kwargs=dict(weight_decay=0.0))
                model_test.save(f"{scvi_save_path}_test", overwrite=True)
            adata_test.obsm[sample_rep] = model_test.get_latent_representation()
            
        print("Control Std:", np.std(adata_control.obsm[sample_rep], axis=0))
        print("Train Std:", np.std(adata_train.obsm[sample_rep], axis=0))
        if adata_test is not None:
            print("Test Std:", np.std(adata_test.obsm[sample_rep], axis=0))

    elif sample_rep == "X_flatvi":
        from .flatvi_wrapper import FlatVIEmbedding

        if flatvi_kwargs is None:
            flatvi_kwargs = {
                'n_latent': n_comps,
                'hidden_dims': [n_hidden, n_hidden, n_comps],
                'fl_weight': 1.0,
                'learning_rate': 1e-3,
                'device': 'cuda'
            }
        flatvi_model = FlatVIEmbedding(**flatvi_kwargs)
        model_path = f"{flatvi_save_path}/model.pt"
        if os.path.exists(model_path):
            flatvi_model.load_model_weights(model_path, in_dim=adata_control.n_vars)
        else:
            flatvi_model.train_model(
                adata_control,
                max_epochs=200,
                batch_size=256,
                save_path=flatvi_save_path
            )
            if flatvi_save_path is not None:
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

    elif sample_rep == "X_scVI_linear":
        import scvi

        scvi.settings.dl_num_workers = 12
        if model_ref is None:
            # 依然是在control上训练
            print("training linearscvi model_ref")
            scvi.model.LinearSCVI.setup_anndata(
                adata_control, 
                layer="counts", 
                batch_key="batch",  # 自动去批次
            )
            model_ref = scvi.model.LinearSCVI(
                adata_control, 
                n_latent=n_comps, 
                gene_likelihood="nb",
            )
            model_ref.train(
                max_epochs=500,        
                early_stopping=True,  
                early_stopping_patience=20, 
                check_val_every_n_epoch=1,
                plan_kwargs={"lr": 5e-4}
            )
            model_ref.save(f"{scvi_save_path}_ref", overwrite=True)
        adata_control.obsm[sample_rep] = model_ref.get_latent_representation()

        print("projecting")
        # train和test project到上面的空间
        if model_train is None:
            model_train = scvi.model.LinearSCVI.load_query_data(
                adata_train, 
                model_ref
            )
            model_train.train(max_epochs=10, plan_kwargs=dict(weight_decay=0.0))
            model_train.save(f"{scvi_save_path}_train", overwrite=True)
        adata_train.obsm[sample_rep] = model_train.get_latent_representation()

        if adata_test is not None:
            if model_test is None:
                model_test = scvi.model.LinearSCVI.load_query_data(
                    adata_test, 
                    model_ref
                )
                model_test.train(max_epochs=10, plan_kwargs=dict(weight_decay=0.0))
                model_test.save(f"{scvi_save_path}_test", overwrite=True)
            adata_test.obsm[sample_rep] = model_test.get_latent_representation()
            
        print("Control Std:", np.std(adata_control.obsm[sample_rep], axis=0))
        print("Train Std:", np.std(adata_train.obsm[sample_rep], axis=0))
        if adata_test is not None:
            print("Test Std:", np.std(adata_test.obsm[sample_rep], axis=0))
        
    elif sample_rep == "??":
        raise ValueError("TODO")
    else:
        raise ValueError("Unimplemented sample_rep")
        

    if condition_rep_dict is not None: 
        print(f"using condition_rep_dict")
        temp = adata_train.obs[condition_keys].map(condition_rep_dict).values
        adata_train.obsm[condition_rep_keys] = convert_mixed_array_to_2d(temp) # 放到obsm中
        if adata_test is not None:
            temp = adata_test.obs[condition_keys].map(condition_rep_dict).values
            adata_test.obsm[condition_rep_keys] = convert_mixed_array_to_2d(temp)
    else:
        raise ValueError("need condition_rep_dict")
    
    
    return adata_control, adata_train, adata_test, model_ref, model_train, model_test
