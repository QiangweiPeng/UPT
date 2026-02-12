import sys
import torch
import numpy as np
import anndata as ad
import scanpy as sc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "external"))
from flatvi.models.base.geometric_vae import GeometricNBVAE
from flatvi.datamodules.sc_datamodule import scDataModule

class FlatVIEmbedding:
    """FlatVI embedding wrapper for WFRFM"""
    
    def __init__(self, 
                 n_latent=100,
                 hidden_dims=[512, 256, 100],
                 batch_norm=True,
                 dropout=True,
                 dropout_p=0.1,
                 fl_weight=1.0,
                 learning_rate=1e-3,
                 device='cuda'):
        
        self.n_latent = n_latent
        self.hidden_dims = hidden_dims
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.dropout_p = dropout_p
        self.fl_weight = fl_weight
        self.learning_rate = learning_rate
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        
    def train_model(self, 
                   adata_train,
                   adata_val=None,
                   max_epochs=500,
                   batch_size=256,
                   save_path=None):
        """训练FlatVI模型"""
        
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
        
        # 准备训练数据
        if adata_val is None:
            # 如果没有提供验证集，从训练集分割
            from sklearn.model_selection import train_test_split
            n_obs = adata_train.n_obs
            
            # 确保验证集至少有一个 batch 的数据
            val_size = max(batch_size, int(n_obs * 0.1))
            if val_size >= n_obs - batch_size:
                # 数据太少，不分割验证集
                adata_val = None
                print(f"Warning: Dataset too small ({n_obs} samples), skipping validation split")
            else:
                train_idx, val_idx = train_test_split(
                    np.arange(n_obs), test_size=val_size, random_state=42
                )
                adata_val = adata_train[val_idx].copy()
                adata_train = adata_train[train_idx].copy()
        
        # 数据预处理（如果还没做）
        if 'counts' not in adata_train.layers:
            adata_train.layers['counts'] = adata_train.X.copy()
            if adata_val is not None:
                adata_val.layers['counts'] = adata_val.X.copy()
        
        # 构建VAE参数
        vae_kwargs = dict(
            in_dim=adata_train.n_vars,
            hidden_dims=self.hidden_dims,
            batch_norm=self.batch_norm,
            dropout=self.dropout,
            dropout_p=self.dropout_p,
            n_epochs_anneal_kl=int(max_epochs * 0.5),
            kl_warmup_fraction=0.5,
            kl_weight=None,  # 使用annealing
            likelihood='nb',
            learning_rate=self.learning_rate,
            model_library_size=True
        )
        
        # 初始化GeometricNBVAE
        self.model = GeometricNBVAE(
            l2=True,
            interpolate_z=False, #无需平滑生成路径
            eta_interp=0.1,
            compute_metrics_every=500000,
            start_jac_after=0,  
            vae_kwargs=vae_kwargs,
            use_c=True,
            detach_theta=True,
            fl_weight=0.01,
            trainable_c=False,
            anneal_fl_weight=True,
            max_fl_weight=0.1,
            n_epochs_anneal_fl= max_epochs,
            fl_anneal_fraction=0.5
        )
        
        # 准备数据加载器
       # 准备数据加载器
        from torch.utils.data import DataLoader, TensorDataset
        
        X_train = torch.FloatTensor(adata_train.X.toarray() if hasattr(adata_train.X, 'toarray') else adata_train.X)
        X_val = torch.FloatTensor(adata_val.X.toarray() if hasattr(adata_val.X, 'toarray') else adata_val.X)
        
        train_dataset = TensorDataset(X_train)
        val_dataset = TensorDataset(X_val)
        
        # ✅ 正确的 collate_fn
        def collate_fn(batch):
            # batch 是一个列表，每个元素是 (tensor,) 形式
            # 需要堆叠所有样本
            X_batch = torch.stack([item[0] for item in batch])
            return {"X": X_batch}
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=12,
            drop_last=True,
            collate_fn=collate_fn  # 通过参数传入
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=12,
            drop_last=True,
            collate_fn=collate_fn  # 通过参数传入
        )

        
        # 配置callbacks
        callbacks = []
        if save_path is not None:
            checkpoint_callback = ModelCheckpoint(
                dirpath=save_path,
                filename='flatvi-{epoch:02d}-{val/loss:.2f}',
                monitor='val/loss',
                mode='min',
                save_top_k=1
            )
            callbacks.append(checkpoint_callback)
        
        early_stopping = EarlyStopping(
            monitor='val/loss',
            patience=30,
            mode='min'
        )
        callbacks.append(early_stopping)
        
        # 训练
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=callbacks,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            log_every_n_steps=10
        )
        
        trainer.fit(
            self.model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        
        return self.model
    
    def load_model(self, checkpoint_path):
        """加载已训练的模型"""
        self.model = GeometricNBVAE.load_from_checkpoint(checkpoint_path)
        self.model.eval()
        self.model.to(self.device)
        return self.model
    
    @torch.no_grad()
    def get_latent_representation(self, adata, batch_size=512):
        """提取潜在表示"""
        if self.model is None:
            raise ValueError("Model not trained or loaded!")
        
        self.model.eval()
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        latents = []
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            # 使用均值而非采样（更稳定）
            z_dict = self.model.encode(batch)
            z = z_dict['mu'] if 'mu' in z_dict else z_dict['z']
            latents.append(z.cpu().numpy())
        
        return np.concatenate(latents, axis=0)
    
    def save_model(self, path):
        """保存模型"""
        if self.model is None:
            raise ValueError("No model to save!")
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model_weights(self, path, in_dim):
        """从权重文件加载"""
        vae_kwargs = dict(
            in_dim=in_dim,
            hidden_dims=self.hidden_dims,
            batch_norm=self.batch_norm,
            dropout=self.dropout,
            dropout_p=self.dropout_p,
            n_epochs_anneal_kl=500,
            kl_warmup_fraction=0.5,
            kl_weight=None,
            likelihood='nb',
            learning_rate=self.learning_rate,
            model_library_size=True
        )
        
        self.model = GeometricNBVAE(
            l2=True,
            interpolate_z=False,
            eta_interp=0.1,
            compute_metrics_every=50,
            start_jac_after=100,
            vae_kwargs=vae_kwargs,
            use_c=True,
            detach_theta=True,
            fl_weight=self.fl_weight,
            trainable_c=False,
            anneal_fl_weight=False,
            max_fl_weight=None,
            n_epochs_anneal_fl=None,
            fl_anneal_fraction=None
        )
        
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        print(f"Model loaded from {path}")
        return self.model