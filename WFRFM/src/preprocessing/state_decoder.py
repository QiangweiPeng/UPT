import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import scipy.sparse as sp
except Exception:
    sp = None


# -----------------------------
# NB log-likelihood (scVI-style parameterization)
# -----------------------------
def nb_log_prob(x, mu, theta, eps: float = 1e-8):
    """
    x:     (B, G) raw counts, >=0
    mu:    (B, G) mean, >0
    theta: (1, G) or (B, G) dispersion, >0
    return: (B, G) log p(x | mu, theta)
    """
    x = x.float()
    mu = mu.clamp_min(eps)
    theta = theta.clamp_min(eps)

    log_theta_mu = torch.log(theta + mu)
    res = (
        torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta) - log_theta_mu)
        + x * (torch.log(mu) - log_theta_mu)
    )
    return res


# -----------------------------
# scVI-style NB Decoder (MLP)
# -----------------------------
class NBDecoder(nn.Module):
    """
    Input:  z (B, z_dim), log_library (B,1)
    Output: mu (B,G), theta (1,G)
    """
    def __init__(self, z_dim: int, n_genes: int, hidden=(256, 256), dropout=0.1):
        super().__init__()
        self.n_genes = n_genes

        layers = []
        in_dim = z_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        # proportions over genes
        self.scale_head = nn.Linear(in_dim, n_genes)

        # gene-wise dispersion (theta_g), unconstrained -> softplus
        self.r_unconstrained = nn.Parameter(torch.randn(n_genes))

    def forward(self, z: torch.Tensor, log_library: torch.Tensor):
        h = self.backbone(z)                                  # (B, H)
        scale = F.softmax(self.scale_head(h), dim=-1)         # (B, G), sum=1
        mu = torch.exp(log_library) * scale                   # (B, G)
        theta = F.softplus(self.r_unconstrained).unsqueeze(0) # (1, G)
        return mu, theta


# -----------------------------
# AnnData Dataset (counts + embedding)
# -----------------------------
class AnnDataCountEmbeddingDataset(Dataset):
    """
    adata.layers[count_layer] : raw counts (cells x genes), dense or sparse
    adata.obsm[emb_key]       : embedding z (cells x z_dim), dense numpy
    """
    def __init__(self, adata, count_layer="count", emb_key="X_state"):
        self.adata = adata
        self.counts = adata.layers[count_layer]
        self.z = adata.obsm[emb_key]

        if not isinstance(self.z, np.ndarray):
            self.z = np.asarray(self.z)

        self.n = adata.n_obs
        self.g = adata.n_vars

        # quick checks
        assert self.z.shape[0] == self.n, f"obsm['{emb_key}'] first dim must match n_obs"
        assert self.z.ndim == 2, "embedding must be 2D (cells x z_dim)"

        # sparse detection
        self.is_sparse = (sp is not None) and sp.issparse(self.counts)

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        # counts row
        if self.is_sparse:
            x = self.counts[idx]              # 1 x G sparse
        else:
            x = self.counts[idx]              # (G,) dense row

        z = self.z[idx]                       # (z_dim,)
        return x, z


def _collate_counts_z(batch):
    """
    batch: list of (x_row, z_row)
    returns:
      x_count: FloatTensor (B,G)
      z:       FloatTensor (B,z_dim)
    """
    xs, zs = zip(*batch)

    # stack z
    z = torch.from_numpy(np.stack(zs, axis=0)).float()

    # stack counts
    x0 = xs[0]
    if sp is not None and sp.issparse(x0):
        # vstack sparse -> dense
        X = sp.vstack(xs).toarray()
    else:
        X = np.stack(xs, axis=0)

    # counts are raw ints; keep float for lgamma etc.
    x = torch.from_numpy(X).float()
    return x, z


# -----------------------------
# Trainer class
# -----------------------------
class NBDecoderTrainer:
    def __init__(
        self,
        decoder: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        grad_clip: float = 5.0,
        device: str | None = None,
        use_amp: bool = True,
    ):
        self.decoder = decoder
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and (self.device.startswith("cuda"))

        self.decoder.to(self.device)
        self.opt = torch.optim.AdamW(self.decoder.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    @torch.no_grad()
    def _make_log_library(self, x_count: torch.Tensor):
        # observed library size: log(sum counts)
        lib = x_count.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B,1)
        return torch.log(lib)

    def _step(self, x_count: torch.Tensor, z: torch.Tensor):
        """
        One optimization step. Returns scalar loss.
        """
        x_count = x_count.to(self.device, non_blocking=True)
        z = z.to(self.device, non_blocking=True)

        log_library = self._make_log_library(x_count)

        self.opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            mu, theta = self.decoder(z, log_library)
            logp = nb_log_prob(x_count, mu, theta)              # (B,G)
            nll = (-logp.sum(dim=1)).mean()                     # scalar

        self.scaler.scale(nll).backward()
        if self.grad_clip is not None and self.grad_clip > 0:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.grad_clip)

        self.scaler.step(self.opt)
        self.scaler.update()

        return float(nll.detach().cpu().item())

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader):
        self.decoder.eval()
        total, n = 0.0, 0
        for x_count, z in dataloader:
            x_count = x_count.to(self.device, non_blocking=True)
            z = z.to(self.device, non_blocking=True)
            log_library = self._make_log_library(x_count)
            mu, theta = self.decoder(z, log_library)
            logp = nb_log_prob(x_count, mu, theta)
            nll = (-logp.sum(dim=1)).mean()
            bs = x_count.size(0)
            total += float(nll.cpu().item()) * bs
            n += bs
        self.decoder.train()
        return total / max(n, 1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 50,
        print_every: int = 1,
    ):
        self.decoder.train()
        for ep in range(1, epochs + 1):
            total, n = 0.0, 0
            for x_count, z in train_loader:
                loss = self._step(x_count, z)
                bs = x_count.size(0)
                total += loss * bs
                n += bs

            train_nll = total / max(n, 1)
            if (ep % print_every) == 0:
                if val_loader is not None:
                    val_nll = self.evaluate(val_loader)
                    print(f"Epoch {ep:03d} | train NLL={train_nll:.4f} | val NLL={val_nll:.4f}")
                else:
                    print(f"Epoch {ep:03d} | train NLL={train_nll:.4f}")

    def save(self, path: str):
        torch.save({"decoder": self.decoder.state_dict()}, path)

    def load(self, path: str, map_location: str | None = None):
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.decoder.load_state_dict(ckpt["decoder"])



def build_train_eval_loaders(
    adata_train,
    adata_eval,
    count_layer="count",
    emb_key="X_state",
    batch_size=256,
    num_workers=0,
    pin_memory=True,
    train_shuffle=True,
):
    """
    Build DataLoaders from two AnnData objects without gene alignment.
    Assumes genes (var order) are already aligned.
    """

    # ---- minimal checks ----
    for adata, name in [(adata_train, "adata_train"), (adata_eval, "adata_eval")]:
        if count_layer not in adata.layers:
            raise KeyError(f"{name}.layers missing '{count_layer}'")
        if emb_key not in adata.obsm:
            raise KeyError(f"{name}.obsm missing '{emb_key}'")
        if adata.obsm[emb_key].shape[0] != adata.n_obs:
            raise ValueError(f"{name}.obsm['{emb_key}'] first dim must equal n_obs")

    if adata_train.n_vars != adata_eval.n_vars:
        raise ValueError("n_vars mismatch between train and eval. (Genes must already be aligned.)")

    if adata_train.obsm[emb_key].shape[1] != adata_eval.obsm[emb_key].shape[1]:
        raise ValueError("Embedding dimension (z_dim) mismatch between train and eval.")

    # ---- datasets ----
    train_ds = AnnDataCountEmbeddingDataset(adata_train, count_layer=count_layer, emb_key=emb_key)
    eval_ds  = AnnDataCountEmbeddingDataset(adata_eval,  count_layer=count_layer, emb_key=emb_key)

    # ---- loaders ----
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_counts_z,
        drop_last=False,
    )

    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_counts_z,
        drop_last=False,
    )

    return train_loader, eval_loader



# ---- quick start (你在 notebook / script 里这样用) ----
# z_dim = adata.obsm["X_state"].shape[1]
# n_genes = adata.n_vars
# decoder = NBDecoder(z_dim=z_dim, n_genes=n_genes, hidden=(256,256), dropout=0.1)
# train_loader, val_loader = build_loaders_from_adata(adata, count_layer="count", emb_key="X_state",
#                                                     batch_size=256, num_workers=0,
#                                                     train_idx=train_idx, val_idx=val_idx)
# trainer = NBDecoderTrainer(decoder, lr=1e-3, device="cuda", use_amp=True)
# trainer.fit(train_loader, val_loader=val_loader, epochs=50)
# trainer.save("nb_decoder.pt")
