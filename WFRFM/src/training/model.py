import torch
import torch.nn as nn
import math

class FiLMResBlock(nn.Module):
    def __init__(self, dim, con_dim, time_dim, cov_dim, bottle_dim=512, expand_ratio=2, activation=None, dropout=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False) 

        hidden_features = int(dim * expand_ratio)
        self.fc1 = nn.Linear(dim, hidden_features)
        self.fc2 = nn.Linear(hidden_features, dim)
        
        if activation == "GELU":
            self.activation = nn.GELU()
        elif activation == "SiLU":
            self.activation = nn.SiLU()
        else:
            raise ValueError("unknown activation")
        
        self.dropout = nn.Dropout(dropout)
        
        # 融合 Condition, Time 和所有 Covariates 的维度
        fuse_dim = con_dim + time_dim + cov_dim
        
        self.film_fuse = nn.Sequential(
            nn.Linear(fuse_dim, bottle_dim),
            self.activation,
            nn.Linear(bottle_dim, 3 * dim)
        )

        # 初始化为0，使初始状态接近恒等变换
        nn.init.zeros_(self.film_fuse[-1].weight)
        nn.init.zeros_(self.film_fuse[-1].bias)

    def forward(self, x, c_emb, t_emb, cov_emb):
        # cov_emb 是所有协变量拼接后的向量
        film_params = self.film_fuse(torch.cat([c_emb, t_emb, cov_emb], dim=-1))
        
        gammas, betas, gates = film_params.chunk(3, dim=-1)
        
        residual = x
        out = self.norm1(x)
        out = out * (1 + gammas) + betas
        
        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = out * gates
        
        return residual + out

class Velocity_GrowthNet(nn.Module):
    def __init__(self, in_dim, out_dim, con_embedding_dim, time_dim, cov_dim, hidden_dim, n_hiddens, bottle_dim, activation=None, dropout=0.1):
        super().__init__()

        self.fc_in = nn.Linear(in_dim, hidden_dim)
        
        self.blocks = nn.ModuleList([
            FiLMResBlock(hidden_dim, con_embedding_dim, time_dim, cov_dim,
                         bottle_dim=bottle_dim, 
                         expand_ratio=2,
                         activation=activation, 
                         dropout=dropout) 
            for _ in range(n_hiddens)
        ])
        
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, c, t_emb, cov_emb):
        h = self.fc_in(x)
        for block in self.blocks:
            h = block(h, c, t_emb, cov_emb)
        return self.fc_out(h)

class TimeEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        device = t.device
        if t.dim() == 1: t = t.unsqueeze(-1)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class FNet(nn.Module):
    def __init__(self, in_out_dim, cov_config, covariate_info, 
                 hidden_dim_v=4096, n_hiddens_v=4, hidden_dim_g=2048, n_hiddens_g=2,
                 condition_dim=1280, con_embedding_dim=512, hidden_dim_con=1024, 
                 time_dim=1024, time_embedding_dim=256, hidden_dim_time=512, 
                 bottle_dim=512, cov_emb_dim=64, # 每个协变量默认分配的维度
                 dropout=0.1, activation='GELU'):
        super().__init__()
        
        self.t_encoder = TimeEncoder(dim=time_dim)
        self.activation = nn.GELU() if activation == "GELU" else nn.SiLU()
        self.dropout = dropout

        # 1. Condition Encoder
        self.condition_encoder = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, hidden_dim_con),
            self.activation,
            nn.Linear(hidden_dim_con, hidden_dim_con),
            self.activation,
            nn.Linear(hidden_dim_con, con_embedding_dim)
        )

        # 2. Time Encoder MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim_time),
            self.activation,
            nn.Linear(hidden_dim_time, time_embedding_dim)
        )

        # 3. Dynamic Covariate Encoders
        self.cov_encoders = nn.ModuleDict()
        self.total_cov_dim = 0
        self.cov_keys = []

        for key, config in cov_config.items():
            if not config.get('use_in_model', True):
                continue
            
            self.cov_keys.append(key)
            if config['type'] == 'categorical':
                # 获取该类别对应的数量
                num_classes = len(covariate_info['categorical_mappings'][key])
                self.cov_encoders[key] = nn.Embedding(num_classes, cov_emb_dim)
                self.total_cov_dim += cov_emb_dim
            else: # continuous
                # 连续变量通过一个小MLP或Linear转为embedding
                self.cov_encoders[key] = nn.Sequential(
                    nn.Linear(1, cov_emb_dim // 2),
                    self.activation,
                    nn.Linear(cov_emb_dim // 2, cov_emb_dim)
                )
                self.total_cov_dim += cov_emb_dim

        # 4. Networks
        self.v_net = Velocity_GrowthNet(
            in_out_dim, in_out_dim, con_embedding_dim, time_embedding_dim, self.total_cov_dim,
            hidden_dim=hidden_dim_v, n_hiddens=n_hiddens_v,
            bottle_dim=bottle_dim, activation=activation, dropout=self.dropout
        )
        
        self.g_net = Velocity_GrowthNet(
            in_out_dim, 1, con_embedding_dim, time_embedding_dim, self.total_cov_dim,
            hidden_dim=hidden_dim_g, n_hiddens=n_hiddens_g,
            bottle_dim=bottle_dim, activation=activation, dropout=self.dropout
        )

    def forward(self, t, z, con, cov_dict):
        """
        cov_dict: 一个字典，包含所有在 cov_config 中 use_in_model 为 True 的 key 对应的 Tensor
        """
        # Encode Condition and Time
        c = self.condition_encoder(con)
        t_emb = self.t_encoder(t)
        t_emb = self.time_mlp(t_emb)

        # Encode Covariates
        cov_embs = []
        for key in self.cov_keys:
            val = cov_dict[key]
            if isinstance(self.cov_encoders[key], nn.Embedding):
                # 类别型，确保是 LongTensor
                cov_embs.append(self.cov_encoders[key](val.long()))
            else:
                # 连续型，确保维度是 [Batch, 1]
                if val.dim() == 1: val = val.unsqueeze(-1)
                cov_embs.append(self.cov_encoders[key](val.float()))
        
        full_cov_emb = torch.cat(cov_embs, dim=-1)

        # Forward through nets
        v = self.v_net(z, c, t_emb, full_cov_emb)
        g = self.g_net(z, c, t_emb, full_cov_emb)

        return v, g
