import torch
import torch.nn as nn
import math

class FiLMResBlock(nn.Module):
    def __init__(self, dim, con_dim, time_dim,bottle_dim=512, activation=None, dropout=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
       # self.norm2 = nn.LayerNorm(dim)
        
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        
        if(activation=="GELU"):
            self.activation = nn.GELU()
        elif(activation=="SiLU"):
            self.activation = nn.SiLU()
        
        self.dropout = nn.Dropout(dropout)
        
        # self.film_gen_c = nn.Linear(con_dim, dim * 2) 
        # self.film_gen_t = nn.Linear(time_dim, dim * 2)
        self.film_fuse = nn.Sequential(
            nn.Linear(con_dim + time_dim, bottle_dim),
            self.activation,
            nn.Linear(bottle_dim, 2 * dim)
        )

        # 初始化为0 tricks
        nn.init.zeros_(self.film_fuse[-1].weight)
        nn.init.zeros_(self.film_fuse[-1].bias)

    def forward(self, x, t_emb, c_emb):
        # x: [Batch, dim]
        # t_emb: [Batch, time_dim]
        # c_emb: [Batch, con_dim]
        
        # params_c = self.film_gen_c(c_emb)
        # params_t = self.film_gen_t(t_emb)
        # film_params = params_c + params_t # 过一层linear然后相加
        film_params = self.film_fuse(torch.cat([c_emb, t_emb], dim=-1))
        
        gammas, betas = film_params.chunk(2, dim=-1)
        
        residual = x
        
        out = self.norm1(x)
        out = out * (1 + gammas) + betas
        
        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return residual + out


class Velocity_GrowthNet(nn.Module):
    def __init__(self, in_dim, out_dim, con_embedding_dim, time_dim, hidden_dim, n_hiddens, bottle_dim, activation=None):
        super().__init__()

        if(activation=="GELU"):
            self.activation = nn.GELU()
        elif(activation=="SiLU"):
            self.activation = nn.SiLU()

        self.fc_in = nn.Linear(in_dim, hidden_dim)
        
        self.blocks = nn.ModuleList([
            FiLMResBlock(hidden_dim, con_embedding_dim, time_dim,bottle_dim, activation=activation) for _ in range(n_hiddens)
        ])
        
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, t, x, c):
        
        h = self.fc_in(x)
        
        for block in self.blocks:
            h = block(h, t, c) 
            
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



# 疑似可能参数有点太多了 但也未必
class FNet(nn.Module):
    def __init__(self, in_out_dim, hidden_dim_v=4096, n_hiddens_v=4, hidden_dim_g=2048, n_hiddens_g=2,
                 condition_dim=1280, con_embedding_dim=512, hidden_dim_con=1024, 
                 time_dim=1024,time_embedding_dim=256, hidden_dim_time=512, 
                 bottle_dim=512, activation='GELU'):
        super().__init__()
        
        self.time_dim = time_dim
        self.time_embedding_dim = time_embedding_dim
        self.t_encoder = TimeEncoder(dim=self.time_dim)
        
        if(activation=="GELU"):
            self.activation = nn.GELU()
        elif(activation=="SiLU"):
            self.activation = nn.SiLU()

        self.con_embedding_dim = con_embedding_dim
        
        self.condition_encoder = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, hidden_dim_con),
            self.activation,
            nn.Linear(hidden_dim_con, hidden_dim_con),
            self.activation,
            nn.Linear(hidden_dim_con, con_embedding_dim)
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(time_dim,hidden_dim_time),
            self.activation,
            nn.Linear(hidden_dim_time,time_embedding_dim)
        )
        
        self.v_net = Velocity_GrowthNet(
            in_out_dim, in_out_dim, self.con_embedding_dim, self.time_embedding_dim, 
            hidden_dim=hidden_dim_v, n_hiddens=n_hiddens_v,
            bottle_dim=bottle_dim, activation = activation
        )
        
        self.g_net = Velocity_GrowthNet(
            in_out_dim, 1, self.con_embedding_dim, self.time_embedding_dim, 
            hidden_dim=hidden_dim_g, n_hiddens=n_hiddens_g,
            bottle_dim=bottle_dim, activation = activation
        )

    def forward(self, t, z, con):
        c = self.condition_encoder(con)
        t_embed = self.t_encoder(t)
        t_embed = self.time_encoder(t_embed)
        
        v = self.v_net(t_embed, z, c)
        g = self.g_net(t_embed, z, c)

        return v, g