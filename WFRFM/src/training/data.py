import torch
import random
import numpy as np
from scipy import sparse  

class DataLoaderHelper:
    """
    这是一个辅助类，用来存储训练所需的静态数据
    避免在 get_batch 循环里反复查找 adata
    """
    def __init__(self, adata_control, adata_treated, 
                 precomputed_results, 
                 sample_rep='X_pca_scaled',
                 condition_keys="target_gene",
                 condition_rep_keys="gene_embeddings"):
        
        self.X_control = adata_control.obsm[sample_rep]
        self.X_treated_all = adata_treated.obsm[sample_rep]

        self.condition_emb_map = {}
        unique_cons = adata_treated.obs[condition_keys].unique()
        
        df = adata_treated.obs[[condition_keys]]
        for con in unique_cons:
            idx = np.where(adata_treated.obs[condition_keys] == con)[0][0]
            self.condition_emb_map[con] = adata_treated.obsm[condition_rep_keys][idx]
            

        self.treated_indices_map = adata_treated.obs.groupby(condition_keys).indices

        # utils里改成了字典
        self.gamma0_plans = precomputed_results["gamma0_plans"]
        self.gamma1_plans = precomputed_results["gamma1_plans"]
        self.delta = precomputed_results["delta"]
        
        self.all_conditions = list(self.gamma0_plans.keys())
        
        print("data_loaded")


def sample_from_ot_plan(ot_plan, x0, x1, batch_size = 256):
    i, j = sample_map(ot_plan, batch_size, replace=True)
    return x0[i], x1[j], i, j  # 只返回索引 i

def sample_map(pi, batch_size, replace=True):
    """
    从 OT plan (pi) 中采样 batch_size 个 (i, j) 对。
    兼容 Dense Numpy Array 和 Sparse Matrix。
    """

    if sparse.issparse(pi):
        pi_coo = pi.tocoo()
        probs = pi_coo.data / pi_coo.data.sum()
        sampled_idx = np.random.choice(len(probs), size=batch_size, p=probs, replace=replace)
        return pi_coo.row[sampled_idx], pi_coo.col[sampled_idx]

    else:
        n_source, n_target = pi.shape
        flat_pi = pi.flatten()
        
        sum_pi = flat_pi.sum()
        if sum_pi == 0:
            probs = np.ones_like(flat_pi) / len(flat_pi)
        else:
            probs = flat_pi / sum_pi
        
        flat_indices = np.random.choice(len(flat_pi), size=batch_size, p=probs, replace=replace)
        
        i = flat_indices // n_target
        j = flat_indices % n_target
        
        return i, j


def compute_xt_ut_gt(t_samp, x0, x1, mass0, mass1, delta):

    EPS = 1e-9
  
    index = torch.norm(x1 - x0, dim=1) < (torch.pi * delta * 0.99)
    # print(torch.sum(index).item()/len(index))

    t_samp = t_samp[index]
    x0 = x0[index]
    x1 = x1[index]
    mass0 = mass0[index]
    mass1 = mass1[index]

    # mass应该大于 0 保证数值稳定
    mass0 = torch.clamp(mass0, min=EPS)
    mass1 = torch.clamp(mass1, min=EPS)
      

    diff = x1 - x0
    norm = torch.norm(diff, dim=1, keepdim=True)   # 按行算模长 (欧氏范数)
    norm_vector = diff / (norm + EPS)  # 避免除以零，保持与 diff 相同的形状


    # 防止出现 tan(pi/2)
    max_dist = torch.pi * delta * (1-EPS)
    norm = torch.clamp(norm, max=max_dist)
    tau = torch.tan(norm / (2 * delta))   # [batch, 1]

    count = torch.sum(norm > (torch.pi * delta))
    if count.item() > 0:
        print('waring')

    # sqrt(m0*m1/(1+tau_sq)) 是 batch 级别的标量
    scale = torch.sqrt(mass0 * mass1 / (1 + tau**2))   # [batch, 1]

    omega = 2 * delta * tau * scale   # [batch, 1]
    omega_vector = omega * norm_vector  # [batch, dim]

    A = mass1 + mass0 - 2 * scale   # [batch, 1]

    B = mass0 - scale # [batch, 1]


    # inv_sqrt_Am0_m_Bsq = 1 / torch.sqrt(A * mass0 - B ** 2 + 1e-9)  # [batch, 1]
    inv_sqrt_Am0_m_Bsq = 2*delta / (omega + EPS) # [batch, 1]

    xt_samp = x0 + omega_vector * (inv_sqrt_Am0_m_Bsq * (torch.arctan((A*t_samp - B)*inv_sqrt_Am0_m_Bsq) - torch.arctan(-B*inv_sqrt_Am0_m_Bsq)))

    # add random noise
    xt_samp = xt_samp + torch.randn_like(xt_samp) * 5e-2

    masst_samp = A*t_samp**2 - 2*B*t_samp + mass0  
    #中间质量同样应该非负
    masst_samp = torch.clamp(masst_samp, min=EPS)
  
    dmasst_dt = (2*A*t_samp - 2*B)
    gt_samp = dmasst_dt / masst_samp
    ut_samp = omega_vector / masst_samp
    # 怀疑这里是数值不稳定的一大原因 实验发现gt和ut经常会变成inf导致loss变成nan

    return xt_samp, gt_samp, ut_samp, masst_samp/mass0, index


def get_batch(helper, #  DataLoaderHelper 
              batch_size_per_condition, 
              batch_size_condition, 
              delta, 
              device):
    """
    我们把adata在dataloader中预取 在get_batch中便可以对numpy切片
    """

    ts, xts, uts, gts, massts, cons = [], [], [], [], [], []
    
    sample_con_names = random.sample(helper.all_conditions, batch_size_condition)
    for cur_con in sample_con_names:
        gamma0_plan = helper.gamma0_plans[cur_con]
        gamma1_plan = helper.gamma1_plans[cur_con]
        
        global_indices = helper.treated_indices_map[cur_con]
        X_treated_cur = helper.X_treated_all[global_indices]
        
        x0, x1, idx_0, idx_1 = sample_from_ot_plan(
            gamma0_plan, 
            helper.X_control, 
            X_treated_cur, 
            batch_size_per_condition
        )
        
        x0_tensor = torch.from_numpy(x0).float().to(device)
        x1_tensor = torch.from_numpy(x1).float().to(device)

        if sparse.issparse(gamma0_plan):
            m0_val = np.asarray(gamma0_plan[idx_0, idx_1]).reshape(-1, 1)
        else:
            m0_val = gamma0_plan[idx_0, idx_1].reshape(-1, 1)
        if sparse.issparse(gamma1_plan):
            m1_val = np.asarray(gamma1_plan[idx_0, idx_1]).reshape(-1, 1)
        else:
            m1_val = gamma1_plan[idx_0, idx_1].reshape(-1, 1)
        
        mass0 = torch.from_numpy(m0_val).float().to(device)
        mass1 = torch.from_numpy(m1_val).float().to(device)
        
        t_samp = torch.rand(x0_tensor.shape[0], 1, device=device)
        
        xt_samp, gt_samp, ut_samp, masst_samp, index = compute_xt_ut_gt(
            t_samp, x0_tensor, x1_tensor, mass0, mass1, delta
        )
        
        ts.append(t_samp[index])
        xts.append(xt_samp)
        uts.append(ut_samp)
        gts.append(gt_samp)
        massts.append(masst_samp)
        
        cur_emb_np = helper.condition_emb_map[cur_con]
        cur_con_tensor = torch.tensor(cur_emb_np, dtype=torch.float32, device=device)
        cur_con_tensor = cur_con_tensor.unsqueeze(0).repeat(len(xt_samp), 1)
        
        cons.append(cur_con_tensor)

    return (torch.cat(ts), torch.cat(xts), torch.cat(uts), 
            torch.cat(gts), torch.cat(massts), torch.cat(cons))