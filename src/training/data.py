import torch
import random
import numpy as np

def sample_map(pi: np.ndarray, batch_size: int = 256, replace: bool = True):
    # 计算行的概率分布
    row_sums = pi.sum(axis=1)
    total_sum = row_sums.sum()
    row_probs = row_sums / total_sum  # 计算 i 的概率

    # 先根据行概率分布采样 i
    i_samples = np.random.choice(pi.shape[0], p=row_probs, size=batch_size, replace=replace)

    # 计算每个 i 对应行的 j 采样
    j_samples = np.zeros(batch_size, dtype=int)
    for idx, i in enumerate(i_samples):
        # 计算当前行的 j 采样概率
        row_p = pi[i] / row_sums[i]  # 归一化，保证概率和为 1
        j_samples[idx] = np.random.choice(pi.shape[1], p=row_p)

    return i_samples, j_samples


def sample_from_ot_plan(ot_plan: np.ndarray, x0: torch.Tensor, x1: torch.Tensor, batch_size: int = 256):
    i, j = sample_map(ot_plan, batch_size, replace=True)
    return x0[i], x1[j], i, j  # 只返回索引 i

def compute_xt_ut_gt(t_samp, x0, x1, mass0, mass1, delta):

    index = torch.norm(x1 - x0, dim=1) < torch.pi * delta
    # print(torch.sum(index).item()/len(index))

    t_samp = t_samp[index]
    x0 = x0[index]
    x1 = x1[index]
    mass0 = mass0[index]
    mass1 = mass1[index]

    diff = x1 - x0
    norm = torch.norm(diff, dim=1, keepdim=True)   # 按行算模长 (欧氏范数)
    norm_vector = diff / (norm + 1e-9)  # 避免除以零，保持与 diff 相同的形状

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
    inv_sqrt_Am0_m_Bsq = 2*delta / (omega + 1e-9) # [batch, 1]

    xt_samp = x0 + omega_vector * (inv_sqrt_Am0_m_Bsq * (torch.arctan((A*t_samp - B)*inv_sqrt_Am0_m_Bsq) - torch.arctan(-B*inv_sqrt_Am0_m_Bsq)))

    # # add random noise
    # xt_samp = xt_samp + torch.randn_like(xt_samp) * 5e-2

    masst_samp = A*t_samp**2 - 2*B*t_samp + mass0
    dmasst_dt = (2*A*t_samp - 2*B)
    gt_samp = dmasst_dt / masst_samp

    ut_samp = omega_vector / masst_samp

    return xt_samp, gt_samp, ut_samp, masst_samp/mass0, index

# def get_batch(X, t_train, batch_size, gamma0_plans, gamma1_plans, delta, ratios):
def get_batch(adata_control, adata_treated, all_conditions, batch_size_per_condition, 
              batch_size_condition, gamma0_plans, gamma1_plans, delta, 
              sample_rep = "X_pca", # X_ae; X_state
              control_key = "is_control",
              condition_keys = "target_gene",
              condition_rep_keys = "gene_embeddings",
              device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    ts = []
    xts = []
    uts = []
    gts = []
    massts = []
    cons = []
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_control = adata_control.obsm[sample_rep]
    
    sample_conditions_index = random.sample(range(len(all_conditions)), batch_size_condition)
    for i in range(len(sample_conditions_index)):        
               
        cur_con_index = sample_conditions_index[i]
        cur_con = all_conditions[cur_con_index]
        cur_adata_treated = adata_treated[adata_treated.obs[condition_keys]==cur_con]

        gamma0_plan = gamma0_plans[cur_con_index]
        gamma1_plan = gamma1_plans[cur_con_index]

        
        x0, x1, idx_0, idx_1 = sample_from_ot_plan(gamma0_plan, X_control, cur_adata_treated.obsm[sample_rep], batch_size_per_condition)

        x0 = torch.from_numpy(x0).float().to(device)
        x1 = torch.from_numpy(x1).float().to(device)

        mass0 = torch.from_numpy(gamma0_plan[idx_0, idx_1].reshape(-1, 1)).float().to(device)
        mass1 = torch.from_numpy(gamma1_plan[idx_0, idx_1].reshape(-1, 1)).float().to(device)
        

        t_samp = torch.rand(x0.shape[0], 1).type_as(x0)

        xt_samp, gt_samp, ut_samp, masst_samp, index = compute_xt_ut_gt(t_samp, x0, x1, mass0, mass1, delta)

        ts.append(t_samp[index])
        xts.append(xt_samp)
        uts.append(ut_samp)
        gts.append(gt_samp)
        massts.append(masst_samp)

        cur_con = cur_adata_treated.obsm[condition_rep_keys][0, :]
        cur_con = torch.tensor(cur_con, dtype=torch.float32).repeat(len(xt_samp), 1).to(device)
        cons.append(cur_con)
    
    return torch.cat(ts), torch.cat(xts), torch.cat(uts), torch.cat(gts), torch.cat(massts), torch.cat(cons)