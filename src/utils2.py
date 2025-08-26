import ot
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

import math

def compute_uot_plans(X, t_train, delta=1, draw=False):
    uot_plans = []
    gamma0_plans = []
    gamma1_plans = []

    for i in tqdm(range(len(t_train)-1), desc="Computing UOT plans..."):
        X_source = X[i]
        X_target = X[i+1]
        n_source, n_target = X_source.shape[0], X_target.shape[0] 
        norm_2_dist = ot.dist(X_source, X_target, metric='euclidean')

        # # 应用公式计算成本矩阵
        # angle = norm_2_dist / (2 * delta)
        # angle = np.minimum(angle, np.pi/2)  # 限制在[0, π/2]范围内
        # cos_val = np.cos(angle)
        # cos_sq = cos_val ** 2
        # # 避免对数计算中的零值（用一个小正数替换零）
        # cos_plus_sq = np.where(cos_sq == 0, 1e-10, cos_sq)
        # # 最终成本矩阵
        # cost_matrix = -np.log(cos_plus_sq)

        cos_sq = np.cos(np.minimum(norm_2_dist / (2 * delta), np.pi/2))**2
        cost_matrix = -np.log(np.where(cos_sq == 0, 1e-10, cos_sq))

        a = np.ones(n_source)
        b = np.ones(n_target)

        G = ot.unbalanced.mm_unbalanced(a, b, cost_matrix, reg_m=[1.0,1.0])

        uot_plans.append(G)
        gamma0_plans.append(((a / G.sum(1))[:, None]) * G)
        gamma1_plans.append((b/G.sum(0))*G)

        print(((gamma1_plans[i]- gamma0_plans[i])<0).any())
        
        if draw:
            source_pred = G.sum(1)
            tar_pred = G.sum(0)
            
            fig = plt.figure(figsize=(15, 5))

            plt.subplot(131)
            plt.plot(a, label = f'source_true_{i}')
            plt.plot(source_pred, label = f'source_pred_{i}')
            plt.legend()

            plt.subplot(132)
            plt.plot(b, label = f'target_true_{i+1}')
            plt.plot(tar_pred, label = f'target_pred_{i+1}')
            plt.legend()
            
            plt.subplot(133)
            plt.scatter(X_source[:,0],X_source[:,1],s=source_pred*10, alpha=0.5)
            plt.show()


    return uot_plans, gamma0_plans, gamma1_plans


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
    i, j = sample_map(ot_plan, batch_size, replace=False)
    return x0[i], x1[j], i, j  # 只返回索引 i

def compute_xt_ut_gt(t_relative, delta_t, x0, x1, mass0, mass1, delta):

    index = torch.norm(x1 - x0, dim=1) < torch.pi * delta

    t_relative = t_relative[index]
    x0 = x0[index]
    x1 = x1[index]
    mass0 = mass0[index]
    mass1 = mass1[index]

    tau = torch.tan((x1 - x0) / (2 * delta))   # [batch, dim]

    diff = x1 - x0
    norm = torch.norm(diff, dim=1)   # 按行算模长 (欧氏范数)
    count = torch.sum(norm > (torch.pi * delta))
    if count.item() > 0:
        print('waring')
    # print(count.item())

    # 每个 batch 求 tau 的平方和，再开方，得到标量
    tau_norm_sq = (tau ** 2).sum(dim=1, keepdim=True)   # [batch, 1]

    # sqrt(m0*m1/(1+tau_norm_sq)) 是 batch 级别的标量
    scale = torch.sqrt(mass0 * mass1 / (1 + tau_norm_sq))   # [batch, 1]

    # 最终 omega 逐维度输出
    omega = 2 * delta * tau * scale   # [batch, dim]

    A = mass1 + mass0 - 2 * scale   # [batch, 1]

    B = mass0 - scale # [batch, 1]

    # inv_sqrt_Am0_m_Bsq = 1 / torch.sqrt(A * mass0 - B ** 2 + 1e-9)  # [batch, 1]
    inv_sqrt_Am0_m_Bsq = 2*delta/(torch.sqrt((omega ** 2).sum(dim=1, keepdim=True)) + 1e-9)

    xt_samp = x0 + omega * (inv_sqrt_Am0_m_Bsq * (torch.arctan((A*t_relative - B)*inv_sqrt_Am0_m_Bsq) - torch.arctan(-B*inv_sqrt_Am0_m_Bsq)))

    masst_samp = A*t_relative**2 - 2*B*t_relative + mass0
    dmasst_dt = 2*A*t_relative - 2*B
    gt_samp = dmasst_dt / masst_samp * (1/delta_t)

    ut_samp = omega / masst_samp * (1/delta_t)

    return xt_samp, gt_samp, ut_samp, index


def get_batch(X, t_train, batch_size, gamma0_plans, gamma1_plans, delta, ratios):
    ts = []
    xts = []
    uts = []
    gts = []
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for t in range(len(t_train)-1): 
               
        gamma0_plan = gamma0_plans[t]
        gamma1_plan = gamma1_plans[t]

        ratio = ratios[t]
        
        # x0, x1, idx = sample_from_ot_plan(uot_plan, X[t], X[t+1], batch_size)
        x0, x1, idx_0, idx_1 = sample_from_ot_plan(gamma0_plan, X[t], X[t+1], batch_size)

        x0 = torch.from_numpy(x0).float().to(device)
        x1 = torch.from_numpy(x1).float().to(device)

        mass0 = torch.from_numpy(gamma0_plan[idx_0, idx_1].reshape(-1, 1)).float().to(device)
        mass1 = torch.from_numpy(gamma1_plan[idx_0, idx_1].reshape(-1, 1)).float().to(device)
        
        delta_t = t_train[t+1] - t_train[t]
        t_relative = torch.rand(x0.shape[0], 1).type_as(x0)
        t_samp = delta_t*t_relative

        xt_samp, gt_samp, ut_samp, index = compute_xt_ut_gt(t_relative, delta_t, x0, x1, mass0, mass1, delta)
        
        ts.append(t_samp[index] + t_train[t])
        xts.append(xt_samp)
        uts.append(ut_samp)
        gts.append(gt_samp)
    
    return torch.cat(ts), torch.cat(xts), torch.cat(uts), torch.cat(gts)