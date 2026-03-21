import torch
import random
import numpy as np
from scipy import sparse  


class DataLoaderHelper:
    """
    这是一个辅助类，用来存储训练所需的静态数据
    支持多维协变量切片下的 Control 和 Treated 动态匹配
    """
    def __init__(self, adata_control, adata_treated, 
                 precomputed_results, cov_config ={},
                 sample_rep='X_pca_scaled',
                 condition_rep_keys="gene_embeddings"):
        
        # 1. 存储全局矩阵
        self.X_control_all = adata_control.obsm[sample_rep]
        self.X_treated_all = adata_treated.obsm[sample_rep]

        # 2. 建立 obs_names 到 全局 integer index 的哈希映射，用于极速查找
        control_obs_to_idx = {name: i for i, name in enumerate(adata_control.obs_names)}
        treated_obs_to_idx = {name: i for i, name in enumerate(adata_treated.obs_names)}

        self.gamma0_plans = precomputed_results["gamma0_plans"]
        self.gamma1_plans = precomputed_results["gamma1_plans"]
        self.delta = precomputed_results["delta"]
        self.all_conditions = list(self.gamma0_plans.keys())
        
        # 3. 解析供每个 condition 使用的具体切片索引
        self.control_indices_dict = {}
        self.treated_indices_dict = {}
        self.condition_emb_map = {}
        
        
        print("正在构建局部到全局的索引映射...")
        for con in self.all_conditions:
            # 取出当前 condition 对应的细胞名
            c_names = precomputed_results["control_obs_names"][con]
            t_names = precomputed_results["treat_obs_names"][con]
            
            # 转化为全局整数索引并缓存
            c_indices = np.array([control_obs_to_idx[name] for name in c_names])
            t_indices = np.array([treated_obs_to_idx[name] for name in t_names])
            
            self.control_indices_dict[con] = c_indices
            self.treated_indices_dict[con] = t_indices
            
            # 动态获取 condition embedding：
            # 既然 t_indices 里的细胞都属于这个 condition，我们直接拿第一个细胞的 embedding 即可
            first_t_idx = t_indices[0]
            self.condition_emb_map[con] = adata_treated.obsm[condition_rep_keys][first_t_idx]

               
        self.cov_control_dict = {}
        if cov_config is not None:
            for key, config in cov_config.items():
                if config.get('use_in_model', True):
                    # 获取该协变量列
                    col_data = adata_control.obs[key]
                    
                    if col_data.dtype.name == 'category' or col_data.dtype == object:
                        # 如果是离散类别（Categorical），提取其内部的整数编码，并转为 int64
                        # 深度学习模型中的 Embedding 层需要 long/int64 类型的输入
                        self.cov_control_dict[key] = col_data.cat.codes.to_numpy(dtype=np.int64)
                    else:
                        # 如果是连续数值（例如 dose），转为标准的 float32 numpy 数组
                        self.cov_control_dict[key] = col_data.to_numpy(dtype=np.float32)



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



def get_batch(helper, 
              batch_size_per_condition, 
              batch_size_condition, 
              delta, 
              device):
    """
    基于预计算的 OT Plan 动态采样
    """
    ts, xts, uts, gts, massts, cons = [], [], [], [], [], []
    
    sample_con_names = random.sample(helper.all_conditions, batch_size_condition)
    for cur_con in sample_con_names:
        gamma0_plan = helper.gamma0_plans[cur_con]
        gamma1_plan = helper.gamma1_plans[cur_con]
        
        # 1. 提取当前 Condition 对应的局部切片
        c_indices = helper.control_indices_dict[cur_con]
        t_indices = helper.treated_indices_dict[cur_con]
        
        X_control_cur = helper.X_control_all[c_indices]
        X_treated_cur = helper.X_treated_all[t_indices]
        
        cov_batch_dict = {k: [] for k in helper.cov_control_dict.keys()}
        
        # 2. 从 OT plan 采样。注意：返回的 idx_0, idx_1 是相对 X_control_cur 和 X_treated_cur 的局部索引
        x0, x1, idx_0_rel, idx_1_rel = sample_from_ot_plan(
            gamma0_plan, 
            X_control_cur, 
            X_treated_cur, 
            batch_size_per_condition
        )
        
        x0_tensor = torch.from_numpy(x0).float().to(device)
        x1_tensor = torch.from_numpy(x1).float().to(device)

        # 3. 提取质量 (Mass)
        # 这里 scipy sparse 的切片会返回 matrix，为了稳妥提取为 numpy array 展平
        if sparse.issparse(gamma0_plan):
            m0_val = np.asarray(gamma0_plan[idx_0_rel, idx_1_rel]).reshape(-1, 1)
            m1_val = np.asarray(gamma1_plan[idx_0_rel, idx_1_rel]).reshape(-1, 1)
        else:
            m0_val = gamma0_plan[idx_0_rel, idx_1_rel].reshape(-1, 1)
            m1_val = gamma1_plan[idx_0_rel, idx_1_rel].reshape(-1, 1)
            
        # 注意如果因为稀疏矩阵返回了矩阵对象，可能需要 .squeeze()。如果原逻辑跑得通就保持不变。
        if isinstance(m0_val, np.matrix): m0_val = m0_val.A
        if isinstance(m1_val, np.matrix): m1_val = m1_val.A
        
        mass0 = torch.from_numpy(m0_val).float().to(device)
        mass1 = torch.from_numpy(m1_val).float().to(device)
        
        t_samp = torch.rand(x0_tensor.shape[0], 1, device=device)
        
        # 4. 计算 Flow Matching 目标值
        xt_samp, gt_samp, ut_samp, masst_samp, index = compute_xt_ut_gt(
            t_samp, x0_tensor, x1_tensor, mass0, mass1, delta
        )
        
        ts.append(t_samp[index])
        xts.append(xt_samp)
        uts.append(ut_samp)
        gts.append(gt_samp)
        massts.append(masst_samp)
        
        # 5. 添加 Condition 向量
        cur_emb_np = helper.condition_emb_map[cur_con]
        cur_con_tensor = torch.tensor(cur_emb_np, dtype=torch.float32, device=device)
        cur_con_tensor = cur_con_tensor.unsqueeze(0).repeat(len(xt_samp), 1)
        cons.append(cur_con_tensor)


        # 6. 处理所有 Covariates 信息
        global_idx_0 = c_indices[idx_0_rel]
        for key, val_all in helper.cov_control_dict.items():
            val_np = val_all[global_idx_0]
            val_np = val_np[index.cpu().numpy()] # 过一遍距离截断 filter
            # 无论连续还是离散，先转成 tensor，后续模型里会自动按类型区分 long() 和 float()
            cov_tensor = torch.from_numpy(val_np).to(device)
            cov_batch_dict[key].append(cov_tensor)

    # 7. 拼接并返回
    final_cov_dict = {k: torch.cat(v) for k, v in cov_batch_dict.items()}
    return (torch.cat(ts), torch.cat(xts), torch.cat(uts), 
            torch.cat(gts), torch.cat(massts), torch.cat(cons),
            final_cov_dict)
