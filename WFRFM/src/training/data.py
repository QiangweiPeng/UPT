import torch
import random
import numpy as np
from scipy import sparse  

import numpy as np

class DataLoaderHelper:
    """
    基于 Global Rulebook 和 Tuple Identity 的训练数据辅助类。
    全面支持按 model_source 动态路由协变量。
    """
    def __init__(self, adata_control, adata_treated, 
                 precomputed_results, 
                 sample_rep='X_pca_scaled',
                 condition_rep_keys="gene_embeddings"):
        
        self.rulebook = adata_treated.uns.get('global_rulebook')
        if self.rulebook is None:
            raise ValueError("未找到 global_rulebook，请先运行 prepare_covariates 预处理数据！")
            
        self.X_control_all = adata_control.obsm[sample_rep]
        self.X_treated_all = adata_treated.obsm[sample_rep]

        # 预加载所有需要的协变量为 Numpy Array，极大加速 __getitem__ 时的切片速度
        self.cov_arrays = {"control": {}, "treated": {}}
        self.cov_model_sources = {} 
        
        print("正在根据 Rulebook 构建协变量路由表...")
        for cov_type in ["categorical", "continuous"]:
            for cov_name, info in self.rulebook["model_inputs"][cov_type].items():
                obs_col = info["obs_col"]        # 例如 "celltype_idx" 或 "dose_value_scaled"
                m_source = info["model_source"]  # "control", "perturbed", 或 "both"
                
                self.cov_model_sources[cov_name] = m_source
                dtype = np.int64 if cov_type == "categorical" else np.float32
                
                # 直接抽取预处理好的完美特征列，存为底层 Array
                self.cov_arrays["control"][cov_name] = adata_control.obs[obs_col].to_numpy(dtype=dtype)
                self.cov_arrays["treated"][cov_name] = adata_treated.obs[obs_col].to_numpy(dtype=dtype)


        # 整数索引
        control_obs_to_idx = {name: i for i, name in enumerate(adata_control.obs_names)}
        treated_obs_to_idx = {name: i for i, name in enumerate(adata_treated.obs_names)}

        self.gamma0_plans = precomputed_results["gamma0_plans"]
        self.gamma1_plans = precomputed_results["gamma1_plans"]
        self.delta = precomputed_results.get("delta", 1.0)
        
        # 这里的 all_conditions 实际上就是精准的 Tuple 字符串列表，例如 ["(DrugA, A549, 1.0)", ...]
        self.all_conditions = list(self.gamma0_plans.keys())
        
        self.control_indices_dict = {}
        self.treated_indices_dict = {}
        self.condition_emb_map = {}
        
        print("正在构建局部到全局的 OT 索引映射...")
        for con in self.all_conditions:
            c_names = precomputed_results["control_obs_names"][con]
            t_names = precomputed_results["treat_obs_names"][con]
            
            c_indices = np.array([control_obs_to_idx[name] for name in c_names])
            t_indices = np.array([treated_obs_to_idx[name] for name in t_names])
            
            self.control_indices_dict[con] = c_indices
            self.treated_indices_dict[con] = t_indices
            
            # 提取 condition embedding
            if condition_rep_keys in adata_treated.obsm:
                first_t_idx = t_indices[0]
                self.condition_emb_map[con] = adata_treated.obsm[condition_rep_keys][first_t_idx]
            else:
                self.condition_emb_map[con] = None 
                
        print("DataLoaderHelper 构建完成！")



    def get_covariates_for_pair(self, c_idx, t_idx):
        """
        根据 rulebook 里的 model_source 规则，自动组装一对 (Control, Treated) 细胞的协变量。
        返回: 离散协变量字典 (cat_covs), 连续协变量字典 (cont_covs)
        """
        cat_covs = {}
        cont_covs = {}
        
        # 遍历 categorical
        for cov_name, info in self.rulebook["model_inputs"]["categorical"].items():
            m_source = self.cov_model_sources[cov_name]
            # 核心路由逻辑：该找谁要数据？
            if m_source == "control" or m_source == "base_cell":
                val = self.cov_arrays["control"][cov_name][c_idx]
            else: # "perturbed", "wishlist", "both"
                val = self.cov_arrays["treated"][cov_name][t_idx]
            cat_covs[cov_name] = val
            
        # 遍历 continuous
        for cov_name, info in self.rulebook["model_inputs"]["continuous"].items():
            m_source = self.cov_model_sources[cov_name]
            if m_source == "control" or m_source == "base_cell":
                val = self.cov_arrays["control"][cov_name][c_idx]
            else:
                val = self.cov_arrays["treated"][cov_name][t_idx]
            cont_covs[cov_name] = val
            
        return cat_covs, cont_covs




def sample_from_ot_plan(ot_plan, x0, x1, batch_size=256):
    i, j = sample_map(ot_plan, batch_size, replace=True)
    return x0[i], x1[j], i, j  

def sample_map(pi, batch_size, replace=True):
    """
    从 OT plan (pi) 中采样 batch_size 个 (i, j) 对。
    兼容 Dense Numpy Array 和 Sparse Matrix，并增加了空矩阵降级保护。
    """
    n_source, n_target = pi.shape
    
    # 极端防崩溃保护：以防某个 condition 匹配到了 0 个细胞
    if n_source == 0 or n_target == 0:
        raise ValueError(f"严重错误: 试图采样的 OT Plan 维度为 ({n_source}, {n_target})，请检查预处理数据切片！")

    if sparse.issparse(pi):
        pi_coo = pi.tocoo()
        
        # 【核心修复】：如果稀疏矩阵全为空（在 OT 计算阶段全被 threshold 过滤掉了）
        if pi_coo.nnz == 0 or pi_coo.data.sum() == 0:
            # 降级策略：不查概率了，直接在 Control 和 Treated 之间进行均匀随机采样
            i = np.random.choice(n_source, size=batch_size, replace=True)
            j = np.random.choice(n_target, size=batch_size, replace=True)
            return i, j
            
        probs = pi_coo.data / pi_coo.data.sum()
        sampled_idx = np.random.choice(len(probs), size=batch_size, p=probs, replace=replace)
        return pi_coo.row[sampled_idx], pi_coo.col[sampled_idx]

    else:
        # 处理 Dense 矩阵
        flat_pi = pi.flatten()
        sum_pi = flat_pi.sum()
        
        # 【优化】：密集矩阵的降级策略同样改成了更高效的直接行列随机，省去构建巨型均分 probs 数组
        if sum_pi == 0:
            i = np.random.choice(n_source, size=batch_size, replace=True)
            j = np.random.choice(n_target, size=batch_size, replace=True)
            return i, j
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


import random
import numpy as np
import torch
from scipy import sparse

def get_batch(helper, 
              batch_size_per_condition, 
              batch_size_condition, 
              delta, 
              device):
    """
    基于预计算的 OT Plan 动态采样。
    彻底打通 Global Rulebook，并根据 model_source 自动从 Control 或 Treated 切片数据。
    """
    ts, xts, uts, gts, massts, cons = [], [], [], [], [], []
    
    cov_batch_dict = {}
    for cov_type in ["categorical", "continuous"]:
        for cov_name in helper.rulebook["model_inputs"][cov_type].keys():
            cov_batch_dict[cov_name] = []
            
    sample_con_names = random.sample(helper.all_conditions, batch_size_condition)
    for cur_con in sample_con_names:
        gamma0_plan = helper.gamma0_plans[cur_con]
        gamma1_plan = helper.gamma1_plans[cur_con]
        
        # 1. 提取当前 Condition 对应的局部切片
        c_indices = helper.control_indices_dict[cur_con]
        t_indices = helper.treated_indices_dict[cur_con]
        
        X_control_cur = helper.X_control_all[c_indices]
        X_treated_cur = helper.X_treated_all[t_indices]
        
        # 2. 从 OT plan 采样。注意：返回的 idx_0_rel, idx_1_rel 是相对局部索引
        x0, x1, idx_0_rel, idx_1_rel = sample_from_ot_plan( # 假设此函数在外部定义
            gamma0_plan, 
            X_control_cur, 
            X_treated_cur, 
            batch_size_per_condition
        )
        
        x0_tensor = torch.from_numpy(x0).float().to(device)
        x1_tensor = torch.from_numpy(x1).float().to(device)

        # 3. 提取质量 (Mass)
        if sparse.issparse(gamma0_plan):
            m0_val = np.asarray(gamma0_plan[idx_0_rel, idx_1_rel]).reshape(-1, 1)
            m1_val = np.asarray(gamma1_plan[idx_0_rel, idx_1_rel]).reshape(-1, 1)
        else:
            m0_val = gamma0_plan[idx_0_rel, idx_1_rel].reshape(-1, 1)
            m1_val = gamma1_plan[idx_0_rel, idx_1_rel].reshape(-1, 1)
            
        if isinstance(m0_val, np.matrix): m0_val = m0_val.A
        if isinstance(m1_val, np.matrix): m1_val = m1_val.A
        
        mass0 = torch.from_numpy(m0_val).float().to(device)
        mass1 = torch.from_numpy(m1_val).float().to(device)
        
        t_samp = torch.rand(x0_tensor.shape[0], 1, device=device)
        
        # 4. 计算 Flow Matching 目标值 (此处的 index 是截断/过滤后的掩码)
        xt_samp, gt_samp, ut_samp, masst_samp, index = compute_xt_ut_gt( # 假设此函数在外部定义
            t_samp, x0_tensor, x1_tensor, mass0, mass1, delta
        )
        
        ts.append(t_samp[index])
        xts.append(xt_samp)
        uts.append(ut_samp)
        gts.append(gt_samp)
        massts.append(masst_samp)
        
        # 5. 添加 Condition 向量
        cur_emb_np = helper.condition_emb_map[cur_con]
        if cur_emb_np is not None:
            cur_con_tensor = torch.tensor(cur_emb_np, dtype=torch.float32, device=device)
            cur_con_tensor = cur_con_tensor.unsqueeze(0).repeat(len(xt_samp), 1)
            cons.append(cur_con_tensor)

        # ==========================================
        # 6. 【杀手锏级提速】：基于底层 Numpy 的向量化协变量路由
        # ==========================================
        # 首先，将采样的相对索引还原为全局的整数索引
        global_idx_0 = c_indices[idx_0_rel]
        global_idx_1 = t_indices[idx_1_rel]
        
        # 提取 Flow Matching 中过滤后的有效掩码
        valid_mask = index.cpu().numpy()
        valid_global_idx_0 = global_idx_0[valid_mask]
        valid_global_idx_1 = global_idx_1[valid_mask]
        
        # 遍历法典中的所有输入变量
        for cov_type in ["categorical", "continuous"]:
            for cov_name in helper.rulebook["model_inputs"][cov_type].keys():
                m_source = helper.cov_model_sources[cov_name]
                
                # 【核心路由分发】：依据 model_source 决定去 Control 还是 Treated 的数组里抓取
                if m_source in ["control", "base_cell"]:
                    val_np = helper.cov_arrays["control"][cov_name][valid_global_idx_0]
                else: 
                    # "perturbed", "wishlist", "both" 等情况去目标端提取
                    val_np = helper.cov_arrays["treated"][cov_name][valid_global_idx_1]
                
                # 动态分配数据类型：Embedding 需要 long (int64)，连续特征需要 float32
                target_dtype = torch.long if cov_type == "categorical" else torch.float32
                cov_tensor = torch.from_numpy(val_np).to(device, dtype=target_dtype)
                
                cov_batch_dict[cov_name].append(cov_tensor)

    # 7. 拼接并返回
    final_cov_dict = {k: torch.cat(v) for k, v in cov_batch_dict.items()}
    cons_tensor = torch.cat(cons) if cons else None
    
    return (torch.cat(ts), torch.cat(xts), torch.cat(uts), 
            torch.cat(gts), torch.cat(massts), cons_tensor,
            final_cov_dict)
