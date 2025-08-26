import math
import warnings
from functools import partial
from typing import Optional
import numpy as np
import ot as pot
import torch
from .utils import torch_wrapper, generate_traj_and_weights
from torchdyn.core import NeuralODE
from tqdm import tqdm
import pandas as pd
import bisect
import random
from .plots import plot_comparisons

def wasserstein_with_weights(x0: torch.Tensor, weights:torch.Tensor, x1: torch.Tensor, method="exact", reg=0.05, power=1) -> float:
    assert power in [1, 2]
    ot_fn = pot.emd2 if method == "exact" else partial(pot.sinkhorn2, reg=reg)
    a, b = pot.unif(x0.shape[0])*weights.squeeze().cpu().numpy()*x0.shape[0], pot.unif(x1.shape[0])
    # print(a.shape)
    # print(b.shape)
    x0 = x0.reshape(x0.shape[0], -1)
    x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1, p=2)
    if power == 2:
        M = M ** 2
    dist = ot_fn(a, b, M.cpu().numpy())
    return math.sqrt(dist) if power == 2 else dist

def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 1,
    **kwargs,
) -> float:
    assert power == 1 or power == 2
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=1e7)
    if power == 2:
        ret = math.sqrt(ret)
    return ret

def test(v, X, t_train, hold_out, save_dir, epoch, type='global',steps=100, draw=True, sample_num=100):
    t_train = [t for t in t_train if t is not None]
    t_train_copy = t_train.copy()
    if hold_out is not None:
        hold_out = int(np.squeeze(hold_out))
        bisect.insort(t_train, hold_out)  
    t = t_train
    # print(t)
    # print(t_train)
    points, points_all = generate_points(v, X, t_train_copy, hold_out, type=type)
    W1 = []
    W2 = []
    if type == 'global':
        X_pred = list(torch.unbind(points, dim=0))
        for i in range(len(X)):
            W1.append(wasserstein(X_pred[i],torch.tensor(X[i]).float(),power=1))
            W2.append(wasserstein(X_pred[i],torch.tensor(X[i]).float(),power=2))
    if type == 'local':
        X_pred = points
        W1.append(wasserstein(X_pred,torch.tensor(X[hold_out]).float(),power=1))
        W2.append(wasserstein(X_pred,torch.tensor(X[hold_out]).float(),power=2))
        
    if draw:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        node = NeuralODE(torch_wrapper(v), solver="dopri5", sensitivity="adjoint", atol=1e-5, rtol=1e-5)
        traj_all = node.trajectory(torch.from_numpy(X[0]).float().to(device),
                            t_span=torch.linspace(t[0], t[-1], steps)
                        ).cpu()
        sample_indices = random.sample(range(traj_all.size(1)), sample_num)
        point_array = np.array(points_all[:, sample_indices,:].detach().numpy().tolist(), dtype=object)
        traj_array = np.array(traj_all[:, sample_indices,:].detach().numpy().tolist(), dtype=object)
        plot_comparisons(X, point_array, traj_array, save_dir, save_name=f'comparison_{epoch}')
    
    return W1, W2

def test2(v,g, X, t_train, hold_out, save_dir, epoch, type='global',steps=100, sample_num=100):
    t_train = [t for t in t_train if t is not None]
    t_train_copy = t_train.copy()
    if hold_out is not None:
        hold_out = int(np.squeeze(hold_out))
        bisect.insort(t_train, hold_out)  
    t = t_train
    # print(t)
    # print(t_train)
    traj_list=[]
    points_list=[]
    weights_list=[]
    for i in range(1,len(t_train)):
        traj, points, weights = generate_traj_and_weights(v, g, X[0],t_train[0],t_train[i])
        traj_list.append(traj)
        points_list.append(points)
        weights_list.append(weights)
    
    W1 = []
    W2 = []
    for i in range(1,len(t_train)):
        w1 = wasserstein_with_weights(points_list[i-1],weights_list[i-1],torch.tensor(X[i]).float(),method="exact",reg=0.05,power=1)
        w2 = wasserstein_with_weights(points_list[i-1],weights_list[i-1],torch.tensor(X[i]).float(),method="exact",reg=0.05,power=2)
        W1.append(w1)
        W2.append(w2)
    
    return W1, W2
        
def generate_points(v, X, t_train, hold_out, type='global'):
    if hold_out is not None:
        hold_out = int(np.squeeze(hold_out))
        bisect.insort(t_train, hold_out)  
    t = t_train
    node = NeuralODE(torch_wrapper(v), solver="dopri5", sensitivity="adjoint", atol=1e-5, rtol=1e-5)
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        points_all = node.trajectory(
                    torch.from_numpy(X[0]).float().to(device),
                    t_span=torch.tensor(t, dtype=torch.float32).to(device),
                ).cpu()
        if type == 'global': # 返回所有时间的点
            points = points_all
        if type == 'local': # 返回hold时间的点
            traj = node.trajectory(
                torch.from_numpy(X[hold_out-1]).float().to(device),
                t_span=torch.tensor([hold_out-1,hold_out], dtype=torch.float32).to(device),
            ).cpu()
            points = traj[-1]
    return points, points_all
