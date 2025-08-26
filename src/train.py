__all__ = ['train']

import os, sys, json, math, itertools
import pandas as pd, numpy as np
import warnings

# from tqdm import tqdm
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm
# from .evaluation import test,test2
import logging
# from .plots import plot_g_values,plot_loss_curve_1
# from .plots import *
import torch

from .utils2 import compute_uot_plans, get_batch
# from .utils import sample, generate_steps
# from .losses import MMD_loss, OT_loss1, OT_loss2, Density_loss, Local_density_loss
# from torchdiffeq import odeint_adjoint as odeint
# from torchdiffeq import odeint as odeint2
from DeepRUOT.models import velocityNet, growthNet, scoreNet, dediffusionNet, indediffusionNet, FNet, ODEFunc2, ODEFunc3
from DeepRUOT.utils import group_extract, sample, to_np, generate_steps, cal_mass_loss, parser, _valid_criterions

def pretrain(model, df, optimizer, n_epoch=1000, test_interval=100,
    batch_size = 256,
    hold_one_out=False,
    hold_out='random',
    logger=None,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_dir=None,
    relative_mass=None,
    delta=1.0,
):
    print('relative_mass',relative_mass)
    time_labels = df['samples'].to_numpy()
    data = df.iloc[:,1:].to_numpy()
    X = [data[time_labels == t] for t in np.unique(time_labels)]
    X_selected = [data[time_labels == t] for t, v in enumerate(np.unique(time_labels)) if v != hold_out]
    t_train = [i for i, v in enumerate(np.unique(time_labels)) if v != hold_out]
    print('t_train:', t_train)
    logger.info('Begin flow and growth matching')

    _, gamma0_plans, gamma1_plans = compute_uot_plans(X_selected, t_train, delta=delta, draw=False)
    progress_bar = tqdm(range(n_epoch), desc="Begin flow and growth matching...", unit="epoch")
    vloss_list = []
    gloss_list = []
    loss_list = []

    for i in progress_bar:
        optimizer.zero_grad()
        t, xt, ut, gt = get_batch(X_selected, t_train, batch_size, gamma0_plans, gamma1_plans, delta, relative_mass)
        vt = model.v_net(t,xt)
        gt_pred = model.g_net(t,xt)

        # print(torch.isnan(vt).any(), torch.isinf(ut).any(), torch.isnan(gt_pred).any(), torch.isinf(gt).any())
        # print(torch.isnan(xt).any())
        # print('---')

        vloss = torch.mean((vt - ut)**2)
        vloss_list.append(vloss.item())
        gloss = torch.mean((gt_pred - gt)**2)
        gloss_list.append(gloss.item())
        loss = vloss + gloss
        loss_list.append(loss.item())
        
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(v.parameters(), max_norm=0.1)
        # torch.nn.utils.clip_grad_norm_(g.parameters(), max_norm=0.1)

        optimizer.step()
        logging.info(f"Epoch {i}: loss={loss.item():.6f}, vloss={vloss.item():.6f}, gloss={gloss.item():.6f}")
        progress_bar.set_postfix({"loss": f"{loss.item():.6f}","vloss": f"{vloss.item():.6f}", "gloss": f"{gloss.item():.6f}"})
        # if (i+1) % test_interval == 0 or i == 0:
        #     W1, W2 = test2(v, g, X, t_train, hold_out, save_dir, epoch=0 if i == 0 else i+1, type='global')
        #     logging.info(f"Epoch {i}, W1={W1}, W2={W2}")
        #     print('\nepoch',0 if i==0 else i+1)        
        #     print('W1:',W1)
        #     print('W2:',W2)
            
    # print('epoch',i+1)        
    # print('W1:',W1)
    # print('W2:',W2)
    # plot_loss_curve_1(loss_list, vloss_list, gloss_list, save_dir)
    # plot_g_values(X, g, save_dir)
    
    return model, vloss_list, gloss_list, loss_list