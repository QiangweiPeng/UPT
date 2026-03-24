import torch
from tqdm import tqdm
import pickle
import logging
import os

import numpy as np

# 假设这俩已经在同级或对应的模块中定义好了
from .data import get_batch
from .data import DataLoaderHelper


def train_model(adata_control, adata_treated, adata_test,
                ot_results_train, ot_results_test,
                model,
                optimizer,
                scheduler=None,
                n_iterations=10000,
                batch_size_per_condition=256,
                batch_size_condition=10,
                sample_rep="X_pca_scaled", 
                condition_rep_keys="gene_embeddings",
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                save_path=None,
                eval_interval=10,
                save_interval=5000,
                save_only_last=False):
    

    train_loader = DataLoaderHelper(
        adata_control, adata_treated, ot_results_train,
        sample_rep=sample_rep,
        condition_rep_keys=condition_rep_keys
    )
    
    test_loader = None
    if adata_test is not None and ot_results_test is not None:
        test_loader = DataLoaderHelper(
            adata_control, adata_test, ot_results_test,
            sample_rep=sample_rep,
            condition_rep_keys=condition_rep_keys
        )
    
    progress_bar = tqdm(range(n_iterations), desc="Begin flow and growth matching...", unit="epoch")
    vloss_list, gloss_list, loss_list, test_loss_list, test_vloss_list, test_gloss_list = [], [], [], [], [], []
    last_ckpt_path = None 

    for i in progress_bar:
        model.train()
        optimizer.zero_grad()
        
        # 完美接收重构后 get_batch 吐出的 cov_dict
        t, xt, ut, gt, masst, cons, cov_dict = get_batch(
            helper=train_loader,
            batch_size_per_condition=batch_size_per_condition,
            batch_size_condition=batch_size_condition,
            delta=train_loader.delta,
            device=device
        )

        # 协变量字典直接喂给模型
        vt, gt_pred = model(t, xt, cons, cov_dict)

        vloss = torch.mean((vt - ut)**2 * masst)
        vloss_list.append(vloss.item())
        gloss = torch.mean((gt_pred - gt)**2 * masst)
        gloss_list.append(gloss.item())
        loss = vloss + gloss
        loss_list.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if (i+1) % eval_interval == 0: 
            model.eval()
            with torch.no_grad():
                if test_loader is not None:
                    t_test, xt_test, ut_test, gt_test, masst_test, cons_test, cov_dict_test = get_batch(
                        helper=test_loader,
                        batch_size_per_condition=batch_size_per_condition,
                        batch_size_condition=batch_size_condition, 
                        delta=test_loader.delta,
                        device=device
                    )
                    vt_test, gt_pred_test = model(t_test, xt_test, cons_test, cov_dict_test)
                    
                    test_vloss = torch.mean((vt_test - ut_test)**2 * masst_test) 
                    test_vloss_list.append(test_vloss.item())
                    test_gloss = torch.mean((gt_pred_test - gt_test)**2 * masst_test)
                    test_gloss_list.append(test_gloss.item())
                    test_loss = test_vloss + test_gloss
                    test_loss_list.append(test_loss.item())
                    
                    logging.info(f"Epoch {i}: loss={loss.item():.3f}, vloss={vloss.item():.3f}, gloss={gloss.item():.3f}, test_loss={test_loss.item():.3f}, test_vloss={test_vloss.item():.3f}, test_gloss={test_gloss.item():.3f}")
                    progress_bar.set_postfix({"loss": f"{loss.item():.3f}","vloss": f"{vloss.item():.3f}", 
                                              "gloss": f"{gloss.item():.3f}", "test_loss": f"{test_loss.item():.3f}"
                                             , "test_vloss":f"{test_vloss.item():.3f}", "test_gloss":f"{test_gloss.item():.3f}"})
        
        # 保存 Checkpoint 逻辑保持不变...
        if (i+1) % save_interval == 0:
            if save_path is not None:
                ckpt_path = f"{save_path}_epoch_{i+1}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(), 
                    'vloss_list': vloss_list,
                    'gloss_list': gloss_list,
                    'loss_list': loss_list,
                    'test_loss_list': test_loss_list,
                    'test_vloss_list': test_vloss_list,
                    'test_gloss_list': test_gloss_list
                }, ckpt_path)
                logging.info(f"Model and training state saved to {ckpt_path}")

                if last_ckpt_path is not None and os.path.isfile(last_ckpt_path) and save_only_last:
                    try:
                        os.remove(last_ckpt_path)
                    except OSError:
                        pass
                last_ckpt_path = ckpt_path

    # 最终保存逻辑保持不变...
    if save_path is not None and n_iterations%save_interval!=0:
        if last_ckpt_path is not None and os.path.isfile(last_ckpt_path) and save_only_last:
            try:
                os.remove(last_ckpt_path)
            except OSError:
                pass
        ckpt_path = f"{save_path}_epoch_{i+1}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 
            'vloss_list': vloss_list,
            'gloss_list': gloss_list,
            'loss_list': loss_list,
            'test_loss_list': test_loss_list,
            'test_vloss_list': test_vloss_list,
            'test_gloss_list': test_gloss_list
        }, ckpt_path)
        logging.info(f"Model and training state saved to {ckpt_path}")

        

    return model # 顺手加个 return 方便外部调用
