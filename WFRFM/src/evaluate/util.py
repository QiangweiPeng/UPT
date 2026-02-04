import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import gc


def draw_loss(load_path = None,eval_interval = 10,cut = 0,save_path = None):

    checkpoint = torch.load(load_path, map_location='cpu')
    
    loss_list = checkpoint['loss_list'][cut:]   
    vloss_list = checkpoint['vloss_list'][cut:] 
    gloss_list = checkpoint['gloss_list'][cut:]
    test_loss_list = checkpoint['test_loss_list'][cut//eval_interval:]
    test_vloss_list = checkpoint['test_vloss_list'][cut//eval_interval:]
    test_gloss_list = checkpoint['test_gloss_list'][cut//eval_interval:]
    
    
    print(f"训练步数: {len(loss_list)}")
    print(f"最后一步 Loss: {loss_list[-1]:.6f}")
    print(f"最后一步 vLoss: {vloss_list[-1]:.6f}")
    print(f"最后一步 gLoss: {gloss_list[-1]:.6f}")
    print(f"最后一步 test_Loss: {test_loss_list[-1]:.6f}")
    print(f"最后一步 test_vLoss: {test_vloss_list[-1]:.6f}")
    print(f"最后一步 test_gLoss: {test_gloss_list[-1]:.6f}")
    

    plt.figure(figsize=(15, 5))

    # train loss vloss gloss
    plt.subplot(1, 2, 1)
    plt.plot(loss_list, label='Total Train Loss', color='blue', alpha=0.6)
    plt.plot(vloss_list, label='Velocity Loss (train)', color='orange', alpha=0.7)
    plt.plot(gloss_list, label='Growth Loss (train)', color='green', alpha=0.7)
    plt.title('Total Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # test loss vloss gloss
    plt.subplot(1, 2, 2)
    plt.plot(test_loss_list, label='Total Test Loss', color='blue', alpha=0.6)
    plt.plot(test_vloss_list, label='Velocity Loss (test)', color='orange', alpha=0.7)
    plt.plot(test_gloss_list, label='Growth Loss (test)', color='green', alpha=0.7)
    plt.title('Total Test Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # length = len(test_loss_list)
    # test_loss_new = np.full(eval_interval * length, np.nan)
    # test_loss_new[eval_interval-1::eval_interval] = test_loss_list
    # test_loss_list = test_loss_new
    
    # test_loss_new = np.full(eval_interval * length, np.nan)
    # test_loss_new[eval_interval-1::eval_interval] = test_vloss_list
    # test_vloss_list = test_loss_new

    # test_loss_new = np.full(eval_interval * length, np.nan)
    # test_loss_new[eval_interval-1::eval_interval] = test_gloss_list
    # test_gloss_list = test_loss_new
    
    # test vloss
    plt.figure(figsize=(7, 5))
    plt.plot(test_vloss_list, label='Test vLoss', color='blue', alpha=0.6)
    plt.title('Test vLoss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    file_name = os.path.basename(load_path).replace(".pt","")
    if save_path is not None:
        save_root = os.path.join(save_path,f"loss_{file_name}_.png")
        plt.savefig(save_root, dpi=300, bbox_inches="tight")
    plt.show()



def get_origin_expression(adata, model, sample_rep, target_library_size=1e4, batch_size=1024):
    """
    分批次重建基因表达，防止显存爆炸 (OOM)。
    """
    device = model.device
    
    # 1. 获取 Embedding (建议先放在 CPU 上，循环时再搬运到 GPU)
    # 使用 .copy() 确保在内存中是连续的
    z_all = adata.obsm[sample_rep].copy()
    n_obs = z_all.shape[0]
    
    recon_list = []
    
    print(f"Starting reconstruction for {n_obs} cells in batches of {batch_size}...")
    
    model.module.eval()
    for i in range(0, n_obs, batch_size):
        end_idx = min(i + batch_size, n_obs)
        z_batch = torch.tensor(z_all[i:end_idx], device=device).float()
        current_batch_size = z_batch.shape[0]
        
        library_batch = torch.zeros((current_batch_size, 1), device=device)
        batch_idx_batch = torch.full((current_batch_size, 1), 0, dtype=torch.long, device=device)
        
        with torch.no_grad():
            outputs = model.module.generative(
                z=z_batch, 
                library=library_batch, 
                batch_index=batch_idx_batch
            )

            px = outputs["px"]
            px_scale = px.scale
            batch_recon = (px_scale * target_library_size).cpu().numpy()
            recon_list.append(batch_recon)
            

        del z_batch, library_batch, batch_idx_batch, outputs, px_scale
        torch.cuda.empty_cache()

    print("Concatenating results...")
    X_recon = np.vstack(recon_list)
    
    adata.X = X_recon
    print(f"Reconstruction finished. Adata X shape: {adata.X.shape}")
    del recon_list
    gc.collect()