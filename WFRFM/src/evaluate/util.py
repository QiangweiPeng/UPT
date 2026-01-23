import torch
import matplotlib.pyplot as plt
import numpy as np
import os


def draw_loss(load_path = None,eval_interval = 10,cut = 0):
    

    checkpoint = torch.load(load_path, map_location='cpu')

    cut = cut
    
    loss_list = checkpoint['loss_list'][cut:]   
    vloss_list = checkpoint['vloss_list'][cut:] 
    gloss_list = checkpoint['gloss_list'][cut:]
    test_loss_list = checkpoint['test_loss_list'][cut:]

    # length = len(test_loss_list)
    # test_loss_new = np.full(eval_interval * length, np.nan)
    # test_loss_new[eval_interval-1::eval_interval] = test_loss_list
    # test_loss_list = test_loss_new
    
    
    print(f"训练步数: {len(loss_list)}")
    print(f"最后一步 Loss: {loss_list[-1]:.6f}")
    print(f"最后一步 vLoss: {vloss_list[-1]:.6f}")
    print(f"最后一步 gLoss: {gloss_list[-1]:.6f}")
    print(f"最后一步 test_Loss: {test_loss_list[-1]:.6f}")
    

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
    
    # train loss test loss
    plt.subplot(1, 2, 2)
    plt.plot(loss_list, label='Total Train Loss', color='orange', alpha=0.7)
    plt.plot(test_loss_list, label='Total Test Loss', color='blue', alpha=0.6,marker='o', markersize=2)
    plt.title('Train vs Test Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    file_name = os.path.basename(load_path).replace(".pt","")
    plt.savefig(f"results/figures/loss_{file_name}.png", dpi=300, bbox_inches="tight")
    plt.show()