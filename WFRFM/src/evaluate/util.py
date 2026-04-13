import torch
import matplotlib.pyplot as plt
import numpy as np
import os


def _summarize_series(values):
    if not values:
        return {
            "count": 0,
            "last": None,
            "min": None,
            "mean_last_100": None,
        }
    tail = values[-100:]
    return {
        "count": len(values),
        "last": float(values[-1]),
        "min": float(np.min(values)),
        "mean_last_100": float(np.mean(tail)),
    }


def draw_loss(load_path=None, eval_interval=10, cut=0, save_path=None, show=True):
    checkpoint = torch.load(load_path, map_location='cpu')

    loss_list = checkpoint['loss_list'][cut:]
    vloss_list = checkpoint['vloss_list'][cut:]
    gloss_list = checkpoint['gloss_list'][cut:]
    test_loss_list = checkpoint.get('test_loss_list', [])[cut // eval_interval:]
    test_vloss_list = checkpoint.get('test_vloss_list', [])[cut // eval_interval:]
    test_gloss_list = checkpoint.get('test_gloss_list', [])[cut // eval_interval:]
    has_test = bool(test_loss_list)

    summary = {
        "load_path": load_path,
        "eval_interval": int(eval_interval),
        "cut": int(cut),
        "has_test_loss": has_test,
        "train": {
            "loss": _summarize_series(loss_list),
            "vloss": _summarize_series(vloss_list),
            "gloss": _summarize_series(gloss_list),
        },
        "test": {
            "loss": _summarize_series(test_loss_list),
            "vloss": _summarize_series(test_vloss_list),
            "gloss": _summarize_series(test_gloss_list),
        },
    }

    print(f"训练步数: {len(loss_list)}")
    if loss_list:
        print(f"最后一步 Loss: {loss_list[-1]:.6f}")
        print(f"最后一步 vLoss: {vloss_list[-1]:.6f}")
        print(f"最后一步 gLoss: {gloss_list[-1]:.6f}")
    if has_test:
        print(f"最后一步 test_Loss: {test_loss_list[-1]:.6f}")
        print(f"最后一步 test_vLoss: {test_vloss_list[-1]:.6f}")
        print(f"最后一步 test_gLoss: {test_gloss_list[-1]:.6f}")
    else:
        print("未检测到 test loss，当前图只绘制 train loss。")

    if has_test:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        train_ax, test_ax = axes
    else:
        fig, train_ax = plt.subplots(1, 1, figsize=(8, 5))
        test_ax = None

    train_ax.plot(loss_list, label='Total Train Loss', color='blue', alpha=0.6)
    train_ax.plot(vloss_list, label='Velocity Loss (train)', color='orange', alpha=0.7)
    train_ax.plot(gloss_list, label='Growth Loss (train)', color='green', alpha=0.7)
    train_ax.set_title('Training Loss')
    train_ax.set_xlabel('Iteration')
    train_ax.set_ylabel('Loss')
    train_ax.grid(True, linestyle='--', alpha=0.5)
    train_ax.legend()

    if has_test and test_ax is not None:
        test_ax.plot(test_loss_list, label='Total Test Loss', color='blue', alpha=0.6)
        test_ax.plot(test_vloss_list, label='Velocity Loss (test)', color='orange', alpha=0.7)
        test_ax.plot(test_gloss_list, label='Growth Loss (test)', color='green', alpha=0.7)
        test_ax.set_title('Test Loss')
        test_ax.set_xlabel('Evaluation Step')
        test_ax.set_ylabel('Loss')
        test_ax.grid(True, linestyle='--', alpha=0.5)
        test_ax.legend()

    fig.tight_layout()
    if save_path is None:
        file_name = os.path.basename(load_path).replace(".pt", "")
        save_path = f"results/figures/loss_{file_name}.png"
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return summary
