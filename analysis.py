import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import numpy as np
import os


def visualize_anchor_batch(gt_anchors: torch.Tensor,
                           pred_anchors: torch.Tensor,
                           roundtrip_anchors: torch.Tensor = None,
                           title: str = "",
                           save_path: str = None):

    gt = gt_anchors.detach().cpu().numpy()
    pred = pred_anchors.detach().cpu().numpy()
    rt = roundtrip_anchors.detach().cpu().numpy() if roundtrip_anchors is not None else None

    M, total = gt.shape
    N = total // 4 - 1

    gt = gt.reshape(M, N + 1, 4)
    pred = pred.reshape(M, N + 1, 4)
    if rt is not None:
        rt = rt.reshape(M, N + 1, 4)

    colors = cm.get_cmap('tab10', M)

    plt.figure(figsize=(6, 6))
    for i in range(M):
        # GT (solid)
        x_gt, y_gt = gt[i, :, 0], gt[i, :, 1]
        plt.plot(x_gt, y_gt, '-o', label=f'GT {i}', color=colors(i))

        # Predicted (dashed)
        x_pred, y_pred = pred[i, :, 0], pred[i, :, 1]
        plt.plot(x_pred, y_pred, '--o', label=f'Pred {i}', color=colors(i), alpha=0.6)

        # Roundtrip (dotted)
        if rt is not None:
            x_rt, y_rt = rt[i, :, 0], rt[i, :, 1]
            plt.plot(x_rt, y_rt, ':o', label=f'RT {i}', color=colors(i), alpha=0.4)

    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper right', fontsize='small', ncol=2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()
