import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import pad_rig, unpad_rig

def plot_rig_vs_predicted_anchor(rig_labels, anchor_preds, inn_model, lengths, save_path=None, num_samples=5):
    """
    Plot: Rig reconstructed from predicted anchor vectors vs. rig from label rig vectors.
    """
    anchor_preds_tensor = torch.from_numpy(anchor_preds).float()
    device = next(inn_model.parameters()).device
    anchor_preds_tensor = anchor_preds_tensor.to(device)
    rig_from_anchors, _ = inn_model.inverse(anchor_preds_tensor)
    rig_from_anchors = rig_from_anchors.detach().cpu().numpy()
    rig_from_anchors_unpadded = unpad_rig(rig_from_anchors, len(lengths))

    for i in range(num_samples):
        mae = np.mean(np.abs(rig_labels[i] - rig_from_anchors_unpadded[i]))
        plt.figure(figsize=(10, 10))
        plot_single_rig(rig_labels[i], lengths, title='Label Rig')
        plot_single_rig(rig_from_anchors_unpadded[i], lengths, title=f'Predicted Rig (from anchor) | MAE: {mae:0.4f}')
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/rig_vs_pred_{i}.png")
        plt.close()

def plot_rig_roundtrip(rig_labels, inn_model, lengths, save_path=None, num_samples=5):
    """
    Plot: Label rig vectors vs. round-tripped rig vectors with noise injection (rig -> anchor + noise -> rig).
    """
    rig_labels_tensor = torch.from_numpy(rig_labels).float()
    device = next(inn_model.parameters()).device
    rig_labels_tensor = rig_labels_tensor.to(device)
    target_dim = (len(lengths) + 1) * 4
    rig_labels_tensor_padded = pad_rig(rig_labels_tensor, target_dim)
    anchor_preds = inn_model(rig_labels_tensor_padded)[0].detach().cpu().numpy()
    rig_roundtripped, _ = inn_model.inverse(torch.from_numpy(anchor_preds).float().to(device))
    rig_roundtripped = rig_roundtripped.detach().cpu().numpy()
    rig_roundtripped_unpadded = unpad_rig(rig_roundtripped, len(lengths))

    for i in range(num_samples):
        mae = np.abs(rig_labels - rig_roundtripped_unpadded).mean()
        plt.figure(figsize=(10, 10))
        plot_single_rig(rig_labels[i], lengths, title='Original Rig')
        plot_single_rig(rig_roundtripped_unpadded[i], lengths, title=f'Round-Tripped Rig | MAE: {mae:0.4f}')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        # Set equal axis limits
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        min_lim = min(xlims[0], ylims[0])
        max_lim = max(xlims[1], ylims[1])
        ax.set_xlim(min_lim, max_lim)
        ax.set_ylim(min_lim, max_lim)
        plt.tight_layout()
        plt.grid("both")
        if save_path:
            plt.savefig(f"{save_path}/rig_roundtrip{i}.png")
        plt.close()

def plot_rig_roundtrip_noise(rig_labels, inn_model, lengths, save_path=None, num_samples=5):
    """
    Plot: Label rig vectors vs. round-tripped rig vectors with noise injection (rig -> anchor + noise -> rig).
    """
    rig_labels_tensor = torch.from_numpy(rig_labels).float()
    device = next(inn_model.parameters()).device
    rig_labels_tensor = rig_labels_tensor.to(device)
    target_dim = (len(lengths) + 1) * 4
    rig_labels_tensor_padded = pad_rig(rig_labels_tensor, target_dim)
    anchor_preds = inn_model(rig_labels_tensor_padded)[0].detach().cpu().numpy()
    anchor_preds += np.random.normal(0, 0.001, anchor_preds.shape)
    rig_roundtripped, _ = inn_model.inverse(torch.from_numpy(anchor_preds).float().to(device))
    rig_roundtripped = rig_roundtripped.detach().cpu().numpy()
    rig_roundtripped_unpadded = unpad_rig(rig_roundtripped, len(lengths))

    for i in range(num_samples):
        mae = np.abs(rig_labels - rig_roundtripped_unpadded).mean()
        plt.figure(figsize=(10, 10))
        plot_single_rig(rig_labels[i], lengths, title='Original Rig')
        plot_single_rig(rig_roundtripped_unpadded[i], lengths, title=f'Round-Tripped Rig w/ Noise | MAE: {mae:0.4f}')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        # Set equal axis limits
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        min_lim = min(xlims[0], ylims[0])
        max_lim = max(xlims[1], ylims[1])
        ax.set_xlim(min_lim, max_lim)
        ax.set_ylim(min_lim, max_lim)
        plt.tight_layout()
        plt.grid("both")
        if save_path:
            plt.savefig(f"{save_path}/rig_roundtrip_noise{i}.png")
        plt.close()

def plot_single_rig(rig_vector, lengths, title=None):
    """
    Plot a single rig skeleton given rig angles and segment lengths.
    """
    N = len(lengths)
    x, y = [0], [0]
    angle = 0
    for j in range(N):
        angle += rig_vector[j]
        x.append(x[-1] + lengths[j] * np.cos(angle))
        y.append(y[-1] + lengths[j] * np.sin(angle))
    plt.plot(x, y, '-o', linewidth=2, markersize=8)
    plt.axis('equal')
    if title:
        plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')

def plot_histogram_labels_vs_preds(labels, preds, title='Histogram', save_path=None):
    """
    Plot histogram of labels and predictions for each joint/angle.
    """
    num_dims = labels.shape[1]
    if preds.shape[1] > num_dims:
        preds = unpad_rig(preds, num_dims)
    for i in range(num_dims):
        plt.figure()
        plt.hist(labels[:, i], bins=30, alpha=0.5, label='Labels')
        plt.hist(preds[:, i], bins=30, alpha=0.5, label='Preds')
        plt.legend()
        plt.title(f'{title} (Dim {i})')
        if save_path:
            plt.savefig(f"{save_path}/hist_dim_{i}.png")
        plt.close()

def plot_scatter_labels_vs_preds(labels, preds, title='Scatter', save_path=None):
    """
    Scatter plot of labels vs predictions for each dimension.
    """
    num_dims = labels.shape[1]
    if preds.shape[1] > num_dims:
        preds = unpad_rig(preds, num_dims)
    for i in range(num_dims):
        mae = np.abs(preds[:, i] - labels[:, i]).mean()
        plt.figure()
        plt.scatter(labels[:, i], preds[:, i], s=4)
        plt.xlabel('Label')
        plt.ylabel('Prediction')
        plt.title(f'{title} (Dim {i}) | MAE: {mae:0.4f}')
        plt.plot([labels[:, i].min(), labels[:, i].max()],
                 [labels[:, i].min(), labels[:, i].max()], 'k--', lw=1)
        max_lim = np.max(np.abs([labels[:, i].max(), labels[:, i].min(), preds[:, i].max(), preds[:, i].min()]))
        plt.xlim(-max_lim, max_lim)
        plt.ylim(-max_lim, max_lim)
        plt.grid("both")
        if save_path:
            plt.savefig(f"{save_path}/scatter_dim_{i}.png")
        plt.close()

def plot_loss_curves(train_losses, val_losses, save_path=None):
    """
    Plot training and validation loss curves.
    """
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid("both")
    if save_path:
        plt.savefig(f"{save_path}/loss_curves.png")
    plt.close()
