import torch
import numpy as np

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pad_rig(rig_batch: torch.Tensor, target_dim: int) -> torch.Tensor:
    batch_size, current_dim = rig_batch.shape
    if current_dim == target_dim:
        return rig_batch
    pad_len = target_dim - current_dim
    pad_tensor = torch.zeros(batch_size, pad_len, device=rig_batch.device, dtype=rig_batch.dtype)
    return torch.cat([rig_batch, pad_tensor], dim=1)

def unpad_rig(rig_padded: np.ndarray, true_dim: int) -> np.ndarray:
    return rig_padded[:, :true_dim]

def maximum_mean_discrepancy(x: torch.Tensor, y: torch.Tensor, kernel='rbf', sigma=1.0) -> torch.Tensor:
    """
    Computes the Maximum Mean Discrepancy (MMD) between two batches x and y.
    Args:
        x: Tensor of shape (n_samples_x, n_features)
        y: Tensor of shape (n_samples_y, n_features)
        kernel: Kernel type ('rbf' only supported)
        sigma: Bandwidth for the RBF kernel
    Returns:
        Scalar tensor with the MMD value
    """
    def rbf_kernel(a, b, sigma):
        a_norm = (a ** 2).sum(dim=1).unsqueeze(1)
        b_norm = (b ** 2).sum(dim=1).unsqueeze(0)
        dist = a_norm + b_norm - 2.0 * torch.mm(a, b.t())
        return torch.exp(-dist / (2 * sigma ** 2))

    xx = rbf_kernel(x, x, sigma)
    yy = rbf_kernel(y, y, sigma)
    xy = rbf_kernel(x, y, sigma)

    # Filter out infinites
    xx = xx[torch.isfinite(xx)]
    yy = yy[torch.isfinite(yy)]
    xy = xy[torch.isfinite(xy)]

    # Ignore NaNs
    mmd = xx.nanmean() + yy.nanmean() - 2 * xy.nanmean()

    return mmd


def loss_compressor(loss, thresh=10, slope=1e-5):

    if loss < thresh:
        return loss
    else:
        return thresh * (1 - slope) + slope * loss
