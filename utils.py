import torch
import numpy as np

def pad_rig(rig_batch: torch.Tensor, target_dim: int) -> torch.Tensor:
    """
    Pads each rig vector in the batch to the target_dim with zeros.
    Args:
        rig_batch: (batch, rig_dim)
        target_dim: int, final length after padding
    Returns:
        (batch, target_dim) tensor
    """
    batch_size, current_dim = rig_batch.shape
    if current_dim == target_dim:
        return rig_batch
    pad_len = target_dim - current_dim
    pad_tensor = torch.zeros(batch_size, pad_len, device=rig_batch.device, dtype=rig_batch.dtype)
    return torch.cat([rig_batch, pad_tensor], dim=1)

def unpad_rig(rig_padded: np.ndarray, true_dim: int) -> np.ndarray:
    """
    Removes zero padding from padded rig vectors.
    Args:
        rig_padded: (batch, padded_dim) array
        true_dim: int, original rig vector length
    Returns:
        (batch, true_dim) array
    """
    return rig_padded[:, :true_dim]
