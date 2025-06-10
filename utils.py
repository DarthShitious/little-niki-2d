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
