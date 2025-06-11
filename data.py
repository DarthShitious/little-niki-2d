import torch
import numpy as np
from torch.utils.data import Dataset

class RandomRigDataset(Dataset):
    """
    Generates random (rig, anchor) pairs for a toy 2D kinematics task.
    """
    def __init__(self, num_samples: int, num_segments: int, angle_range=(-np.pi/2, np.pi/2), lengths=None):
        """
        Args:
            num_samples: Number of (rig, anchor) pairs to generate.
            num_segments: Number of segments in the rig.
            angle_range: Tuple (min, max) for segment angles.
            lengths: Optional fixed lengths for each segment.
        """
        self.num_samples = num_samples
        self.num_segments = num_segments
        self.angle_min, self.angle_max = angle_range
        self.lengths = lengths if lengths is not None else np.ones(num_segments, dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rig_vector = np.random.uniform(self.angle_min, self.angle_max, size=(self.num_segments,)).astype(np.float32)
        anchor_vector = rig_to_anchor(rig_vector[None, :], self.lengths)[0]  # shape: (4*(N+1),)
        return torch.from_numpy(rig_vector), torch.from_numpy(anchor_vector)

def rig_to_anchor(rig_vectors: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """
    Converts batch of rig vectors (angles) to flattened anchor vectors [x, y, cos, sin] for each joint.
    Args:
        rig_vectors: (M, N) batch of rigs
        lengths: (N,) segment lengths
    Returns:
        anchors: (M, 4*(N+1)) flattened joint arrays
    """
    M, N = rig_vectors.shape
    assert lengths.shape == (N,), "Lengths must have shape (N,)"
    anchors = np.zeros((M, N + 1, 4), dtype=np.float32)
    anchors[:, 0, :] = [0.0, 0.0, 1.0, 0.0]  # root joint

    global_angles = np.cumsum(rig_vectors, axis=1)  # (M, N)
    x = np.zeros(M)
    y = np.zeros(M)

    for j in range(N):
        theta = global_angles[:, j]
        dx = lengths[j] * np.cos(theta)
        dy = lengths[j] * np.sin(theta)
        x += dx
        y += dy
        anchors[:, j + 1, 0] = x
        anchors[:, j + 1, 1] = y
        anchors[:, j + 1, 2] = np.cos(theta)
        anchors[:, j + 1, 3] = np.sin(theta)
    return anchors.reshape(M, 4 * (N + 1))

import torch

def rig_to_anchor_torch(rig_vectors: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Converts batch of rig vectors (angles) to flattened anchor vectors [x, y, cos, sin] for each joint.
    Args:
        rig_vectors: (M, N) batch of rigs
        lengths: (N,) segment lengths
    Returns:
        anchors: (M, 4*(N+1)) flattened joint arrays
    """
    M, N = rig_vectors.shape
    assert lengths.shape == (N,), "Lengths must have shape (N,)"
    device = rig_vectors.device

    lengths = lengths.to(device)

    anchors = torch.zeros((M, N + 1, 4), dtype=rig_vectors.dtype, device=device)
    anchors[:, 0, :] = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=rig_vectors.dtype, device=device)  # root joint

    # (M, N) global angles
    global_angles = torch.cumsum(rig_vectors, dim=1)

    # Cos/sin for all joints
    cos_theta = torch.cos(global_angles).to(device)
    sin_theta = torch.sin(global_angles).to(device)

    # (M, N) dx/dy for each joint
    dx = lengths.unsqueeze(0) * cos_theta
    dy = lengths.unsqueeze(0) * sin_theta

    # (M, N) x/y positions for each joint (cumulative sum over joints)
    x = torch.cumsum(dx, dim=1)
    y = torch.cumsum(dy, dim=1)

    # Fill anchors for each joint (skip root joint at index 0)
    anchors[:, 1:, 0] = x
    anchors[:, 1:, 1] = y
    anchors[:, 1:, 2] = cos_theta
    anchors[:, 1:, 3] = sin_theta

    return anchors.reshape(M, 4 * (N + 1))

