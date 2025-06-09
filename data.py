import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import Dataset, DataLoader

class RandomRigDataset(Dataset):
    def __init__(self, num_samples: int, num_segments: int, angle_range=(-np.pi/2, np.pi/2), lengths=None):
        """
        Args:
            num_samples: number of (rig, anchor) pairs to generate
            num_segments: number of segments in the rig
            angle_range: range of angles (min, max) to sample from
            lengths: optional custom segment lengths (shape: (num_segments,))
        """
        self.num_samples = num_samples
        self.num_segments = num_segments
        self.angle_min, self.angle_max = angle_range
        self.lengths = lengths if lengths is not None else np.ones(num_segments, dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Sample random rig vector
        rig_vector = np.random.uniform(self.angle_min, self.angle_max, size=(self.num_segments,)).astype(np.float32)

        # Compute anchor vector
        rig_vector_batch = rig_vector[np.newaxis, :]  # shape: (1, N)
        anchor_vector = rig_to_anchor(rig_vector_batch, self.lengths)[0]  # extract (N+1)*4 flat vector

        # Convert to torch tensors
        return torch.from_numpy(rig_vector), torch.from_numpy(anchor_vector)



def rig_to_anchor(rig_vectors: np.ndarray, lengths: np.ndarray) -> np.ndarray:

    M, N = rig_vectors.shape
    assert lengths.shape == (N,), "Lengths must have shape (N,)"

    # Initialize anchor array: one more joint (root)
    anchors = np.zeros((M, N + 1, 4), dtype=np.float32)

    # Root joint is at origin with angle 0 (cos=1, sin=0)
    anchors[:, 0, 0] = 0.0  # x
    anchors[:, 0, 1] = 0.0  # y
    anchors[:, 0, 2] = 1.0  # cos(0)
    anchors[:, 0, 3] = 0.0  # sin(0)

    # Compute global angles
    global_angles = np.cumsum(rig_vectors, axis=1)  # shape: (M, N)

    # Initialize cumulative position and angle
    x = np.zeros(M)
    y = np.zeros(M)

    for j in range(N):
        θ = global_angles[:, j]
        dx = lengths[j] * np.cos(θ)
        dy = lengths[j] * np.sin(θ)

        x += dx
        y += dy

        anchors[:, j + 1, 0] = x
        anchors[:, j + 1, 1] = y
        anchors[:, j + 1, 2] = np.cos(θ)
        anchors[:, j + 1, 3] = np.sin(θ)

    # Flatten final output: (M, 4*(N+1))
    return anchors.reshape(M, 4 * (N + 1))


def plot_anchor_vectors(anchor_vectors: np.ndarray):

    M, total = anchor_vectors.shape
    N = total // 4 - 1
    anchors = anchor_vectors.reshape(M, N + 1, 4)

    colors = cm.get_cmap('tab10', M)

    plt.figure(figsize=(6, 6))
    for i in range(M):
        x = anchors[i, :, 0]
        y = anchors[i, :, 1]
        dx = anchors[i, :, 2]
        dy = anchors[i, :, 3]

        # Plot bones
        plt.plot(x, y, '-o', label=f'Rig {i}', color=colors(i))

        # Plot orientation arrows
        for j in range(N + 1):
            plt.arrow(x[j], y[j], 0.1 * dx[j], 0.1 * dy[j],
                      head_width=0.05, color=colors(i), alpha=0.6)

    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.legend()
    plt.title('Anchor Vectors / Forward Kinematics')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('test5.png')


def plot_rig_vectors(rig_angles: np.ndarray, lengths: np.ndarray):
    anchors = rig_to_anchor_vectors(rig_angles, lengths)
    plot_anchor_vectors(anchors)


if __name__ == '__main__':

    # Configuration
    num_samps = 1
    num_segs = 20
    lengths = np.ones(num_segs)

    # Generate rigvectors
    rig_vectors = np.random.uniform(-np.pi/2, np.pi/2, (num_samps, num_segs))
    # rig_vectors = np.array([[0, 0, 0], [np.radians(45), np.radians(45), np.radians(45)]])

    # Convert to anchor vectors
    anchor_vectors = rig_to_anchor(rig_vectors=rig_vectors, lengths=lengths)

    # Plot
    plot_anchor_vectors(anchor_vectors=anchor_vectors)

    print()



