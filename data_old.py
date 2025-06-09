import numpy as np
import matplotlib.pyplot as plt

class RigVector:
    def __init__(self, rig_vectors: np.ndarray):
        rig_vectors = np.array(rig_vectors)
        assert rig_vectors.ndim == 2, "Need a batch of rig vectors!"
        assert (rig_vectors.shape[1] - 2) % 2 == 0, "rig_vectors must be of shape (B, 2 + 2*N)"

        self.rig_vectors = rig_vectors
        self.batch_size = rig_vectors.shape[0]
        self.num_bones = (rig_vectors.shape[1] - 2) // 2
        self.locations = rig_vectors[:, :2]
        self.rotations = rig_vectors[:, 2:].reshape(self.batch_size, self.num_bones, 2)

    def normalize(self) -> 'RigVector':
        norms = np.linalg.norm(self.rotations, axis=-1, keepdims=True)
        normalized_rotations = self.rotations / np.clip(norms, 1e-8, None)
        rig_vecs = np.concatenate([self.locations, normalized_rotations.reshape(self.batch_size, -1)], axis=1)
        return RigVector(rig_vecs)

    def __repr__(self):
        return f"RigVector(batch_size={self.batch_size}, num_bones={self.num_bones})"


class AnchorVector:
    def __init__(self, anchor_vectors: np.ndarray):
        anchor_vectors = np.array(anchor_vectors)
        assert anchor_vectors.ndim == 2 and anchor_vectors.shape[1] % 4 == 0, "Anchor vectors are all fucked up!"
        self.anchor_vectors = anchor_vectors
        self.batch_size = anchor_vectors.shape[0]
        self.num_bones = anchor_vectors.shape[1] // 4

    @property
    def positions(self):
        return self.anchor_vectors.reshape(self.batch_size, self.num_bones, 4)[..., :2]

    @property
    def directions(self):
        return self.anchor_vectors.reshape(self.batch_size, self.num_bones, 4)[..., 2:]

    def normalize(self) -> 'AnchorVector':
        reshaped = self.anchor_vectors.reshape(self.batch_size, self.num_bones, 4)
        dirs = reshaped[..., 2:]
        norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
        unit_dirs = dirs / np.clip(norms, 1e-8, None)
        normalized = np.concatenate([reshaped[..., :2], unit_dirs], axis=-1)
        flat = normalized.reshape(self.batch_size, -1)
        return AnchorVector(flat)

    def __repr__(self):
        return f"AnchorVector(batch_size={self.batch_size}, num_bones={self.num_bones})"


class Armature:
    def __init__(self, bone_lengths: list[float]):
        self.bone_lengths = bone_lengths
        self.num_bones = len(bone_lengths)

    def render(self, rig_vectors: RigVector):
        assert rig_vectors.num_bones == self.num_bones
        rig_vectors = rig_vectors.normalize()

        plt.figure(figsize=(6, 6))
        for i in range(rig_vectors.batch_size):
            root = rig_vectors.locations[i]
            relative_dirs = rig_vectors.rotations[i]
            rel_angles = np.arctan2(relative_dirs[:, 1], relative_dirs[:, 0])
            cum_angles = np.cumsum(rel_angles)
            directions = np.stack([np.cos(cum_angles), np.sin(cum_angles)], axis=-1)

            points = [root]
            current = root.copy()
            for j in range(self.num_bones):
                offset = directions[j] * self.bone_lengths[j]
                current = current + offset
                points.append(current.copy())

            points = np.array(points)
            plt.plot(points[:, 0], points[:, 1], marker='o', label=f'Rig {i}' if i < 5 else None)

        plt.axis('equal')
        plt.grid(True)
        plt.title("2D Armature Rendering with Relative Bone Angles")
        if rig_vectors.batch_size <= 5:
            plt.legend()
        plt.savefig("armature2.png")

    def rig_to_anchor(self, rig_vectors: RigVector) -> AnchorVector:
        rig_vectors = rig_vectors.normalize()
        batch_size = rig_vectors.batch_size
        anchors_flat = []

        for i in range(batch_size):
            root = rig_vectors.locations[i]
            relative_dirs = rig_vectors.rotations[i]
            rel_angles = np.arctan2(relative_dirs[:, 1], relative_dirs[:, 0])
            cum_angles = np.cumsum(rel_angles)
            directions = np.stack([np.cos(cum_angles), np.sin(cum_angles)], axis=-1)

            current = root.copy()
            positions = []
            for j in range(self.num_bones):
                positions.append(current.copy())
                offset = directions[j] * self.bone_lengths[j]
                current += offset

            anchor_i = np.concatenate([
                np.stack(positions, axis=0),
                directions
            ], axis=-1).reshape(-1)  # shape (N×4,)
            anchors_flat.append(anchor_i)

        anchors = np.stack(anchors_flat, axis=0)  # (B, N×4)
        return AnchorVector(anchors)

    def anchor_to_rig(self, anchors: AnchorVector) -> RigVector:
        anchors = anchors.normalize()
        batch_size = anchors.batch_size
        positions = anchors.positions
        directions = anchors.directions
        rigs = []

        for i in range(batch_size):
            root = positions[i, 0]
            global_dirs = directions[i]
            global_angles = np.arctan2(global_dirs[:, 1], global_dirs[:, 0])
            rel_angles = np.concatenate([[global_angles[0]], np.diff(global_angles)])
            rel_dirs = np.stack([np.cos(rel_angles), np.sin(rel_angles)], axis=-1)
            rig_i = np.concatenate([root, rel_dirs.flatten()])
            rigs.append(rig_i)

        rig_array = np.stack(rigs, axis=0)  # (B, 2 + 2×N)
        return RigVector(rig_array)

if __name__ == "__main__":
    num_samps = 4
    num_bones = 3
    bone_lengths = [1.0, 0.8, 0.6]

    # rand_rots = np.random.randn(num_samps, num_bones * 2)
    # rand_pos = np.random.randn(num_samps, 2)
    # rand_rigs = np.concatenate([rand_pos, rand_rots], axis=1)
    a = np.sqrt(2) / 2
    rig_batch = RigVector([[0, 0, a, a, a, a, a, a]]).normalize()

    arm = Armature(bone_lengths)

    anchors = arm.rig_to_anchor(rig_batch)

    rig_reconstructed = arm.anchor_to_rig(anchors)
    arm.render(rig_reconstructed)


    print()

