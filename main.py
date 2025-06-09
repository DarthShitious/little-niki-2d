import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime

from data import RandomRigDataset, rig_to_anchor
from models import build_inn
from analysis import visualize_anchor_batch


def pad_to_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    batch_size, current_dim = x.shape
    if current_dim == target_dim:
        return x
    pad_len = target_dim - current_dim
    pad_tensor = torch.zeros(batch_size, pad_len, device=x.device, dtype=x.dtype)
    return torch.cat([x, pad_tensor], dim=1)

def depad_rig_vectors(rig_padded: np.ndarray, original_num_segments: int) -> np.ndarray:
    """
    Remove padding from rig vectors.

    Args:
      rig_padded: (M, padded_length) numpy array with zeros padded at the end
      original_num_segments: int, the true number of segments without padding

    Returns:
      rig: (M, original_num_segments) numpy array with padding removed
    """
    return rig_padded[:, :original_num_segments]


def train_inn():
    # Configuration
    batch_size = 128
    num_segments = 16
    lr = 4e-4
    weight_decay = 1e-4
    num_epochs = 100
    vis_interval = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join("results", f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # Datasets & loaders
    train_dataset = RandomRigDataset(num_samples=10000, num_segments=num_segments)
    val_dataset = RandomRigDataset(num_samples=2000, num_segments=num_segments)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Build INN model
    model = build_inn(num_segments).to(device)
    print(f"Learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0.0

        for rig_batch, anchor_batch in train_loader:
            target_dim = (num_segments + 1) * 4  # 68 in this case

            # Padding rig vectors before feeding into model:
            rig_batch = pad_to_dim(rig_batch, target_dim)

            rig_batch = rig_batch.to(device)
            anchor_batch = anchor_batch.to(device)

            optimizer.zero_grad()
            pred, _ = model(rig_batch)
            loss = loss_fn(pred, anchor_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for rig_batch, anchor_batch in val_loader:
                target_dim = (num_segments + 1) * 4  # 68 in this case

                # Padding rig vectors before feeding into model:
                rig_batch = pad_to_dim(rig_batch, target_dim)

                rig_batch = rig_batch.to(device)
                anchor_batch = anchor_batch.to(device)

                pred, _ = model(rig_batch)
                loss = loss_fn(pred, anchor_batch)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # Visualization and roundtrip test
        if epoch % vis_interval == 0:
            model.eval()
            rig_batch = next(iter(val_loader))[0][:2].to(device)
            target_dim = (num_segments + 1) * 4  # 68 in this case

            # Padding rig vectors before feeding into model:
            rig_batch = pad_to_dim(rig_batch, target_dim)
            with torch.no_grad():
                pred_anchors, _ = model(rig_batch)
                pred_anchors = pred_anchors.cpu()
                recon_rig, _ = model(pred_anchors.to(device), rev=True)
                recon_rig = recon_rig.cpu()
                recon_rig = depad_rig_vectors(recon_rig, num_segments)

            rig_batch = depad_rig_vectors(rig_batch, num_segments)
            gt_anchors = rig_to_anchor(rig_batch.cpu().numpy(), lengths=np.ones(num_segments))

            # Visualize prediction vs ground truth
            visualize_anchor_batch(
                torch.tensor(gt_anchors), pred_anchors,
                title=f"Epoch {epoch}",
                save_path=os.path.join(results_dir, f"epoch_{epoch:03d}.png")
            )

            # Log round-trip reconstruction error
            recon_loss = ((rig_batch.cpu() - recon_rig) ** 2).mean().item()
            print(f"Roundtrip MSE: {recon_loss:.6f}")

    # Plot losses
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.title("Training and Validation Loss")
    plt.savefig(os.path.join(results_dir, "loss_curve.png"))
    plt.close()


if __name__ == '__main__':
    train_inn()
