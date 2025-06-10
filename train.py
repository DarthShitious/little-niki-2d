import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datetime import datetime

from data import RandomRigDataset
from models import build_inn
import analysis
from utils import pad_rig, unpad_rig

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # --- Config ---
    batch_size = 128
    num_segments = 16
    lr = 4e-4
    weight_decay = 1e-4
    num_epochs = 100
    vis_interval = 10
    seed = 42

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Output directory ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join("results", f"{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # --- Data ---
    lengths = np.ones(num_segments, dtype=np.float32)
    train_dataset = RandomRigDataset(num_samples=10000, num_segments=num_segments, lengths=lengths)
    val_dataset = RandomRigDataset(num_samples=2000, num_segments=num_segments, lengths=lengths)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- Model ---
    model = build_inn(num_segments).to(device)
    print(f"Learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_model_path = os.path.join(results_dir, "best_model.pt")

    target_dim = (num_segments + 1) * 4  # model input/output dim

    for epoch in range(1, num_epochs + 1):
        # --- Training ---
        model.train()
        total_train_loss = 0.0
        for rig_batch, anchor_batch in train_loader:
            rig_batch = pad_rig(rig_batch, target_dim)
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

        # --- Validation ---
        model.eval()
        total_val_loss = 0.0
        all_val_rig = []
        all_val_pred = []
        with torch.no_grad():
            for rig_batch, anchor_batch in val_loader:
                rig_batch = pad_rig(rig_batch, target_dim)
                rig_batch = rig_batch.to(device)
                anchor_batch = anchor_batch.to(device)
                pred, _ = model(rig_batch)
                loss = loss_fn(pred, anchor_batch)
                total_val_loss += loss.item()
                all_val_rig.append(rig_batch.cpu().numpy())
                all_val_pred.append(pred.cpu().numpy())
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # --- Save best model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)

        # --- Organized Visualization ---
        if epoch % vis_interval == 0 or epoch == num_epochs:
            val_rig = np.concatenate(all_val_rig, axis=0)
            val_pred = np.concatenate(all_val_pred, axis=0)
            val_rig_unpadded = unpad_rig(val_rig, num_segments)

            epoch_dir = os.path.join(results_dir, f"{epoch:03d}")

            # Plot types and their directories
            plot_configs = [
                ("rig_vs_pred", analysis.plot_rig_vs_predicted_anchor, dict(num_samples=3)),
                ("rig_roundtrip", analysis.plot_rig_roundtrip, dict(num_samples=3)),
                ("hist", analysis.plot_histogram_labels_vs_preds, dict(title='Validation Rig')),
                ("scatter", analysis.plot_scatter_labels_vs_preds, dict(title='Validation Rig'))
            ]
            for plot_type, func, kwargs in plot_configs:
                plot_dir = os.path.join(epoch_dir, plot_type)
                os.makedirs(plot_dir, exist_ok=True)
                if plot_type == "rig_vs_pred":
                    func(val_rig_unpadded, val_pred, model, lengths, save_path=plot_dir, **kwargs)
                elif plot_type == "rig_roundtrip":
                    func(val_rig_unpadded, model, lengths, save_path=plot_dir, **kwargs)
                elif plot_type == "hist":
                    func(val_rig_unpadded, val_pred, save_path=plot_dir, **kwargs)
                elif plot_type == "scatter":
                    func(val_rig_unpadded, val_pred, save_path=plot_dir, **kwargs)

            # Loss curve at the root results_dir
            analysis.plot_loss_curves(train_losses, val_losses, save_path=results_dir)

    print(f"Training complete! Best model saved at {best_model_path}")

if __name__ == "__main__":
    main()
