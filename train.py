import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from datetime import datetime
from tqdm import tqdm
from typing import Optional, Dict, Union, Callable, List
import matplotlib.pyplot as plt
from pathlib import Path

from analysis import plot_single_rig, plot_rigs
from models import build_inn
from data import RandomRigDataset
from utils import pad_rig, unpad_rig, set_seed

# Loss function type
LossFn = Union[nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]



class Trainer:

    def __init__(
        self,
        config: Dict,
        loss_function: LossFn,
        model: nn.Module,
        device: torch.device,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        ) -> None:

        self.config = config
        self.loss_function = loss_function
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.num_segments = config["NUM_SEGMENTS"]
        self.target_dim = (self.num_segments + 1) * 4

        self.train_losses = []
        self.test_losses = []
        self.epoch_train_preds = []
        self.epoch_train_labels = []
        self.epoch_test_preds = []
        self.epoch_test_labels = []


    def _run_epoch(self, dataloader: DataLoader, mode: str) -> None:
        mode = mode.lower()

        assert mode in ['train', 'test'], \
            f"Invalid mode '{mode}'. Expected 'train' or 'test'."

        is_train = (mode == 'train')
        self.model.train() if is_train else self.model.eval()

        total_loss = 0.0
        with torch.enable_grad() if is_train else torch.no_grad():

            for rigs, anchors in tqdm(dataloader):

                # Put on device
                rigs = rigs.to(self.device)
                anchors = anchors.to(self.device)

                # Zero gradients if training
                if is_train:
                    self.optimizer.zero_grad()

                # Inverse: predict rig vectors and error latent from anchor_labels (ignore ljd)
                anchors_noisy = anchors + torch.randn_like(anchors) * 0.1
                rig_inv_preds_and_errs, _ = self.model.inverse(anchors_noisy)
                rig_inv_preds = rig_inv_preds_and_errs[:, :self.num_segments]
                errs_inv = rig_inv_preds_and_errs[:, self.num_segments:]
                loss_inv = self.loss_function.inverse_loss(rig_inv_preds, rigs)
                
                # Forward: predict anchors from rigs and error latent
                anchors_fwd, _ = self.model(torch.concatenate([rig_inv_preds, errs_inv], axis=1))
                loss_fwd = self.loss_function.forward_loss(anchors_fwd, anchors)

                # Independence: predicted rig vectors independent of error latent vectors
                loss_ind = self.loss_function.independence_loss(rig_inv_preds, errs_inv)

                # Inverse boundary:
                rig_inv_bnd_and_errs, _ = self.model.inverse(anchors)
                rig_inv_bnd_preds = rig_inv_bnd_and_errs[:, :self.num_segments]
                errs_inv_bnd = rig_inv_bnd_and_errs[:, self.num_segments:]
                loss_inv_bnd = self.loss_function.inverse_boundary_loss(rig_inv_bnd_preds, rigs, errs_inv_bnd)

                # Forward boundary:
                anchor_fwd_bnd_preds, _ = self.model(pad_rig(rigs, self.target_dim))
                loss_fwd_bnd = self.loss_function.forward_boundary_loss(anchor_fwd_bnd_preds, anchors)

                print(f"loss_fwd: {loss_fwd.item():.6f}, loss_inv: {loss_inv.item():.6f}, loss_ind: {loss_ind.item():.6f}, loss_inv_bnd: {loss_inv_bnd.item():.6f}, loss_fwd_bnd: {loss_fwd_bnd.item():.6f}")

                loss = self.config["LAMBDA_FWD"] * loss_fwd + \
                       self.config["LAMBDA_INV"] * loss_inv + \
                       self.config["LAMBDA_IND"] * loss_ind + \
                       self.config["LAMBDA_INV_BND"] * loss_inv_bnd + \
                       self.config["LAMBDA_FWD_BND"] * loss_fwd_bnd

                # If training, calculate gradients and backpropagate
                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.get("GRAD_CLIP", 5.0))
                    self.optimizer.step()

                # If scheduler exists, step it
                if self.scheduler is not None:
                    self.scheduler.step()

                # Add to total loss
                total_loss += loss.item()

            # Detach and move to CPU
                pred_np = anchors_fwd.detach().cpu().numpy()
                label_np = anchors.detach().cpu().numpy()

                if is_train:
                    self.epoch_train_preds.append(pred_np)
                    self.epoch_train_labels.append(label_np)
                else:
                    self.epoch_test_preds.append(pred_np)
                    self.epoch_test_labels.append(label_np)

        # Calculate average loss
        avg_loss = total_loss / len(dataloader)
        print(f"Average loss: {avg_loss}")

        # Concatenate predictions and labels
        if is_train:
            self.epoch_train_preds = np.concatenate(self.epoch_train_preds, axis=0)
            self.epoch_train_labels = np.concatenate(self.epoch_train_labels, axis=0)
            self.train_losses.append(avg_loss)
        else:
            self.epoch_test_preds = np.concatenate(self.epoch_test_preds, axis=0)
            self.epoch_test_labels = np.concatenate(self.epoch_test_labels, axis=0)
            self.test_losses.append(avg_loss)


    def train_epoch(self, dataloader: DataLoader) -> None:
        print("[INFO] Training...")
        self.epoch_train_preds = []
        self.epoch_train_labels = []
        self._run_epoch(dataloader=dataloader, mode="train")

    def test_epoch(self, dataloader: DataLoader) -> None:
        print("[INFO] Testing...")
        self.epoch_test_preds = []
        self.epoch_test_labels = []
        self._run_epoch(dataloader=dataloader, mode="test")

    def plot_losses(self, save_dir: Optional[Path]=None) -> None:

        fig = plt.figure(figsize=(20, 10))

        plt.plot(
            np.arange(0, len(self.train_losses)),
            self.train_losses,
            linewidth=2,
            color='red',
            label='Training Loss'
        )

        plt.plot(
            np.arange(0, len(self.test_losses)),
            self.test_losses,
            linewidth=2,
            color='cornflowerblue',
            label='Testing Loss'
        )

        plt.grid("both")
        plt.title("Loss History", fontsize=22)
        plt.xlabel("Epoch", fontsize=18)
        plt.ylabel("Loss", fontsize=18)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.join(save_dir, 'loss_history.png')
            plt.savefig(filename)
        else:
            print("No save path defined!")
            plt.show()

        plt.close(fig)

    def plot_scatter(self, save_dir: Optional[Path]=None):

        def _scatter(preds, labels, prefix: str, save_dir: str):
            num_dims = preds.shape[1]
            for d in range(num_dims):
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.scatter(preds[:, d], labels[:, d], alpha=0.5, s=4, color='cornflowerblue')
                ax.plot(
                    [labels[:, d].min(), labels[:, d].max()],
                    [labels[:, d].min(), labels[:, d].max()],
                    'k--', lw=2
                )
                ax.set_title(f"{prefix.capitalize()} Scatter Plot: {d}")
                ax.set_xlabel("Predictions", fontsize=12)
                ax.set_ylabel("Labels", fontsize=12)
                ax.grid("Both")

                if save_dir is not None:
                    os.makedirs(save_dir, exist_ok=True)
                    filename = os.path.join(save_dir, f"{prefix}_scatter_{d:03d}.png")
                    plt.savefig(filename)
                else:
                    plt.show()

                plt.close(fig)


        if len(self.epoch_train_preds) and len(self.epoch_train_labels):
            _scatter(self.epoch_train_preds, self.epoch_train_labels, "train", save_dir)

        if len(self.epoch_test_preds) and len(self.epoch_test_labels):
            _scatter(self.epoch_test_preds, self.epoch_test_labels, "test", save_dir)

    def plot_analysis(self, dataloader: DataLoader, lengths: np.ndarray, save_dir: Optional[Path]=None) -> None:

        self.model.eval()
        
        with torch.no_grad():

            # Plot round trip
            rig_inputs, anchor_labels = next(iter(dataloader))
            anchor_preds, ljd_fwd = self.model(pad_rig(rig_inputs + torch.randn_like(rig_inputs)*0, self.target_dim).to(self.device))

            rig_fal_latent_error, ljd_fal = self.model.inverse(anchor_labels.to(self.device))
            rig_fal = rig_fal_latent_error[:, :self.num_segments]
            error_latent_fal = rig_fal_latent_error[:, self.num_segments:]


            rig_recon_latent_error, ljd_inv = self.model.inverse(anchor_preds)
            rig_recon = rig_recon_latent_error[:, :self.num_segments]
            error_latent_recon = rig_recon_latent_error[:, self.num_segments:]

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                filename = os.path.join(save_dir, f"rig_recon.png")
                plt.savefig(filename)
            plot_rigs(rigs=[rig_inputs[0], rig_recon[0], rig_fal[0]], lengths=lengths, save_path=filename)



            # plot_single_rig(rig_inputs[0], lengths=lengths, save_path=os.path.join())

            # plot_single_rig(rig_recon[0].cpu().detach().numpy(), lengths=lengths)



    @property
    def train_preds(self):
        return self.epoch_train_preds

    @property
    def train_labels(self):
        return self.epoch_train_labels

    @property
    def test_preds(self):
        return self.epoch_test_preds

    @property
    def test_labels(self):
        return self.epoch_test_labels

    def __repr__(self) -> str:
        return f"Trainer(model={self.model.__class__.__name__}, device={self.device})"


# def main():
#     # Configuration
#     batch_size = 128
#     train_size = 16384*4
#     test_size = 4096
#     num_segments = 128
#     lr = 1e-4
#     weight_decay = 1e-4
#     num_epochs = 1000
#     vis_interval = 1
#     seed = 1337

#     # For padding and unpadding
#     target_dim = (num_segments + 1) * 4  # model input/output dim

#     # Set seed
#     set_seed(seed)

#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Define output directory for results
#     timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
#     results_dir = os.path.join("results", f"{timestamp}")
#     os.makedirs(results_dir, exist_ok=True)

#     # Instantiate model
#     model = build_inn(num_segments).to(device)
#     print(f"Learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

#     # Define optimizer
#     optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

#     # Define loss function
#     loss_fn = torch.nn.MSELoss()

#     # Get ready for training
#     train_losses = []
#     val_losses = []
#     best_val_loss = float("inf")
#     best_model_path = os.path.join(results_dir, "best_model.pt")

#     # Train
#     for epoch in range(1, num_epochs + 1):

#         # Create epoch of data samples
#         lengths = np.ones(num_segments, dtype=np.float32) * 0.1
#         train_dataset = RandomRigDataset(num_samples=train_size, num_segments=num_segments, lengths=lengths)
#         val_dataset = RandomRigDataset(num_samples=test_size, num_segments=num_segments, lengths=lengths)
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#         # --- Training ---
#         model.train()
#         total_train_loss = 0.0
#         for rig_batch, anchor_batch in tqdm(train_loader):
#             rig_batch = pad_rig(rig_batch, target_dim)
#             rig_batch = rig_batch.to(device)
#             anchor_batch = anchor_batch.to(device)

#             optimizer.zero_grad()
#             pred, _ = model(rig_batch)
#             loss = loss_fn(pred, anchor_batch)
#             loss.backward()
#             optimizer.step()

#             total_train_loss += loss.item()

#         avg_train_loss = total_train_loss / len(train_loader)
#         train_losses.append(avg_train_loss)

#         # --- Validation ---
#         model.eval()
#         total_val_loss = 0.0
#         all_val_rig = []
#         all_val_pred = []
#         all_val_label = []
#         with torch.no_grad():
#             for rig_batch, anchor_batch in tqdm(val_loader):
#                 rig_batch = pad_rig(rig_batch, target_dim)
#                 rig_batch = rig_batch.to(device)
#                 anchor_batch = anchor_batch.to(device)
#                 pred, _ = model(rig_batch)
#                 loss = loss_fn(pred, anchor_batch)
#                 total_val_loss += loss.item()
#                 all_val_rig.append(rig_batch.cpu().numpy())
#                 all_val_pred.append(pred.cpu().numpy())
#                 all_val_label.append(anchor_batch.cpu().numpy())
#         avg_val_loss = total_val_loss / len(val_loader)
#         val_losses.append(avg_val_loss)

#         print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

#         # --- Save best model ---
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             torch.save(model.state_dict(), best_model_path)

#         # --- Organized Visualization ---
#         if epoch % vis_interval == 0 or epoch == num_epochs:
#             val_rig = np.concatenate(all_val_rig, axis=0)
#             val_pred = np.concatenate(all_val_pred, axis=0)
#             val_label = np.concatenate(all_val_label, axis=0)
#             val_rig_unpadded = unpad_rig(val_rig, num_segments)

#             epoch_dir = os.path.join(results_dir, f"{epoch:03d}")

#             # Plot types and their directories
#             plot_configs = [
#                 ("rig_vs_pred", analysis.plot_rig_vs_predicted_anchor, dict(num_samples=3)),
#                 ("rig_roundtrip", analysis.plot_rig_roundtrip, dict(num_samples=3)),
#                 ("rig_roundtrip_noise", analysis.plot_rig_roundtrip_noise, dict(num_samples=3)),
#                 ("hist", analysis.plot_histogram_labels_vs_preds, dict(title='Validation Rig')),
#                 ("scatter", analysis.plot_scatter_labels_vs_preds, dict(title='Validation Rig'))
#             ]
#             for plot_type, func, kwargs in plot_configs:
#                 plot_dir = os.path.join(epoch_dir, plot_type)
#                 os.makedirs(plot_dir, exist_ok=True)
#                 if plot_type == "rig_vs_pred":
#                     func(val_rig_unpadded, val_pred, model, lengths, save_path=plot_dir, **kwargs)
#                 elif plot_type == "rig_roundtrip":
#                     func(val_rig_unpadded, model, lengths, save_path=plot_dir, **kwargs)
#                 elif plot_type == "rig_roundtrip_noise":
#                     func(val_rig_unpadded, model, lengths, save_path=plot_dir, **kwargs)
#                 elif plot_type == "hist":
#                     func(val_label, val_pred, save_path=plot_dir, **kwargs)
#                 elif plot_type == "scatter":
#                     func(val_label, val_pred, save_path=plot_dir, **kwargs)

#             # Loss curve at the root results_dir
#             analysis.plot_loss_curves(train_losses, val_losses, save_path=results_dir)

#     print(f"Training complete! Best model saved at {best_model_path}")

