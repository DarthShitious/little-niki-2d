import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, Union, Callable
import matplotlib.pyplot as plt
from pathlib import Path
from functools import partial

from analysis import plot_rigs
from data import rig_to_anchor_torch
from utils import pad_rig, loss_compressor

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

        self.fk_model = partial(rig_to_anchor_torch, lengths=torch.Tensor(config["LENGTHS_ARRAY"]))

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
                anchors_noisy = anchors + torch.randn_like(anchors) * self.config["NOISE_INJ"]
                rig_inv_preds_and_errs, _ = self.model.inverse(anchors_noisy)
                rig_inv_preds = rig_inv_preds_and_errs[:, :self.num_segments]
                errs_inv = rig_inv_preds_and_errs[:, self.num_segments:]
                loss_inv = self.loss_function.inverse_loss(rig_inv_preds, rigs, self.fk_model)

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

                loss = self.config["LAMBDA_FWD"] * loss_fwd + \
                       self.config["LAMBDA_INV"] * loss_inv + \
                       self.config["LAMBDA_IND"] * loss_ind + \
                       self.config["LAMBDA_INV_BND"] * loss_inv_bnd + \
                       self.config["LAMBDA_FWD_BND"] * loss_fwd_bnd

                # Print losses
                print(f"loss: {loss.item():.6f}, loss_fwd: {loss_fwd.item():.6f}, loss_inv: {loss_inv.item():.6f}, loss_ind: {loss_ind.item():.6f}, loss_inv_bnd: {loss_inv_bnd.item():.6f}, loss_fwd_bnd: {loss_fwd_bnd.item():.6f}")

                # If training, calculate gradients and backpropagate
                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.get("GRAD_CLIP", 3.0))
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
            maxlim = np.nanmax(np.abs([preds.max(), preds.min(), labels.max(), labels.min()]))
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
                ax.set_xlim(-maxlim, maxlim)
                ax.set_ylim(-maxlim, maxlim)
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

        os.makedirs(save_dir, exist_ok=True)

        self.model.eval()

        with torch.no_grad():

            # Hodge-podge of perriwinkle blue blankets
            rig_inputs, anchor_labels = next(iter(dataloader))
            anchor_preds, ljd_fwd = self.model(pad_rig(rig_inputs + torch.randn_like(rig_inputs)*0, self.target_dim).to(self.device))

            rig_fal_latent_error, ljd_fal = self.model.inverse(anchor_labels.to(self.device))
            rig_fal = rig_fal_latent_error[:, :self.num_segments]
            error_latent_fal = rig_fal_latent_error[:, self.num_segments:]


            rig_recon_latent_error, ljd_inv = self.model.inverse(anchor_preds)
            rig_recon = rig_recon_latent_error[:, :self.num_segments]
            error_latent_recon = rig_recon_latent_error[:, self.num_segments:]
            anchor_noise = anchor_labels + torch.randn_like(anchor_labels) * 0.1
            rig_noise_latent_error, ljd_noise = self.model.inverse(anchor_noise.to(self.device))
            rig_noise = rig_noise_latent_error[:, :self.num_segments]
            error_latent_noise = rig_noise_latent_error[:, self.num_segments:]


            # Rig plots
            for n in range(4):
                mae_recon = np.abs(rig_inputs.detach().cpu().numpy() - rig_recon.detach().cpu().numpy()).mean(1)
                mae_fal = np.abs(rig_inputs.detach().cpu().numpy() - rig_fal.detach().cpu().numpy()).mean(1)
                mae_noise = np.abs(rig_inputs.detach().cpu().numpy() - rig_noise.detach().cpu().numpy()).mean(1)
                rig_filename = os.path.join(save_dir, f"rig_plots_{n:02d}.png")
                plot_rigs(
                    rigs=[rig_inputs[n], rig_recon[n], rig_fal[n], rig_noise[n]],
                    labels=[
                        "Rig Inputs",
                        f"Rig Inputs -> Anchor Pred -> Rig Recon | MAE: {mae_recon[n]:0.4f} | Total MAE: {mae_recon.mean():.4f}",
                        f"Anchor Labels -> Rig | MAE: {mae_fal[n]:0.4f} | Total MAE: {mae_fal.mean():.4f}",
                        f"Anchor Labels + Noise -> Rig | MAE: {mae_noise[n]:0.4f} | Total MAE: {mae_noise.mean():.4f}"
                    ],
                    lengths=lengths,
                    save_path=rig_filename
                )

            # Error Latent
            error_latent_recon_filename = os.path.join(save_dir, f"error_latents_recon.png")
            fig = plt.figure(figsize=(10, 10))
            plt.hist(
                error_latent_recon.flatten().detach().cpu().numpy(),
                bins=100,
                density=True
            )
            plt.grid("both")
            plt.savefig(error_latent_recon_filename)
            plt.close(fig)

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



