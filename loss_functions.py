import torch

from utils import maximum_mean_discrepancy as mmd


class LittleNIKILoss(torch.nn.Module):
    def __init__(self, config):
        self.config = config
        self.lambda_fwd = config["LAMBDA_FWD"]
        self.lambda_inv = config["LAMBDA_INV"]
        self.lambda_fwd_bnd = config["LAMBDA_FWD_BND"]
        self.lambda_inv_bnd = config["LAMBDA_INV_BND"]

    def forward_loss(self, anchor_preds, anchor_labels):
        return torch.pow(anchor_preds - anchor_labels, 2).mean()

    def inverse_loss(self, rig_preds, rig_labels):
        return torch.pow(rig_preds - rig_labels, 2).mean()

    def forward_boundary_loss(self, anchor_preds_bnd, anchor_labels):
        return torch.pow(anchor_preds_bnd - anchor_labels, 2).mean()

    def inverse_boundary_loss(self, rig_preds_bnd, rig_labels, error_latent):
        error_l2 = (error_latent**2).sum(dim=1).mean()
        bnd_l2 = torch.pow(rig_preds_bnd - rig_labels, 2).mean()
        return error_l2 + bnd_l2


    def independence_loss(self, rig_preds, error_latent):
        # rig_preds: [batch, dim1], error_latent: [batch, dim2]
        # Joint: concat along last dim
        joint = torch.cat([rig_preds, error_latent], dim=1)
        # Marginals: rig_preds with shuffled error_latent
        error_latent_shuffled = error_latent[torch.randperm(error_latent.size(0))]
        marginal = torch.cat([rig_preds, error_latent_shuffled], dim=1)
        # Compare joint vs. marginal
        mmd_joint_marginal = mmd(joint, marginal)
        # Regularize error_latent to standard normal
        z = torch.randn_like(error_latent)
        mmd_z = mmd(error_latent, z)
        print(f'mmd_joint_marginal: {mmd_joint_marginal}')
        print(f'mmd_z: {mmd_z}')
        return mmd_joint_marginal + mmd_z
        


    def niki_loss(self, anchor_preds, anchor_preds_bnd, anchor_labels, rig_preds, rig_preds_bnd, rig_labels, error_latent):
        return self.lambda_fwd * self.forward_loss(anchor_preds, anchor_labels) + \
               self.lambda_inv * self.inverse_loss(rig_preds, rig_labels) +\
               self.lambda_fwd_bnd * self.forward_boundary_loss(anchor_preds_bnd, anchor_labels) + \
               self.lambda_inv_bnd * self.inverse_boundary_loss(rig_preds_bnd, rig_labels, error_latent)