# Copyright 2023 Toyota Research Institute.  All rights reserved.

from abc import ABC

import torch
import torch.nn as nn

from vidar.arch.losses.BaseLoss import BaseLoss
from vidar.utils.data import get_mask_from_list, get_from_list
from vidar.utils.depth import depth2inv


def get_criterion(method):
    """Determines the supervised loss to be used"""
    if method == 'l1':
        return nn.L1Loss()
    else:
        raise ValueError('Unknown supervised loss {}'.format(method))


class LossWrapper(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.criterion = get_criterion(method)

    def forward(self, pred, gt, soft_mask=None, logvar=None):
        loss = self.criterion(pred, gt)
        if soft_mask is not None:
            loss = loss * soft_mask.detach().view(-1, 1)
        if logvar is not None:
            loss = loss * torch.exp(-logvar)
        return loss.mean()


class SupervisedOpticalFlowLoss(BaseLoss, ABC):
    """Supervised Loss"""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.criterion = LossWrapper(cfg.method)
        self.inverse = cfg.has('inverse', False)

    def calculate(self, pred, gt, mask=None, soft_mask=None, logvar=None):
        """Calculates loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction (BxCxHxW)
        gt : torch.Tensor
            Ground truth (BxHxW)
        mask : torch.Tensor
            Valid Mask (BxHxW)
        soft_mask : torch.Tensor
            Soft mask for per-pixel loss calculation (BxHxW)
        logvar : torch.Tensor
            Log-variance value for uncertainty weighting (BxHxW)

        Returns
        -------
        loss : torch.Tensor
            Loss value
        """
        # Interpolations
        pred = self.interp_nearest(pred, gt)
        mask = self.interp_nearest(mask, gt)
        soft_mask = self.interp_bilinear(soft_mask, gt)

        from vidar.utils.write import write_image
        write_image('mask.png', mask[0])

        # Flatten tensors
        pred, gt, mask, soft_mask = self.flatten(pred, gt, mask, soft_mask)

        # Calculate loss
        return self.criterion(
            pred[mask], gt[mask],
            soft_mask=soft_mask[mask] if soft_mask is not None else None
        )

    def forward(self, pred, gt, mask=None, soft_mask=None, logvar=None):
        """Forward pass for optical flow loss.

        Parameters
        ----------
        pred : list of torch.Tensor
            List with predictions (BxCxHxW)
        gt : torch.Tensor
            Ground truth optical flow (BxHxW)
        mask : torch.Tensor
            Valid mask (BxHxW)
        soft_mask : list of torch.Tensor
            List with soft masks for per-pixel loss calculation (BxHxW)
        logvar : list of torch.Tensor
            List with log-variance value for uncertainty weighting (BxHxW)

        Returns
        -------
        dict
            Dictionary containing loss and metrics
        """
        if self.inverse:
            pred, gt = depth2inv(pred), depth2inv(gt)

        scales = self.get_scales(pred)
        weights = self.get_weights(scales)

        losses, metrics = [], {}

        for i in range(scales):
            pred_i, gt_i = get_from_list(pred, i), get_from_list(gt, i)
            mask_i = get_mask_from_list(mask, i, return_ones=gt_i)
            soft_mask_i = get_mask_from_list(soft_mask, i)
            logvar_i = get_from_list(logvar)

            loss_i = weights[i] * self.calculate(pred_i, gt_i, mask_i, soft_mask_i, logvar_i)

            metrics[f'supervised_depth_loss/{i}'] = loss_i.detach()
            losses.append(loss_i)

        loss = sum(losses) / len(losses)

        return {
            'loss': loss,
            'metrics': metrics,
        }
