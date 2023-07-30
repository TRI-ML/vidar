# Copyright 2023 Toyota Research Institute.  All rights reserved.

from abc import ABC
from functools import partial

import torch.nn as nn

from vidar.arch.losses.BaseLoss import BaseLoss
from vidar.utils.data import get_mask_from_list, get_from_list
from vidar.utils.tensor import same_shape, interpolate
from vidar.utils.types import is_list


class LossWrapper(nn.Module):
    """Wrapper for loss functions."""
    def __init__(self, method):
        super().__init__()
        loss_dict = {
            'mse': nn.MSELoss(reduction='none'),
            'l1': nn.L1Loss(reduction='none'),
        }
        self.criterion = loss_dict[method]

    def forward(self, pred, gt, soft_mask=None):
        """Forward pass for loss wrapper.

        Parameters
        ----------
        pred : list of torch.Tensor
            List with predictions (BxCxHxW)
        gt : torch.Tensor
            Ground truth image (BxHxW)
        soft_mask : list of torch.Tensor
            List with soft masks for per-pixel loss calculation (BxHxW)

        Returns
        -------
        loss : torch.Tensor
            Loss value
        """
        loss = self.criterion(pred, gt)
        if soft_mask is not None:
            loss = loss * soft_mask.detach().view(-1, 1)
        return loss.mean(1).mean(0)


class SupervisedImageLoss(BaseLoss, ABC):
    """Supervised Loss"""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.criterion = LossWrapper(cfg.method)

    def calculate(self, pred, gt, mask=None, soft_mask=None):
        """Calculates loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction (BxCxHxW)
        gt : torch.Tensor
            Ground truth image (BxHxW)
        mask : torch.Tensor
            Valid mask (BxHxW)
        soft_mask : list of torch.Tensor
            List with soft masks for per-pixel loss calculation (BxHxW)

        Returns
        -------
        loss : torch.Tensor
            Loss value
        """
        # Interpolations
        pred = self.interp_bilinear(pred, gt)
        # mask = self.interp_nearest(mask, gt)
        mask = self.interp_nearest(mask.float(), gt).bool()
        soft_mask = self.interp_bilinear(soft_mask, gt)

        # Masks
        mask = self.mask_sparse(mask, gt)

        # Flatten tensors
        pred, gt, mask, soft_mask, _ = self.flatten(pred, gt, mask, soft_mask)

        # Calculate loss
        return self.criterion(
            pred[mask], gt[mask],
            soft_mask=soft_mask[mask] if soft_mask is not None else None,
        )

    def forward(self, pred, gt, mask=None, soft_mask=None):
        """Forward pass for image loss.

        Parameters
        ----------
        pred : list of torch.Tensor
            List with predictions (BxCxHxW)
        gt : torch.Tensor
            Ground truth image (BxHxW)
        mask : torch.Tensor
            Valid mask (BxHxW)
        soft_mask : list of torch.Tensor
            List with soft masks for per-pixel loss calculation (BxHxW)

        Returns
        -------
        dict
            Dictionary containing loss and metrics
        """
        scales = self.get_scales(pred)
        weights = self.get_weights(scales)

        losses, metrics = [], {}

        for i in range(scales):
            pred_i, gt_i = get_from_list(pred, i), get_from_list(gt, i)
            mask_i = get_mask_from_list(mask, i, return_ones=gt_i)
            soft_mask_i = get_mask_from_list(soft_mask, i)

            loss_i = weights[i] * self.calculate(pred_i, gt_i, mask_i, soft_mask_i)

            metrics[f'supervised_image_loss/{i}'] = loss_i.detach()
            losses.append(loss_i)

        loss = sum(losses) / len(losses)

        return {
            'loss': loss,
            'metrics': metrics,
        }
