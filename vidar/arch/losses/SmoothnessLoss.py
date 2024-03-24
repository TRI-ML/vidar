# Copyright 2023 Toyota Research Institute.  All rights reserved.

from abc import ABC

import torch

from knk_vision.vidar.vidar.arch.losses.BaseLoss import BaseLoss
from knk_vision.vidar.vidar.utils.data import get_mask_from_list
from knk_vision.vidar.vidar.utils.tensor import same_shape, interpolate_image


class SmoothnessLoss(BaseLoss, ABC):
    """ Smoothness loss class, to enforce smoothness in depth predictions."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.normalize = cfg.normalize

    def calculate(self, rgb, depth, mask):
        """Calculates loss.

        Parameters
        ----------
        rgb : torch.Tensor
            Input image (BxCxHxW)
        depth : torch.Tensor
            Depth map (BxHxW)
        mask : torch.Tensor
            Valid mask (BxHxW)

        Returns
        -------
        loss : torch.Tensor
            Loss value
        """
        rgb = self.interp_nearest(rgb, depth)

        if self.normalize:
            mean_depth = depth.mean(2, True).mean(3, True)
            norm_depth = depth / (mean_depth + 1e-7)
        else:
            norm_depth = depth

        grad_depth_x = torch.abs(norm_depth[:, :, :, :-1] - norm_depth[:, :, :, 1:])
        grad_depth_y = torch.abs(norm_depth[:, :, :-1, :] - norm_depth[:, :, 1:, :])

        grad_rgb_x = torch.mean(torch.abs(rgb[:, :, :, :-1] - rgb[:, :, :, 1:]), 1, keepdim=True)
        grad_rgb_y = torch.mean(torch.abs(rgb[:, :, :-1, :] - rgb[:, :, 1:, :]), 1, keepdim=True)

        grad_depth_x *= torch.exp(-1.0 * grad_rgb_x)
        grad_depth_y *= torch.exp(-1.0 * grad_rgb_y)

        return grad_depth_x.mean() + grad_depth_y.mean()

    def forward(self, rgb, depth, mask=None):
        """Forward pass for the smoothness loss.

        Parameters
        ----------
        rgb : torch.Tensor
            Input image (BxCxHxW)
        depth : list of torch.Tensor
            List of depth maps (BxHxW)
        mask : torch.Tensor
            Valid mask (BxHxW)

        Returns
        -------
        dict
            Dictionary containing loss and metrics
        """
        scales = self.get_scales(depth)
        weights = self.get_weights(scales)

        losses, metrics = [], {}

        for i in range(scales):
            rgb_i, depth_i = rgb, depth[i]
            mask_i = get_mask_from_list(mask, i, return_ones=rgb_i)

            loss_i = weights[i] * self.calculate(rgb_i, depth_i, mask_i)

            metrics[f'smoothness_loss/{i}'] = loss_i.detach()
            losses.append(loss_i)

        loss = sum(losses) / len(losses)

        return {
            'loss': loss,
            'metrics': metrics,
        }
