# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from abc import ABC

import torch

from vidar.arch.losses.BaseLoss import BaseLoss
from vidar.utils.tensor import same_shape, interpolate_image


class SmoothnessLoss(BaseLoss, ABC):
    """
    Smoothness loss class

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.normalize = cfg.normalize

    def calculate(self, rgb, depth):
        """
        Calculate smoothness loss

        Parameters
        ----------
        rgb : torch.Tensor
            Input image [B,3,H,W]
        depth : torch.Tensor
            Predicted depth map [B,1,H,W]

        Returns
        -------
        loss : torch.Tensor
            Smoothness loss [1]
        """
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

    def forward(self, rgb, depth):
        """
        Calculate smoothness loss

        Parameters
        ----------
        rgb : list[torch.Tensor]
            Input images [B,3,H,W]
        depth : list[torch.Tensor]
            Predicted depth maps [B,1,H,W]

        Returns
        -------
        output : Dict
            Dictionary with loss and metrics
        """
        scales = self.get_scales(rgb)
        weights = self.get_weights(scales)

        losses, metrics = [], {}

        for i in range(scales):
            rgb_i, depth_i = rgb[i], depth[i]
            if not same_shape(rgb_i.shape[-2:], depth_i.shape[-2:]):
                rgb_i = interpolate_image(rgb_i, shape=depth_i.shape[-2:])

            loss_i = weights[i] * self.calculate(rgb_i, depth_i)

            metrics[f'smoothness_loss/{i}'] = loss_i.detach()
            losses.append(loss_i)

        loss = sum(losses) / len(losses)

        return {
            'loss': loss,
            'metrics': metrics,
        }
