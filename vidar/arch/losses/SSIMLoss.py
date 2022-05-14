# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from abc import ABC

import torch
import torch.nn as nn

from vidar.arch.losses.BaseLoss import BaseLoss


class SSIMLoss(BaseLoss, ABC):
    """SSIM (Structural Similarity Index Metric) loss class"""
    def __init__(self):
        super().__init__()

        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)

        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        """
        Calculates SSIM loss

        Parameters
        ----------
        x : torch.Tensor
            Input image 1 [B,3,H,W]
        y : torch.Tensor
            Input image 2 [B,3,H,W]

        Returns
        -------
        output : Dict
            Dictionary with loss
        """
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        loss = torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

        return {
            'loss': loss,
        }

