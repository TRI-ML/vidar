# Copyright 2023 Toyota Research Institute.  All rights reserved.

from abc import ABC
from functools import partial

import torch

from vidar.arch.losses.BaseLoss import BaseLoss
from vidar.arch.losses.SSIMLoss import SSIMLoss
from vidar.utils.tensor import interpolate


class PhotometricLoss(BaseLoss, ABC):
    """Photometric loss calss, to calculate the similarity between two images"""
    def __init__(self, cfg):
        super().__init__()
        self.alpha = cfg.alpha
        self.ssim_loss = SSIMLoss()
        self.interpolate = partial(interpolate, scale_factor=None, mode='bilinear')

    def forward(self, pred, gt):
        """Photometric loss forward pass"""
        pred = self.interpolate(pred, size=gt)
        l1_loss = torch.abs(pred - gt).mean(1, True)

        if self.alpha == 0.0:
            photometric_loss = l1_loss
        else:
            ssim_loss = self.ssim_loss(pred, gt)['loss'].mean(1, True)
            photometric_loss = self.alpha * ssim_loss + (1.0 - self.alpha) * l1_loss

        return {
            'loss': photometric_loss,
        }

