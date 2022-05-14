# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from abc import ABC
from collections import OrderedDict
from functools import partial

import torch.nn as nn

from vidar.utils.tensor import same_shape, interpolate


class BaseLoss(nn.Module, ABC):
    """
    Base class for loss calculation

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg=None):
        super().__init__()

        self.losses = OrderedDict()
        self.blocks = OrderedDict()

        self.nearest = partial(interpolate, scale_factor=None, mode='nearest', align_corners=None)
        self.bilinear = partial(interpolate, scale_factor=None, mode='bilinear', align_corners=True)

        if cfg is not None:
            self.gamma = cfg.has('gamma', 1.0)
            self.weight = cfg.has('weight', 1.0)
            self.scales = cfg.has('scales', 99)

            self.flag_mask_sparse = cfg.has('mask_sparse', False)
            self.flag_mask_range = cfg.has('mask_range', None)

    def forward(self, *args, **kwargs):
        """Forward method"""
        raise NotImplementedError('Forward not implemented for {}'.format(self.__name__))

    def get_weights(self, scales):
        """Get scale weights"""
        return [self.weight * self.gamma ** i for i in range(scales)]

    def get_scales(self, scales):
        """Get number of scales"""
        return min(self.scales, len(scales))

    @staticmethod
    def interp(dst, src, fn):
        """Interpolate dst to match src using fn"""
        if dst is None or dst.dim() == 3:
            return dst
        assert dst.dim() == src.dim()
        if dst.dim() == 4 and not same_shape(dst.shape, src.shape):
            dst = fn(dst, size=src)
        return dst

    def interp_bilinear(self, dst, src):
        """Bilinear interpolation"""
        return self.interp(dst, src, self.bilinear)

    def interp_nearest(self, dst, src):
        """Nearest-neighbor interpolation"""
        return self.interp(dst, src, self.nearest)

    def mask_sparse(self, mask, gt):
        """Mask based on sparse GT"""
        if mask is None:
            return mask
        if self.flag_mask_sparse:
            mask *= gt.sum(1) > 0
        return mask

    def mask_range(self, mask, gt):
        """Mask based on depth range"""
        if mask is None:
            return mask
        if self.flag_mask_range is None:
            return mask
        mask *= (gt.sum(1) >= self.flag_mask_range[0]) & \
                (gt.sum(1) <= self.flag_mask_range[1])
        return mask

    @staticmethod
    def flatten(pred, gt, mask=None, soft_mask=None):
        """
        Flatten 2D inputs for loss calculation

        Parameters
        ----------
        pred : torch.Tensor
            Input predictions
        gt : torch.Tensor
            Input ground-truth
        mask : torch.Tensor or None
            Input mask (binary)
        soft_mask : torch.Tensor or None
            Input soft mask (probability)

        Returns
        -------
        pred, gt, mask, soft_mask : torch.Tensor
            Flattened inputs
        """
        if pred.dim() == 4:
            pred = pred.permute(0, 2, 3, 1)
        pred = pred.reshape(-1, pred.shape[-1])

        if gt.dim() == 4:
            gt = gt.permute(0, 2, 3, 1)
        gt = gt.reshape(-1, gt.shape[-1])

        if mask is not None:
            if mask.dim() == 4:
                mask = mask.permute(0, 2, 3, 1)
            mask = mask.reshape(-1)

        if soft_mask is not None:
            if soft_mask.dim() == 4:
                soft_mask = soft_mask.permute(0, 2, 3, 1)
            soft_mask = soft_mask.reshape(-1)

        return pred, gt, mask, soft_mask
