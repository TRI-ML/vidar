# Copyright 2023 Toyota Research Institute.  All rights reserved.

from abc import ABC
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn

from vidar.utils.tensor import same_shape, interpolate
from vidar.utils.types import is_dict, is_list


class BaseLoss(nn.Module, ABC):
    """Base loss class, with basic functionality. All losses should inherit from this class."""
    def __init__(self, cfg=None):
        super().__init__()

        self.losses = OrderedDict()
        self.blocks = OrderedDict()

        self.nearest = partial(interpolate, scale_factor=None, mode='nearest')
        self.bilinear = partial(interpolate, scale_factor=None, mode='bilinear')

        if cfg is not None:
            self.gamma = cfg.has('gamma', 1.0)
            self.weight = cfg.has('weight', 1.0)
            self.scales = cfg.has('scales', 99)

            self.flag_mask_sparse = cfg.has('mask_sparse', False)
            self.flag_mask_range = cfg.has('mask_range', None)

            self.flag_fade_in = cfg.has('fade_in', None)
            self.flag_fade_out = cfg.has('fade_out', None)

    def fade_in(self, epoch=None):
        """Fades a loss in, from 0 to 1, over a number of epochs."""
        if epoch is None or self.flag_fade_in is None:
            return 1.0

        if not is_list(self.flag_fade_in):
            st, fn = 1, self.flag_fade_in
        elif len(self.flag_fade_in) == 1:
            st, fn = 1, self.flag_fade_in[1]
        else:
            st, fn = self.flag_fade_in

        if epoch < st:
            value = 0.0
        elif epoch >= fn:
            value = 1.0
        else:
            value = 1.0 - (fn - epoch) / (fn - st + 1)
        return value

    def fade_out(self, epoch):
        """Fades a loss out, from 1 to 0, over a number of epochs."""
        if self.flag_fade_out is None:
            return 1.0

        if not is_list(self.flag_fade_out):
            st, fn = 1, self.flag_fade_out
        elif len(self.flag_fade_out) == 1:
            st, fn = 1, self.flag_fade_out[0]
        else:
            st, fn = self.flag_fade_out[:2]

        if is_list(self.flag_fade_out) and len(self.flag_fade_out) == 3:
            final = self.flag_fade_out[2]
        else:
            final = 0.0

        if epoch < st:
            value = 1.0
        elif epoch >= fn:
            value = final
        else:
            value = (1.0 - final) * (fn - epoch) / (fn - st + 1) + final
        return value

    def forward(self, *args, **kwargs):
        """Loss forward pass (not implemented here)"""
        raise NotImplementedError('Forward not implemented for {}'.format(self.__name__))

    def get_weights(self, scales, epoch=None):
        """Get loss weights for each scale, based on the current epoch."""
        fade_in = self.fade_in(epoch)
        fade_out = self.fade_out(epoch)
        return [fade_in * fade_out * self.weight * self.gamma ** i for i in range(scales)]

    def get_scales(self, scales):
        """Get the number of scales to use for the loss."""
        while is_dict(scales):
            scales = list(scales.values())[0]
        return min(self.scales, len(scales))

    @staticmethod
    def interp(dst, src, fn):
        """Interpolate a tensor dst to match the size of another tensor src given an interpolation function fn."""
        if dst is None or dst.dim() == 3:
            return dst
        assert dst.dim() == src.dim()
        if dst.dim() == 4 and not same_shape(dst.shape, src.shape):
            is_bool = dst.dtype == torch.bool
            dst = dst.float() if is_bool else dst
            dst = fn(dst, size=src)
            dst = dst.bool() if is_bool else dst
        return dst

    def interp_bilinear(self, dst, src):
        """Bilinear interpolation."""
        return self.interp(dst, src, self.bilinear)
        

    def interp_nearest(self, dst, src):
        """Nearest interpolation."""
        return self.interp(dst, src, self.nearest)

    def mask_sparse(self, mask, gt):
        """Mask out loss pixels that are not present in the ground truth."""
        if mask is None:
            return mask
        if self.flag_mask_sparse:
            mask *= gt.sum(1, keepdim=True) > 0
        return mask

    def mask_range(self, mask, gt):
        """Mask loss based on range."""
        if mask is None or gt is None:
            return mask
        if self.flag_mask_range is None:
            return mask
        mask *= (gt.sum(1, keepdim=True) >= self.flag_mask_range[0]) & \
                (gt.sum(1, keepdim=True) <= self.flag_mask_range[1])
        return mask

    @staticmethod
    def flatten(pred, gt, mask=None, soft_mask=None, logvar=None):
        """Flatten 2D loss tensors to 1D."""

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

        if logvar is not None:
            if logvar.dim() == 4:
                logvar = logvar.permute(0, 2, 3, 1)
            logvar = logvar.reshape(-1)

        return pred, gt, mask, soft_mask, logvar
