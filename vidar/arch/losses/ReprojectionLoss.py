# Copyright 2023 Toyota Research Institute.  All rights reserved.

from abc import ABC
from functools import partial

import torch

from vidar.arch.losses.BaseLoss import BaseLoss
from vidar.utils.config import cfg_has
from vidar.utils.data import get_from_list, get_mask_from_list, get_scale_from_dict
from vidar.utils.tensor import interpolate, multiply_args, masked_average
from vidar.utils.types import is_list, is_dict


class ReprojectionLoss(BaseLoss, ABC):
    """ Reprojection loss class, to warp images for photometric similarity"""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.automasking = cfg.automasking
        self.reprojection_reduce_op = cfg.reprojection_reduce_op
        self.jitter_identity_reprojection = cfg.jitter_identity_reprojection
        self.logvar_weight = cfg_has(cfg, 'logvar_weight', 0.0)

        self.interpolate = partial(interpolate, mode='bilinear', scale_factor=None)

        self.inf = 1e6

    @staticmethod
    def compute_reprojection_mask(reprojection_loss, identity_reprojection_loss):
        """Compute reprojection mask based on reprojection loss and identity reprojection loss"""
        if identity_reprojection_loss is None:
            reprojection_mask = torch.ones_like(reprojection_loss)
        else:
            all_losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=1)
            idxs = torch.argmin(all_losses, dim=1, keepdim=True)
            reprojection_mask = (idxs == 0)
        return reprojection_mask

    def reduce_reprojection(self, reprojection_losses, overlap_mask=None):
        """ Reduce reprojection losses using the specified operation"""
        if is_list(reprojection_losses):
            reprojection_losses = torch.cat(reprojection_losses, 1)
        if self.reprojection_reduce_op == 'mean':
            assert overlap_mask is None, 'Not implemented yet'
            reprojection_loss = reprojection_losses.mean(1, keepdim=True)
        elif self.reprojection_reduce_op == 'min':
            if overlap_mask is not None:
                if is_list(overlap_mask):
                    overlap_mask = torch.cat(overlap_mask, 1)
                reprojection_losses[~overlap_mask.bool()] = self.inf
            reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)
            overlap_mask = reprojection_loss < self.inf
            reprojection_loss[~overlap_mask] = 0.0 # For visualization purposes
        else:
            raise ValueError(
                f'Invalid reprojection reduce operation: {self.reprojection_reduce_op}')
        return reprojection_loss, overlap_mask

    def calculate(self, rgb, rgb_ctx, warps, logvar=None,
                  valid_mask=None, overlap_mask=None):
        """Calculate reprojection loss"""

        reprojection_losses = [
            self.losses['photometric'](warp, rgb)['loss'] for warp in warps]
        reprojection_loss, overlap_mask = self.reduce_reprojection(
            reprojection_losses, overlap_mask=overlap_mask)

        if self.automasking:
            reprojection_identity_losses = [
                self.losses['photometric'](ctx, rgb)['loss'] for ctx in rgb_ctx]
            reprojection_identity_loss, _ = self.reduce_reprojection(
                reprojection_identity_losses)
            if self.jitter_identity_reprojection > 0:
                reprojection_identity_loss += self.jitter_identity_reprojection * \
                    torch.randn(reprojection_identity_loss.shape, device=reprojection_identity_loss.device)
        else:
            reprojection_identity_loss = None

        reprojection_mask = self.compute_reprojection_mask(
            reprojection_loss, reprojection_identity_loss,
        )
        reprojection_mask = multiply_args(reprojection_mask, valid_mask, overlap_mask)

        if logvar is not None and self.logvar_weight > 0.0:
            logvar = self.interpolate(logvar, reprojection_loss.shape[-2:])
            reprojection_loss = reprojection_loss * torch.exp(-logvar)

        average_loss = masked_average(reprojection_loss, reprojection_mask)
        # reprojection_loss *= reprojection_mask # REMOVE FOR VISUALIZATION

        if logvar is not None and self.logvar_weight > 0.0:
            average_loss += self.logvar_weight * masked_average(logvar, reprojection_mask)

        return average_loss, reprojection_mask, reprojection_loss, overlap_mask

    def forward(self, rgb, rgb_ctx, warps, logvar=None,
                valid_mask=None, overlap_mask=None):
        """ Forward pass for the reprojection loss"""
        
        scales = self.get_scales(warps)
        weights = self.get_weights(scales)

        losses, masks, photos, overlaps, metrics = [], [], [], [], {}

        dict_data = is_dict(warps)

        if dict_data:
            keys = list(warps.keys())
            rgb_ctx = [rgb_ctx[key] for key in keys]

        for i in range(scales):
            if dict_data:
                warps_i = [warps[key][i] for key in keys]
                valid_mask_i = get_scale_from_dict(valid_mask, i)
                overlap_mask_i = get_scale_from_dict(overlap_mask, i)
                logvar_i = get_scale_from_dict(logvar, i)

                loss_i, mask_i, photo_i, overlap_mask_i = self.calculate(
                    rgb, rgb_ctx, warps_i, logvar=logvar_i,
                    valid_mask=valid_mask_i, overlap_mask=overlap_mask_i)
                loss_i = weights[i] * loss_i
            else:
                rgb_i, rgb_context_i, warps_i = rgb[0], rgb_ctx[0], warps[i]
                valid_mask_i = get_mask_from_list(valid_mask, i)
                overlap_mask_i = get_mask_from_list(overlap_mask, i)
                logvar_i = get_from_list(logvar, i)

                loss_i, mask_i, photo_i, overlap_mask_i = self.calculate(
                    rgb_i, rgb_context_i, warps_i, logvar=logvar_i,
                    valid_mask=valid_mask_i, overlap_mask=overlap_mask_i)
                loss_i = weights[i] * loss_i
                
            metrics[f'reprojection_loss/{i}'] = loss_i.detach()

            losses.append(loss_i)
            masks.append(mask_i)
            photos.append(photo_i)
            overlaps.append(overlap_mask_i)

        loss = sum(losses) / len(losses)

        return {
            'loss': loss,
            'metrics': metrics,
            'mask': masks,
            'photo': photos,
            'overlap': overlaps,
        }