# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from abc import ABC
from functools import partial

import torch
import torch.nn as nn

from knk_vision.vidar.vidar.utils.flow import coords_from_optical_flow
from knk_vision.vidar.vidar.utils.tensor import grid_sample, interpolate
from knk_vision.vidar.vidar.utils.types import is_list


class ViewSynthesis(nn.Module, ABC):
    """
    Class for view synthesis calculation based on image warping

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg=None):
        super().__init__()
        self.grid_sample = partial(
            grid_sample, mode='bilinear', padding_mode='border')
        self.interpolate = partial(
            interpolate, mode='bilinear', scale_factor=None)
        self.grid_sample_zeros = partial(
            grid_sample, mode='nearest', padding_mode='zeros')
        self.upsample_depth = cfg.has('upsample_depth', True) if cfg is not None else True

    @staticmethod
    def get_num_scales(depths, optical_flow):
        """Return number of scales based on input"""
        if depths is not None:
            return len(depths)
        if optical_flow is not None:
            return len(optical_flow)
        else:
            raise ValueError('Invalid inputs for view synthesis')

    @staticmethod
    def get_tensor_ones(depths, optical_flow, scale):
        """Return unitary tensor based on input"""
        if depths is not None:
            return torch.ones_like(depths[scale])
        elif optical_flow is not None:
            b, _, h, w = optical_flow[scale].shape
            return torch.ones((b, 1, h, w), device=optical_flow[scale].device)
        else:
            raise ValueError('Invalid inputs for view synthesis')

    def get_coords(self, rgbs, depths, cams, optical_flow, context, scale, tgt):
        """
        Calculate projection coordinates for warping

        Parameters
        ----------
        rgbs : list[torch.Tensor]
            Input images (for dimensions) [B,3,H,W]
        depths : list[torch.Tensor]
            Target depth maps [B,1,H,W]
        cams : list[Camera]
            Input cameras
        optical_flow : list[torch.Tensor]
            Input optical flow for alternative warping
        context : list[Int]
            Context indices
        scale : Int
            Current scale
        tgt : Int
            Target index

        Returns
        -------
        output : Dict
            Dictionary containing warped images and masks
        """
        if depths is not None and cams is not None:
            cams_tgt = cams[0] if is_list(cams) else cams
            cams_ctx = cams[1] if is_list(cams) else cams
            depth = self.interpolate(depths[scale], size=rgbs[tgt][0].shape[-2:]) \
                if self.upsample_depth else depths[scale]
            return {
                ctx: cams_tgt[tgt].coords_from_depth(depth, cams_ctx[ctx]) for ctx in context
            }
        elif optical_flow is not None:
            return {
                ctx: coords_from_optical_flow(
                    optical_flow[scale]).permute(0, 2, 3, 1) for ctx in context
            }
        else:
            raise ValueError('Invalid input for view synthesis')

    def forward(self, rgbs, depths=None, cams=None,
                optical_flow=None, return_masks=False, tgt=(0, 0)):
        """Forward pass for view synthesis.

        Parameters
        ----------
        rgbs : list of torch.Tensor
            Input images (BxCxHxW)  
        depths : list of torch.Tensor, optional
            Depth maps for warping, by default None
        cams : list of Camera, optional
            Cameras for warping, by default None
        optical_flow : List of tensor.Tensor, optional
            Optical flow maps for warping, by default None
        return_masks : bool, optional
            True if overlap mas are returned, by default False
        tgt : tuple, optional
            Target key, by default (0, 0)

        Returns
        -------
        dict
            Dictionary containing warped images and masks
        """
        context = [key for key in rgbs.keys() if key != tgt]
        num_scales = self.get_num_scales(depths, optical_flow)

        warps, masks = [], []
        for scale in range(num_scales):
            src = 0 if self.upsample_depth else scale
            coords = self.get_coords(rgbs, depths, cams, optical_flow, context, scale, tgt=tgt)
            warps.append([self.grid_sample(
                rgbs[ctx][src], coords[ctx].type(rgbs[ctx][src].dtype)) for ctx in context])
            if return_masks:
                ones = self.get_tensor_ones(depths, optical_flow, scale)
                masks.append([self.grid_sample_zeros(
                    ones, coords[ctx].type(ones.dtype)) for ctx in context])

        return {
            'warps': warps,
            'masks': masks if return_masks else None
        }
