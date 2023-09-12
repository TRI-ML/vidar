# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch

from vidar.geometry.camera import Camera
from vidar.utils.decorators import iterate1
from vidar.utils.tensor import interpolate_image
from vidar.utils.types import is_dict, is_int


@iterate1
def make_rgb_scales(rgb, pyramid=None, ratio_scales=None):
    """Interpolate RGB images to match depth scales"""
    assert pyramid is None or ratio_scales is None
    if pyramid is not None:
        if is_int(pyramid):
            return [rgb] * pyramid
        while is_dict(pyramid):
            pyramid = list(pyramid.values())[0]
        return [interpolate_image(rgb, shape=pyr.shape[-2:]) for pyr in pyramid]
    elif ratio_scales is not None:
        return [interpolate_image(rgb, scale_factor=ratio_scales[0] ** i)
                for i in range(ratio_scales[1])]
    else:
        raise NotImplementedError('Invalid option')


def break_context(dic, tgt=0, ctx=None, scl=None, stack=False):
    """Break context into target and context"""
    # Get remaining frames if context is not provided
    if ctx is None:
        ctx = [key for key in dic.keys() if key != tgt]
    # Get all scales or a single scale
    if scl is None:
        tgt, ctx = dic[tgt], [dic[key] for key in ctx if key != tgt]
    else:
        tgt, ctx = dic[tgt][scl], [dic[key][scl] for key in ctx if key != tgt]
    # Stack context if requested
    if stack:
        ctx = torch.stack(ctx, 1)
    # Return target and context
    return tgt, ctx


def create_cameras(rgb, intrinsics, pose, zero_origin=True, scaled=None):
    """Create cameras from intrinsics and pose"""
    if pose is None:
        return None
    cams = {key: Camera(
        K=intrinsics[key] if is_dict(intrinsics) else intrinsics,
        Twc=pose[key],
        hw=rgb[key] if is_dict(rgb) else rgb,
    ).scaled(scaled).to(pose[key].device) for key in pose.keys()}
    if zero_origin:
        cams[0] = Camera(
            K=intrinsics[0] if is_dict(intrinsics) else intrinsics,
            hw=rgb[0] if is_dict(rgb) else rgb,
        ).scaled(scaled).to(rgb.device)
    return cams

def apply_rgb_mask(synthesised_masks: dict, mask_rgb_tgt: list,
                   mode='nearest', align_corners=None) -> dict:
    """
    Merge masking tensor will be used for photometric loss calculation (that decides where to be ignored).

    Parameters
    ----------
    synthesised_masks : Dict[tuple, List[torch.Tensor]]
        Masks from view-synthesis
    mask_rgb_tgt : List[torch.Tensor]
        Self-occlusion masks to be merged, that shapes ScaleList[ Tensor(Bx1xHxW) ]

    Returns
    -------
    Dict[tuple, List[torch.Tensor]]
        Merged masks which has the same shape with synthesised_masks

    """
    for key in synthesised_masks.keys():  # broken_context keys corresponding to the target
        for scale in range(len(synthesised_masks[key])):  # Scale list
            synthesised_masks[key][scale] *= torch.nn.functional.interpolate(
                mask_rgb_tgt[scale], size=synthesised_masks[key][scale].shape[-2:], mode=mode,
                align_corners=align_corners)
    return synthesised_masks