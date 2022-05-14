# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import torch

from vidar.geometry.camera_nerf import CameraNerf
from vidar.utils.decorators import iterate1
from vidar.utils.tensor import interpolate_image
from vidar.utils.types import is_dict


@iterate1
def make_rgb_scales(rgb, pyramid=None, ratio_scales=None):
    """
    Create different RGB scales to correspond with predictions

    Parameters
    ----------
    rgb : torch.Tensor
        Input image [B,3,H,W]
    pyramid : list[torch.Tensor]
        List with tensors at different scales
    ratio_scales : Tuple
        Alternatively, you can provide how many scales and the downsampling ratio for each

    Returns
    -------
    pyramid : list[torch.Tensor]
        List with the input image at the same resolutions as pyramid
    """
    assert pyramid is None or ratio_scales is None
    if pyramid is not None:
        return [interpolate_image(rgb, shape=pyr.shape[-2:]) for pyr in pyramid]
    elif ratio_scales is not None:
        return [interpolate_image(rgb, scale_factor=ratio_scales[0] ** i)
                for i in range(ratio_scales[1])]
    else:
        raise NotImplementedError('Invalid option')


def break_context(dic, tgt=0, ctx=None, scl=None, stack=False):
    """
    Separate a dictionary between target and context information

    Parameters
    ----------
    dic : Dict
        Input dictionary
    tgt : Int
        Which key corresponds to target
    ctx : Int
        Which key corresponds to context (if None, use everything else)
    scl : Int
        Which scale should be used (it None, assume there are no scales)
    stack : Bool
        Stack output context or not

    Returns
    -------
    tgt : torch.Tensor
        Target information
    ctx : list[torch.Tensor] or torch.Tensor
        Context information (list or stacked)
    """
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
    """
    Create cameras from batch information
    Parameters
    ----------
    rgb : Dict
        Dictionary with images
    intrinsics : Dict
        Dictionary with camera intrinsics
    pose : Dict
        Dictionary with camera poses
    zero_origin : Bool
        Zero target camera to the origin or not
    scaled : Float
        Scale factor for the output cameras

    Returns
    -------
    cams : Dict
        Dictionary with output cameras
    """
    if pose is None:
        return None
    cams = {key: CameraNerf(
        K=intrinsics[key] if is_dict(intrinsics) else intrinsics,
        Twc=pose[key],
        hw=rgb[key] if is_dict(rgb) else rgb,
    ).scaled(scaled).to(pose[key].device) for key in pose.keys()}
    if zero_origin:
        cams[0] = CameraNerf(
            K=intrinsics[0] if is_dict(intrinsics) else intrinsics,
            hw=rgb[0] if is_dict(rgb) else rgb,
        ).scaled(scaled).to(rgb.device)
    return cams
