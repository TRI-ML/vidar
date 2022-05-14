# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from functools import reduce

import torch
import torch.nn.functional as tfn

from vidar.utils.decorators import iterate1
from vidar.utils.types import is_tensor, is_dict, is_seq


@iterate1
def interpolate(tensor, size, scale_factor, mode, align_corners):
    """
    Interpolate a tensor to a different resolution

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor [B,?,H,W]
    size : Tuple
        Interpolation size (H,W)
    scale_factor : Float
        Scale factor for interpolation
    mode : String
        Interpolation mode
    align_corners : Bool
        Corner alignment flag

    Returns
    -------
    tensor : torch.Tensor
        Interpolated tensor [B,?,h,w]
    """
    if is_tensor(size):
        size = size.shape[-2:]
    return tfn.interpolate(
        tensor, size=size, scale_factor=scale_factor,
        mode=mode, align_corners=align_corners, recompute_scale_factor=False,
    )


def masked_average(loss, mask, eps=1e-7):
    """Calculates the average of a tensor considering mask information"""
    return (loss * mask).sum() / (mask.sum() + eps)


def multiply_mask(data, mask):
    """Multiplies a tensor with a mask"""
    return data if (data is None or mask is None) else data * mask


def multiply_args(*args):
    """Multiplies input arguments"""
    valids = [v for v in args if v is not None]
    return None if not valids else reduce((lambda x, y: x * y), valids)


def grid_sample(tensor, grid, padding_mode, mode, align_corners):
    return tfn.grid_sample(tensor, grid,
        padding_mode=padding_mode, mode=mode, align_corners=align_corners)


def pixel_grid(hw, b=None, with_ones=False, device=None, normalize=False):
    """
    Creates a pixel grid for image operations

    Parameters
    ----------
    hw : Tuple
        Height/width of the grid
    b : Int
        Batch size
    with_ones : Bool
        Stack an extra channel with 1s
    device : String
        Device where the grid will be created
    normalize : Bool
        Whether the grid is normalized between [-1,1]

    Returns
    -------
    grid : torch.Tensor
        Output pixel grid [B,2,H,W]
    """
    if is_tensor(hw):
        b, hw = hw.shape[0], hw.shape[-2:]
    if is_tensor(device):
        device = device.device
    hi, hf = 0, hw[0] - 1
    wi, wf = 0, hw[1] - 1
    yy, xx = torch.meshgrid([torch.linspace(hi, hf, hw[0], device=device),
                             torch.linspace(wi, wf, hw[1], device=device)], indexing='ij')
    if with_ones:
        grid = torch.stack([xx, yy, torch.ones(hw, device=device)], 0)
    else:
        grid = torch.stack([xx, yy], 0)
    if b is not None:
        grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)
    if normalize:
        grid = norm_pixel_grid(grid)
    return grid


def norm_pixel_grid(grid, hw=None, in_place=False):
    """
    Normalize a pixel grid to be between [0,1]

    Parameters
    ----------
    grid : torch.Tensor
        Grid to be normalized [B,2,H,W]
    hw : Tuple
        Height/Width for normalization
    in_place : Bool
        Whether the operation is done in place or not

    Returns
    -------
    grid : torch.Tensor
        Normalized grid [B,2,H,W]
    """
    if hw is None:
        hw = grid.shape[-2:]
    if not in_place:
        grid = grid.clone()
    grid[:, 0] = 2.0 * grid[:, 0] / (hw[1] - 1) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / (hw[0] - 1) - 1.0
    return grid


def unnorm_pixel_grid(grid, hw=None, in_place=False):
    """
    Unnormalize pixel grid to be between [0,H] and [0,W]

    Parameters
    ----------
    grid : torch.Tensor
        Grid to be normalized [B,2,H,W]
    hw : Tuple
        Height/width for unnormalization
    in_place : Bool
        Whether the operation is done in place or not

    Returns
    -------
    grid : torch.Tensor
        Unnormalized grid [B,2,H,W]
    """
    if hw is None:
        hw = grid.shape[-2:]
    if not in_place:
        grid = grid.clone()
    grid[:, 0] = 0.5 * (hw[1] - 1) * (grid[:, 0] + 1)
    grid[:, 1] = 0.5 * (hw[0] - 1) * (grid[:, 1] + 1)
    return grid


def match_scales(image, targets, num_scales,
                 mode='bilinear', align_corners=True):
    """
    Creates multiple resolution versions of the input to match another list of tensors

    Parameters
    ----------
    image : torch.Tensor
        Input image [B,?,H,W]
    targets : list[torch.Tensor]
        Target resolutions
    num_scales : int
        Number of scales to consider
    mode : String
        Interpolation mode
    align_corners : Bool
        Corner alignment flag

    Returns
    -------
    images : list[torch.Tensor]
        List containing tensors in the required resolutions
    """
    # For all scales
    images = []
    image_shape = image.shape[-2:]
    for i in range(num_scales):
        target_shape = targets[i].shape
        # If image shape is equal to target shape
        if same_shape(image_shape, target_shape):
            images.append(image)
        else:
            # Otherwise, interpolate
            images.append(interpolate_image(
                image, target_shape, mode=mode, align_corners=align_corners))
    # Return scaled images
    return images


def cat_channel_ones(tensor, n=1):
    """
    Concatenate tensor with an extra channel of ones

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to be concatenated
    n : Int
        Which channel will be concatenated

    Returns
    -------
    cat_tensor : torch.Tensor
        Concatenated tensor
    """
    # Get tensor shape with 1 channel
    shape = list(tensor.shape)
    shape[n] = 1
    # Return concatenation of tensor with ones
    return torch.cat([tensor, torch.ones(shape,
                      device=tensor.device, dtype=tensor.dtype)], n)


def same_shape(shape1, shape2):
    """Checks if two shapes are the same"""
    if len(shape1) != len(shape2):
        return False
    for i in range(len(shape1)):
        if shape1[i] != shape2[i]:
            return False
    return True


def interpolate_image(image, shape=None, scale_factor=None, mode='bilinear',
                      align_corners=True, recompute_scale_factor=False):
    """
    Interpolate an image to a different resolution

    Parameters
    ----------
    image : torch.Tensor
        Image to be interpolated [B,?,h,w]
    shape : torch.Tensor or tuple
        Output shape [H,W]
    scale_factor : Float
        Scale factor for output shape
    mode : String
        Interpolation mode
    align_corners : Bool
        True if corners will be aligned after interpolation
    recompute_scale_factor : Bool
        True if scale factor is recomputed

    Returns
    -------
    image : torch.Tensor
        Interpolated image [B,?,H,W]
    """
    assert shape is not None or scale_factor is not None, 'Invalid option for interpolate_image'
    if mode == 'nearest':
        align_corners = None
    # Take last two dimensions as shape
    if shape is not None:
        if is_tensor(shape):
            shape = shape.shape
        if len(shape) > 2:
            shape = shape[-2:]
        # If the shapes are the same, do nothing
        if same_shape(image.shape[-2:], shape):
            return image
    # Interpolate image to match the shape
    return tfn.interpolate(image, size=shape, scale_factor=scale_factor,
                           mode=mode, align_corners=align_corners,
                           recompute_scale_factor=recompute_scale_factor)


def check_assert(pred, gt, atol=1e-5, rtol=1e-5):
    """
    Check two dictionaries with allclose assertions

    Parameters
    ----------
    pred : Dict
        Dictionary with predictions
    gt : Dict
        Dictionary with ground-truth
    atol : Float
        Absolute tolerance
    rtol : Float
        Relative tolerance
    """
    for key in gt.keys():
        if key in pred.keys():
            # assert key in pred and key in gt
            if is_dict(pred[key]):
                check_assert(pred[key], gt[key])
            elif is_seq(pred[key]):
                for val1, val2 in zip(pred[key], gt[key]):
                    if is_tensor(val1):
                        assert torch.allclose(val1, val2, atol=atol, rtol=rtol), \
                            f'Assert error in {key} : {val1.mean().item()} x {val2.mean().item()}'
                    else:
                        assert val1 == val2, \
                            f'Assert error in {key} : {val1} x {val2}'
            else:
                if is_tensor(pred[key]):
                    assert torch.allclose(pred[key], gt[key], atol=atol, rtol=rtol), \
                        f'Assert error in {key} : {pred[key].mean().item()} x {gt[key].mean().item()}'
                else:
                    assert pred[key] == gt[key], \
                        f'Assert error in {key} : {pred[key]} x {gt[key]}'


def interleave(data, b):
    """Interleave data considering multiple batches"""
    data_interleave = data.unsqueeze(1).expand(-1, b, *data.shape[1:])
    return data_interleave.reshape(-1, *data.shape[1:])
