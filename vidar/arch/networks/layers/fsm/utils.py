# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn.functional as tfunc

from vidar.utils.types import is_list


def coords_from_motion(ref_camera, tgt_depth, tgt_camera, scene_flow=None):
    """
    Get coordinates from motion (depth + ego-motion) information

    Parameters
    ----------
    ref_camera : Camera
        Reference camera
    tgt_depth : Tensor
        Target depth map [B,1,H,W]
    tgt_camera : Camera
        Target camera
    scene_flow : Tensor
        Target optical flow

    Returns
    -------
    coords : torch.Tensor
        Warping coordinates [B,2,H,W]
    """
    # If there are multiple reference cameras, iterate for each
    if is_list(ref_camera):
        return [coords_from_motion(camera, tgt_depth, tgt_camera, scene_flow)
                for camera in ref_camera]
    # If there are multiple depth maps, iterate for each
    if is_list(tgt_depth):
        return [coords_from_motion(ref_camera, depth, tgt_camera, scene_flow)
                for depth in tgt_depth]
    # Reconstruct and reproject points to generate warping coordinates
    world_points = tgt_camera.reconstruct(tgt_depth, frame='w', scene_flow=scene_flow)
    return ref_camera.project(world_points, frame='w').permute(0, 3, 1, 2).contiguous()


def mask_from_coords(coords):
    """
    Get overlap mask from coordinates

    Parameters
    ----------
    coords : Tensor
        Warping coordinates [B,2,H,W]

    Returns
    -------
    mask : Tensor
        Overlap mask [B,1,H,W]
    """
    # If there are multiple warping coordinates, iterate for each
    if is_list(coords):
        return [mask_from_coords(coord) for coord in coords]
    # Create and return mask
    b, _, h, w = coords.shape
    mask = torch.ones((b, 1, h, w), dtype=torch.float32, device=coords.device, requires_grad=False)
    mask = warp_from_coords(mask, coords, mode='bilinear', padding_mode='zeros', align_corners=True)
    return mask.bool()


def warp_from_coords(tensor, coords, mask=False, mode='bilinear',
                     padding_mode='zeros', align_corners=True):
    """
    Warp an image from a coordinate map

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor for warping [B,?,H,W]
    coords : torch.Tensor
        Warping coordinates [B,2,H,W]
    mask : Bool
        Whether the warped tensor is masked for non-overlapping regions
    mode : String
        Warping mode
    padding_mode : String
        Padding mode
    align_corners : Bool
        Align corners flag

    Returns
    -------
    warp : torch.Tensor
        Warped tensor [B,?,H,W]
    """
    # Sample grid from data with coordinates
    warp = tfunc.grid_sample(tensor, coords.permute(0, 2, 3, 1).contiguous(),
                             mode=mode, padding_mode=padding_mode,
                             align_corners=align_corners)
    # If masking
    if mask:
        mask = torch.ones_like(tensor, requires_grad=False)
        mask = tfunc.grid_sample(mask, coords.permute(0, 2, 3, 1).contiguous())
        warp = warp * (mask >= 1.0).detach()
    # Returned warped tensor
    return warp


def filter_dict(dictionary, keywords):
    """
    Returns only the keywords that are part of a dictionary

    Parameters
    ----------
    dictionary : dict
        Dictionary for filtering
    keywords : list of str
        Keywords that will be filtered

    Returns
    -------
    keywords : list of str
        List containing the keywords that are keys in dictionary
    """
    return [key for key in keywords if key in dictionary]


def merge_outputs(*outputs):
    """
    Merges model outputs for logging

    Parameters
    ----------
    outputs : tuple of dict
        Outputs to be merged

    Returns
    -------
    output : dict
        Dictionary with a "metrics" key containing a dictionary with various metrics and
        all other keys that are not "loss" (it is handled differently).
    """
    ignore = ['loss']  # Keys to ignore
    combine = ['metrics']  # Keys to combine
    merge = {key: {} for key in combine}
    for output in outputs:
        # Iterate over all keys
        for key, val in output.items():
            # Combine these keys
            if key in combine:
                for sub_key, sub_val in output[key].items():
                    assert sub_key not in merge[key].keys(), \
                        'Combining duplicated key {} to {}'.format(sub_key, key)
                    merge[key][sub_key] = sub_val
            # Ignore these keys
            elif key not in ignore:
                assert key not in merge.keys(), \
                    'Adding duplicated key {}'.format(key)
                merge[key] = val
    return merge


def flip_batch_input(batch):
    """
    Flip batch input information (copies data first)

    Parameters
    ----------
    batch : dict
        Batch information

    Returns
    -------
    batch : dict
        Flipped batch
    """
    # Flip images and input depth
    for key in filter_dict(batch, [
        'rgb', 'input_depth'
    ]):
        batch[key] = flip_lr(batch[key])
    # Flip context images
    for key in filter_dict(batch, [
        'rgb_context',
    ]):
        batch[key] = [flip_lr(img) for img in batch[key]]
    # Flip intrinsics
    for key in filter_dict(batch, [
        'intrinsics'
    ]):
        batch[key] = batch[key].clone()
        batch[key][:, 0, 2] = batch['rgb'].shape[3] - batch[key][:, 0, 2]
    # Return flipped batch
    return batch


def flip_output(output):
    """
    Flip output information

    Parameters
    ----------
    output : dict
        Dictionary of model outputs (e.g. with keys like 'inv_depths' and 'uncertainty')

    Returns
    -------
    output : dict
        Flipped output
    """
    # Flip list of tensors
    for key in filter_dict(output, [
        'inv_depths', 'uncertainty', 'logits_semantic'
    ]):
        output[key] = [flip_lr(val) for val in output[key]]
    return output


class CameraNormalizer:
    """
    Camera normalizer class.
    Initialized with a desired focal lenght, and will normalize images to follow these values.
    These images can then be unormalized to return to the original resolution/intrinsics

    Parameters
    ----------
    focal : tuple
        Focal lengths (fx, fy)
    """
    def __init__(self, focal, mode='reflect'):
        self.focal = focal
        self.mode = mode
        self.diffs = []

    def normalize(self, rgb, intrinsics):
        """
        Normalize input image

        Parameters
        ----------
        rgb : torch.Tensor
            Input image [B,3,H,W]
        intrinsics : torch.Tensor
            Input intrinsics [B,3,3]

        Returns
        -------
        rgb_pad : torch.Tensor
            Normalized image with padding [B,3,H,W]
        """
        rgb_pad = []
        self.diffs.clear()
        # Process each image independently
        for i in range(len(rgb)):
            rgb_i = rgb[i].unsqueeze(0)
            intrinsics_i = intrinsics[i]
            wh_orig = rgb_i.shape[2:]
            # Get resize ratio
            ratio = [float(self.focal[1] / intrinsics_i[1, 1]),
                     float(self.focal[0] / intrinsics_i[0, 0])]
            wh_norm = [int(o * r) for o, r in zip(wh_orig, ratio)]
            # Resize image
            rgb_i_norm = torch.nn.functional.interpolate(
                rgb_i, size=wh_norm, mode='bilinear', align_corners=True)
            # Pad image
            diff = [int(o - n) for o, n in zip(wh_orig, wh_norm)]
            rgb_i_pad = torch.nn.functional.pad(
                rgb_i_norm, pad=[diff[1] // 2, (diff[1] + 1) // 2,
                                 diff[0] // 2, (diff[0] + 1) // 2], mode=self.mode)
            rgb_pad.append(rgb_i_pad)
            self.diffs.append(diff)
        # Return concatenation of all images
        return torch.cat(rgb_pad, 0)

    def unormalize(self, rgb):
        """
        Unormalize image following the previous normalization.

        Parameters
        ----------
        rgb : torch.Tensor
            Normalized image with padding [B,3,H,W]

        Returns
        -------
        orig_rgb : torch.Tensor
            Original image
        """
        # If it's a list, unnormalize each one
        if is_list(rgb):
            return [self.unormalize(r) for r in rgb]
        rgb_orig = []
        hw = rgb.shape[2:]
        for i in range(len(rgb)):
            rgb_i = rgb[i].unsqueeze(0)
            diff_i = self.diffs[i]
            rgb_i = rgb_i[:, :,
                    (diff_i[0] // 2): (hw[0] - (diff_i[0] + 1) // 2),
                    (diff_i[1] // 2): (hw[1] - (diff_i[1] + 1) // 2)]
            rgb_i_orig = torch.nn.functional.interpolate(
                rgb_i, size=hw, mode='bilinear', align_corners=True)
            rgb_orig.append(rgb_i_orig)
        return torch.cat(rgb_orig, 0)