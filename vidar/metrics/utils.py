# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import torch

from vidar.utils.tensor import interpolate_image


def scale_output(pred, gt, scale_fn):
    """
    Match depth maps to ground-truth resolution

    Parameters
    ----------
    pred : torch.Tensor
        Predicted depth maps [B,1,w,h]
    gt : torch.tensor
        Ground-truth depth maps [B,1,H,W]
    scale_fn : String
        How to scale output to GT resolution
            Resize: Nearest neighbors interpolation
            top-center: Pad the top of the image and left-right corners with zeros

    Returns
    -------
    pred : torch.tensor
        Uncropped predicted depth maps [B,1,H,W]
    """
    if pred.dim() == 5 and gt.dim() == 5:
        return torch.stack([scale_output(pred[:, i], gt[:, i], scale_fn) for i in range(pred.shape[1])], 1)
    # Return depth map if scaling is not required
    if scale_fn == 'none':
        return pred
    elif scale_fn == 'resize':
        # Resize depth map to GT resolution
        return interpolate_image(pred, gt.shape, mode='bilinear', align_corners=True)
    else:
        # Create empty depth map with GT resolution
        pred_uncropped = torch.zeros(gt.shape, dtype=pred.dtype, device=pred.device)
        # Uncrop top vertically and center horizontally
        if scale_fn == 'top-center':
            top, left = gt.shape[2] - pred.shape[2], (gt.shape[3] - pred.shape[3]) // 2
            pred_uncropped[:, :, top:(top + pred.shape[2]), left:(left + pred.shape[3])] = pred
        else:
            raise NotImplementedError('Depth scale function {} not implemented.'.format(scale_fn))
        # Return uncropped depth map
        return pred_uncropped


def create_crop_mask(crop, gt):
    """
    Create crop mask for evaluation

    Parameters
    ----------
    crop : String
        Type of crop
    gt : torch.Tensor
        Ground-truth depth map (for dimensions)

    Returns
    -------
    crop_mask: torch.Tensor
        Mask for evaluation
    """
    # Return None if mask is not required
    if crop in ('', None):
        return None
    # Create empty mask
    batch_size, _, gt_height, gt_width = gt.shape
    crop_mask = torch.zeros(gt.shape[-2:]).byte().type_as(gt)
    # Get specific mask
    if crop == 'eigen_nyu':
        crop_mask[20:459, 24:615] = 1
    elif crop == 'bts_nyu':
        crop_mask[45:471, 41:601] = 1
    elif crop == 'garg':
        y1, y2 = int(0.40810811 * gt_height), int(0.99189189 * gt_height)
        x1, x2 = int(0.03594771 * gt_width), int(0.96405229 * gt_width)
        crop_mask[y1:y2, x1:x2] = 1
    elif crop == 'eigen':
        y1, y2 = int(0.3324324 * gt_height), int(0.91351351 * gt_height)
        x1, x2 = int(0.03594771 * gt_width), int(0.96405229 * gt_width)
        crop_mask[y1:y2, x1:x2] = 1
    # Return crop mask
    return crop_mask
