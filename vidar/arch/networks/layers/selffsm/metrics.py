# Copyright 2023 Toyota Research Institute.  All rights reserved.

from functools import partial

import torch
from torch import linalg as LA

from knk_vision.vidar.vidar.geometry.pose_utils import get_geodesic_err, multicam_translations

__PI = 3.1416


def define_metrics():
    """
    Methods that get `gt` and `pred` shaping (Bxcamx4x4), and return the error tensor (Bxcam)

    t: Translation Error || pred_i - gt_i ||
    tn: Normalized Translation Error || pred_i - gt_i || / | gt_i |
    ts: Translation error with gt scaling, || c_i*pred_i - gt_i || / | gt_i |, c_i is |gt_i| / |pred_i|
    ta: Translation angular difference, cos similarity (gt_i, pred_i)
    tad: Cosine similarity with degree
    rot: Rotation comparison via Geodesic error
    rod: Geodesic (degree)
    """
    return {
        "t": partial(multicam_transl_err, borrow_gt_scale=False, normalize=False),
        "tn": partial(multicam_transl_err, borrow_gt_scale=False, normalize=True),
        "ts": partial(multicam_transl_err, borrow_gt_scale=True, normalize=True),
        "ta": partial(multicam_transl_cos_simularity, replace_nan_by=3.1416),
        "tad": partial(multicam_transl_cos_simularity, replace_nan_by=3.1416, to_degree=True),
        "rot": partial(multicam_geodesic_errors, to_degree=False),
        "rod": partial(multicam_geodesic_errors, to_degree=True)
    }


""" 
###############################
Translation metrics 
###############################
"""


def cam_norm(arg_tensor: torch.Tensor) -> torch.Tensor:
    """Translation vector norms which shape (B,), from the pose tensor of (Bx4x4)"""
    t_vec = arg_tensor[:, :3, -1]  # (B,3)
    t_norms = LA.norm(t_vec, dim=1)  # (B, )
    return t_norms


def multicam_norm(arg_tensor: torch.Tensor) -> torch.Tensor:
    """Translation vector norms which shapes (B, cam), from multi-camera poses of (Bxcamx4x4)"""
    b_size = arg_tensor.shape[0]
    t_vec = multicam_translations(arg_tensor)  # (B,cam,3)
    t_norms = LA.norm(t_vec.reshape(-1, 3), dim=1).view(b_size, -1)  # (B, cam)
    return t_norms


def multicam_transl_err(gt: torch.Tensor, pred: torch.Tensor,
                        borrow_gt_scale: bool = False, normalize=True) -> torch.Tensor:
    """
    Get the estimation error of relative poses that base coordinate is the location of the canonical camera.

    Parameters
    ----------
    gt : torch.Tensor
        The ground-truth pose tensor, (Bxcamx4x4)
    pred : torch.Tensor
        The predicted pose (Bxcamx4x4). That translation may be unscaled
    borrow_gt_scale : bool
        Flag whether scaling is applied
    normalize : bool
        Flag whether a normalization is applied

    Returns
    -------
    torch.Tensor
        The error tensor per cameras, (Bxcam)

    """
    b_size = gt.shape[0]
    gt_t, pred_t = [multicam_translations(item) for item in (gt, pred)]  # (B,cam,3) x2
    gt_t_norms = LA.norm(gt_t.reshape(-1, 3), dim=1).view(b_size, -1)  # (B, cam)
    if not borrow_gt_scale:
        ret = LA.norm((gt_t - pred_t).reshape(-1, 3), dim=1).view(b_size, -1)  # (B,cam,3)
    else:
        pred_t_norms = LA.norm(pred_t.reshape(-1, 3), dim=1).view(b_size, -1)  # (B, cam)
        multiply_to_pred = gt_t_norms / (1e-7 + pred_t_norms)  # (B, cam), reshaped to (B, cam, 1) after this process
        ret = LA.norm((gt_t - multiply_to_pred.unsqueeze(2) * pred_t).reshape(-1, 3), dim=1).view(b_size, -1)
    if normalize:
        return ret / gt_t_norms
    else:
        return ret


def multicam_transl_cos_simularity(gt: torch.Tensor, pred: torch.Tensor,
                                   replace_nan_by: float = __PI,
                                   to_degree=False) -> torch.Tensor:
    """
    Vector angle between the ground-truth and prediction that are represented as a homogeneous transformation matrix.

    Parameters
    ----------
    gt : torch.Tensor
        Pose tensor, (Bxcamx4x4)
    pred : torch.Tensor
        Pose tensor, (Bxcamx4x4)
    replace_nan_by : float
        Replace the value if the cosine similarity return Nan
    to_degree : bool
        Decide the unit to represent

    Returns
    -------
    torch.Tensor
        Arc-cosine distance of the two vectors [rad], (Bxcam)
    """
    b_size = gt.shape[0]
    gt_t = gt[:, :, :3, 3]  # (B, cam_num, 3)
    pred_t = pred[:, :, :3, 3]  # (B, cam_num, 3)
    err_per_cam = clamped_acos_simularity(
        gt_t.reshape(-1, 3),
        pred_t.reshape(-1, 3),
        replace_nan_by, to_degree=to_degree).view(b_size, -1)  # (B, cam_num)
    return err_per_cam


def clamped_acos_simularity(gt: torch.Tensor, pd: torch.Tensor, replace_nan_by: float, to_degree=False):
    """ Core implementation for arc-cosine distance; Return pi as a distance to indicate the "Far" place"""
    _dtype_latter = pd.dtype
    if gt.dtype != pd.dtype:  # The more accurate, the more stable to work ...
        gt = gt.to(torch.float64)  # (Bx3)
        pd = pd.to(torch.float64)  # (Bx3)
        _dtype_latter = torch.float64
    dots = torch.diagonal(torch.matmul(gt, torch.t(pd)), 0)  # dot vectors (B,)
    gt_norms = LA.norm(gt, dim=1)
    pd_norms = LA.norm(pd, dim=1)
    norms_product = gt_norms * pd_norms  # norm vectors (B,)
    indices_zero_norm = (norms_product < 1e-7).nonzero().squeeze(1)
    arc_cos = torch.acos(torch.clamp(dots / norms_product, -1., 1.))
    arc_cos[indices_zero_norm] = torch.tensor(replace_nan_by).to(_dtype_latter).to(pd.device)
    if to_degree:
        arc_cos = arc_cos * torch.ones_like(arc_cos) * 180. / torch.pi
    return arc_cos


""" 
###############################
Rotation metrics 
###############################
"""


def multicam_geodesic_errors(gt: torch.Tensor, pred: torch.Tensor, to_degree=False) -> torch.Tensor:
    """Get Geodesic error (B,cam) from pose tensor (B,cam,4,4)"""
    # Rotation; Geodesic error
    b_size = gt.shape[0]
    gt_pose_rot = gt[:, :, :3, :3]  # (B, cam_num, 3, 3)
    pred_pose_rot = pred[:, :, :3, :3]  # (B, cam_num, 3, 3)
    err_per_cam = get_geodesic_err(
        gt_pose_rot.reshape(-1, 3, 3),
        pred_pose_rot.reshape(-1, 3, 3)).view(b_size, -1)  # (B, cam_num)
    if to_degree:
        err_per_cam = err_per_cam * torch.ones_like(err_per_cam) * 180. / torch.pi
    return err_per_cam
