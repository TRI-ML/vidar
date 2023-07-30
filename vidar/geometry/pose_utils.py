# Copyright 2023 Toyota Research Institute.  All rights reserved.

import numpy as np
import torch
import torch.nn.functional as F

from vidar.utils.decorators import iterate1


def mat2euler(M, cy_thresh=None):
    """Convert rotation matrix to euler angles"""
    _FLOAT_EPS_4 = np.finfo(float).eps * 4.0

    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.cpu().data.numpy().dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4

    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flatten()
    cy = torch.sqrt(r33 * r33 + r23 * r23)
    if cy > cy_thresh:
        z = torch.atan2(-r12, r11)
        y = torch.atan2(r13, cy)
        x = torch.atan2(-r23, r33)
    else:
        z = torch.atan2(r21, r22)
        y = torch.atan2(r13, cy)
        x = 0.0
    return torch.tensor([x, y, z], device=M.device)


def to_global_pose(pose, zero_origin=False):
    """Get global pose coordinates from current and context poses"""
    tgt = 0 if 0 in pose else (0, 0)
    base = None if zero_origin else pose[tgt].T[[0]].clone()
    pose[tgt].T[[0]] = torch.eye(4, device=pose[tgt].device, dtype=pose[tgt].dtype)
    for b in range(1, len(pose[tgt])):
        pose[tgt].T[[b]] = (pose[tgt][b] * pose[tgt][0]).T.float()
    for key in pose.keys():
        if key != tgt:
            pose[key] = pose[key] * pose[tgt]
    if not zero_origin:
        for key in pose.keys():
            for b in range(len(pose[key])):
                pose[key].T[[b]] = pose[key].T[[b]] @ base
    return pose


def to_global_pose_broken(pose, zero_origin=False):
    """Get global pose coordinates from current and context poses"""
    tgt = 0 if 0 in pose else (0, 0)
    base = None if zero_origin else pose[tgt].T.clone()
    pose[tgt].T = torch.eye(
        4, device=pose[tgt].device, dtype=pose[tgt].dtype).repeat(pose[tgt].T.shape[0], 1, 1)

    keys = pose.keys()
    # steps = sorted(set([key[0] for key in keys]))
    steps = {key[1]: [key2[0] for key2 in keys if key2[1] == key[1]] for key in keys}
    cams = sorted(set([key[1] for key in keys]))
    
    for cam in cams:
        if cam != tgt[1]:
            pose[(tgt[0], cam)].T = (pose[(tgt[0], cam)] * pose[tgt]).T.float()
    for cam in cams:
        for step in steps[cam]:
            if step != tgt[0]:
                pose[(step, cam)] = (pose[(step, cam)] * pose[(tgt[0], cam)])
    # for step in steps:
    #     if step != tgt[0]:
    #         for cam in cams:
    #             pose[(step, cam)] = (pose[(step, cam)] * pose[(tgt[0], cam)])
    if not zero_origin:
        for key in keys:
            pose[key].T = pose[key].T @ base
    return pose


def euler2mat(angle):
    """Convert euler angles to rotation matrix"""
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack([ cosz, -sinz, zeros,
                         sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).view(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([ cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).view(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).view(B, 3, 3)

    rot_mat = xmat.bmm(ymat).bmm(zmat)
    return rot_mat


def pose_vec2mat(vec, mode='euler'):
    """Convert Euler parameters to transformation matrix."""
    if mode is None:
        return vec
    trans, rot = vec[:, :3].unsqueeze(-1), vec[:, 3:]
    if mode == 'euler':
        rot_mat = euler2mat(rot)
    else:
        raise ValueError('Rotation mode not supported {}'.format(mode))
    mat = torch.cat([rot_mat, trans], dim=2)  # [B,3,4]
    return mat


@iterate1
def invert_pose(T):
    """Inverts a [B,4,4] torch.tensor pose"""
    Tinv = torch.eye(4, device=T.device, dtype=T.dtype).repeat([len(T), 1, 1])
    Tinv[:, :3, :3] = torch.transpose(T[:, :3, :3], -2, -1)
    Tinv[:, :3, -1] = torch.bmm(-1. * Tinv[:, :3, :3], T[:, :3, -1].unsqueeze(-1)).squeeze(-1)
    return Tinv
    # return torch.linalg.inv(T)


def tvec_to_translation(tvec):
    """Converts a [B,3] torch.tensor translation vector to a [B,4,4] torch.tensor pose"""
    batch_size = tvec.shape[0]
    T = torch.eye(4).to(device=tvec.device).repeat(batch_size, 1, 1)
    t = tvec.contiguous().view(-1, 3, 1)
    T[:, :3, 3, None] = t
    return T


def euler2rot(euler):
    """Converts a [B,3] torch.tensor euler angle vector to a [B,4,4] torch.tensor pose"""

    euler_norm = torch.norm(euler, 2, 2, True)
    axis = euler / (euler_norm + 1e-7)

    cos_a = torch.cos(euler_norm)
    sin_a = torch.sin(euler_norm)
    cos1_a = 1 - cos_a

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    x_sin = x * sin_a
    y_sin = y * sin_a
    z_sin = z * sin_a
    x_cos1 = x * cos1_a
    y_cos1 = y * cos1_a
    z_cos1 = z * cos1_a

    xx_cos1 = x * x_cos1
    yy_cos1 = y * y_cos1
    zz_cos1 = z * z_cos1
    xy_cos1 = x * y_cos1
    yz_cos1 = y * z_cos1
    zx_cos1 = z * x_cos1

    batch_size = euler.shape[0]
    rot = torch.zeros((batch_size, 4, 4)).to(device=euler.device)

    rot[:, 0, 0] = torch.squeeze(xx_cos1 + cos_a)
    rot[:, 0, 1] = torch.squeeze(xy_cos1 - z_sin)
    rot[:, 0, 2] = torch.squeeze(zx_cos1 + y_sin)
    rot[:, 1, 0] = torch.squeeze(xy_cos1 + z_sin)
    rot[:, 1, 1] = torch.squeeze(yy_cos1 + cos_a)
    rot[:, 1, 2] = torch.squeeze(yz_cos1 - x_sin)
    rot[:, 2, 0] = torch.squeeze(zx_cos1 - y_sin)
    rot[:, 2, 1] = torch.squeeze(yz_cos1 + x_sin)
    rot[:, 2, 2] = torch.squeeze(zz_cos1 + cos_a)
    rot[:, 3, 3] = 1

    return rot


def vec2mat(euler, translation, invert=False):
    """Converts a [B,3] torch.tensor euler angle vector to a [B,4,4] torch.tensor pose"""

    R = euler2rot(euler)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = tvec_to_translation(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def rot2quat(R):
    """Converts a [B,3,3] torch.tensor rotation matrix to a [B,4] torch.tensor quaternion"""

    b, _, _ = R.shape
    q = torch.ones((b, 4), device=R.device)

    R00 = R[:, 0, 0]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R11 = R[:, 1, 1]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]
    R22 = R[:, 2, 2]

    q[:, 3] = torch.sqrt(1.0 + R00 + R11 + R22) / 2
    q[:, 0] = (R21 - R12) / (4 * q[:, 3])
    q[:, 1] = (R02 - R20) / (4 * q[:, 3])
    q[:, 2] = (R10 - R01) / (4 * q[:, 3])

    return q


def quat2rot(q):
    """Converts a [B,4] torch.tensor quaternion to a [B,3,3] torch.tensor rotation matrix"""

    b, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((b, 3, 3), device=q.device)

    qr = q[:, 0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (qj ** 2 + qk ** 2)
    R[:, 0, 1] = 2 * (qj * qi - qk * qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1 - 2 * (qi ** 2 + qk ** 2)
    R[:, 1, 2] = 2 * (qj * qk - qi * qr)
    R[:, 2, 0] = 2 * (qk * qi - qj * qr)
    R[:, 2, 1] = 2 * (qj * qk + qi * qr)
    R[:, 2, 2] = 1 - 2 * (qi ** 2 + qj ** 2)

    return R
