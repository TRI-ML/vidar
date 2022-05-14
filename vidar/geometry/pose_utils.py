# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn.functional as F

from vidar.utils.decorators import iterate1


def to_global_pose(pose, zero_origin=False):
    """Get global pose coordinates from current and context poses"""
    if zero_origin:
        pose[0].T[[0]] = torch.eye(4, device=pose[0].device, dtype=pose[0].dtype)
    for b in range(1, len(pose[0])):
        pose[0].T[[b]] = (pose[0][b] * pose[0][0]).T.float()
    for key in pose.keys():
        if key != 0:
            pose[key] = pose[key] * pose[0]
    return pose


# def to_global_pose(pose, zero_origin=False):
#     """Get global pose coordinates from current and context poses"""
#     if zero_origin:
#         pose[(0, 0)].T = torch.eye(4, device=pose[(0, 0)].device, dtype=pose[(0, 0)].dtype). \
#             repeat(pose[(0, 0)].shape[0], 1, 1)
#     for key in pose.keys():
#         if key[0] == 0 and key[1] != 0:
#             pose[key].T = (pose[key] * pose[(0, 0)]).T
#     for key in pose.keys():
#         if key[0] != 0:
#             pose[key] = pose[key] * pose[(0, 0)]
#     return pose


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
    """Convert translation and Euler rotation to a [B,4,4] torch.Tensor transformation matrix"""
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
    """Invert a [B,4,4] torch.Tensor pose"""
    Tinv = torch.eye(4, device=T.device, dtype=T.dtype).repeat([len(T), 1, 1])
    Tinv[:, :3, :3] = torch.transpose(T[:, :3, :3], -2, -1)
    Tinv[:, :3, -1] = torch.bmm(-1. * Tinv[:, :3, :3], T[:, :3, -1].unsqueeze(-1)).squeeze(-1)
    return Tinv
    # return torch.linalg.inv(T)


def tvec_to_translation(tvec):
    """Convert translation vector to translation matrix (no rotation)"""
    batch_size = tvec.shape[0]
    T = torch.eye(4).to(device=tvec.device).repeat(batch_size, 1, 1)
    t = tvec.contiguous().view(-1, 3, 1)
    T[:, :3, 3, None] = t
    return T


def euler2rot(euler):
    """Convert Euler parameters to a [B,3,3] torch.Tensor rotation matrix"""
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
    """Convert Euler rotation and translation to a [B,4,4] torch.Tensor transformation matrix"""
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
    """Convert a [B,3,3] rotation matrix to [B,4] quaternions"""
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
    """Convert [B,4] quaternions to [B,3,3] rotation matrix"""
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
