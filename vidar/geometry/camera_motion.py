# Copyright 2023 Toyota Research Institute.  All rights reserved.

import random
import numpy as np
import torch

from pyquaternion import Quaternion
from vidar.utils.types import is_seq, is_dict
from vidar.geometry.camera import Camera


def slerp(cams, n=10, keep_edges=False, use_first=False, perturb=False):
    """Interpolate between cameras using SLERP

    Parameters
    ----------
    cams : list of Camera or dict of Camera
        List of cameras
    n : int, optional
        Number of interpolated cameras, by default 10
    keep_edges : bool, optional
        Keep original camera poses in the final interpolation, by default False
    use_first : bool, optional
        Use the first camera location, by default False
    perturb : bool, optional
        Perturb camera locations with random noise, by default False

    Returns
    -------
    list of Camera
        Interpolated cameras
    """

    if is_dict(cams):
        cams = list(cams.values())

    cams_n = []
    hw, K = cams[0].hw, cams[0].K

    n = n + 1
    delta = 1. / n

    if keep_edges:
        cams_n.append(cams[0])

    Rbase = None

    for j in range(len(cams) - 1):

        pose0 = cams[j].Tcw.T
        pose1 = cams[j+1].Tcw.T

        t0 = pose0[:, :3, -1]
        t1 = pose1[:, :3, -1]

        R0 = pose0[:, :3, :3]
        R1 = pose1[:, :3, :3]

        if Rbase is None:
            Rbase = R0

        quat0 = Quaternion(matrix=R0[0].cpu().numpy(), atol=1e-1, rtol=1e-1)
        quat1 = Quaternion(matrix=R1[0].cpu().numpy(), atol=1e-1, rtol=1e-1)

        for i in range(1, n):
            ni = float(i) / n

            if perturb:
                # If it's the first sample and there is only one sample
                if i == 1 and n == 2:
                    ni = random.random()
                # If it's the first sample
                elif i == 1:
                    ni = 0.0 + 1.5 * delta * random.random()
                # If it's the last sample
                elif i == n - 1:
                    ni = 1.0 - 1.5 * delta * random.random()
                # If it's a middle sample
                else:
                    ni = ni + 0.5 * delta * (2 * random.random() - 1)

            ti = t0 + (t1 - t0) * ni

            if not use_first:
                qi = Quaternion.slerp(quat0, quat1, ni)
                qi = torch.tensor(qi.rotation_matrix).unsqueeze(0).to(cams[0].device)
            else:
                qi = Rbase

            Ti = torch.eye(4).unsqueeze(0).to(cams[0].device)
            Ti[:, :3, :3] = qi
            Ti[:, :3, -1] = ti

            cam = Camera(K=K, hw=hw, Tcw=Ti)
            cams_n.append(cam)

        if keep_edges:
            cams_n.append(cams[j+1])

    return cams_n


def wander(cam, num_frames=60, max_disp=48.):
    """Wander around a camera pose

    Parameters
    ----------
    cam : Camera
        Initial camera pose
    num_frames : int, optional
        Number of interpolation frames, by default 60
    max_disp : float, optional
        Maximum noise value, by default 48.

    Returns
    -------
    list of torch.Tensor
        Interpolated poses
    """

    device = cam.device
    c2w = cam.Tcw.T.cpu().numpy()[0]
    fx = cam.fx

    max_trans = max_disp / fx

    output_poses = []
    for i in range(num_frames):
        x_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_frames))
        y_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) # / 3.0
        z_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) # / 3.0

        i_pose = np.concatenate([
            np.concatenate([np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]], axis=1),
            np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]
        ], axis=0)

        i_pose = np.linalg.inv(i_pose)

        ref_pose = np.concatenate([c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0)

        render_pose = np.dot(ref_pose, i_pose)
        output_poses.append(torch.tensor(render_pose, device=device).unsqueeze(0))

    return output_poses


def circle(cam, n, dists, mode='fwd_right', look_at=None, shift_left=False, shift_right=False):
    """Moves a camera in circle

    Parameters
    ----------
    cam : Camera
        Initial camera pose
    n : int 
        Number of interpolation frames
    dists : list of float
        dimensions of the circle
    mode : str, optional
        Circle direction, by default 'fwd_right'
    look_at : torch.Tensor, optional
        Location the cameras will be looking at, by default None
    shift_left : bool, optional
        Which side the camera will be moving (left), by default False
    shift_right : bool, optional
        Which side the camera will be moving (right), by default False

    Returns
    -------
    list of Camera
        Interpolated cameras
    """
    pose = cam.Twc
    pose_fwd = pose.clone().translateForward(1).T[0, :3, -1]
    pose_right = pose.clone().translateRight(1).T[0, :3, -1]
    pose_up = pose.clone().translateUp(1).T[0, :3, -1]
    pose = pose.T[0, :3, -1]

    sign = 1 if shift_right else -1

    if not is_seq(dists):
        dists = [dists] * 3

    diff_fwd = (pose_fwd - pose) * dists[0]
    diff_right = (pose_right - pose) * dists[1]
    diff_up = (pose_up - pose) * dists[2]

    poses = []
    for i in range(n):
        rad = 2 * np.pi * float(i) / float(n)
        rad2 = 8 * np.pi * float(i) / float(n)
        pose_i = cam.clone().Twc.T
        if shift_left:
            pose_i[:, :3, -1] += diff_right
        if shift_right:
            pose_i[:, :3, -1] -= diff_right
        if mode == 'fwd_right':
            pose_i[:, 0, -1] += sign * diff_right[0] * np.sin(rad)
            pose_i[:, 2, -1] += sign * diff_fwd[2] * np.cos(rad)
            pose_i[:, 1, -1] += sign * diff_up[1] * np.sin(rad2)
        elif mode == 'right_up':
            pose_i[:, 0, -1] += sign * diff_right[0] * np.sin(rad)
            pose_i[:, 1, -1] += sign * diff_up[1] * np.cos(rad)
        elif mode == 'fwd_up':
            pose_i[:, 1, -1] += sign * diff_up[1] * np.sin(rad)
            pose_i[:, 2, -1] += sign * diff_fwd[2] * np.cos(rad)
        poses.append(Camera(K=cam.K, hw=cam.hw, Twc=pose_i))

    if look_at is not None:
        pose_fwd = cam.Twc.clone().translateForward(look_at).inverse().T[0, :3, -1]
        for i in range(len(poses)):
            poses[i].look_at(at=pose_fwd)

    return poses
