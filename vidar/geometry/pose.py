# Copyright 2023 Toyota Research Institute.  All rights reserved.

from copy import deepcopy

import torch

from vidar.geometry.pose_utils import invert_pose, pose_vec2mat, to_global_pose, euler2mat, to_global_pose_broken
from vidar.utils.types import is_int, is_tensor


def from_dict_sample(T, to_global=False, zero_origin=False, to_matrix=False, broken=False):
    """Helper function to convert sample poses to Pose objects"""
    pose = {key: Pose(val) for key, val in T.items()}
    if to_global:
        to_global_pose_fn = to_global_pose_broken if broken else to_global_pose
        pose = to_global_pose_fn(pose, zero_origin=zero_origin)
    if to_matrix:
        pose = {key: val.T for key, val in pose.items()}
    return pose


def from_dict_batch(T, **kwargs):
    """Helper function to convert a dicionary of tensor poses to Pose objects"""
    pose_batch = [from_dict_sample({key: val[b] for key, val in T.items()}, **kwargs)
                  for b in range(T[0].shape[0])]
    return {key: torch.stack([v[key] for v in pose_batch], 0) for key in pose_batch[0]}


class Pose:
    """
    Pose class, that encapsulates a [4,4] transformation matrix
    for a specific reference frame
    """
    def __init__(self, T=1):
        """
        Initializes a Pose object.

        Parameters
        ----------
        T : int or torch.Tensor
            Transformation matrix [B,4,4]
        """
        if is_int(T):
            T = torch.eye(4).repeat(T, 1, 1)
        self.T = T if T.dim() == 3 else T.unsqueeze(0)

    def __len__(self):
        """Batch size of the transformation matrix"""
        return len(self.T)

    def __getitem__(self, idx):
        """Return batch-wise pose"""
        if not is_tensor(idx):
            idx = [idx]
        return Pose(self.T[idx])

    def __mul__(self, data):
        """Transforms the input (Pose or 3D points) using this object"""
        if isinstance(data, Pose):
            return Pose(self.T.bmm(data.T))
        elif isinstance(data, torch.Tensor):
            return self.T[:, :3, :3].bmm(data) + self.T[:, :3, -1].unsqueeze(-1)
        else:
            raise NotImplementedError()

    def detach(self):
        """Detach pose from graph"""
        return Pose(self.T.detach())

    def clone(self):
        """Clone pose"""
        return type(self)(
            T=self.T.clone()
        )

    @property
    def shape(self):
        """Return pose shape"""
        return self.T.shape

    @property
    def device(self):
        """Return pose device"""
        return self.T.device

    @property
    def dtype(self):
        """Return pose type"""
        return self.T.dtype

    @classmethod
    def identity(cls, N=1, device=None, dtype=torch.float):
        """Initializes as a [4,4] identity matrix"""
        return cls(torch.eye(4, device=device, dtype=dtype).repeat([N,1,1]))

    @staticmethod
    def from_dict(T, to_global=False, zero_origin=False, to_matrix=False, broken=False):
        if T is None:
            return None
        tgt = (0, 0) if broken else 0
        if T[tgt].dim() == 3:
            return from_dict_sample(
                T, to_global=to_global, zero_origin=zero_origin, to_matrix=to_matrix, broken=broken)
        elif T[tgt].dim() == 4:
            return from_dict_batch(
                T, to_global=to_global, zero_origin=zero_origin, to_matrix=True, broken=broken)

    @classmethod
    def from_vec(cls, vec, mode):
        """Initializes from a [B,6] batch vector"""
        mat = pose_vec2mat(vec, mode)
        pose = torch.eye(4, device=vec.device, dtype=vec.dtype).repeat([len(vec), 1, 1])
        pose[:, :3, :3] = mat[:, :3, :3]
        pose[:, :3, -1] = mat[:, :3, -1]
        return cls(pose)

    def repeat(self, *args, **kwargs):
        """Repeats the transformation matrix multiple times"""
        self.T = self.T.repeat(*args, **kwargs)
        return self

    def inverse(self):
        """Returns a new Pose that is the inverse of this one"""
        return Pose(invert_pose(self.T))

    def to(self, *args, **kwargs):
        """Moves object to a specific device"""
        self.T = self.T.to(*args, **kwargs)
        return self

    def cuda(self, *args, **kwargs):
        """Moves pose to GPU"""
        self.to('cuda')
        return self

    def translate(self, xyz):
        """Translate pose by a vector"""
        self.T[:, :3, -1] = self.T[:, :3, -1] + xyz.to(self.device)
        return self

    def rotate(self, rpw):
        """Rotate pose by a vector"""
        rot = euler2mat(rpw)
        T = invert_pose(self.T).clone()
        T[:, :3, :3] = T[:, :3, :3] @ rot.to(self.device)
        self.T = invert_pose(T)
        return self

    def rotateRoll(self, r):
        """Rotate pose by a roll angle"""
        return self.rotate(torch.tensor([[0, 0, r]]))

    def rotatePitch(self, p):
        """Rotate pose by a pitch angle"""
        return self.rotate(torch.tensor([[p, 0, 0]]))

    def rotateYaw(self, w):
        """Rotate pose by a yaw angle"""
        return self.rotate(torch.tensor([[0, w, 0]]))

    def translateForward(self, t):
        """Translate pose on its z axis (forward)"""
        return self.translate(torch.tensor([[0, 0, -t]]))

    def translateBackward(self, t):
        """Translate pose on its z axis (backward)"""
        return self.translate(torch.tensor([[0, 0, +t]]))

    def translateLeft(self, t):
        """Translate pose on its x axis (left)"""
        return self.translate(torch.tensor([[+t, 0, 0]]))

    def translateRight(self, t):
        """Translate pose on its x axis (right)"""
        return self.translate(torch.tensor([[-t, 0, 0]]))

    def translateUp(self, t):
        """Translate pose on its y axis (up)"""
        return self.translate(torch.tensor([[0, +t, 0]]))

    def translateDown(self, t):
        """Translate pose on its y axis (down)"""
        return self.translate(torch.tensor([[0, -t, 0]]))
