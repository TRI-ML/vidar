# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import torch

from vidar.geometry.pose_utils import invert_pose, pose_vec2mat, to_global_pose, euler2mat
from vidar.utils.types import is_int


def from_dict_sample(T, to_global=False, zero_origin=False, to_matrix=False):
    """
    Create poses from a sample dictionary

    Parameters
    ----------
    T : Dict
        Dictionary containing input poses [B,4,4]
    to_global : Bool
        Whether poses should be converted to global frame of reference
    zero_origin : Bool
        Whether the target camera should be the center of the frame of reference
    to_matrix : Bool
        Whether output poses should be classes or tensors

    Returns
    -------
    pose : Dict
        Dictionary containing output poses
    """
    pose = {key: Pose(val) for key, val in T.items()}
    if to_global:
        pose = to_global_pose(pose, zero_origin=zero_origin)
    if to_matrix:
        pose = {key: val.T for key, val in pose.items()}
    return pose


def from_dict_batch(T, **kwargs):
    """Create poses from a batch dictionary"""
    pose_batch = [from_dict_sample({key: val[b] for key, val in T.items()}, **kwargs)
                  for b in range(T[0].shape[0])]
    return {key: torch.stack([v[key] for v in pose_batch], 0) for key in pose_batch[0]}


class Pose:
    """
    Pose class for 3D operations

    Parameters
    ----------
    T : torch.Tensor or Int
        Transformation matrix [B,4,4], or batch size (poses initialized as identity)
    """
    def __init__(self, T=1):
        if is_int(T):
            T = torch.eye(4).repeat(T, 1, 1)
        self.T = T if T.dim() == 3 else T.unsqueeze(0)

    def __len__(self):
        """Return batch size"""
        return len(self.T)

    def __getitem__(self, i):
        """Return batch-wise pose"""
        return Pose(self.T[[i]])

    def __mul__(self, data):
        """Transforms data (pose or 3D points)"""
        if isinstance(data, Pose):
            return Pose(self.T.bmm(data.T))
        elif isinstance(data, torch.Tensor):
            return self.T[:, :3, :3].bmm(data) + self.T[:, :3, -1].unsqueeze(-1)
        else:
            raise NotImplementedError()

    def detach(self):
        """Return detached pose"""
        return Pose(self.T.detach())

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
    def from_dict(T, to_global=False, zero_origin=False, to_matrix=False):
        """Create poses from a dictionary"""
        if T[0].dim() == 3:
            return from_dict_sample(T, to_global=to_global, zero_origin=zero_origin, to_matrix=to_matrix)
        elif T[0].dim() == 4:
            return from_dict_batch(T, to_global=to_global, zero_origin=zero_origin, to_matrix=True)

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
        """Copy pose to device"""
        self.T = self.T.to(*args, **kwargs)
        return self

    def cuda(self, *args, **kwargs):
        """Copy pose to CUDA"""
        self.to('cuda')
        return self

    def translate(self, xyz):
        """Translate pose"""
        self.T[:, :3, -1] = self.T[:, :3, -1] + xyz.to(self.device)
        return self

    def rotate(self, rpw):
        """Rotate pose"""
        rot = euler2mat(rpw)
        T = invert_pose(self.T).clone()
        T[:, :3, :3] = T[:, :3, :3] @ rot.to(self.device)
        self.T = invert_pose(T)
        return self

    def rotateRoll(self, r):
        """Rotate pose in the roll axis"""
        return self.rotate(torch.tensor([[0, 0, r]]))

    def rotatePitch(self, p):
        """Rotate pose in the pitcv axis"""
        return self.rotate(torch.tensor([[p, 0, 0]]))

    def rotateYaw(self, w):
        """Rotate pose in the yaw axis"""
        return self.rotate(torch.tensor([[0, w, 0]]))

    def translateForward(self, t):
        """Translate pose forward"""
        return self.translate(torch.tensor([[0, 0, -t]]))

    def translateBackward(self, t):
        """Translate pose backward"""
        return self.translate(torch.tensor([[0, 0, +t]]))

    def translateLeft(self, t):
        """Translate pose left"""
        return self.translate(torch.tensor([[+t, 0, 0]]))

    def translateRight(self, t):
        """Translate pose right"""
        return self.translate(torch.tensor([[-t, 0, 0]]))

    def translateUp(self, t):
        """Translate pose up"""
        return self.translate(torch.tensor([[0, +t, 0]]))

    def translateDown(self, t):
        """Translate pose down"""
        return self.translate(torch.tensor([[0, -t, 0]]))
