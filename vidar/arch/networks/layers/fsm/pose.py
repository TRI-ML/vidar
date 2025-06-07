# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch

from knk_vision.vidar.vidar.geometry.pose_utils import invert_pose, pose_vec2mat


class Pose:
    """
    Pose class, that encapsulates a [4,4] transformation matrix
    for a specific reference frame
    """
    def __init__(self, mat):
        """
        Initializes a Pose object.

        Parameters
        ----------
        mat : torch.Tensor
            Transformation matrix [B,4,4]
        """
        assert tuple(mat.shape[-2:]) == (4, 4)
        if mat.dim() == 2:
            mat = mat.unsqueeze(0)
        assert mat.dim() == 3
        self.mat = mat

    def __len__(self):
        """Batch size of the transformation matrix"""
        return len(self.mat)

    def __getitem__(self, i):
        return Pose(self.mat[i].unsqueeze(0)).to(self.device)

    @property
    def device(self):
        """Return pose device"""
        return self.mat.device

    @classmethod
    def identity(cls, N=1, device=None, dtype=torch.float):
        """Initializes as a [4,4] identity matrix"""
        return cls(torch.eye(4, device=device, dtype=dtype).repeat([N,1,1]))

    @classmethod
    def from_vec(cls, vec, mode):
        """Initializes from a [B,6] batch vector"""
        mat = pose_vec2mat(vec, mode)  # [B,3,4]
        pose = torch.eye(4, device=vec.device, dtype=vec.dtype).repeat([len(vec), 1, 1])
        pose[:, :3, :3] = mat[:, :3, :3]
        pose[:, :3, -1] = mat[:, :3, -1]
        return cls(pose)

    @property
    def shape(self):
        """Returns the transformation matrix shape"""
        return self.mat.shape

    def item(self):
        """Returns the transformation matrix"""
        return self.mat

    def repeat(self, *args, **kwargs):
        """Repeats the transformation matrix multiple times"""
        self.mat = self.mat.repeat(*args, **kwargs)
        return self

    def inverse(self):
        """Returns a new Pose that is the inverse of this one"""
        return Pose(invert_pose(self.mat))

    def to(self, *args, **kwargs):
        """Moves object to a specific device"""
        self.mat = self.mat.to(*args, **kwargs)
        return self

    def transform_pose(self, pose):
        """Creates a new pose object that compounds this and another one (self * pose)"""
        assert tuple(pose.shape[-2:]) == (4, 4)
        return Pose(self.mat.bmm(pose.item()))

    def transform_points(self, points):
        """Transforms 3D points using this object"""
        assert 2 < points.dim() <= 4 and points.shape[1] == 3, \
            'Wrong dimensions for transform_points'
        # Determine if input is a grid
        is_grid = points.dim() == 4
        # If it's a grid, flatten it
        points_flat = points.view(points.shape[0], 3, -1) if is_grid else points
        # Tranform points
        out = self.mat[:, :3, :3].bmm(points_flat) + \
              self.mat[:, :3, -1].unsqueeze(-1)
        # Return transformed points
        return out.view(points.shape) if is_grid else out

    def __matmul__(self, other):
        """Transforms the input (Pose or 3D points) using this object"""
        if isinstance(other, Pose):
            return self.transform_pose(other)
        elif isinstance(other, torch.Tensor):
            if other.shape[1] == 3 and other.dim() > 2:
                assert other.dim() == 3 or other.dim() == 4
                return self.transform_points(other)
            else:
                raise ValueError('Unknown tensor dimensions {}'.format(other.shape))
        else:
            raise NotImplementedError()

