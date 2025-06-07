# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch

from knk_vision.vidar.vidar.geometry.cameras.base import CameraBase
from knk_vision.vidar.vidar.utils.types import is_seq


class CameraUCM(CameraBase):
    """UCM camera class (Unified Camera Model)"""
    def __init__(self, K, *args, **kwargs):
        self._K = K
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_list(cams):
        """Create a single camera from a list of cameras"""
        K = torch.cat([cam.K for cam in cams], 0)
        Twc = torch.cat([cam.Twc.T for cam in cams], 0)
        return CameraUCM(K=K, Twc=Twc, hw=cams[0].hw)

    @staticmethod
    def from_dict(K, hw, Twc=None):
        """Create a single camera from a dictionary of cameras"""
        return {key: CameraUCM(
            K=K[key] if key in K else K[0],
            hw=hw[key], Twc=val
        ) for key, val in Twc.items()}

    @staticmethod
    def from_list(cams):
        """Create a single camera from a list of cameras"""
        K = torch.cat([cam.K for cam in cams], 0)
        Twc = torch.cat([cam.Twc.T for cam in cams], 0)
        return CameraUCM(K=K, Twc=Twc, hw=cams[0].hw)

    @staticmethod
    def from_dict(K, hw, Twc=None):
        """Create a single camera from a dictionary of cameras"""
        return {key: CameraUCM(K=K[0], hw=hw[0], Twc=val) for key, val in Twc.items()}

    @property
    def fx(self):
        """Focal length in x"""
        return self._K[:, 0].unsqueeze(1)

    @property
    def fy(self):
        """Focal length in y"""
        return self._K[:, 1].unsqueeze(1)

    @property
    def cx(self):
        """Principal point in x"""
        return self._K[:, 2].unsqueeze(1)

    @property
    def cy(self):
        """Principal point in y"""
        return self._K[:, 3].unsqueeze(1)

    @property
    def alpha(self):
        """alpha in UCM model"""
        return self._K[:, 4].unsqueeze(1)

    def scaled(self, scale_factor):
        """Scale the camera intrinsics"""
        if scale_factor is None or scale_factor == 1:
            return self
        if is_seq(scale_factor):
            if len(scale_factor) == 4:
                scale_factor = scale_factor[-2:]
            scale_factor = [float(scale_factor[i]) / float(self._hw[i]) for i in range(2)]
        else:
            scale_factor = [scale_factor] * 2
        K = self._K.clone()
        K[:, 0] *= scale_factor[0]
        K[:, 1] *= scale_factor[1]
        K[:, 2] *= scale_factor[0]
        K[:, 3] *= scale_factor[1]
        return type(self)(
            K=K,
            hw=[int(self._hw[i] * scale_factor[i]) for i in range(len(self._hw))],
            Twc=self._Twc
        )

    def lift(self, grid):
        """Lift a grid of pixels to 3D points"""
        fx, fy, cx, cy, alpha = self.fx, self.fy, self.cx, self.cy, self.alpha

        u = grid[:, 0]
        v = grid[:, 1]

        mx = (u - cx) / fx * (1 - alpha)
        my = (v - cy) / fy * (1 - alpha)
        r_square = mx ** 2 + my ** 2
        xi = alpha / (1 - alpha)
        coeff = (xi + torch.sqrt(1 + (1 - xi ** 2) * r_square)) / (1 + r_square)

        x = coeff * mx
        y = coeff * my
        z = coeff * 1 - xi
        z = z.clamp(min=1e-7)

        x_norm = x / z
        y_norm = y / z
        z_norm = z / z

        return torch.stack((x_norm, y_norm, z_norm), dim=1).float()

    def unlift(self, points, from_world, euclidean=False):
        """Unlift a grid of pixels to 3D points"""
        if from_world:
            points = self.Twc * points

        d = torch.norm(points, dim=1)
        fx, fy, cx, cy, alpha = self.fx, self.fy, self.cx, self.cy, self.alpha
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        z = z.clamp(min=1e-7)

        x = fx * x / (alpha * d + (1 - alpha) * z + 1e-7) + cx
        y = fy * y / (alpha * d + (1 - alpha) * z + 1e-7) + cy
        xy = torch.stack([x, y], dim=1)

        return xy, z

    def get_origin(self, flatten=False):
        """Get the origin of the camera in world coordinates"""
        orig = self.Tcw.T[:, :3, -1].view(len(self), 3, 1, 1).repeat(1, 1, *self.hw)
        if flatten:
            orig = orig.reshape(len(self), 3, -1).permute(0, 2, 1)
        return orig

    def get_viewdirs(self, normalize=None, to_world=None, flatten=False, reflect=False, grid=None):
        """Get the view directions of the camera in world coordinates"""
        ones = torch.ones((len(self), 1, *self.hw), dtype=self.dtype, device=self.device)
        rays = self.reconstruct_depth_map(ones, to_world=False, grid=grid)

        if reflect:
            rays[:, 1] = - rays[:, 1]
            rays[:, 2] = - rays[:, 2]

        if normalize is True or normalize == 'unit':
            rays = rays / torch.norm(rays, dim=1).unsqueeze(1)
        if normalize == 'plane':
            rays = rays / torch.norm(rays, dim=1).unsqueeze(1)
            rays = rays / rays[:, [2]]

        if to_world:
            # rays = self.to_world(rays).reshape(len(self), 3, *self.hw)
            rays = self.no_translation().to_world(rays).reshape(len(self), 3, *self.hw)

        if flatten:
            rays = rays.reshape(len(self), 3, -1).permute(0, 2, 1)

        return rays

    def offset_start(self, start):
        """Offset the camera by a given amount"""
        new_cam = self.clone()
        if is_seq(start):
            new_cam.K[:, 2] -= start[1]
            new_cam.K[:, 3] -= start[0]
        else:
            start = start.to(self.device)
            new_cam.K[:, 2] -= start[:, 1]
            new_cam.K[:, 3] -= start[:, 0]
        return new_cam
