# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch
from torch_scatter import scatter_min

from knk_vision.vidar.vidar.geometry.camera_utils import invert_intrinsics, scale_intrinsics
from knk_vision.vidar.vidar.geometry.cameras.base import CameraBase
from knk_vision.vidar.vidar.utils.tensor import same_shape, cat_channel_ones, unnorm_pixel_grid
from knk_vision.vidar.vidar.utils.types import is_seq


class CameraPinhole(CameraBase):
    """Pinhole camera model"""
    def __init__(self, K, *args, **kwargs):
        # Intrinsics
        if same_shape(K.shape[-2:], (3, 3)):
            self._K = torch.eye(4, dtype=K.dtype, device=K.device).repeat(K.shape[0], 1, 1)
            self._K[:, :3, :3] = K
        else:
            self._K = K
        super().__init__(*args, **kwargs)

        self.convert_matrix = torch.tensor(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        ).unsqueeze(0)

    @staticmethod
    def from_list(cams):
        """Create a single camera from a list of cameras"""
        K = torch.cat([cam.K for cam in cams], 0)
        Twc = torch.cat([cam.Twc.T for cam in cams], 0)
        return CameraPinhole(K=K, Twc=Twc, hw=cams[0].hw)

    @staticmethod
    def from_dict(K, hw, Twc=None, broken=False):
        """Create a single camera from a dictionary of cameras"""
        if broken:
            return {key: CameraPinhole(
                K=K[key] if key in K else K[(0, key[1])],
                hw=hw[key], Twc=val
            ) for key, val in Twc.items()}
        else:
            return {key: CameraPinhole(
                K=K[key] if key in K else K[0],
                hw=hw[key], Twc=val
            ) for key, val in Twc.items()}

    @property
    def fx(self):
        """Focal length in x direction"""
        return self._K[:, 0, 0]

    @property
    def fy(self):
        """Focal length in y direction"""
        return self._K[:, 1, 1]

    @property
    def cx(self):
        """Principal point in x direction"""
        return self._K[:, 0, 2]

    @property
    def cy(self):
        """Principal point in y direction"""
        return self._K[:, 1, 2]

    @property
    def fxy(self):
        """Focal length in x and y direction"""
        return torch.tensor([self.fx, self.fy], dtype=self.dtype, device=self.device)

    @property
    def cxy(self):
        """Principal point in x and y direction"""
        return torch.tensor([self.cx, self.cy], dtype=self.dtype, device=self.device)

    @property
    def invK(self):
        """Inverse of camera intrinsics"""
        return invert_intrinsics(self._K)

    def offset_start(self, start):
        """Offset the principal point"""
        new_cam = self.clone()
        if is_seq(start):
            new_cam.K[:, 0, 2] -= start[1]
            new_cam.K[:, 1, 2] -= start[0]
        else:
            start = start.to(self.device)
            new_cam.K[:, 0, 2] -= start[:, 1]
            new_cam.K[:, 1, 2] -= start[:, 0]
        return new_cam

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
        return type(self)(
            K=scale_intrinsics(self._K, scale_factor),
            hw=[int(self._hw[i] * scale_factor[i]) for i in range(len(self._hw))],
            Twc=self._Twc
        )

    def lift(self, grid):
        """Lift a grid of points to 3D"""
        return torch.matmul(self.invK[:, :3, :3], grid)

    def unlift(self, points, from_world=True, euclidean=False):
        """Unlift a grid of points to 2D"""
        projected = torch.matmul(self.Pwc(from_world), cat_channel_ones(points, 1))
        coords = projected[:, :2] / (projected[:, 2].unsqueeze(1) + 1e-7)
        if not euclidean:
            depth = projected[:, 2]
        else:
            points = self.from_world(points) if from_world else points
            depth = torch.linalg.vector_norm(points, dim=1, keepdim=True)
        return coords, depth

    def switch(self):
        """Switch the camera from world to camera coordinates"""
        T = self.convert_matrix.to(self.device)
        Twc = T @ self.Twc.T @ T
        return type(self)(K=self.K, Twc=Twc, hw=self.hw)

    def bwd(self):
        """Switch the camera from world to camera coordinates"""
        T = self.convert_matrix.to(self.device)
        Tcw = T @ self.Twc.T @ T
        return type(self)(K=self.K, Tcw=Tcw, hw=self.hw)

    def fwd(self):
        """Switch the camera from world to camera coordinates"""
        T = self.convert_matrix.to(self.device)
        Twc = T @ self.Tcw.T @ T
        return type(self)(K=self.K, Twc=Twc, hw=self.hw)

    def up(self):
        """Get the up vector of the camera"""
        up = self.clone()
        up.Twc.translateUp(1)
        return up.get_center() - self.get_center()

    def forward(self):
        """Get the forward vector of the camera"""
        forward = self.clone()
        forward.Twc.translateForward(1)
        return forward.get_center() - self.get_center()

    def look_at(self, at, up=None):
        """Look at a point"""

        if up is None:
            up = self.up()

        eps = 1e-5
        eye = self.get_center()

        at = at.unsqueeze(0)
        up = up.unsqueeze(0).to(at.device)
        up /= up.norm(dim=-1, keepdim=True) + eps

        z_axis = at - eye
        z_axis /= z_axis.norm(dim=-1, keepdim=True) + eps

        up = up.expand(z_axis.shape)
        x_axis = torch.cross(up, z_axis)
        x_axis /= x_axis.norm(dim=-1, keepdim=True) + eps

        y_axis = torch.cross(z_axis, x_axis)
        y_axis /= y_axis.norm(dim=-1, keepdim=True) + eps

        R = torch.stack((x_axis, y_axis, z_axis), dim=-1)

        Tcw = self.Tcw
        Tcw.T[:, :3, :3] = R
        self.Twc = Tcw.inverse()

    def get_center(self):
        """Get the center of the camera"""
        return self.Tcw.T[:, :3, -1]

    def get_origin(self, flatten=False):
        """Get the origin of the camera"""
        orig = self.get_center().view(len(self), 3, 1, 1).repeat(1, 1, *self.hw)
        if flatten:
            orig = orig.reshape(len(self), 3, -1).permute(0, 2, 1)
        return orig

    def get_viewdirs(self, normalize=None, to_world=None, flatten=False, reflect=False, grid=None):
        """Get the view directions of the camera"""

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

    def get_render_rays(self, near=None, far=None, n_rays=None, gt=None):
        """Get the rays for rendering"""

        b = len(self)

        ones = torch.ones((b, 1, *self.hw), dtype=self.dtype, device=self.device)

        rays = self.reconstruct_depth_map(ones, to_world=False)
        rays = rays / torch.norm(rays, dim=1).unsqueeze(1)

        rays[:, 1] = - rays[:, 1]
        rays[:, 2] = - rays[:, 2]

        orig = self.pose[:, :3, -1].view(b, 3, 1, 1).repeat(1, 1, *self.hw)
        rays = self.no_translation().inverted_pose().to_world(rays).reshape(b, 3, *self.hw)

        info = [orig, rays]
        if near is not None:
            info = info + [near * ones]
        if far is not None:
            info = info + [far * ones]
        if gt is not None:
            info = info + [gt]

        rays = torch.cat(info, 1)
        rays = rays.permute(0, 2, 3, 1).reshape(b, -1, rays.shape[1])

        if n_rays is not None:
            idx = torch.randint(0, self.n_pixels, (n_rays,))
            rays = rays[:, idx, :]

        return rays

    def get_plucker(self):
        """Get the plucker coordinates of the camera"""

        b = len(self)
        ones = torch.ones((b, 1, *self.hw), dtype=self.dtype, device=self.device)
        rays = self.reconstruct_depth_map(ones, to_world=False)
        rays = rays / torch.norm(rays, dim=1).unsqueeze(1)
        orig = self.get_center().view(b, 3, 1, 1).repeat(1, 1, *self.hw)

        orig = orig.view(1, 3, -1).permute(0, 2, 1)
        rays = rays.view(1, 3, -1).permute(0, 2, 1)

        cross = torch.cross(orig, rays, dim=-1)
        plucker = torch.cat((rays, cross), dim=-1)

        return plucker

    def project_pointcloud(self, pcl_src, rgb_src=None, thr=1):
        """Project a pointcloud to the image plane"""

        if rgb_src is not None and rgb_src.dim() == 4:
            rgb_src = rgb_src.view(*rgb_src.shape[:2], -1)

        # Get projected coordinates and depth values
        uv_all, z_all = self.project_points(pcl_src, return_z=True, from_world=True)

        rgbs_tgt, depths_tgt = [], []

        b = pcl_src.shape[0]
        for i in range(b):
            uv, z = uv_all[i].reshape(-1, 2), z_all[i].reshape(-1, 1)

            # Remove out-of-bounds coordinates and points behind the camera
            idx = (uv[:, 0] >= -1) & (uv[:, 0] <= 1) & \
                  (uv[:, 1] >= -1) & (uv[:, 1] <= 1) & (z[:, 0] > 0.0)

            # Unormalize and stack coordinates for scatter operation
            uv = (unnorm_pixel_grid(uv[idx], self.hw)).round().long()
            uv = uv[:, 0] + uv[:, 1] * self.hw[1]

            # Min scatter operation (only keep the closest depth)
            depth_tgt = 1e10 * torch.ones((self.hw[0] * self.hw[1], 1), device=pcl_src.device)
            depth_tgt, argmin = scatter_min(src=z[idx], index=uv.unsqueeze(1), dim=0, out=depth_tgt)
            depth_tgt[depth_tgt == 1e10] = 0.

            num_valid = (depth_tgt > 0).sum()
            if num_valid > thr:

                # Substitute invalid values with zero
                invalid = argmin == argmin.max()
                argmin[invalid] = 0
                if rgb_src is not None:
                    rgb_tgt = rgb_src[i].permute(1, 0)[idx][argmin]
                    rgb_tgt[invalid] = 0

            else:

                if rgb_src is not None:
                    rgb_tgt = -1 * torch.ones(1, self.n_pixels, 3, device=self.device, dtype=self.dtype)

            # Reshape outputs
            depth_tgt = depth_tgt.reshape(1, 1, self.hw[0], self.hw[1])
            depths_tgt.append(depth_tgt)

            if rgb_src is not None:
                rgb_tgt = rgb_tgt.reshape(1, self.hw[0], self.hw[1], 3).permute(0, 3, 1, 2)
                rgbs_tgt.append(rgb_tgt)

        if rgb_src is not None:
            rgb_tgt = torch.cat(rgbs_tgt, 0)
        else:
            rgb_tgt = None

        depth_tgt = torch.cat(depths_tgt, 0)

        return rgb_tgt, depth_tgt

    def reconstruct_depth_map_rays(self, depth, to_world=False):
        """Reconstruct a depth map from a depth image"""
        if depth is None:
            return None
        b, _, h, w = depth.shape
        rays = self.get_viewdirs(normalize=True, to_world=False)
        points = (rays * depth).view(b, 3, -1)
        if to_world and self.Tcw is not None:
            points = self.Tcw * points
        return points.view(b, 3, h, w)

    def to_ndc_rays(self, rays_o, rays_d, near=1.0):
        """Transform rays from camera coordinates to NDC coordinates"""
        H, W = self.hw
        focal = self.fy[0].item()

        t = - (near + rays_o[..., 2]) / rays_d[..., 2]
        rays_o = rays_o + t[..., None] * rays_d

        o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
        o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
        o2 = 1. + 2. * near / rays_o[..., 2]

        d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
        d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
        d2 = -2. * near / rays_o[..., 2]

        rays_o = torch.stack([o0, o1, o2], -1)
        rays_d = torch.stack([d0, d1, d2], -1)

        return rays_o, rays_d

    def from_ndc(self, xyz_ndc):
        """Transform points from NDC coordinates to camera coordinates"""
        wh = self.wh
        fx = fy = self.fy[0].item()

        z_e = 2. / (xyz_ndc[..., 2:3] - 1. + 1e-6)
        x_e = - xyz_ndc[..., 0:1] * z_e * wh[0] / (2. * fx)
        y_e = - xyz_ndc[..., 1:2] * z_e * wh[1] / (2. * fy)

        return torch.cat([x_e, y_e, z_e], -1)
