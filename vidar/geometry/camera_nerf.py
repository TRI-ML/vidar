# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import torch
from torch_scatter import scatter_min

from vidar.geometry.camera import Camera
from vidar.utils.tensor import unnorm_pixel_grid


class CameraNerf(Camera):
    """Camera class with additional functionality"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.convert_matrix = torch.tensor(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        ).unsqueeze(0)

    @staticmethod
    def from_list(cams):
        """Create cameras from a list"""
        K = torch.cat([cam.K for cam in cams], 0)
        Twc = torch.cat([cam.Twc.T for cam in cams], 0)
        return CameraNerf(K=K, Twc=Twc, hw=cams[0].hw)

    def switch(self):
        """Switch camera between conventions"""
        T = self.convert_matrix.to(self.device)
        Twc = T @ self.Twc.T @ T
        return type(self)(K=self.K, Twc=Twc, hw=self.hw)

    def bwd(self):
        """Switch camera to the backwards convention"""
        T = self.convert_matrix.to(self.device)
        Tcw = T @ self.Twc.T @ T
        return type(self)(K=self.K, Tcw=Tcw, hw=self.hw)

    def fwd(self):
        """Switch camera to the forward convention"""
        T = self.convert_matrix.to(self.device)
        Twc = T @ self.Tcw.T @ T
        return type(self)(K=self.K, Twc=Twc, hw=self.hw)

    def look_at(self, at, up=torch.Tensor([0, 1, 0])):
        """
        Set a direction for the camera to point (in-place)

        Parameters
        ----------
        at : torch.Tensor
            Where the camera should be pointing at [B,3]
        up : torch.Tensor
            Up direction [B,3]
        """
        eps = 1e-5
        eye = self.Tcw.T[:, :3, -1]

        at = at.unsqueeze(0)
        up = up.unsqueeze(0).to(at.device)

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

    def get_origin(self, flatten=False):
        """Return camera origin"""
        orig = self.Tcw.T[:, :3, -1].view(len(self), 3, 1, 1).repeat(1, 1, *self.hw)
        if flatten:
            orig = orig.reshape(len(self), 3, -1).permute(0, 2, 1)
        return orig

    def get_viewdirs(self, normalize=False, flatten=False, to_world=False):
        """Return camera viewing rays"""
        ones = torch.ones((len(self), 1, *self.hw), dtype=self.dtype, device=self.device)
        rays = self.reconstruct_depth_map(ones, to_world=False)
        if normalize:
            rays = rays / torch.norm(rays, dim=1).unsqueeze(1)
        if to_world:
            rays = self.to_world(rays).reshape(len(self), 3, *self.hw)
        if flatten:
            rays = rays.reshape(len(self), 3, -1).permute(0, 2, 1)
        return rays

    def get_render_rays(self, near=None, far=None, n_rays=None, gt=None):
        """
        Get render rays

        Parameters
        ----------
        near : Float
            Near plane
        far : Float
            Far plane
        n_rays : Int
            Number of rays
        gt : torch.Tensor
            Ground-truth values for concatenation

        Returns
        -------
        rays : torch.Tensor
            Camera viewing rays
        """
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
        """Get plucker vectors"""
        b = len(self)
        ones = torch.ones((b, 1, *self.hw), dtype=self.dtype, device=self.device)
        rays = self.reconstruct_depth_map(ones, to_world=False)
        rays = rays / torch.norm(rays, dim=1).unsqueeze(1)
        orig = self.Tcw.T[:, :3, -1].view(b, 3, 1, 1).repeat(1, 1, *self.hw)

        orig = orig.view(1, 3, -1).permute(0, 2, 1)
        rays = rays.view(1, 3, -1).permute(0, 2, 1)

        cross = torch.cross(orig, rays, dim=-1)
        plucker = torch.cat((rays, cross), dim=-1)

        return plucker

    def project_pointcloud(self, pcl_src, rgb_src, thr=1):
        """
        Project pointcloud to the camera plane

        Parameters
        ----------
        pcl_src : torch.Tensor
            Input 3D pointcloud
        rgb_src : torch.Tensor
            Pointcloud color information
        thr : Int
            Threshold for the number of valid points

        Returns
        -------
        rgb_tgt : torch.Tensor
            Projected image [B,3,H,W]
        depth_tgt : torch.Tensor
            Projected depth map [B,1,H,W]
        """
        if rgb_src.dim() == 4:
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
                rgb_tgt = rgb_src[i].permute(1, 0)[idx][argmin]
                rgb_tgt[invalid] = -1

            else:

                rgb_tgt = -1 * torch.ones(1, self.n_pixels, 3, device=self.device, dtype=self.dtype)

            # Reshape outputs
            rgb_tgt = rgb_tgt.reshape(1, self.hw[0], self.hw[1], 3).permute(0, 3, 1, 2)
            depth_tgt = depth_tgt.reshape(1, 1, self.hw[0], self.hw[1])

            rgbs_tgt.append(rgb_tgt)
            depths_tgt.append(depth_tgt)

        rgb_tgt = torch.cat(rgbs_tgt, 0)
        depth_tgt = torch.cat(depths_tgt, 0)

        return rgb_tgt, depth_tgt
