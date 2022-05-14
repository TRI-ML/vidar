# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from abc import ABC
from copy import deepcopy

import torch
import torch.nn as nn
from einops import rearrange

from vidar.geometry.camera_utils import invert_intrinsics, scale_intrinsics
from vidar.geometry.pose import Pose
from vidar.geometry.pose_utils import invert_pose
from vidar.utils.tensor import pixel_grid, same_shape, cat_channel_ones, norm_pixel_grid, interpolate, interleave
from vidar.utils.types import is_tensor, is_seq


class Camera(nn.Module, ABC):
    """
    Camera class for 3D reconstruction

    Parameters
    ----------
    K : torch.Tensor
        Camera intrinsics [B,3,3]
    hw : Tuple
        Camera height and width
    Twc : Pose or torch.Tensor
        Camera pose (world to camera) [B,4,4]
    Tcw : Pose or torch.Tensor
        Camera pose (camera to world) [B,4,4]
    """
    def __init__(self, K, hw, Twc=None, Tcw=None):
        super().__init__()

        # Asserts

        assert Twc is None or Tcw is None

        # Fold if multi-batch

        if K.dim() == 4:
            K = rearrange(K, 'b n h w -> (b n) h w')
            if Twc is not None:
                Twc = rearrange(Twc, 'b n h w -> (b n) h w')
            if Tcw is not None:
                Tcw = rearrange(Tcw, 'b n h w -> (b n) h w')

        # Intrinsics

        if same_shape(K.shape[-2:], (3, 3)):
            self._K = torch.eye(4, dtype=K.dtype, device=K.device).repeat(K.shape[0], 1, 1)
            self._K[:, :3, :3] = K
        else:
            self._K = K

        # Pose

        if Twc is None and Tcw is None:
            self._Twc = torch.eye(4, dtype=K.dtype, device=K.device).unsqueeze(0).repeat(K.shape[0], 1, 1)
        else:
            self._Twc = invert_pose(Tcw) if Tcw is not None else Twc
        if is_tensor(self._Twc):
            self._Twc = Pose(self._Twc)

        # Resolution

        self._hw = hw
        if is_tensor(self._hw):
            self._hw = self._hw.shape[-2:]

    def __getitem__(self, idx):
        """Return batch-wise pose"""
        if is_seq(idx):
            return type(self).from_list([self.__getitem__(i) for i in idx])
        else:
            return type(self)(
                K=self._K[[idx]],
                Twc=self._Twc[[idx]] if self._Twc is not None else None,
                hw=self._hw,
            )

    def __len__(self):
        """Return length as intrinsics batch"""
        return self._K.shape[0]

    def __eq__(self, cam):
        """Check if two cameras are the same"""
        if not isinstance(cam, type(self)):
            return False
        if self._hw[0] != cam.hw[0] or self._hw[1] != cam.hw[1]:
            return False
        if not torch.allclose(self._K, cam.K):
            return False
        if not torch.allclose(self._Twc.T, cam.Twc.T):
            return False
        return True

    def clone(self):
        """Return a copy of this camera"""
        return deepcopy(self)

    @property
    def pose(self):
        """Return camera pose (world to camera)"""
        return self._Twc.T

    @property
    def K(self):
        """Return camera intrinsics"""
        return self._K

    @K.setter
    def K(self, K):
        """Set camera intrinsics"""
        self._K = K

    @property
    def invK(self):
        """Return inverse of camera intrinsics"""
        return invert_intrinsics(self._K)

    @property
    def batch_size(self):
        """Return batch size"""
        return self._Twc.T.shape[0]

    @property
    def hw(self):
        """Return camera height and width"""
        return self._hw

    @hw.setter
    def hw(self, hw):
        """Set camera height and width"""
        self._hw = hw

    @property
    def wh(self):
        """Get camera width and height"""
        return self._hw[::-1]

    @property
    def n_pixels(self):
        """Return number of pixels"""
        return self._hw[0] * self._hw[1]

    @property
    def fx(self):
        """Return horizontal focal length"""
        return self._K[:, 0, 0]

    @property
    def fy(self):
        """Return vertical focal length"""
        return self._K[:, 1, 1]

    @property
    def cx(self):
        """Return horizontal principal point"""
        return self._K[:, 0, 2]

    @property
    def cy(self):
        """Return vertical principal point"""
        return self._K[:, 1, 2]

    @property
    def fxy(self):
        """Return focal length"""
        return torch.tensor([self.fx, self.fy], dtype=self.dtype, device=self.device)

    @property
    def cxy(self):
        """Return principal points"""
        return self._K[:, :2, 2]
        # return torch.tensor([self.cx, self.cy], dtype=self.dtype, device=self.device)

    @property
    def Tcw(self):
        """Return camera pose (camera to world)"""
        return None if self._Twc is None else self._Twc.inverse()

    @Tcw.setter
    def Tcw(self, Tcw):
        """Set camera pose (camera to world)"""
        self._Twc = Tcw.inverse()

    @property
    def Twc(self):
        """Return camera pose (world to camera)"""
        return self._Twc

    @Twc.setter
    def Twc(self, Twc):
        """Set camera pose (world to camera)"""
        self._Twc = Twc

    @property
    def dtype(self):
        """Return tensor type"""
        return self._K.dtype

    @property
    def device(self):
        """Return device"""
        return self._K.device

    def detach_pose(self):
        """Detach pose from the graph"""
        return type(self)(K=self._K, hw=self._hw,
                          Twc=self._Twc.detach() if self._Twc is not None else None)

    def detach_K(self):
        """Detach intrinsics from the graph"""
        return type(self)(K=self._K.detach(), hw=self._hw, Twc=self._Twc)

    def detach(self):
        """Detach camera from the graph"""
        return type(self)(K=self._K.detach(), hw=self._hw,
                          Twc=self._Twc.detach() if self._Twc is not None else None)

    def inverted_pose(self):
        """Invert camera pose"""
        return type(self)(K=self._K, hw=self._hw,
                          Twc=self._Twc.inverse() if self._Twc is not None else None)

    def no_translation(self):
        """Return new camera without translation"""
        Twc = self.pose.clone()
        Twc[:, :-1, -1] = 0
        return type(self)(K=self._K, hw=self._hw, Twc=Twc)

    @staticmethod
    def from_dict(K, hw, Twc=None):
        """Create cameras from a pose dictionary"""
        return {key: Camera(K=K[0], hw=hw[0], Twc=val) for key, val in Twc.items()}

    # @staticmethod
    # def from_dict(K, hw, Twc=None):
    #     return {key: Camera(K=K[(0, 0)], hw=hw[(0, 0)], Twc=val) for key, val in Twc.items()}

    @staticmethod
    def from_list(cams):
        """Create cameras from a list"""
        K = torch.cat([cam.K for cam in cams], 0)
        Twc = torch.cat([cam.Twc.T for cam in cams], 0)
        return Camera(K=K, Twc=Twc, hw=cams[0].hw)

    def scaled(self, scale_factor):
        """Return a scaled camera"""
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

    def offset_start(self, start):
        """Offset camera intrinsics based on a crop"""
        new_cam = self.clone()
        start = start.to(self.device)
        new_cam.K[:, 0, 2] -= start[:, 1]
        new_cam.K[:, 1, 2] -= start[:, 0]
        return new_cam

    def interpolate(self, rgb):
        """Interpolate an image to fit the camera"""
        if rgb.dim() == 5:
            rgb = rearrange(rgb, 'b n c h w -> (b n) c h w')
        return interpolate(rgb, scale_factor=None, size=self.hw, mode='bilinear', align_corners=True)

    def interleave_K(self, b):
        """Interleave camera intrinsics to fit multiple batches"""
        return type(self)(
            K=interleave(self._K, b),
            Twc=self._Twc,
            hw=self._hw,
        )

    def interleave_Twc(self, b):
        """Interleave camera pose to fit multiple batches"""
        return type(self)(
            K=self._K,
            Twc=interleave(self._Twc, b),
            hw=self._hw,
        )

    def interleave(self, b):
        """Interleave camera to fit multiple batches"""
        return type(self)(
            K=interleave(self._K, b),
            Twc=interleave(self._Twc, b),
            hw=self._hw,
        )

    def Pwc(self, from_world=True):
        """Return projection matrix"""
        return self._K[:, :3] if not from_world or self._Twc is None else \
            torch.matmul(self._K, self._Twc.T)[:, :3]

    def to_world(self, points):
        """Transform points to world coordinates"""
        if points.dim() > 3:
            points = points.reshape(points.shape[0], 3, -1)
        return points if self.Tcw is None else self.Tcw * points

    def from_world(self, points):
        """Transform points back to camera coordinates"""
        if points.dim() > 3:
            points = points.reshape(points.shape[0], 3, -1)
        return points if self._Twc is None else \
            torch.matmul(self._Twc.T, cat_channel_ones(points, 1))[:, :3]

    def to(self, *args, **kwargs):
        """Copy camera to device"""
        self._K = self._K.to(*args, **kwargs)
        if self._Twc is not None:
            self._Twc = self._Twc.to(*args, **kwargs)
        return self

    def cuda(self, *args, **kwargs):
        """Copy camera to CUDA"""
        return self.to('cuda')

    def relative_to(self, cam):
        """Create a new camera relative to another camera"""
        return Camera(K=self._K, hw=self._hw, Twc=self._Twc * cam.Twc.inverse())

    def global_from(self, cam):
        """Create a new camera in global coordinates relative to another camera"""
        return Camera(K=self._K, hw=self._hw, Twc=self._Twc * cam.Twc)

    def reconstruct_depth_map(self, depth, to_world=False):
        """
        Reconstruct a depth map from the camera viewpoint

        Parameters
        ----------
        depth : torch.Tensor
            Input depth map [B,1,H,W]
        to_world : Bool
            Transform points to world coordinates

        Returns
        -------
        points : torch.Tensor
            Output 3D points [B,3,H,W]
        """
        if depth is None:
            return None
        b, _, h, w = depth.shape
        grid = pixel_grid(depth, with_ones=True, device=depth.device).view(b, 3, -1)
        points = depth.view(b, 1, -1) * torch.matmul(self.invK[:, :3, :3], grid)
        if to_world and self.Tcw is not None:
            points = self.Tcw * points
        return points.view(b, 3, h, w)

    def reconstruct_cost_volume(self, volume, to_world=True, flatten=True):
        """
        Reconstruct a cost volume from the camera viewpoint

        Parameters
        ----------
        volume : torch.Tensor
            Input depth map [B,1,D,H,W]
        to_world : Bool
            Transform points to world coordinates
        flatten: Bool
            Flatten volume points

        Returns
        -------
        points : torch.Tensor
            Output 3D points [B,3,D,H,W]
        """
        c, d, h, w = volume.shape
        grid = pixel_grid((h, w), with_ones=True, device=volume.device).view(3, -1).repeat(1, d)
        points = torch.stack([
            (volume.view(c, -1) * torch.matmul(invK[:3, :3].unsqueeze(0), grid)).view(3, d * h * w)
            for invK in self.invK], 0)
        if to_world and self.Tcw is not None:
            points = self.Tcw * points
        if flatten:
            return points.view(-1, 3, d, h * w).permute(0, 2, 1, 3)
        else:
            return points.view(-1, 3, d, h, w)

    def project_points(self, points, from_world=True, normalize=True, return_z=False):
        """
        Project points back to image plane

        Parameters
        ----------
        points : torch.Tensor
            Input 3D points [B,3,H,W] or [B,3,N]
        from_world : Bool
            Whether points are in the global frame
        normalize : Bool
            Whether projections should be normalized to [-1,1]
        return_z : Bool
            Whether projected depth is return as well

        Returns
        -------
        coords : torch.Tensor
            Projected 2D coordinates [B,2,H,W]
        depth : torch.Tensor
            Projected depth [B,1,H,W]
        """
        is_depth_map = points.dim() == 4
        hw = self._hw if not is_depth_map else points.shape[-2:]

        if is_depth_map:
            points = points.reshape(points.shape[0], 3, -1)
        b, _, n = points.shape

        points = torch.matmul(self.Pwc(from_world), cat_channel_ones(points, 1))

        coords = points[:, :2] / (points[:, 2].unsqueeze(1) + 1e-7)
        depth = points[:, 2]

        if not is_depth_map:
            if normalize:
                coords = norm_pixel_grid(coords, hw=self._hw, in_place=True)
                invalid = (coords[:, 0] < -1) | (coords[:, 0] > 1) | \
                          (coords[:, 1] < -1) | (coords[:, 1] > 1) | (depth < 0)
                coords[invalid.unsqueeze(1).repeat(1, 2, 1)] = -2
            if return_z:
                return coords.permute(0, 2, 1), depth
            else:
                return coords.permute(0, 2, 1)

        coords = coords.view(b, 2, *hw)
        depth = depth.view(b, 1, *hw)

        if normalize:
            coords = norm_pixel_grid(coords, hw=self._hw, in_place=True)
            invalid = (coords[:, 0] < -1) | (coords[:, 0] > 1) | \
                      (coords[:, 1] < -1) | (coords[:, 1] > 1) | (depth[:, 0] < 0)
            coords[invalid.unsqueeze(1).repeat(1, 2, 1, 1)] = -2

        if return_z:
            return coords.permute(0, 2, 3, 1), depth
        else:
            return coords.permute(0, 2, 3, 1)

    def project_cost_volume(self, points, from_world=True, normalize=True):
        """
        Project points back to image plane

        Parameters
        ----------
        points : torch.Tensor
            Input 3D points [B,3,H,W] or [B,3,N]
        from_world : Bool
            Whether points are in the global frame
        normalize : Bool
            Whether projections should be normalized to [-1,1]

        Returns
        -------
        coords : torch.Tensor
            Projected 2D coordinates [B,2,H,W]
        """
        if points.dim() == 4:
            points = points.permute(0, 2, 1, 3).reshape(points.shape[0], 3, -1)
        b, _, n = points.shape

        points = torch.matmul(self.Pwc(from_world), cat_channel_ones(points, 1))

        coords = points[:, :2] / (points[:, 2].unsqueeze(1) + 1e-7)
        coords = coords.view(b, 2, -1, *self._hw).permute(0, 2, 3, 4, 1)

        if normalize:
            coords[..., 0] /= self._hw[1] - 1
            coords[..., 1] /= self._hw[0] - 1
            return 2 * coords - 1
        else:
            return coords

    def coords_from_cost_volume(self, volume, ref_cam=None):
        """
        Get warp coordinates from a cost volume

        Parameters
        ----------
        volume : torch.Tensor
            Input cost volume [B,1,D,H,W]
        ref_cam : Camera
            Optional to generate cross-camera coordinates

        Returns
        -------
        coords : torch.Tensor
            Projected 2D coordinates [B,2,H,W]
        """
        if ref_cam is None:
            return self.project_cost_volume(self.reconstruct_cost_volume(volume, to_world=False), from_world=True)
        else:
            return ref_cam.project_cost_volume(self.reconstruct_cost_volume(volume, to_world=True), from_world=True)

    def coords_from_depth(self, depth, ref_cam=None):
        """
        Get warp coordinates from a depth map

        Parameters
        ----------
        depth : torch.Tensor
            Input cost volume [B,1,D,H,W]
        ref_cam : Camera
            Optional to generate cross-camera coordinates

        Returns
        -------
        coords : torch.Tensor
            Projected 2D coordinates [B,2,H,W]
        """
        if ref_cam is None:
            return self.project_points(self.reconstruct_depth_map(depth, to_world=False), from_world=True)
        else:
            return ref_cam.project_points(self.reconstruct_depth_map(depth, to_world=True), from_world=True)
