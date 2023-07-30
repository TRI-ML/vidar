# Copyright 2023 Toyota Research Institute.  All rights reserved.

from abc import ABC
from copy import deepcopy

import torch
import torch.nn as nn

from vidar.geometry.camera_utils import invert_intrinsics, scale_intrinsics
from vidar.geometry.pose import Pose
from vidar.geometry.pose_utils import invert_pose
from vidar.utils.tensor import pixel_grid, same_shape, cat_channel_ones, norm_pixel_grid, interpolate, interleave
from vidar.utils.types import is_tensor, is_seq, is_tuple
from einops import rearrange


class CameraBase(nn.Module, ABC):
    """Base camera class

    Parameters
    ----------
    hw : tuple
        Camera resolution
    Twc : torch.Tensor, optional
        Camera pose (world to camera), by default None
    Tcw : torch.Tensor, optional
        Camera pose (camera to world), by default None
    """
    def __init__(self, hw, Twc=None, Tcw=None):
        super().__init__()
        assert Twc is None or Tcw is None

        # Pose

        if Twc is None and Tcw is None:
            self._Twc = torch.eye(
                4, dtype=self._K.dtype, device=self._K.device).unsqueeze(0).repeat(self._K.shape[0], 1, 1)
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
            if not is_tensor(idx):
                idx = [idx]
            return type(self)(
                K=self._K[idx],
                Twc=self._Twc[idx] if self._Twc is not None else None,
                hw=self._hw,
            )

    def __len__(self):
        """Return length as intrinsics batch"""
        return self._K.shape[0]

    def __eq__(self, cam):
        """Check camera equality"""
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
        """Clone camera"""
        return type(self)(
            K=self.K.clone(),
            Twc=self.Twc.clone(),
            hw=[v for v in self._hw],
        )

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
    def batch_size(self):
        """Return batch size"""
        return self._Twc.T.shape[0]

    @property
    def b(self):
        """Return batch size"""
        return self._Twc.T.shape[0]

    @property
    def bhw(self):
        """Return batch size and resolution"""
        return self.b, self.hw

    @property
    def bdhw(self):
        """Return batch size, device, and resolution"""
        return self.b, self.device, self.hw

    @property
    def hw(self):
        """Return camera resolution"""
        return self._hw

    @hw.setter
    def hw(self, hw):
        """Set camera resolution"""
        self._hw = hw

    @property
    def wh(self):
        """Return camera resolution"""
        return self._hw[::-1]

    @property
    def n_pixels(self):
        """Return number of pixels"""
        return self._hw[0] * self._hw[1]

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
        """Get camera data type"""
        return self._K.dtype

    @property
    def device(self):
        """Get camera device"""
        return self._K.device

    def detach_pose(self):
        """Detach camera pose"""
        return type(self)(K=self._K, hw=self._hw,
                          Twc=self._Twc.detach() if self._Twc is not None else None)

    def detach_K(self):
        """Detach camera intrinsics"""
        return type(self)(K=self._K.detach(), hw=self._hw, Twc=self._Twc)

    def detach(self):
        """Detach camera"""
        return type(self)(K=self._K.detach(), hw=self._hw,
                          Twc=self._Twc.detach() if self._Twc is not None else None)

    def inverted_pose(self):
        """Return camera with inverted pose"""
        return type(self)(K=self._K, hw=self._hw,
                          Twc=self._Twc.inverse() if self._Twc is not None else None)

    def no_translation(self):
        """Return camera with no translation"""
        Twc = self.pose.clone()
        Twc[:, :-1, -1] = 0
        return type(self)(K=self._K, hw=self._hw, Twc=Twc)

    def no_pose(self):
        """Return camera with no pose"""
        return type(self)(K=self._K, hw=self._hw)

    def interpolate(self, rgb):
        """Interpolate RGB image"""
        if rgb.dim() == 5:
            rgb = rearrange(rgb, 'b n c h w -> (b n) c h w')
        return interpolate(rgb, scale_factor=None, size=self.hw, mode='bilinear')

    def interleave_K(self, b):
        """Interleave intrinsics for multiple batches"""
        return type(self)(
            K=interleave(self._K, b),
            Twc=self._Twc,
            hw=self._hw,
        )

    def interleave_Twc(self, b):
        """Interleave pose for multiple batches"""
        return type(self)(
            K=self._K,
            Twc=interleave(self._Twc, b),
            hw=self._hw,
        )

    def interleave(self, b):
        """Interleave camera for multiple batches"""
        return type(self)(
            K=interleave(self._K, b),
            Twc=interleave(self._Twc, b),
            hw=self._hw,
        )

    def repeat_bidir(self):
        """Repeat camera for bidirectional training"""
        return type(self)(
            K=self._K.repeat(2, 1, 1),
            Twc=torch.cat([self._Twc.T, self.Tcw.T], 0),
            hw=self.hw,
        )

    def Pwc(self, from_world=True):
        """Return camera projection matrix (world to camera)"""
        return self._K[:, :3] if not from_world or self._Twc is None else \
            torch.matmul(self._K, self._Twc.T)[:, :3]

    def to_world(self, points):
        """Moves pointcloud to world coordinates"""
        if points.dim() > 3:
            points = points.reshape(points.shape[0], 3, -1)
        return points if self.Tcw is None else self.Tcw * points

    def from_world(self, points):
        """Moves pointcloud to camera coordinates"""
        if points.dim() > 3:
            shape = points.shape
            points = points.reshape(points.shape[0], 3, -1)
        else:
            shape = None
        local_points = points if self._Twc is None else \
            torch.matmul(self._Twc.T, cat_channel_ones(points, 1))[:, :3]
        return local_points if shape is None else local_points.view(shape)
    
    def from_world2(self, points):
        """Moves pointcloud to camera coordinates"""
        if points.dim() > 3:
            points = points.reshape(points.shape[0], 3, -1)
        return points if self._Twc is None else \
            torch.matmul(self._Twc.T[:, :3, :3], points[:, :3]) + self._Twc.T[:, :3, 3:]

    def to(self, *args, **kwargs):
        """Moves camera to device"""
        self._K = self._K.to(*args, **kwargs)
        if self._Twc is not None:
            self._Twc = self._Twc.to(*args, **kwargs)
        return self

    def cuda(self, *args, **kwargs):
        """Moves camera to GPU"""
        return self.to('cuda')

    def relative_to(self, cam):
        """Returns camera relative to another camera"""
        return type(self)(K=self._K, hw=self._hw, Twc=self._Twc * cam.Twc.inverse())

    def global_from(self, cam):
        """Returns global camera from one relative to another camera"""
        return type(self)(K=self._K, hw=self._hw, Twc=self._Twc * cam.Twc)

    def pixel_grid(self, shake=False):
        """Returns pixel grid"""
        return pixel_grid(
            b=self.batch_size, hw=self.hw, with_ones=True,
            shake=shake, device=self.device).view(self.batch_size, 3, -1)

    def reconstruct_depth_map(self, depth, to_world=False, grid=None, scene_flow=None, world_scene_flow=None):
        """Reconstruct 3D pointcloud from z-buffer depth map"""
        if depth is None:
            return None
        b, _, h, w = depth.shape
        if grid is None:
            grid = pixel_grid(depth, with_ones=True, device=depth.device).view(b, 3, -1)
        points = self.lift(grid) * depth.view(depth.shape[0], 1, -1)
        if scene_flow is not None:
            points = points + scene_flow.view(b, 3, -1)
        if to_world and self.Tcw is not None:
            points = self.Tcw * points
            if world_scene_flow is not None:
                points = points + world_scene_flow.view(b, 3, -1)
        return points.view(b, 3, h, w)

    def reconstruct_euclidean(self, depth, to_world=False, grid=None, scene_flow=None, world_scene_flow=None):
        """Reconstruct 3D pointcloud from euclidean depth map"""
        if depth is None:
            return None
        b, _, h, w = depth.shape
        rays = self.get_viewdirs(normalize=True, to_world=False, grid=grid).view(b, 3, -1)
        points = rays * depth.view(depth.shape[0], 1, -1)
        if scene_flow is not None:
            points = points + scene_flow.view(b, 3, -1)
        if to_world and self.Tcw is not None:
            points = self.Tcw * points
            if world_scene_flow is not None:
                points = points + world_scene_flow.view(b, 3, -1)
        return points.view(b, 3, h, w)

    def reconstruct_volume(self, depth, euclidean=False, **kwargs):
        """Reconstruct 3D pointcloud from depth volume"""
        if euclidean:
            return torch.stack([self.reconstruct_euclidean(depth[:, :, i], **kwargs)
                                for i in range(depth.shape[2])], 2)
        else:
            return torch.stack([self.reconstruct_depth_map(depth[:, :, i], **kwargs)
                                for i in range(depth.shape[2])], 2)

    def reconstruct_cost_volume(self, volume, to_world=True, flatten=True):
        """Reconstruct 3D pointcloud from cost volume"""
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

    def project_points(self, points, from_world=True, normalize=True,
                       return_z=False, return_e=False, flag_invalid=True):
        """Projects 3D points to 2D image plane"""

        is_depth_map = points.dim() == 4
        hw = self._hw if not is_depth_map else points.shape[-2:]
        return_depth = return_z or return_e

        if is_depth_map:
            points = points.reshape(points.shape[0], 3, -1)
        b, _, n = points.shape

        coords, depth = self.unlift(points, from_world=from_world, euclidean=return_e)

        if not is_depth_map:
            if normalize:
                coords = norm_pixel_grid(coords, hw=self._hw, in_place=True)
                if flag_invalid:
                    invalid = (coords[:, 0] < -1) | (coords[:, 0] > 1) | \
                              (coords[:, 1] < -1) | (coords[:, 1] > 1) | (depth < 0)
                    coords[invalid.unsqueeze(1).repeat(1, 2, 1)] = -2
            if return_depth:
                return coords.permute(0, 2, 1), depth
            else:
                return coords.permute(0, 2, 1)

        coords = coords.view(b, 2, *hw)
        depth = depth.view(b, 1, *hw)

        if normalize:
            coords = norm_pixel_grid(coords, hw=self._hw, in_place=True)
            if flag_invalid:
                invalid = (coords[:, 0] < -1) | (coords[:, 0] > 1) | \
                          (coords[:, 1] < -1) | (coords[:, 1] > 1) | (depth[:, 0] < 0)
                coords[invalid.unsqueeze(1).repeat(1, 2, 1, 1)] = -2

        if return_depth:
            return coords.permute(0, 2, 3, 1), depth
        else:
            return coords.permute(0, 2, 3, 1)

    def project_cost_volume(self, points, from_world=True, normalize=True):
        """Projects 3D points from a cost volume to 2D image plane"""
    
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

    def create_radial_volume(self, bins, to_world=True):
        """Create a volume of radial depth bins"""
        ones = torch.ones((1, *self.hw), device=self.device)
        volume = torch.stack([depth * ones for depth in bins], 1).unsqueeze(0)
        return self.reconstruct_volume(volume, to_world=to_world)

    def project_volume(self, volume, from_world=True):
        """Project a volume to 2D image plane"""
        b, c, d, h, w = volume.shape
        return self.project_points(volume.view(b, c, -1), from_world=from_world).view(b, d, h, w, 2)

    def coords_from_cost_volume(self, volume, ref_cam=None):
        """Project a cost volume to 2D image plane"""
        if ref_cam is None:
            return self.project_cost_volume(self.reconstruct_cost_volume(volume, to_world=False), from_world=True)
        else:
            return ref_cam.project_cost_volume(self.reconstruct_cost_volume(volume, to_world=True), from_world=True)

    def z2e(self, z_depth):
        """Convert z-buffer depth to euclidean depth"""
        points = self.reconstruct_depth_map(z_depth, to_world=False)
        return self.project_points(points, from_world=False, return_e=True)[1]

    def e2z(self, e_depth):
        """Convert euclidean depth to z-buffer depth"""
        points = self.reconstruct_euclidean(e_depth, to_world=False)
        return self.project_points(points, from_world=False, return_z=True)[1]

    def control(self, draw, tvel=0.2, rvel=0.1):
        """Control camera with keyboard (requires camviz)"""
        change = False
        if draw.UP:
            self.Twc.translateForward(tvel)
            change = True
        if draw.DOWN:
            self.Twc.translateBackward(tvel)
            change = True
        if draw.LEFT:
            self.Twc.translateLeft(tvel)
            change = True
        if draw.RIGHT:
            self.Twc.translateRight(tvel)
            change = True
        if draw.KEY_Z:
            self.Twc.translateUp(tvel)
            change = True
        if draw.KEY_X:
            self.Twc.translateDown(tvel)
            change = True
        if draw.KEY_A:
            self.Twc.rotateYaw(-rvel)
            change = True
        if draw.KEY_D:
            self.Twc.rotateYaw(+rvel)
            change = True
        if draw.KEY_W:
            self.Twc.rotatePitch(+rvel)
            change = True
        if draw.KEY_S:
            self.Twc.rotatePitch(-rvel)
            change = True
        if draw.KEY_Q:
            self.Twc.rotateRoll(-rvel)
            change = True
        if draw.KEY_E:
            self.Twc.rotateRoll(+rvel)
            change = True
        return change
