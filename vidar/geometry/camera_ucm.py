# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from functools import lru_cache
import torch
import torch.nn as nn

from vidar.geometry.pose import Pose
from vidar.utils.tensor import pixel_grid


class UCMCamera(nn.Module):
    """
    Differentiable camera class implementing reconstruction and projection
    functions for the unified camera model (UCM).
    """
    def __init__(self, I, Tcw=None):
        """
        Initializes the Camera class

        Parameters
        ----------
        I : torch.Tensor [5]
            Camera intrinsics parameter vector
        Tcw : Pose
            Camera -> World pose transformation
        """
        super().__init__()
        self.I = I
        if Tcw is None:
            self.Tcw = Pose.identity(len(I))
        elif isinstance(Tcw, Pose):
            self.Tcw = Tcw
        else:
            self.Tcw = Pose(Tcw)

        self.Tcw.to(self.I.device)

    def __len__(self):
        """Batch size of the camera intrinsics"""
        return len(self.I)

    def to(self, *args, **kwargs):
        """Moves object to a specific device"""
        self.I = self.I.to(*args, **kwargs)
        self.Tcw = self.Tcw.to(*args, **kwargs)
        return self

    @property
    def fx(self):
        """Focal length in x"""
        return self.I[:, 0].unsqueeze(1).unsqueeze(2)

    @property
    def fy(self):
        """Focal length in y"""
        return self.I[:, 1].unsqueeze(1).unsqueeze(2)

    @property
    def cx(self):
        """Principal point in x"""
        return self.I[:, 2].unsqueeze(1).unsqueeze(2)

    @property
    def cy(self):
        """Principal point in y"""
        return self.I[:, 3].unsqueeze(1).unsqueeze(2)

    @property
    def alpha(self):
        """alpha in UCM model"""
        return self.I[:, 4].unsqueeze(1).unsqueeze(2)

    @property
    @lru_cache()
    def Twc(self):
        """World -> Camera pose transformation (inverse of Tcw)"""
        return self.Tcw.inverse()

    def reconstruct(self, depth, frame='w'):
        """
        Reconstructs pixel-wise 3D points from a depth map.

        Parameters
        ----------
        depth : torch.Tensor [B,1,H,W]
            Depth map for the camera
        frame : 'w'
            Reference frame: 'c' for camera and 'w' for world

        Returns
        -------
        points : torch.tensor [B,3,H,W]
            Pixel-wise 3D points
        """

        if depth is None:
            return None
        b, c, h, w = depth.shape
        assert c == 1

        grid = pixel_grid(depth, with_ones=True, device=depth.device)

        # Estimate the outward rays in the camera frame
        fx, fy, cx, cy, alpha = self.fx, self.fy, self.cx, self.cy, self.alpha # [B,1,1]

        if torch.any(torch.isnan(alpha)):
            raise ValueError('alpha is nan')

        u = grid[:,0,:,:]
        v = grid[:,1,:,:]

        mx = (u - cx) / fx * (1 - alpha)
        my = (v - cy) / fy * (1 - alpha)
        r_square = mx ** 2 + my ** 2
        xi = alpha / (1 - alpha) # [B, 1, 1]
        coeff = (xi + torch.sqrt(1 + (1 - xi ** 2) * r_square)) / (1 + r_square) # [B, H, W]
        
        x = coeff * mx
        y = coeff * my
        z = coeff * 1 - xi
        z = z.clamp(min=1e-7)
        
        x_norm = x / z
        y_norm = y / z
        z_norm = z / z
        xnorm = torch.stack(( x_norm, y_norm, z_norm ), dim=1).float()

        # Scale rays to metric depth
        Xc = xnorm * depth

        # If in camera frame of reference
        if frame == 'c':
            return Xc
        # If in world frame of reference
        elif frame == 'w':
            return (self.Twc * Xc.view(b, 3, -1)).view(b,3,h,w)
        # If none of the above
        else:
            raise ValueError('Unknown reference frame {}'.format(frame))

    def project(self, X, frame='w'):
        """
        Projects 3D points onto the image plane

        Parameters
        ----------
        X : torch.Tensor [B,3,H,W]
            3D points to be projected
        frame : 'w'
            Reference frame: 'c' for camera and 'w' for world

        Returns
        -------
        points : torch.Tensor [B,H,W,2]
            2D projected points that are within the image boundaries
        """
        B, C, H, W = X.shape
        assert C == 3

        # Project 3D points onto the camera image plane
        if frame == 'c':
            X = X
        elif frame == 'w':
            X = (self.Tcw * X.view(B,3,-1)).view(B,3,H,W)
        else:
            raise ValueError('Unknown reference frame {}'.format(frame))
        
        d = torch.norm(X, dim=1)
        fx, fy, cx, cy, alpha = self.fx, self.fy, self.cx, self.cy, self.alpha
        x, y, z = X[:,0,:], X[:,1,:], X[:,2,:]
        z = z.clamp(min=1e-7)
        
        Xnorm = fx * x / (alpha * d + (1 - alpha) * z + 1e-7) + cx
        Ynorm = fy * y / (alpha * d + (1 - alpha) * z + 1e-7) + cy
        Xnorm = 2 * Xnorm / (W-1) - 1
        Ynorm = 2 * Ynorm / (H-1) - 1

        coords = torch.stack([Xnorm, Ynorm], dim=-1).permute(0,3,1,2)
        z = z.unsqueeze(1)

        invalid = (coords[:, 0] < -1) | (coords[:, 0] > 1) | \
                      (coords[:, 1] < -1) | (coords[:, 1] > 1) | (z[:, 0] < 0)
        coords[invalid.unsqueeze(1).repeat(1, 2, 1, 1)] = -2

        # Return pixel coordinates
        return coords.permute(0, 2, 3, 1)

    def reconstruct_depth_map(self, depth, to_world=True):
        if to_world:
            return self.reconstruct(depth, frame='w')
        else:
            return self.reconstruct(depth, frame='c')

    def project_points(self, points, from_world=True, normalize=True, return_z=False):
        if from_world:
            return self.project(points, frame='w')
        else:
            return self.project(points, frame='c')

    def coords_from_depth(self, depth, ref_cam=None):
        if ref_cam is None:
            return self.project_points(self.reconstruct_depth_map(depth, to_world=False), from_world=True)
        else:
            return ref_cam.project_points(self.reconstruct_depth_map(depth, to_world=True), from_world=True)