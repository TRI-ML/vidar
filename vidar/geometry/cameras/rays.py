# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch
from torch_scatter import scatter_min

from vidar.geometry.camera_utils import invert_intrinsics, scale_intrinsics
from vidar.geometry.cameras.base import CameraBase
from vidar.utils.tensor import same_shape, cat_channel_ones, unnorm_pixel_grid
from vidar.utils.types import is_seq


class CameraRays(CameraBase):
    """Rays camera class (no parametric model)"""
    def __init__(self, K, *args, **kwargs):
        # Intrinsics
        self._K = K / K[:, [2]]
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_list(cams):
        """Create a single camera from a list of cameras"""
        K = torch.cat([cam.K for cam in cams], 0)
        Twc = torch.cat([cam.Twc.T for cam in cams], 0)
        return CameraRays(K=K, Twc=Twc, hw=cams[0].hw)

    @staticmethod
    def from_dict(K, hw, Twc=None, broken=False):
        """Create a single camera from a dictionary of cameras"""
        if broken:
            return {key: CameraRays(
                K=K[key] if key in K else K[(0, key[1])], 
                hw=hw[key], Twc=val
            ) for key, val in Twc.items()}
        else:
            return {key: CameraRays(
                K=K[key] if key in K else K[0], 
                hw=hw[key], Twc=val
            ) for key, val in Twc.items()}


    def lift(self, grid=None):
        """Lift a grid of points to 3D"""
        return self._K.reshape(self.batch_size, 3, -1)

