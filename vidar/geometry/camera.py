# Copyright 2023 Toyota Research Institute.  All rights reserved.

from vidar.geometry.cameras.pinhole import CameraPinhole
from vidar.geometry.cameras.ucm import CameraUCM


def Camera(K, hw, Twc=None, Tcw=None, geometry='pinhole'):
    """Create a camera object"""
    if geometry == 'pinhole':
        return CameraPinhole(K, hw, Twc, Tcw)
    elif geometry == 'ucm':
        return CameraUCM(K, hw, Twc, Tcw)
    else:
        raise ValueError('Invalid camera geometry')