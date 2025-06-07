# Copyright 2023 Toyota Research Institute.  All rights reserved.

from packnet_sfm.geometry.pose import Pose
from knk_vision.vidar.vidar.arch.models.BaseModel import BaseModel


class PoseModel(BaseModel):
    """Base pose network, for two-frame ego-motion estimation

    Parameters
    ----------
    BaseModel : Config
        Configuration file
    """
    def __init__(self, cfg):
        super().__init__()
        self.rotation_mode = cfg.rotation_mode
        self._network_requirements = [
            'pose_net',
        ]

    def compute_pose_net(self, image, contexts):
        """Compute pose predictions from a pose network"""
        pose_vec = self.pose_net(image, contexts)
        return [Pose.from_vec(pose_vec[:, i], self.rotation_mode)
                for i in range(pose_vec.shape[1])]

    def forward(self, batch, return_logs=False, force_flip=False):
        """Model forward pass, given a batch dictionary"""
        
        # Generate pose predictions if available
        pose_output = None
        if 'rgb_context' in batch and self.pose_net is not None:
            pose_output = self.compute_pose_net(
                batch['rgb'], batch['rgb_context'])
        # Return output dictionary
        return {
            'poses': pose_output,
        }
