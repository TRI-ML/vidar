# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from abc import ABC
from vidar.arch.models.BaseModel import BaseModel

class SelfSupervisedModelInference(BaseModel, ABC):
    """
    Self-supervised depth estimation model

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        #self.view_synthesis = ViewSynthesis()
        self.set_attr(cfg.model, 'use_gt_pose', False)
        self.set_attr(cfg.model, 'use_gt_intrinsics', True)

    #def forward(self, batch, epoch=0):
    def forward(self, input_rgb, epoch=0):
        """Model forward pass"""

        #rgb = batch['rgb']
        #rgb = 
  
        depth_output = self.networks['depth'](rgb=input_rgb)
        pred_depth = depth_output['depths']

        return pred_depth