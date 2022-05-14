# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from vidar.arch.models.BaseModel import BaseModel


class DepthModel(BaseModel):
    """
    Base depth estimation model

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__()

    def forward(self, batch, **kwargs):
        """Model forward pass"""

        depth_output = self.networks['depth'](batch['rgb'][0])
        return {
            'loss': 0.0,
            'metrics': {},
            'predictions': {
                'depth': {0: depth_output['depths']}
            }
        }
