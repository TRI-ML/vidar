# Copyright 2023 Toyota Research Institute.  All rights reserved.

from knk_vision.vidar.vidar.arch.models.BaseModel import BaseModel


class DepthModel(BaseModel):
    """Basic depth estimation model, for inference only

    Parameters
    ----------
    cfg : Config
        Copnfiguration file for model generation
    """

    required_networks = ('depth',)

    def __init__(self, cfg):
        super().__init__()

    def forward(self, batch, **kwargs):
        depth_output = self.networks['depth'](batch['rgb'][0], batch['intrinsics'][0])
        return {
            'loss': 0.0,
            'metrics': {},
            'predictions': {
                'depth': {0: depth_output['depths']}
            }
        }
