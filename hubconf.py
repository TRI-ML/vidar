# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.


dependencies = ["torch"]

import torch

from vidar.core.evaluator import Evaluator
from vidar.utils.config import read_config
from vidar.utils.setup import setup_arch
from vidar.utils.config import cfg_has
from imageio import imread, imsave
import torch.nn.functional as F
import torch
import urllib


def PackNet(pretrained=True, **kwargs):
    """# This docstring shows up in hub.help()
    PackNet model for monocular depth estimation
    pretrained (bool): load pretrained weights into model
    """

    cfg_url = "https://raw.githubusercontent.com/TRI-ML/vidar/main/configs/papers/packnet/hub_packnet.yaml"
    cfg = urllib.request.urlretrieve(cfg_url, "packnet_config.yaml")
    cfg = read_config("packnet_config.yaml")
    model = Evaluator(cfg)

    if pretrained:
        url = "https://tri-ml-public.s3.amazonaws.com/github/vidar/models/PackNet_MR_selfsup_KITTI.ckpt"
        model = setup_arch(cfg.arch, verbose=True)
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state_dict["state_dict"], strict=False)

    return model
