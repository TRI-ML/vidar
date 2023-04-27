# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.


dependencies = ["torch"]

import urllib
import torch
import torch.nn.functional as F

from vidar.core.evaluator import Evaluator
from vidar.utils.config import read_config
from vidar.utils.setup import setup_arch


def DeFiNe(pretrained=True, **kwargs):
    """
    DeFiNe model for monocular depth estimation
    pretrained (bool): load pretrained weights into model
    Usage:
        batch = {}
        batch["rgb"] = # a list of images as 13HW torch.tensors
        batch["intrinsics"] = # a list of 133 torch.tensor intrinsics matrices (one for each image)
        batch["pose"] = # a batch of 144 relative poses to reference frame (one will be identity)
        depth_preds = define_model(batch) # list of depths, one for each image
    """

    cfg_url = "https://raw.githubusercontent.com/IgorVasiljevic-TRI/vidar/main/configs/papers/define/hub_define_temporal.yaml"
    cfg = urllib.request.urlretrieve(cfg_url, "define_config.yaml")
    cfg = read_config("define_config.yaml")
    model = Evaluator(cfg)

    if pretrained:
        url = "https://tri-ml-public.s3.amazonaws.com/github/vidar/models/define_temporal.ckpt"
        model = setup_arch(cfg.arch, verbose=True)
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")

        prefix = "module."
        n_clip = len(prefix)
        adapted_dict = {
            k[n_clip:]: v
            for k, v in state_dict["state_dict"].items()
            if k.startswith(prefix)
        }
        model.load_state_dict(adapted_dict, strict=False)

    return model


def PackNet(pretrained=True, **kwargs):
    """
    PackNet model for monocular depth estimation
    pretrained (bool): load pretrained weights into model

    Usage:
        model = torch.hub.load("TRI-ML/vidar", "PackNet", pretrained=True)
        rgb_image = torch.rand(1, 3, H, W)
        depth_pred = model(rgb_image)
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
