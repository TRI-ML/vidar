# Copyright 2023 Toyota Research Institute.  All rights reserved.

from abc import ABC

from knk_vision.vidar.vidar.arch.models.BaseModel import BaseModel
from knk_vision.vidar.vidar.geometry.camera_nerf import CameraNerf
from knk_vision.vidar.vidar.geometry.pose import Pose


class DefineInferenceModel(BaseModel, ABC):
    """DeFiNe model class, focusing on inference. (https://arxiv.org/abs/2207.14287)

    Parameters
    ----------
    cfg : Config
        Configuration file for model generation
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        from knk_vision.vidar.vidar.arch.networks.perceiver.DeFiNeNet import DeFiNeNet

        self.networks["perceiver"] = DeFiNeNet(cfg.model.network)
        self.weights = cfg.model.task_weights
        self.use_pose_noise = cfg.model.use_pose_noise
        self.use_virtual_cameras = cfg.model.use_virtual_cameras
        self.virtual_cameras_eval = cfg.model.virtual_cameras_eval
        self.use_virtual_rgb = cfg.model.use_virtual_rgb
        self.augment_canonical = cfg.model.augment_canonical
        self.scale_loss = cfg.model.scale_loss

        self.encode_train = cfg.model.encode_train
        self.decode_train = cfg.model.decode_train

        self.encode_eval = cfg.model.encode_eval
        self.decode_eval = cfg.model.decode_eval

        if self.encode_eval == "same":
            self.encode_eval = self.encode_train
        if self.decode_eval == "same":
            self.decode_eval = self.decode_train

        self.decode_encodes = cfg.model.decode_encodes
        self.sample_decoded_queries = cfg.model.sample_decoded_queries

    def forward(self, batch):
        rgb = batch["rgb"]
        intrinsics = batch["intrinsics"]
        pose = batch["pose"]

        encode_data = []

        for i in range(len(rgb)):
            cam_i = CameraNerf(
                K=intrinsics[i], Twc=Pose(pose[i]).to(rgb[0].device), hw=rgb[0]
            )
            cam_dict = {"rgb": rgb[i], "cam": cam_i}
            encode_data.append(cam_dict)

        decode_data = encode_data

        # Run PerceiverIO (encode and decode)
        perceiver_output = self.networks["perceiver"](
            encode_data=encode_data,
            decode_data=decode_data,
            sample_queries=self.sample_decoded_queries,
            filter_invalid=False,
        )
        return perceiver_output["output"]["depth"]
