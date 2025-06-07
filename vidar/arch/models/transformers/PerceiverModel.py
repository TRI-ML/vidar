# Copyright 2023 Toyota Research Institute.  All rights reserved.

from abc import ABC

import torch
import random

from knk_vision.vidar.vidar.arch.losses.PhotometricLoss import PhotometricLoss
from knk_vision.vidar.vidar.arch.models.BaseModel import BaseModel
from knk_vision.vidar.vidar.geometry.camera import Camera
from knk_vision.vidar.vidar.geometry.pose import Pose
from knk_vision.vidar.vidar.utils.config import Config
from knk_vision.vidar.vidar.utils.data import fold_batch
from collections import OrderedDict


def get_all_index(data):
    """Get all possible index combinations"""
    output = []
    for i in data.keys():
        for j in range(len(data[i])):
            output.append([i, j])
    return output


def get_index(data, idx_list):
    """Get index combinations from a list"""
    output = []
    n_train, n_test = 0, 0
    for i in data.keys():
        for j in range(len(data[i])):
            test = False
            for idx in idx_list:
                if i == idx[0] and j == idx[1]:
                    output.append([i, j, n_test, 'test'])
                    n_test += 1
                    test = True
                    break
            if not test:
                output.append([i, j, n_train, 'train'])
                n_train += 1
    return output


def get_data(data, indices):
    """Get data from index combinations"""
    train, test = [], []
    for idx in indices:
        if idx[3] == 'train':
            train.append(data[idx[0]][[idx[1]]])
        elif idx[3] == 'test':
            test.append(data[idx[0]][[idx[1]]])
    return torch.cat(train, 0), torch.cat(test, 0)


def restore_data(train, test, indices):
    """Restore data from index combinations"""
    output = OrderedDict()
    for idx in indices:
        if idx[0] not in output.keys():
            output[idx[0]] = []
        if idx[-1] == 'train':
            output[idx[0]].append(train[[idx[2]]])
        elif idx[-1] == 'test':
            output[idx[0]].append(test[[idx[2]]])
    for key in output.keys():
        output[key] = [torch.cat(output[key], 0)]
    return output


class PerceiverModel(BaseModel, ABC):
    """Perceiver model class, as the basis for related Perceiver IO models.

    Parameters
    ----------
    cfg : Config
        Configuration file for model generation
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        # Photometric loss for image warping
        self.photo_loss = PhotometricLoss(Config(alpha=0.85))

    def forward(self, batch, epoch=0):
        """Model forward pass, given a batch dictionary"""

        batch = fold_batch(batch)

        rgb = batch['rgb']
        pose = batch['pose']

        intrinsics = batch['intrinsics']
        for key in rgb.keys():
            intrinsics[key] = intrinsics[0]

        # Convert pose to global coordinates
        pose = Pose.from_dict(pose, to_global=True, zero_origin=True)
        pose = {key: val.T for key, val in pose.items()}

        n_context = 6

        all_idx = get_all_index(rgb)
        rand = torch.randperm(len(all_idx))[n_context:]
        # test_index = [[0, 0]]
        test_index = [all_idx[i] for i in rand]

        idx = get_index(rgb, test_index)

        # Get training and testing data
        rgb_train, rgb_test = get_data(rgb, idx)
        pose_train, pose_test = get_data(pose, idx)
        intrinsics_train, intrinsics_test = get_data(intrinsics, idx)

        # Create camera objects
        cam_train = Camera(K=intrinsics_train, hw=rgb_train, Twc=pose_train)
        cam_test = Camera(K=intrinsics_test, hw=rgb_test, Twc=pose_test)

        # Encode and decode
        latent = self.networks['nerf'].encode(rgb_train, cam_train)
        output_train = self.networks['nerf'].decode_rgb(cam_train, latent)
        output_test = self.networks['nerf'].decode_rgb(cam_test, latent)

        predictions_rgb = restore_data(output_train['rgb'], output_test['rgb'], idx)

        predictions = {
            'rgb_perceiver': predictions_rgb,
        }

        # Return predictions if not training
        if not self.training:
            return {
                'predictions': predictions,
            }

        # Calculate losses
        loss_image = self.photo_loss(rgb_test, output_test['rgb'])['loss'].abs().mean()
        loss = loss_image

        return {
            'loss': loss,
            'metrics': {},
            'predictions': predictions,
        }

