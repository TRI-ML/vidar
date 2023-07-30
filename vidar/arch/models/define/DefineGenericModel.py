# Copyright 2023 Toyota Research Institute.  All rights reserved.

import random
from abc import ABC

import numpy as np
import torch

from vidar.arch.models.BaseModel import BaseModel
from vidar.arch.models.define.DefineModel import augment_canonical, create_virtual_cameras
from vidar.geometry.camera import Camera
from vidar.geometry.pose import Pose
from vidar.utils.data import flatten
from vidar.utils.types import is_list
from vidar.arch.networks.perceiver.DeFiNeNet import DeFiNeNet


def sample_pred_gt(output, gt, pred):
    """Sample predicted and ground truth values given the query indices"""
    for i in range(len(gt)):
        b, n, h, w = gt[i].shape
        query_idx = output['query_idx'][i]
        gt[i] = torch.stack([gt[i][j].view(n, -1)[:, query_idx[j]]
                            for j in range(b)], 0)
        pred[i][0] = pred[i][0].permute(0, 2, 1)
    return gt, pred


def parse_idx(all_idx, idxs):
    """Parse indices given a list of all indices and a list of indices to parse"""
    new_idxs = []
    for idx in idxs:
        if is_list(idx[0]) and is_list(idx[1]):
            for i in idx[0]:
                for j in idx[1]:
                    new_idxs.append([i, j])
        elif is_list(idx[0]):
            for i in idx[0]:
                new_idxs.append([i, idx[1]])
        elif is_list(idx[1]):
            for i in idx[1]:
                new_idxs.append([idx[0], i])
        elif idx[0] == '*':
            for i in all_idx:
                if i[1] == idx[1]:
                    new_idxs.append(i)
        elif idx[1] == '*':
            for i in all_idx:
                if i[0] == idx[0]:
                    new_idxs.append(i)
        else:
            new_idxs.append(idx)
    return new_idxs


class DefineGenericModel(BaseModel, ABC):
    """DeFiNe model class. (https://arxiv.org/abs/2207.14287)

    Parameters
    ----------
    cfg : Config
        Configuration file for model generation
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        self.networks['perceiver'] = DeFiNeNet(cfg.model.network)
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

        if self.encode_eval == 'same':
            self.encode_eval = self.encode_train
        if self.decode_eval == 'same':
            self.decode_eval = self.decode_train

        self.decode_encodes = cfg.model.decode_encodes
        self.sample_decoded_queries = cfg.model.sample_decoded_queries

    def get_idx(self, all_idx):
        """Get encode and decode indices given all indices"""

        n = len(all_idx)

        num_encodes = self.encode_train if self.training else self.encode_eval
        num_decodes = self.decode_train if self.training else self.decode_eval

        encode_idx = None
        if is_list(num_encodes):
            if num_encodes[0].startswith('+'):
                encode_idx = parse_idx(all_idx, num_encodes[1:])
            elif num_encodes[0].startswith('-'):
                num_encodes_parse = parse_idx(all_idx, num_encodes[1:])
                encode_idx = [
                    idx for idx in all_idx if idx not in num_encodes_parse]
            if len(num_encodes[0]) > 1:
                num = int(num_encodes[0][1:])
                encode_idx = np.random.permutation(encode_idx)
                encode_idx = encode_idx[:num]
        elif num_encodes == 'all':
            num_encodes = n
        elif num_encodes < 0:
            num_encodes = n + num_encodes

        decode_idx = None
        if is_list(num_decodes):
            if num_decodes[0].startswith('+'):
                decode_idx = parse_idx(all_idx, num_decodes[1:])
            elif num_decodes[0].startswith('-'):
                num_decodes_parse = parse_idx(all_idx, num_decodes[1:])
                decode_idx = [
                    idx for idx in all_idx if idx not in num_decodes_parse]
            if len(num_decodes[0]) > 1:
                num = int(num_decodes[0][1:])
                decode_idx = np.random.permutation(decode_idx)
                decode_idx = decode_idx[:num]
        elif num_decodes == 'all':
            num_decodes = n
        elif num_decodes == 'remaining':
            num_decodes = n - num_encodes
        elif num_decodes < 0:
            num_encodes = n + num_decodes

        if self.training:
            # Shuffle indices and separate encode and decode indices
            all_idx = np.random.permutation(all_idx)

        if encode_idx is None:
            encode_idx = all_idx[:num_encodes]
        if decode_idx is None:
            decode_idx = all_idx[-num_decodes:]

        encode_idx = [list(idx) for idx in encode_idx]
        decode_idx = [list(idx) for idx in decode_idx]

        if self.decode_encodes:
            decode_idx += [idx for idx in encode_idx if idx not in decode_idx]

        return encode_idx, decode_idx

    def parse_output(self, output, key, encode_idx, predictions):
        """Parse output given a key and the encode indices"""
        if key in output.keys():
            pred = [[val] for val in output[key]]
            if not self.training:
                predictions_key = {}
                for idx, (i, j) in enumerate(encode_idx):
                    if j not in predictions_key.keys():
                        predictions_key[j] = []
                    predictions_key[j].append(output[key][idx])
                predictions_key = {key: [torch.stack(val, 1)]
                                   for key, val in predictions_key.items()}
                predictions[key] = predictions_key
        else:
            pred = None
        return pred

    def forward(self, batch, epoch=0, collate=True):
        """Forward pass of the model, given a batch dictionary"""

        # Run on list of b

        if is_list(batch):
            output = [self.forward(b, collate=False) for b in batch]
            loss = [out['loss'] for out in output]
            return {'loss': sum(loss) / len(loss), 'predictions': {}, 'metrics': {}}

        if not collate:
            for key in ['rgb', 'intrinsics', 'pose', 'depth']:
                batch[key] = {k: v.unsqueeze(0) for k, v in batch[key].items()}

        # Unsqueeze batch data if there is only one camera

        key_dim = {'rgb': 4, 'depth': 4, 'intrinsics': 3, 'pose': 3}
        for key in ['rgb', 'depth', 'intrinsics', 'pose']:
            for ctx in batch[key].keys():
                if batch[key][ctx].dim() == key_dim[key]:
                    batch[key][ctx] = batch[key][ctx].unsqueeze(1)
        for key in ['intrinsics', 'pose']:
            for ctx in batch[key].keys():
                if batch[key][ctx].dim() == 3:
                    batch[key][ctx] = batch[key][ctx].unsqueeze(1)

        ###

        rgb = batch['rgb']
        intrinsics = batch['intrinsics']
        pose = batch['pose']
        depth = batch['depth']

        # Get context keys, batch size and number of cameras
        ctx = [key for key in rgb.keys() if key != 'virtual']
        b, n = rgb[0].shape[:2]

        # Create all indices in a list
        ii, jj = list(range(n)), ctx
        all_idx = flatten([[[i, j] for i in ii] for j in jj])
        encode_idx, decode_idx = self.get_idx(all_idx)

        # Prepare pose and add jittering if requested

        pose = [{j: pose[j][i] for j in ctx} for i in range(b)]

        pose0 = [Pose().to(rgb[0].device) for _ in range(b)]
        if self.training and len(self.use_pose_noise) > 0.0:
            if random.random() < self.use_pose_noise[0]:
                for i in range(b):
                    pose0[i].translateUp(
                        self.use_pose_noise[1] * (2 * random.random() - 1))
                    pose0[i].translateLeft(
                        self.use_pose_noise[1] * (2 * random.random() - 1))
                    pose0[i].translateForward(
                        self.use_pose_noise[1] * (2 * random.random() - 1))
                    pose0[i].rotateRoll(
                        torch.pi * self.use_pose_noise[2] * (2 * random.random() - 1))
                    pose0[i].rotatePitch(
                        torch.pi * self.use_pose_noise[2] * (2 * random.random() - 1))
                    pose0[i].rotateYaw(
                        torch.pi * self.use_pose_noise[2] * (2 * random.random() - 1))
        for i in range(b):
            pose[i][0][[0]] = pose0[i].T

        pose = [Pose.from_dict(
            p, to_global=True, zero_origin=False, to_matrix=True) for p in pose]
        pose = {j: torch.stack([pose[i][j] for i in range(b)], 0) for j in ctx}

        # Augment canonical pose if requested
        if self.training and self.augment_canonical:
            pose = augment_canonical(pose)

        # Separate batch data per camera
        rgb = [{j: rgb[j][:, i] for j in ctx} for i in range(n)]
        intrinsics = [{j: intrinsics[0][:, i] for j in ctx} for i in range(n)]
        pose = [{j: pose[j][:, i] for j in ctx} for i in range(n)]
        depth = [{j: depth[j][:, i] for j in ctx} for i in range(n)]

        # Create camera with batch information
        cams = [{j: Camera(K=intrinsics[i][0], Twc=pose[i][j], hw=rgb[i][j])
                 for j in ctx} for i in range(n)]

        # Create encode dictionary
        encode_data = [{
            'rgb': rgb[i][j],
            'cam': cams[i][j],
            'gt_depth': depth[i][j],
        } for i, j in encode_idx]

        # Create decode dictionary
        decode_data = [{
            'rgb': rgb[i][j],
            'cam': cams[i][j],
            'gt_depth': depth[i][j],
        } for i, j in decode_idx]

        # Run PerceiverIO (encode and decode)
        perceiver_output = self.networks['perceiver'](
            encode_data=encode_data,
            decode_data=decode_data,
            sample_queries=self.sample_decoded_queries,
            filter_invalid=False,
        )
        output = perceiver_output['output']

        predictions = {}

        # DEPTH

        # Get predicted depths
        pred_depths = self.parse_output(
            output, 'depth', decode_idx, predictions)
        # Get predicted monocular depths
        pred_depths_mono = self.parse_output(
            output, 'depth_mono', decode_idx, predictions)

        # RGB

        # Get predicted RGB
        pred_rgbs = self.parse_output(output, 'rgb', decode_idx, predictions)

        # VIRTUAL

        if len(self.use_virtual_cameras) > 0 and (self.training or self.virtual_cameras_eval):

            virtual_data = create_virtual_cameras(
                decode_data,
                n_samples=self.use_virtual_cameras[1],
                cam_noise=self.use_virtual_cameras[2:-1],
                center_noise=self.use_virtual_cameras[-1],
                thr=0.1, tries=10, decay=0.9,
            )
            virtual_output = self.networks['perceiver'].decode(
                latent=perceiver_output['latent'], data=virtual_data,
                sources=['cam'], field='cam',
                sample_queries=self.sample_decoded_queries,
                filter_invalid=True
            )['output']

            gt_virtual_cams = [data['cam'] for data in virtual_data]

            if pred_depths is not None:
                pred_depths_virtual = [[depth]
                                       for depth in virtual_output['depth']]
                gt_depths_virtual = [data['gt_depth'] for data in virtual_data]

                if not self.training:
                    batch['depth']['virtual'] = torch.stack(
                        gt_depths_virtual, 1)
                    predictions['depth']['virtual'] = [torch.stack(
                        [pred[0] for pred in pred_depths_virtual], 1)]

                if 'query_idx' in virtual_output:
                    gt_depths_virtual, pred_depths_virtual = sample_pred_gt(
                        virtual_output, gt_depths_virtual, pred_depths_virtual)
            else:
                pred_depths_virtual, gt_depths_virtual = None, None

            if pred_rgbs is not None:
                pred_rgbs_virtual = [[rgb] for rgb in virtual_output['rgb']]
                gt_rgbs_virtual = [data['rgb'] for data in virtual_data]

                if not self.training:
                    batch['rgb']['virtual'] = torch.stack(gt_rgbs_virtual, 1)
                    predictions['rgb']['virtual'] = [torch.stack(
                        [pred[0] for pred in pred_rgbs_virtual], 1)]

                if 'query_idx' in virtual_output:
                    gt_rgbs_virtual, pred_rgbs_virtual = sample_pred_gt(
                        virtual_output, gt_rgbs_virtual, pred_rgbs_virtual)
            else:
                pred_rgbs_virtual, gt_rgbs_virtual = None, None

        else:

            virtual_data = virtual_output = None

        if not self.training:
            return {
                'predictions': predictions,
                'batch': batch,
            }

        # Get GT images and depths
        gt_depths = [depth[i][j] for i, j in decode_idx]
        gt_rgbs = [rgb[i][j] for i, j in decode_idx]

        if 'query_idx' in output:
            if pred_depths is not None:
                gt_depths, pred_depths = sample_pred_gt(
                    output, gt_depths, pred_depths)
            if pred_rgbs is not None:
                gt_rgbs, pred_rgbs = sample_pred_gt(output, gt_rgbs, pred_rgbs)

        loss, metrics = self.compute_loss_and_metrics(
            pred_rgbs, gt_rgbs,
            pred_depths, gt_depths,
        )

        if len(self.use_virtual_cameras) > 0:
            virtual_loss, _ = self.compute_loss_and_metrics(
                pred_rgbs_virtual if self.use_virtual_rgb else None,
                gt_rgbs_virtual if self.use_virtual_rgb else None,
                pred_depths_virtual, gt_depths_virtual,
            )
            loss = loss + self.use_virtual_cameras[0] * virtual_loss

        if pred_depths_mono is not None:
            mono_loss, _ = self.compute_loss_and_metrics(
                None, None, pred_depths_mono, gt_depths,
            )
            loss = loss + mono_loss

        return {
            'loss': loss,
            'metrics': metrics,
            'predictions': predictions,
        }

    def compute_loss_and_metrics(self, pred_rgbs, gt_rgbs, pred_depths, gt_depths):
        """Compute loss and metrics given predictions and ground truth"""

        loss, metrics = [], {}

        # Calculate RGB losses
        if pred_rgbs is not None and 'rgb' in self.losses and self.weights[0] > 0.0:
            loss_rgb = []
            for pred, gt in zip(pred_rgbs, gt_rgbs):
                rgb_output = self.losses['rgb'](pred, gt)
                loss_rgb.append(self.weights[0] * rgb_output['loss'])
            loss.append(sum(loss_rgb) / len(loss_rgb))

        # Calculate depth losses
        if pred_depths is not None and 'depth' in self.losses and self.weights[1] > 0.0:
            loss_depth = []
            for pred, gt in zip(pred_depths, gt_depths):
                depth_output = self.losses['depth'](pred, gt)
                loss_depth.append(self.weights[1] * depth_output['loss'])
            loss.append(sum(loss_depth) / len(loss_depth))

        if len(loss) == 2 and self.scale_loss:
            ratio_rgb_depth = loss[1].item() / loss[0].item()
            loss[0] = loss[0] * ratio_rgb_depth

        loss = sum(loss) / len(loss)

        return loss, metrics
