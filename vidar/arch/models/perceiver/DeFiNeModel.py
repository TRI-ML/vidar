# Copyright 2023 Toyota Research Institute.  All rights reserved.

import random
from abc import ABC
from functools import partial

import numpy as np
import torch

from knk_vision.vidar.vidar.arch.blocks.image.ViewSynthesis import ViewSynthesis
from knk_vision.vidar.vidar.arch.losses.PhotometricLoss import PhotometricLoss
from knk_vision.vidar.vidar.arch.models.BaseModel import BaseModel
from knk_vision.vidar.vidar.arch.models.utils import make_rgb_scales
from knk_vision.vidar.vidar.geometry.camera import Camera
from knk_vision.vidar.vidar.geometry.pose import Pose
from knk_vision.vidar.vidar.geometry.pose_utils import invert_pose
from knk_vision.vidar.vidar.utils.config import Config
from knk_vision.vidar.vidar.utils.data import expand_batch, break_key
from knk_vision.vidar.vidar.utils.depth import calculate_normals, to_depth
from knk_vision.vidar.vidar.utils.depth import index2depth
from knk_vision.vidar.vidar.utils.tensor import grid_sample, interpolate
from knk_vision.vidar.vidar.utils.types import is_list, is_int, is_dict
from knk_vision.vidar.vidar.utils.viz import viz_photo, viz_depth
from knk_vision.vidar.vidar.utils.volume import compute_depth_bins
from knk_vision.vidar.vidar.utils.write import write_image
from knk_vision.vidar.vidar.utils.networks import freeze_layers_and_norms


def augment_canonical(pose):
    """Augment canonical pose with random rotation and translation"""

    ctx = list(pose.keys())
    num = list(range(pose[0].shape[1]))

    i = random.choice(ctx)
    j = random.choice(num)

    base = Pose(pose[i][:, j]).inverse().T
    for key in ctx:
        for n in num:
            pose[key][:, n] = pose[key][:, n] @ base

    return pose


def create_virtual_cameras(encode_data, n_samples=1, cam_noise=None, center_noise=None,
                           downsample=1.0, thr=0.1, tries=10, decay=0.9, prediction=None, cams=None):
    """Create virtual cameras from the given data"""

    gt_cams_raw = [val['cam_raw'] for val in encode_data.values()]
    gt_cams_mod = [val['cam_mod'] for val in encode_data.values()]
    gt_depths = [val['depth' if prediction is None else prediction] for val in encode_data.values()]
    gt_rgbs = [val['rgb'] for val in encode_data.values()]

    if not is_int(thr):
        n = gt_rgbs[0].shape[-2] * gt_rgbs[0].shape[-1]
        thr = int(n * thr)

    if gt_depths[0] is None:

        virtual_data = {}
        for key, gt_cam_raw, gt_cam_mod in zip(encode_data.keys(), gt_cams_raw, gt_cams_mod):
            for i in range(n_samples):

                cam_raw = gt_cam_raw.clone()
                cam_mod = gt_cam_mod.clone()

                up_raw = cam_raw.clone().Twc.translateUp(1).inverse().T[:, :3, -1] - cam_raw.clone().Tcw.T[:, :3, -1]
                up_mod = cam_mod.clone().Twc.translateUp(1).inverse().T[:, :3, -1] - cam_mod.clone().Tcw.T[:, :3, -1]

                center_raw = cam_raw.clone().Twc.translateForward(1).inverse().T[:, :3, -1]
                center_mod = cam_mod.clone().Twc.translateForward(1).inverse().T[:, :3, -1]

                if cam_noise is not None:

                    up = cam_noise[0] * (2 * random.random() - 1)
                    left = cam_noise[1] * (2 * random.random() - 1)
                    forward = cam_noise[2] * (2 * random.random() - 1)

                    cam_raw.Twc.translateUp(up)
                    cam_mod.Twc.translateUp(up)
                    cam_raw.Twc.translateLeft(left)
                    cam_mod.Twc.translateLeft(left)
                    cam_raw.Twc.translateForward(forward)
                    cam_mod.Twc.translateForward(forward)

                    cam_raw.look_at(center_raw, up=up_raw)
                    cam_mod.look_at(center_mod, up=up_mod)

                virtual_data[(key, i)] = {
                    'cam_raw': cam_raw,
                    'cam_mod': cam_mod,
                }

        return virtual_data

    # Make projected pointclouds and colors 
    pcls_proj = [cam.scaled(downsample).reconstruct_depth_map(depth, to_world=True)
                for cam, depth in zip(gt_cams, gt_depths)]
    pcls_proj = [pcl.reshape(*pcl.shape[:2], -1) for pcl in pcls_proj]
    pcl_proj = torch.cat([pcl for pcl in pcls_proj], -1)
    clr_proj = torch.cat([rgb.reshape(*rgb.shape[:2], -1) for rgb in gt_rgbs], -1)

    if cams is not None:

        virtual_data = []
        for cam in cams:
            rgb_proj, depth_proj = cam.project_pointcloud(pcl_proj, clr_proj)
            virtual_data.append({
                'cam': cam,
                'rgb': rgb_proj.contiguous(),
                'depth': depth_proj.contiguous(),
                'camera': cam,
            })

        return virtual_data

    # Get pointcloud centers
    gt_pcl_centers = [pcl.mean(-1) for pcl in pcls_proj]

    virtual_data = {}
    for key, gt_rgb, gt_depth, gt_cam, gt_pcl_center in \
            zip(encode_data.keys(), gt_rgbs, gt_depths, gt_cams, gt_pcl_centers):
        for i in range(n_samples):

            cam = gt_cam.clone()
            pcl_center = gt_pcl_center.clone()

            weight = 1.0
            rgb_proj = depth_proj = None

            # Try multiple times in case projections are empty
            for j in range(tries):

                if center_noise is not None:
                    pcl_center_noise = weight * center_noise * (2 * torch.rand_like(gt_pcl_center) - 1)
                    pcl_center = gt_pcl_center + pcl_center_noise

                if cam_noise is not None:
                    cam.look_at(pcl_center)
                    cam.Twc.translateUp(weight * cam_noise[0] * (2 * random.random() - 1))
                    cam.Twc.translateLeft(weight * cam_noise[1] * (2 * random.random() - 1))
                    cam.Twc.translateForward(weight * cam_noise[2] * (2 * random.random() - 1))
                    cam.look_at(pcl_center)

                rgb_proj_try, depth_proj_try = cam.project_pointcloud(pcl_proj, clr_proj)

                valid = (depth_proj_try > 0).sum() > thr

                if valid:
                    rgb_proj, depth_proj = rgb_proj_try, depth_proj_try
                    break
                else:
                    weight = weight * decay

            # If all tries failed, use the ground truth
            if rgb_proj is None and depth_proj is None:
                rgb_proj, depth_proj = gt_rgb, gt_depth
                cam = gt_cam.clone()

            virtual_data[(key, i)] = {
                'cam': cam,
                'rgb': rgb_proj.contiguous(),
                'depth': depth_proj.contiguous(),
                'camera': cam,
            }

    return virtual_data


def remove_none(data):
    """Remove None values from a dictionary"""
    return {key: val for key, val in data.items() if val is not None}


def parse_idx(all_idx, idxs):
    """Parse indices from a list of all indices"""
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
            new_idxs.append(tuple(idx))
    return new_idxs


class DeFiNeModel(BaseModel, ABC):
    """DeFiNe model class. (https://arxiv.org/abs/2207.14287)

    Parameters
    ----------
    cfg : Config
        Configuration file for model generation
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        # Loss functions

        self.view_synthesis = ViewSynthesis(Config(upsample_depth=True))

        # Augmentations

        self.use_pose_noise = cfg.model.augmentations.use_pose_noise
        self.use_virtual_cameras = cfg.model.augmentations.use_virtual_cameras
        self.virtual_cameras_eval = cfg.model.augmentations.virtual_cameras_eval
        self.use_virtual_rgb = cfg.model.augmentations.use_virtual_rgb
        self.augment_canonical = cfg.model.augmentations.augment_canonical
        self.evaluate_projection = cfg.model.augmentations.evaluate_projection
        self.zero_origin = cfg.model.augmentations.zero_origin

        # Weights

        self.scale_loss = cfg.model.weights.scale_loss

        # Samples

        self.encode_train = cfg.model.samples.encode_train
        self.decode_train = cfg.model.samples.decode_train
        self.encode_eval = cfg.model.samples.encode_eval
        self.decode_eval = cfg.model.samples.decode_eval
        if self.encode_eval == 'same':
            self.encode_eval = self.encode_train
        if self.decode_eval == 'same':
            self.decode_eval = self.decode_train
        self.decode_encodes = cfg.model.samples.decode_encodes

        self.sample_encoder = cfg.model.samples.encoder_rays
        self.sample_decoder = cfg.model.samples.decoder_rays

        self.explicit_encodes = cfg.model.has('explicit_encodes', False)

        self.with_monodepth = cfg.model.has('with_monodepth', False)
        if self.with_monodepth:
            self.depth_net = cfg.model.with_monodepth.depth_net
            self.monodepth_supervision = cfg.model.with_monodepth.has('depth_supervision', None)
            self.monodepth_freeze = cfg.model.with_monodepth.has('freeze', False)
        else:
            self.monodepth_supervision = None

        self.with_perceiver = cfg.model.has('with_perceiver', True)

        if self.with_perceiver:
            self.is_variational = cfg.networks.perceiver.has('variational')
            self.multi_evaluation = cfg.networks.perceiver.has('multi_evaluation', 1)
        else:
            self.is_variational = False
            self.multi_evaluation = 1

        self.distill = None if not cfg.model.has('distill') else {
            'no_supervision': cfg.model.distill.no_supervision,
            'start_epoch': cfg.model.distill.start_epoch,
        }

        self.interpolate_bilinear = partial(interpolate,
            scale_factor=None, mode='bilinear', align_corners=True)
        self.interpolate_nearest = partial(interpolate,
            scale_factor=None, mode='nearest', align_corners=None)

    @property
    def tasks(self):
        """Return the tasks of the model"""
        if self.with_perceiver:
            return self.networks['perceiver'].tasks
        else:
            return []

    def get_idx(self, all_idx, set_encodes=None):
        """Get the indices for encoding and decoding"""

        encode_idx = None
        decode_idx = None

        if set_encodes is not None:
            encode_idx = all_idx[-set_encodes:]
            all_idx = all_idx[:-set_encodes]

        n = len(all_idx)

        num_encodes = self.encode_train if self.training else self.encode_eval
        num_decodes = self.decode_train if self.training else self.decode_eval

        if encode_idx is None:
            if is_list(num_encodes):
                if num_encodes[0].startswith('+'):
                    encode_idx = parse_idx(all_idx, num_encodes[1:])
                elif num_encodes[0].startswith('-'):
                    num_encodes_parse = parse_idx(all_idx, num_encodes[1:])
                    encode_idx = [idx for idx in all_idx if idx not in num_encodes_parse]
                if len(num_encodes[0]) > 1:
                    num = int(num_encodes[0][1:])
                    encode_idx = np.random.permutation(encode_idx)
                    encode_idx = encode_idx[:num]
            elif num_encodes == 'all':
                num_encodes = n
            elif num_encodes < 0:
                num_encodes = n + num_encodes

        if decode_idx is None:
            if is_list(num_decodes):
                if num_decodes[0].startswith('+'):
                    decode_idx = parse_idx(all_idx, num_decodes[1:])
                elif num_decodes[0].startswith('-'):
                    num_decodes_parse = parse_idx(all_idx, num_decodes[1:])
                    decode_idx = [idx for idx in all_idx if idx not in num_decodes_parse]
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
            all_idx = np.random.permutation(all_idx)

        if encode_idx is None:
            encode_idx = all_idx[:num_encodes]
        if decode_idx is None:
            decode_idx = all_idx[-num_decodes:]

        encode_idx = [tuple(idx[::-1]) for idx in encode_idx]
        decode_idx = [tuple(idx[::-1]) for idx in decode_idx]

        if self.decode_encodes and self.training:
            decode_idx += [idx for idx in encode_idx if idx not in decode_idx]

        return encode_idx, decode_idx

    def create_predictions(self, idx, batch, output, decode_data, virtual_output=None):
        """Create predictions from the model output"""

        predictions = {}

        # Do nothing if it's training
        if self.training:
            return predictions

        # Remove volumetric predictions
        for task in list(output.keys()):
            for pop_key in ['volumetric', 'multiview', 'mpi']:
                if task.startswith(pop_key):
                    output.pop(task)

        # Convert and stack predictions
        gt_keys = []
        for task in output.keys():
            if 'info' not in task:
                predictions_key = {}
                for k in idx:
                    j, i = k
                    if j not in predictions_key.keys():
                        predictions_key[j] = []
                    predictions_key[j].append(output[task][k])
                predictions_key = {key: [torch.stack(val, 1)] if not is_list(val[0]) else
                                        [torch.stack([v[i] for v in val], 1) for i in range(len(val[0]))]
                                   for key, val in predictions_key.items()}
                predictions[task] = remove_none(predictions_key)

                gt_task = task.split('_')[0]
                if gt_task not in gt_keys and gt_task in batch.keys():
                    batch_key = {}
                    for k in idx:
                        j, i = k
                        if j not in batch_key.keys():
                            batch_key[j] = []
                        batch_key[j].append(batch[gt_task][j][:, i])
                    batch_key = {key: torch.stack(val, 1) for key, val in batch_key.items()}
                    batch[gt_task] = remove_none(batch_key)
                    gt_keys.append(gt_task)

        # Process virtual predictions
        if virtual_output is not None:
            for task in output.keys():
                for key in output[task].keys():
                    batch[task]['virtual_%d_%d' % key] = torch.stack(
                        [val for key2, val in virtual_output[task]['gt'].items() if key == key2[0]], 1)
                    predictions[task]['virtual_%d_%d' % key] = [
                        torch.stack([val[0] for key2, val in virtual_output[task]['pred'].items() if key == key2[0]], 1)
                    ]

        return batch, predictions

    def prep_pose(self, pose, augment=True):
        """Prepare pose for training"""

        pose = {key: val.clone() for key, val in pose.items()}

        b = pose[0].shape[0]
        ctx = [key for key in pose.keys() if key != 'virtual']
        pose = [{j: pose[j][i] for j in ctx} for i in range(b)]

        if self.zero_origin and augment:
            pose0 = [Pose().to(pose[0][0].device) for _ in range(b)]
        else:
            pose0 = [Pose(pose[i][0][[0]]) for i in range(b)]

        for i in range(b):
            pose[i][0][[0]] = pose0[i].T

        pose = [Pose.from_dict(p, to_global=True, zero_origin=False, to_matrix=True) for p in pose]
        pose = {j: torch.stack([pose[i][j] for i in range(b)], 0) for j in ctx}

        if self.training and self.augment_canonical and augment:
            pose = augment_canonical(pose)

        if self.training and len(self.use_pose_noise) > 0.0 and augment:
            if random.random() < self.use_pose_noise[0]:

                base = pose[0][:, 0].clone()
                base_inv = invert_pose(base)

                for key in pose.keys():
                    for n in range(pose[key].shape[1]):
                        pose[key][:, n] = pose[key][:, n] @ base_inv

                jittered = []
                for i in range(b):
                    jitter = Pose(base[[i]])
                    jitter.translateUp(self.use_pose_noise[1] * (2 * random.random() - 1))
                    jitter.translateLeft(self.use_pose_noise[1] * (2 * random.random() - 1))
                    jitter.translateForward(self.use_pose_noise[1] * (2 * random.random() - 1))
                    jitter.rotateRoll(torch.pi * self.use_pose_noise[2] * (2 * random.random() - 1))
                    jitter.rotatePitch(torch.pi * self.use_pose_noise[2] * (2 * random.random() - 1))
                    jitter.rotateYaw(torch.pi * self.use_pose_noise[2] * (2 * random.random() - 1))
                    jittered.append(jitter.T)
                jittered = torch.cat(jittered, 0)

                for key in pose.keys():
                    for n in range(pose[key].shape[1]):
                        pose[key][:, n] = pose[key][:, n] @ jittered

        return pose

    def virtual_cameras(self, encode_data, decode_data, perceiver_output):
        """Create virtual cameras for training"""

        if len(self.use_virtual_cameras) > 0 and (self.training or self.virtual_cameras_eval):

            virtual_data = create_virtual_cameras(
                decode_data,
                n_samples=self.use_virtual_cameras[1],
                cam_noise=self.use_virtual_cameras[2:-1],
                center_noise=self.use_virtual_cameras[-1],
                thr=0.1, tries=10, decay=0.9,
            )

            virtual_output = self.networks['perceiver'].decode(
                encoded=perceiver_output, encode_data=encode_data, decode_data=virtual_data,
                sample_decoder=self.sample_decoder,
            )['output']

            output = {
                'data': virtual_data,
                'cam_raw': {key: val['cam_raw'] for key, val in virtual_data.items()},
                'cam_mod': {key: val['cam_mod'] for key, val in virtual_data.items()},
            }

            for task in virtual_output.keys():
                if 'info' not in task:
                    if 'volumetric' in task or 'multiview' in task:
                        output[task] = {
                            'pred': virtual_output[task],
                        }
                    else:
                        task_key = task.split('_')[0]
                        output[task] = {
                            'pred': virtual_output[task],
                            'gt': {key: val[task_key] if task_key in val else None
                                   for key, val in virtual_data.items()}
                        }

            return output

        else:

            return None

    def encode(self, data, sources=None, scene=None):
        """Encode data to the perceiver network"""
        return self.networks['perceiver'].encode(
            data=data, sources=sources,
            sample_encoder=self.sample_encoder, scene=scene,
        )

    def single_decode(self, encoded, encode_data, decode_data,
                      sources=None, scene=None, monodepth=None):
        """Decode data from the perceiver network"""
        return self.networks['perceiver'].decode(
            encoded=encoded, encode_data=encode_data, decode_data=decode_data, sources=sources,
            sample_decoder=self.sample_decoder, scene=scene, monodepth=monodepth,
        )

    def multi_decode(self, encoded, encode_data, decode_data,
                     sources=None, num_evaluations=None, scene=None, monodepth=None):
        """Decode data multiple times for statistical analysis"""
        return self.networks['perceiver'].multi_decode(
            encoded=encoded, encode_data=encode_data, decode_data=decode_data, sources=sources,
            sample_decoder=self.sample_decoder, num_evaluations=num_evaluations, scene=scene, monodepth=monodepth,
        )

    def decode(self, encoded, encode_data, decode_data, sources=None, scene=None, monodepth=None):
        """Decode data from the perceiver network"""
        if not self.training and self.multi_evaluation > 1:
            return self.multi_decode(
                encoded, encode_data, decode_data, sources,
                num_evaluations=self.multi_evaluation, scene=scene, monodepth=monodepth)
        else:
            return self.single_decode(
                encoded, encode_data, decode_data, sources,
                scene=scene, monodepth=monodepth)

    def forward(self, batch, epoch=0):
        """Forward pass of the model, using a batch of data"""

        # Mixed batch training

        if is_list(batch):
            output = [self.forward(b, epoch) for b in batch]
            return {'loss': sum([out['loss'] for out in output]), 'metrics': {}, 'predictions': {}}

        # Remove explicit encoder

        if not self.explicit_encodes and 'enc' in batch['rgb'].keys():
            for key in batch.keys():
                if is_dict(batch[key]) and 'enc' in batch[key].keys():
                    batch[key].pop('enc')

        # Unsqueeze batch data if there is only one camera

        batch = expand_batch(batch, 'camera')

        # Incorporate explicit encode information onto the batch if available

        if 'enc' in batch['rgb'].keys():
            set_encodes = batch['rgb']['enc'].shape[1]
            for key in batch.keys():
                if is_dict(batch[key]) and 'enc' in batch[key].keys():
                    batch[key][0] = torch.cat([batch[key][0], batch[key]['enc']], 1)
                    batch[key].pop('enc')
        else:
            set_encodes = None

        # Get relevant keys

        has_rgb = 'rgb' in batch
        has_pose = 'pose' in batch
        has_intrinsics = 'intrinsics' in batch
        has_depth = 'depth' in batch
        has_normals = has_depth and 'normals' in self.tasks
        has_semantic = 'semantic' in batch
        has_meta = 'meta' in batch

        rgb = batch['rgb'] if has_rgb else None
        pose = batch['pose'] if has_pose else None
        intrinsics = batch['intrinsics'] if has_intrinsics else None
        depth = batch['depth'] if has_depth else None
        semantic = batch['semantic'] if has_semantic else None
        meta = batch['meta'] if has_meta else None

        scene = batch['scene'] if 'scene' in batch else None

        if semantic is not None:
            for key in semantic.keys():
                semantic[key] = semantic[key].unsqueeze(2)

        # Prepare poses

        pose_raw = self.prep_pose(pose, augment=False)
        pose_mod = self.prep_pose(pose, augment=True)

        if 'pose' in self.networks:
            pose = self.networks['pose'](batch['filename'])
            pose_raw[0] = pose
            pose_mod[0] = pose

        # Separate batch data and create cameras

        rgb = break_key(rgb, 4)
        intrinsics = break_key(intrinsics, 3)
        depth = break_key(depth, 4)
        semantic = break_key(semantic, 4)
        meta = break_key(meta)

        pose_raw = break_key(pose_raw, 3)
        pose_mod = break_key(pose_mod, 3)

        cams_raw = {key: Camera(K=intrinsics[(0, key[1])], Twc=pose_raw[key], hw=rgb[key])
                    for key in rgb.keys()}
        cams_mod = {key: Camera(K=intrinsics[(0, key[1])], Twc=pose_mod[key], hw=rgb[key])
                    for key in rgb.keys()}

        ###

        normals = {key: calculate_normals(depth[key], cams_mod[key]) for key in cams_mod.keys()} \
            if has_normals else None

        # Create all indices in a list

        all_idx = [key[::-1] for key in pose_raw.keys()]
        encode_idx, decode_idx = self.get_idx(all_idx, set_encodes)

        # Create encode dictionary

        encode_data = {key: {
            'cam_raw': cams_raw[key],
            'cam_mod': cams_mod[key],
            'meta': meta[key] if has_meta else None,
            'rgb': rgb[key] if has_rgb else None,
            'depth': depth[key] if has_depth else None,
            'normals': normals[key] if has_normals else None,
            'semantic': semantic[key] if has_semantic else None,
            'camera': None,
        } for key in encode_idx}

        # Create decode dictionary

        decode_data = {key: {
            'cam_raw': cams_raw[key],
            'cam_mod': cams_mod[key],
            'meta': meta[key] if has_meta else None,
            'rgb': rgb[key] if has_rgb else None,
            'depth': depth[key] if has_depth else None,
            'normals': normals[key] if has_normals else None,
            'semantic': semantic[key] if has_semantic else None,
            'camera': None,
        } for key in decode_idx}

        # Monodepth

        if self.with_monodepth:
            rgb_mono = {key: val['rgb'] for key, val in decode_data.items()}
            cam_mono = {key: val['cam_raw'] for key, val in decode_data.items()}
            freeze_layers_and_norms(self.networks[self.depth_net], ['ALL'], flag_freeze=self.monodepth_freeze)
            output_monodepth = {key: self.networks[self.depth_net](val) for key, val in rgb_mono.items()}
            output_monodepth = {
                'depth': {key: val['depths'] for key, val in output_monodepth.items()},
                'info': {key: {'cam_scaled': val} for key, val in cam_mono.items()},
                'rgb': {key: [val] for key, val in rgb_mono.items()}
            }
        else:
            output_monodepth = None

        # Perceiver IO

        if self.with_perceiver:
            # Run PerceiverIO (encode)
            perceiver_encoded = self.encode(encode_data, scene=scene)
            # Virtual camera processing
            virtual_output = self.virtual_cameras(encode_data, decode_data, perceiver_encoded)
            # Run PerceiverIO (decode)
            perceiver_decoded = self.decode(
                perceiver_encoded, encode_data=encode_data, decode_data=decode_data,
                scene=scene, monodepth=output_monodepth
            )

            perceiver_losses, output = perceiver_decoded['losses'], perceiver_decoded['output']
        else:
            output, virtual_output = {}, None
            perceiver_losses = None

        if self.with_monodepth:
            for key in output_monodepth.keys():
                output['%s_mono' % key] = output_monodepth[key]

        # Convert to predictions

        if not self.training and 'depth' in output.keys():
            output['depth'] = to_depth(output['depth'], output['info'])

        if not self.training:
            batch, predictions = self.create_predictions(decode_idx, batch, output, decode_data, virtual_output)

            return {
                'predictions': predictions,
                'batch': batch,
            }

        # Prepare GT for loss calculation

        batch_gt = {'all_rgb': rgb, 'all_cam': cams_mod}
        for task_key in ['rgb', 'depth', 'normals', 'semantic', 'camera']:
            batch_gt[task_key] = {key: val[task_key] for key, val in decode_data.items()}
        if self.with_monodepth:
            for task in output_monodepth.keys():
                if 'info' not in task:
                    batch_gt['%s_mono' % task] = {key: val[task] for key, val in decode_data.items()}

        # Loss calculation

        loss = self.training_losses(
            batch_gt, encode_data, decode_data, output, virtual_output, epoch, perceiver_losses)
        predictions = metrics = {}

        # Return losses, metrics (empty) and predictions (empty)

        return {
            'loss': loss,
            'metrics': metrics,
            'predictions': predictions,
        }

    def training_losses(self, gt_data, encode_data, decode_data,
                        output, virtual_output, epoch, perceiver_losses=None):
        """Calculate the training losses"""

        # Start prediction dictionary
        pred_dict = {
            'all_rgb': gt_data['all_rgb'],
            'all_cam': gt_data['all_cam'],
        }

        # Add predictions and ground-truths
        for task in output.keys():
            if 'info' not in task:
                if task.startswith('volumetric') or task.startswith('multiview') or task.startswith('mpi'):
                    pred_dict[task] = {
                        'gt': {
                            'rgb': gt_data['rgb'],
                            'depth': gt_data['depth'],
                        },
                        'pred': output[task]
                    }
                else:
                    pred_dict[task] = {
                        'gt': {key: val for key, val in gt_data[task.split('_')[0]].items()},
                        'pred': output[task]
                    }
            else:
                pred_dict[task] = output[task]

        if self.monodepth_supervision is not None:
            for key1 in self.monodepth_supervision[1]:
                for key2 in pred_dict[key1]['pred'].keys():
                    pred_dict[key1]['gt'][key2] = pred_dict[
                        self.monodepth_supervision[0]]['pred'][key2][0].detach()

        # Calculate losses
        loss = self.compute_losses(pred_dict, encode_data, decode_data, epoch, is_virtual=False)

        # Calculate virtual losses
        if virtual_output is not None:
            virtual_loss = self.compute_losses(virtual_output, encode_data, decode_data, epoch, is_virtual=True)
            loss = loss + self.use_virtual_cameras[0] * virtual_loss

        # Add network losses
        if perceiver_losses is not None:
            loss += sum([val for val in perceiver_losses.values()])

        # Return loss
        return loss

    def compute_losses(self, pred_dict, encode_data, decode_data, epoch, is_virtual):
        """Compute individual losses for each task"""

        losses = []

        # Replace GT with predicted values
        if self.distill:

            sup_suffix = '1_vol2'
            vol_suffixes = ['1_vol1', '1_vol2']

            if is_virtual:
                task_keys = ['rgb', 'depth']
            else:
                task_keys = ['depth']

            for key in pred_dict['depth_single']['pred'].keys():
                for task_key in task_keys:
                    if epoch >= self.distill['start_epoch']:
                        gt = pred_dict[f'{task_key}_{sup_suffix}']['pred'][key][0].detach()
                        if gt is not None and gt.sum() > 0:
                            pred_dict[f'{task_key}_single']['gt'][key] = gt
                    if self.distill['no_supervision']:
                        for vol_suffix in vol_suffixes:
                            vol_key = f'{task_key}_{vol_suffix}'
                            if vol_key in pred_dict.keys():
                                pred_dict[vol_key]['gt'][key] = None
                        if self.with_monodepth:
                            mono_key = f'{task_key}_mono'
                            if mono_key in pred_dict.keys():
                                pred_dict[mono_key]['gt'][key] = None

        # RGB losses
        for key in pred_dict.keys():
            if key.startswith('rgb') and 'mono' not in key:
                losses.append(self.compute_rgb_losses(
                    pred_dict, key_pred=key, is_virtual=is_virtual))

        # Depth losses
        for key in pred_dict.keys():
            if key.startswith('depth'):  # if key.startswith('depth_mono'):
                losses.append(self.compute_depth_losses(
                    encode_data, decode_data, pred_dict, key_pred=key, is_virtual=is_virtual))

        # Volumetric losses
        for key in pred_dict.keys():
            if key.startswith('volumetric'):  # if key.startswith('depth_mono'):
                losses.append(self.compute_volumetric_losses(
                    encode_data, decode_data, pred_dict, key_pred=key, is_virtual=is_virtual))

        # Normals losses
        for key in pred_dict.keys():
            if key.startswith('normals'):
                losses.append(self.compute_normals_losses(
                    pred_dict, key_pred=key, is_virtual=is_virtual))

        # Semantic losses
        for key in pred_dict.keys():
            if key.startswith('semantic'):
                losses.append(self.compute_semantic_losses(
                    pred_dict, key_pred=key, is_virtual=is_virtual))

        # Camera losses
        for key in pred_dict.keys():
            if key.startswith('camera'):
                losses.append(self.compute_camera_losses(
                    pred_dict, key_pred=key, is_virtual=is_virtual))

        # Remove invalid losses
        losses = [loss for loss in losses if loss is not None]

        # Scale loss
        if len(losses) > 1 and self.scale_loss:
            max_loss = max([loss.item() for loss in losses])
            for i in range(len(losses)):
                ratio_rgb_depth = max_loss / losses[i].item()
                losses[i] = losses[i] * ratio_rgb_depth

        return 0 if len(losses) == 0 else sum(losses) / len(losses)

    def parse_stddev(self, pred):
        """Parse the standard deviation of the predictions"""
        weight = 1.0 if not self.is_variational else self.networks['perceiver'].variational_params['soft_mask_weight']
        stddev = {key: None if len(val) == 1 or weight == 1.0 else torch.stack(val).std(0).sum(
            1 if val[0].dim() == 4 else -1, keepdim=True) for key, val in pred.items()}
        return {key: None if val is None else
            weight + (val / (val.max() + 1e-6)) * (1.0 - weight) for key, val in stddev.items()}

    def compute_rgb_losses(self, pred_dict, is_virtual, key_pred='rgb'):
        """Compute RGB losses"""

        losses = []

        if 'rgb_sup' in self.losses:
            pred, gt = pred_dict[key_pred]['pred'], pred_dict[key_pred]['gt']
            soft_mask = self.parse_stddev(pred)

            for key in pred.keys():
                if gt[key] is not None:
                    output = self.losses['rgb_sup'](
                        pred[key], gt[key], soft_mask=soft_mask[key])
                    losses.append(output['loss'])

        return sum(losses) / len(losses) if len(losses) > 0 else None

    def compute_depth_losses(self, encode_data, decode_data, pred_dict, is_virtual, key_pred='depth'):
        """Compute depth losses"""

        pred, gt = pred_dict[key_pred]['pred'], pred_dict[key_pred]['gt']
        soft_mask = self.parse_stddev(pred)

        losses = []

        if 'depth_sup' in self.losses:
            for key in pred.keys():
                if gt[key] is not None:
                    output = self.losses['depth_sup'](
                        pred[key], gt[key], soft_mask=soft_mask[key])
                    losses.append(output['loss'])

        if 'depth_normals_sup' in self.losses:
            info = pred_dict[key_pred.replace('depth', 'info')]
            cam_scaled = {key: val['cam_scaled'] for key, val in info.items()}
            for key in pred.keys():
                if (gt[key] <= 0).sum() == 0:  # Only possible if it's dense
                    output = self.losses['depth_normals_sup'](pred[key], gt[key], cam_scaled[key])
                    losses.append(output['loss'])

        if 'depth_selfsup' in self.losses and not is_virtual:
            info = pred_dict['info_' + key_pred.split('_')[-1]]
            rgb = pred_dict[key_pred.replace('depth', 'rgb')]['gt']
            all_rgb, all_cam = pred_dict['all_rgb'], pred_dict['all_cam']  # DOUBLE CHECK CAMERA REFERENCE
            cam_scaled = {key: val['cam_scaled'] for key, val in info.items()}
            for key in pred.keys():
                output = self.depth_selfsup(
                    all_rgb, rgb, pred, [cam_scaled, all_cam], key, key_pred)
                losses.append(output['loss'])

        return sum(losses) / len(losses) if len(losses) > 0 else None

    def compute_volumetric_losses(self, encode_data, decode_data, pred_dict, is_virtual, key_pred='volumetric'):
        """Compute volumetric losses"""

        if 'gt' not in pred_dict[key_pred].keys():
            return None

        pred, gt = pred_dict[key_pred]['pred'], pred_dict[key_pred]['gt']
        gt_rgb, gt_depth = gt['rgb'], gt['depth']

        losses = []

        if 'volumetric_sup' in self.losses:
            info = pred_dict[key_pred.replace('volumetric', 'info')]
            rgb = pred_dict[key_pred.replace('volumetric', 'rgb')]['gt']
            all_rgb, all_cam = pred_dict['all_rgb'], pred_dict['all_cam']
            cam_scaled = {key: val['cam_scaled'] for key, val in info.items()}
            z_samples = {key: val['z_samples'] for key, val in info.items()}
            for key in pred.keys():
                if gt_depth[key] is not None:
                    output = self.losses['volumetric_sup'](
                        all_rgb, rgb, pred, [cam_scaled, all_cam], z_samples, key)
                    losses.append(output['loss'])

        return sum(losses) / len(losses) if len(losses) > 0 else None

    def compute_normals_losses(self, pred_dict, is_virtual, key_pred='normals'):
        """Compute normals losses"""

        losses = []

        if 'normals_sup' in self.losses:
            pred, gt = pred_dict[key_pred]['pred'], pred_dict[key_pred]['gt']
            for key in pred.keys():
                output = self.losses['normals_sup'](pred[key], gt[key])
                losses.append(output['loss'])

        return sum(losses) / len(losses) if len(losses) > 0 else None

    def compute_semantic_losses(self, pred_dict, is_virtual, key_pred='semantic'):
        """Compute semantic losses"""

        losses = []

        if 'semantic_sup' in self.losses:
            pred, gt = pred_dict[key_pred]['pred'], pred_dict[key_pred]['gt']
            for key in pred.keys():
                output = self.losses['semantic_sup'](pred[key], gt[key])
                losses.append(output['loss'])

        return sum(losses) / len(losses) if len(losses) > 0 else None

    def compute_camera_losses(self, pred_dict, is_virtual, key_pred='camera'):
        """Compute camera losses"""

        losses = []

        if 'camera_sup' in self.losses:
            pred, gt = pred_dict[key_pred]['pred'], pred_dict[key_pred]['gt']
            for key in pred.keys():
                output = self.losses['camera_sup'](pred[key], gt[key])
                losses.append(output['loss'])

        return sum(losses) / len(losses) if len(losses) > 0 else None

    def depth_selfsup(self, rgb_all, rgb_gt, depths, cams, tgt, key_pred):
        """Depth self-supervision loss"""

        ctx = [key for key in rgb_all.keys() if key != tgt]

        depths_tgt = depths[tgt]
        num_scales = self.get_num_scales(depths_tgt)

        rgbs_all = {key: [val] * num_scales for key, val in rgb_all.items()}
        rgbs_gt = make_rgb_scales(rgb_gt, pyramid=depths_tgt)

        rgb_tgt = [rgbs_gt[tgt][i] for i in range(num_scales)]
        rgb_ctx = [[rgbs_all[j][i] for j in ctx] for i in range(num_scales)]

        synth = self.view_synthesis(
            rgbs_all, depths=depths_tgt, cams=cams, return_masks=True, tgt=tgt)

        output = self.losses['depth_selfsup'](
            rgb_tgt, rgb_ctx, synth['warps'], overlap_mask=synth['masks'])

        return output

    def run_monodepth(self, data):
        """Run monodepth network"""

        b = [d['rgb'].shape[0] for d in data]
        rgb = torch.cat([d['rgb'] for d in data], 0)
        depth = self.networks[self.depth_net](rgb)['depths']
        depth = [torch.split(d, b) for d in depth]
        return [[d[i] for d in depth] for i in range(len(depth[0]))]
