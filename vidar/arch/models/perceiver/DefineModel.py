# Copyright 2021 Toyota Research Institute.  All rights reserved.

import random
from abc import ABC

import torch

from vidar.arch.models.BaseModel import BaseModel
from vidar.geometry.camera_nerf import CameraNerf
from vidar.geometry.pose import Pose
from vidar.utils.data import flatten
from vidar.utils.types import is_list, is_int


def augment_canonical(pose):

    ctx = list(pose.keys())
    num = list(range(pose[0].shape[1]))

    i = random.choice(ctx)
    j = random.choice(num)

    base = Pose(pose[i][:, j]).inverse().T
    for key in ctx:
        for n in num:
            pose[key][:, n] = pose[key][:, n] @ base

    return pose


def parse_output(output, key, encode_idx, predictions):
    if key in output.keys():
        pred = [[val] for val in output[key]]
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


def create_virtual_cameras(encode_data, n_samples=1, cam_noise=None, center_noise=None,
                           downsample=1.0, thr=0.1, tries=10, decay=0.9):

    gt_cams = [datum['cam'] for datum in encode_data]
    gt_depths = [datum['gt_depth'] for datum in encode_data]
    gt_rgbs = [datum['rgb'] for datum in encode_data]

    if not is_int(thr):
        n = gt_rgbs[0].shape[-2] * gt_rgbs[0].shape[-1]
        thr = int(n * thr)

    pcls_proj = [cam.scaled(downsample).reconstruct_depth_map(depth, to_world=True)
                 for cam, depth in zip(gt_cams, gt_depths)]
    pcls_proj = [pcl.reshape(*pcl.shape[:2], -1) for pcl in pcls_proj]
    pcl_proj = torch.cat([pcl for pcl in pcls_proj], -1)
    clr_proj = torch.cat([rgb.reshape(*rgb.shape[:2], -1)
                         for rgb in gt_rgbs], -1)

    gt_pcl_centers = [pcl.mean(-1) for pcl in pcls_proj]

    virtual_data = []
    for gt_rgb, gt_depth, gt_cam, gt_pcl_center in zip(gt_rgbs, gt_depths, gt_cams, gt_pcl_centers):
        for i in range(n_samples):

            cam = gt_cam.clone()
            pcl_center = gt_pcl_center.clone()

            weight = 1.0
            rgb_proj = depth_proj = None
            for j in range(tries):

                if center_noise is not None:
                    pcl_center_noise = weight * center_noise * \
                        (2 * torch.rand_like(gt_pcl_center) - 1)
                    pcl_center = gt_pcl_center + pcl_center_noise

                if cam_noise is not None:
                    cam.look_at(pcl_center)
                    cam.Twc.translateUp(
                        weight * cam_noise[0] * (2 * random.random() - 1))
                    cam.Twc.translateLeft(
                        weight * cam_noise[1] * (2 * random.random() - 1))
                    cam.Twc.translateForward(
                        weight * cam_noise[2] * (2 * random.random() - 1))
                    cam.look_at(pcl_center)

                rgb_proj_try, depth_proj_try = cam.project_pointcloud(
                    pcl_proj, clr_proj)

                valid = (depth_proj_try > 0).sum() > thr
                if valid:
                    rgb_proj, depth_proj = rgb_proj_try, depth_proj_try
                    break
                else:
                    weight = weight * decay

            if rgb_proj is None and depth_proj is None:
                rgb_proj, depth_proj = gt_rgb, gt_depth
                cam = gt_cam.clone()

            virtual_data.append({
                'cam': cam,
                'rgb': rgb_proj.contiguous(),
                'gt_depth': depth_proj.contiguous(),
            })

    return virtual_data


def ControlVidarCamera(draw, cam, tvel=0.2, rvel=0.1):
    change = False
    if draw.UP:
        cam.Twc.translateForward(tvel)
        change = True
    if draw.DOWN:
        cam.Twc.translateBackward(tvel)
        change = True
    if draw.LEFT:
        cam.Twc.translateLeft(tvel)
        change = True
    if draw.RIGHT:
        cam.Twc.translateRight(tvel)
        change = True
    if draw.PGUP:
        cam.Twc.translateUp(tvel)
        change = True
    if draw.PGDOWN:
        cam.Twc.translateDown(tvel)
        change = True
    if draw.KEY_A:
        cam.Twc.rotateYaw(-rvel)
        change = True
    if draw.KEY_D:
        cam.Twc.rotateYaw(+rvel)
        change = True
    if draw.KEY_W:
        cam.Twc.rotatePitch(+rvel)
        change = True
    if draw.KEY_S:
        cam.Twc.rotatePitch(-rvel)
        change = True
    if draw.KEY_Q:
        cam.Twc.rotateRoll(-rvel)
        change = True
    if draw.KEY_E:
        cam.Twc.rotateRoll(+rvel)
        change = True
    return change


class HuggingModel(BaseModel, ABC):

    def __init__(self, cfg):
        super().__init__(cfg)

        from vidar.arch.networks.perceiver.HuggingNet import HuggingNet
        self.networks['perceiver'] = HuggingNet(cfg.model.network)
        self.weights = [1.0, 1.0]
        self.use_pose_noise = cfg.model.use_pose_noise
        self.use_virtual_cameras = cfg.model.use_virtual_cameras
        self.use_virtual_rgb = cfg.model.use_virtual_rgb
        self.augment_canonical = cfg.model.augment_canonical

    def forward(self, batch, epoch=0, collate=True):

        if is_list(batch):
            output = [self.forward(b, collate=False) for b in batch]
            loss = [out['loss'] for out in output]
            return {'loss': sum(loss) / len(loss), 'predictions': {}, 'metrics': {}}

        if not collate:
            for key in ['rgb', 'intrinsics', 'pose', 'depth']:
                batch[key] = {k: v.unsqueeze(0) for k, v in batch[key].items()}

        rgb = batch['rgb']
        intrinsics = batch['intrinsics']
        pose = batch['pose']
        depth = batch['depth']

        ctx = [0]  # rgb.keys()
        b, n = rgb[0].shape[:2]

        ii, jj = range(n), ctx

        pose = [{key: val[i] for key, val in pose.items()} for i in range(b)]

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
        pose = {key: torch.stack([pose[i][key] for i in range(
            len(pose))], 0) for key in pose[0].keys()}

        if self.training and self.augment_canonical:
            pose = augment_canonical(pose)

        rgb = [{key: val[:, i] for key, val in rgb.items()} for i in range(n)]
        intrinsics = [{key: val[:, i]
                       for key, val in intrinsics.items()} for i in range(n)]
        pose = [{key: val[:, i] for key, val in pose.items()}
                for i in range(n)]
        depth = [{key: val[:, i] for key, val in depth.items()}
                 for i in range(n)]

        cams = [{j: CameraNerf(K=intrinsics[i][0], Twc=pose[i][j], hw=rgb[i][j])
                 for j in ctx} for i in range(n)]

        encode_idx = flatten([[[i, j] for i in ii] for j in jj])

        encode_data = [{
            'rgb': rgb[i][j],
            'cam': cams[i][j],
            'gt_depth': depth[i][j],
        } for i, j in encode_idx]

        perceiver_output = self.networks['perceiver'](
            encode_data=encode_data,
        )

        output = perceiver_output['output']

        predictions = {}

        # DEPTH

        pred_depths = parse_output(output, 'depth', encode_idx, predictions)
        pred_depths_mono = parse_output(
            output, 'depth_mono', encode_idx, predictions)

        if pred_depths is not None and pred_depths_mono is not None:
            for i in range(len(pred_depths)):
                pred_depths[i] += pred_depths_mono[i]

        # RGB

        pred_rgbs = parse_output(output, 'rgb', encode_idx, predictions)

        # VIRTUAL

        if len(self.use_virtual_cameras) > 0:

            virtual_data = create_virtual_cameras(
                encode_data,
                n_samples=self.use_virtual_cameras[1],
                cam_noise=self.use_virtual_cameras[2:-1],
                center_noise=self.use_virtual_cameras[-1],
                thr=0.1, tries=10, decay=0.9,
            )
            virtual_output = self.networks['perceiver'].decode(
                latent=perceiver_output['latent'], data=virtual_data,
                sources=['cam'], field='cam',
            )['output']

            gt_virtual_cams = [data['cam'] for data in virtual_data]

            if pred_depths is not None:
                pred_depths_virtual = [[depth]
                                       for depth in virtual_output['depth']]
                gt_depths_virtual = [data['gt_depth'] for data in virtual_data]

                batch['depth']['virtual'] = torch.stack(gt_depths_virtual, 1)
                predictions['depth']['virtual'] = [torch.stack(
                    [pred[0] for pred in pred_depths_virtual], 1)]
            else:
                pred_depths_virtual, gt_depths_virtual = None, None

            if pred_rgbs is not None:
                pred_rgbs_virtual = [[rgb] for rgb in virtual_output['rgb']]
                gt_rgbs_virtual = [data['rgb'] for data in virtual_data]

                batch['rgb']['virtual'] = torch.stack(gt_rgbs_virtual, 1)
                predictions['rgb']['virtual'] = [torch.stack(
                    [pred[0] for pred in pred_rgbs_virtual], 1)]
            else:
                pred_rgbs_virtual, gt_rgbs_virtual = None, None

        ##########################################################

        # display_data_interactive(self.networks, encode_data, encode_data, output, virtual_data)
        # display_data(rgb, depth, cams, ctx, n, batch, predictions, gt_virtual_cams)

        ##########################################################

        if not self.training:
            return {
                'predictions': predictions,
                'batch': batch,
            }

        gt_depths = [depth[i][j] for i, j in encode_idx]
        gt_rgbs = [rgb[i][j] for i, j in encode_idx]

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

        return {
            'loss': loss,
            'metrics': metrics,
            'predictions': predictions,
        }

    def compute_loss_and_metrics(self, pred_rgbs, gt_rgbs, pred_depths, gt_depths):

        loss, metrics = [], {}

        if pred_rgbs is not None and 'rgb' in self.losses and self.weights[0] > 0.0:
            loss_rgb = []
            for pred, gt in zip(pred_rgbs, gt_rgbs):
                rgb_output = self.losses['rgb'](pred, gt)
                loss_rgb.append(self.weights[0] * rgb_output['loss'])
            loss.append(sum(loss_rgb) / len(loss_rgb))
        if pred_depths is not None and 'depth' in self.losses and self.weights[1] > 0.0:
            loss_depth = []
            for pred, gt in zip(pred_depths, gt_depths):
                depth_output = self.losses['depth'](pred, gt)
                loss_depth.append(self.weights[1] * depth_output['loss'])
            loss.append(sum(loss_depth) / len(loss_depth))

        loss = sum(loss) / len(loss)

        return loss, metrics
