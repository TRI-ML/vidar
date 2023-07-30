# TRI-VIDAR - Copyright 2023 Toyota Research Institute.  All rights reserved.

from abc import ABC

from vidar.arch.blocks.image.ViewSynthesisSelfSup import ViewSynthesis
from vidar.arch.models.BaseModel import BaseModel
from vidar.arch.models.utils import make_rgb_scales, create_cameras
from vidar.utils.data import get_from_dict
from vidar.utils.config import cfg_has


class SelfSupervisedModel(BaseModel, ABC):
    """
    Self-supervised depth estimation model

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.view_synthesis = ViewSynthesis()
        self.set_attr(cfg.model, 'use_gt_pose', False)
        self.set_attr(cfg.model, 'use_gt_intrinsics', True)

        # Define camera model for self-calibration
        if not self.use_gt_intrinsics:
            self.camera_model = cfg_has(cfg.networks.intrinsics, 'camera_model', 'ucm')
            if self.camera_model == 'ucm':
                from vidar.geometry.camera_ucm import UCMCamera
                self.camera_class = UCMCamera
            elif self.camera_model == 'eucm':
                from vidar.geometry.camera_eucm import EUCMCamera
                self.camera_class = EUCMCamera
            elif self.camera_model == 'ds':
                from vidar.geometry.camera_ds import DSCamera
                self.camera_class = DSCamera
            else:
                raise NotImplementedError('Invalid camera type')

    def forward(self, batch, epoch=0):
        """Model forward pass, given a batch dictionary"""

        tgt = (0, 0)

        rgb = batch['rgb']
        if self.use_gt_intrinsics:
            intrinsics = get_from_dict(batch, 'intrinsics')
        else:
            intrinsics = self.networks['intrinsics'](rgb=rgb[tgt])

        valid_mask = get_from_dict(batch, 'mask')

        if self.use_gt_intrinsics:
            depth_output = self.networks['depth'](rgb=rgb[tgt], intrinsics=intrinsics[tgt])
        else:
            depth_output = self.networks['depth'](rgb=rgb[tgt])
        pred_depth = depth_output['depths']

        predictions = {
            'depth': {tgt: pred_depth},
        }

        pred_logvar = get_from_dict(depth_output, 'logvar')
        if pred_logvar is not None:
            predictions['logvar'] = {tgt: pred_logvar}

        # Return only predictions if not training 
        
        if not self.training:
            return {
                'predictions': predictions,
            }

        # IF using GT, get it from batch
        if self.use_gt_pose:
            assert 'pose' in batch, 'You need input pose'
            pose = batch['pose']
        # Otherwise, calculate it
        elif 'pose' in self.networks:
            pose = self.compute_pose(rgb, self.networks['pose'], tgt=tgt, invert=True)
            predictions['pose'] = pose
        else:
            pose = None

        # If not using GT, get camera class we are using
        if not self.use_gt_intrinsics:
            cams = {tgt: self.camera_class(I=intrinsics)}
            for key in pose.keys():
                cams[key] = self.camera_class(I=intrinsics, Tcw=pose[key])
        else:
            cams = create_cameras(rgb[tgt], intrinsics[tgt], pose)

        gt_depth = None if 'depth' not in batch else batch['depth'][tgt]
        loss, metrics = self.compute_loss_and_metrics(
            rgb, pred_depth, cams, gt_depth=gt_depth,
            logvar=pred_logvar, valid_mask=valid_mask
        )

        # If not using GT intrinsics, get them from the network
        if not self.use_gt_intrinsics:
            if self.camera_model == 'ucm':
                fx, fy, cx, cy, alpha = intrinsics[0].squeeze()
                intrinsics_metrics = {'fx': fx, 'fy':fy, 'cx':cx, 'cy':cy, 'alpha':alpha}
                metrics.update(intrinsics_metrics)
            elif self.camera_model == 'eucm':
                fx, fy, cx, cy, alpha, beta = intrinsics[0].squeeze()
                intrinsics_metrics = {'fx': fx, 'fy':fy, 'cx':cx, 'cy':cy, 'alpha':alpha, 'beta':beta}
                metrics.update(intrinsics_metrics)
            elif self.camera_model == 'ds':
                fx, fy, cx, cy, xi, alpha = intrinsics[0].squeeze()
                intrinsics_metrics = {'fx': fx, 'fy':fy, 'cx':cx, 'cy':cy, 'xi':xi, 'alpha':alpha}
                metrics.update(intrinsics_metrics)
            else:
                raise NotImplementedError('Invalid camera type')

        return {
            'loss': loss,
            'metrics': metrics,
            'predictions': predictions,
        }

    def compute_loss_and_metrics(self, rgb, depths, cams, gt_depth=None,
                                 logvar=None, valid_mask=None):
        """
        Compute loss and metrics for training

        Parameters
        ----------
        rgb : Dict
            Dictionary with input images [B,3,H,W]
        depths : list[torch.Tensor]
            List with target depth maps in different scales [B,1,H,W]
        cams : Dict
            Dictionary with cameras for each input image
        gt_depth : torch.Tensor
            Ground-truth depth map for supervised training
        logvar : list[torch.Tensor]
            Log-variance maps for uncertainty training
        valid_mask : list[torch.Tensor]
            Binary mask for masking out invalid pixels [B,1,H,W]

        Returns
        -------
        loss : torch.Tensor
            Training loss
        metrics : Dict
            Dictionary with training metrics
        """
        tgt = (0, 0)
        ctx = [key for key in rgb.keys() if key != tgt]

        num_scales = self.get_num_scales(depths)

        rgbs = make_rgb_scales(rgb, pyramid=depths)
        rgb_tgt = [rgbs[tgt][i] for i in range(num_scales)]
        rgb_ctx = [[rgbs[j][i] for j in ctx] for i in range(num_scales)]

        loss, metrics = [], {}

        if 'reprojection' in self.losses:
            synth = self.view_synthesis(
                rgbs, depths=depths, cams=cams, return_masks=True)
            reprojection_output = self.losses['reprojection'](
                rgb_tgt, rgb_ctx, synth['warps'], logvar=logvar,
                valid_mask=valid_mask, overlap_mask=synth['masks'])
            loss.append(reprojection_output['loss'])
            metrics.update(**reprojection_output['metrics'])
        if 'smoothness' in self.losses:
            smoothness_output = self.losses['smoothness'](rgb_tgt[0], depths)
            loss.append(smoothness_output['loss'])
            metrics.update(**smoothness_output['metrics'])
        if 'supervision' in self.losses and gt_depth is not None:
            supervision_output = self.losses['supervision'](depths, gt_depth)
            loss.append(supervision_output['loss'])
            metrics.update(**supervision_output['metrics'])
        if 'normals' in self.losses and gt_depth is not None:
            normals_output = self.losses['normals'](depths, gt_depth, cams[0])
            loss.append(normals_output['loss'])
            metrics.update(**normals_output['metrics'])

        loss = sum(loss)

        return loss, metrics
    

