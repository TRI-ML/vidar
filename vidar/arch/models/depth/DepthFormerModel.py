# Copyright 2023 Toyota Research Institute.  All rights reserved.

import random
from abc import ABC
from functools import partial

import torch

from knk_vision.vidar.vidar.arch.blocks.image.ViewSynthesis import ViewSynthesis
from knk_vision.vidar.vidar.arch.models.BaseModel import BaseModel
from knk_vision.vidar.vidar.arch.models.utils import make_rgb_scales, break_context, create_cameras
from knk_vision.vidar.vidar.utils.data import get_from_dict
from knk_vision.vidar.vidar.utils.depth import inv2depth
from knk_vision.vidar.vidar.utils.tensor import interpolate, multiply_args
from knk_vision.vidar.vidar.utils.types import is_str, is_tuple, is_list, is_dict


def fix_predictions(predictions):
    fixed_predictions = {}
    for key, val in predictions.items():
        if is_dict(val):
            fixed_predictions[key] = {k if is_tuple(k) else (k, 0): v for k, v in val.items()}   
    return fixed_predictions


def curr_stereo(val):
    return is_str(val) and val.startswith('0')


class DepthFormerModel(BaseModel, ABC):
    """
    Depthformer base model (https://arxiv.org/abs/2204.07616)

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        self.set_attr(cfg.model, 'warp_context', None)
        self.set_attr(cfg.model, 'match_context', None)

        self.motion_masking = cfg.model.motion_masking
        self.matching_augmentation = cfg.model.matching_augmentation
        self.freeze_teacher_and_pose = cfg.model.freeze_teacher_and_pose

        self.view_synthesis = ViewSynthesis()

        self.interpolate_nearest = partial(
            interpolate, mode='nearest', scale_factor=None)

        self.set_attr(cfg.model, 'spatial_weight', [0.0, 0.0])
        self.set_attr(cfg.model, 'spatio_temporal_weight', [0.0, 0.0])
        self.set_attr(cfg.model, 'run_mono', True)
        self.set_attr(cfg.model, 'run_multi', True)

        self.set_attr(cfg.model, 'use_gt_pose', False)
        self.set_attr(cfg.model, 'use_gt_depth', False)
        self.set_attr(cfg.model, 'display', False)
        self.set_attr(cfg.model, 'mono_type', 'mono')
        self.set_attr(cfg.model, 'multi_temporal_only', False)

        self.set_attr(cfg.model, 'apply_consistency', True)

    @staticmethod
    def process_stereo(batch):
        """Process batch to recover stereo / monocular information"""
        
        batch = {key: val for key, val in batch.items()}
        new_intrinsics = {0: batch['intrinsics'][0]}
        for key in batch['pose'].keys():
            if not is_str(key) and key != 0:
                new_intrinsics[key] = batch['intrinsics'][0]
        batch['intrinsics'] = new_intrinsics
        suffixes = ['', 'r', 's', 't', 'u', 'v']
        if batch['rgb'][0].dim() == 5:
            # Change RGB
            rgb_stereo = {}
            for key, val in batch['rgb'].items():
                rgb_stereo[key] = val[:, 0]
                for i in range(1, val.shape[1]):
                    rgb_stereo['%d%s' % (key, suffixes[i])] = val[:, i]
            batch['rgb'] = rgb_stereo
            # Change pose
            if 'pose' in batch:
                pose_stereo = {}
                for key, val in batch['pose'].items():
                    pose_stereo[key] = val[:, 0]
                    for i in range(1, val.shape[1]):
                        pose_stereo['%d%s' % (key, suffixes[i])] = val[:, i]
                batch['pose'] = pose_stereo
            # Change intrinsics
            if 'intrinsics' in batch:
                intrinsics_stereo = {}
                for key, val in batch['intrinsics'].items():
                    intrinsics_stereo[key] = val[:, 0]
                    for i in range(1, val.shape[1]):
                        intrinsics_stereo['%d%s' % (key, suffixes[i])] = val[:, i]
                for key in batch['pose'].keys():
                    if not is_str(key) and key != 0:
                        for suffix in ['r', 's', 't', 'u', 'v']:
                            if '0%s' % suffix in intrinsics_stereo.keys():
                                intrinsics_stereo['%d%s' % (key, suffix)] = intrinsics_stereo['0%s' % suffix]
                batch['intrinsics'] = intrinsics_stereo
        return batch

    @staticmethod
    def get_stereo_pose(batch, pose, context):
        """Get poses from stereo images"""
        for key in context:
            if key not in pose.keys() and curr_stereo(key):
                pose[key] = batch['pose'][key]
        return pose

    @staticmethod
    def pose_context(pose, context):
        """Extract context poses from a pose dictionary"""
        for key in context:
            if not is_str(key):
                for key2 in list(pose.keys()):
                    if curr_stereo(key2):
                        new_key = '%d%s' % (key, key2[1:])
                        if new_key in context:
                            pose[new_key] = pose[key2] @ pose[key]
        return pose

    @staticmethod
    def conf_mask(depth1, depth2, thr=1.0):
        """Calculate confidence masks"""
        mask1 = ((depth1 - depth2) / depth2).abs() < thr
        mask2 = ((depth2 - depth1) / depth1).abs() < thr
        return mask1 * mask2

    def forward(self, batch, epoch=0):
        """Model forward pass"""

        tgt = (0, 0)

        mono_depth_string = 'mono_depth'
        multi_depth_string = 'multi_depth'

        for key in ['rgb', 'pose', 'intrinsics']:
            if key in batch:
                batch[key] = {k[0] if is_tuple(k) else k: v for k, v in batch[key].items()}
 
        predictions = {}

        batch = {key: val for key, val in batch.items()}
        batch['rgb'] = {key: val for key, val in batch['rgb'].items()}

        ### TRANSFORMER

        batch = self.process_stereo(batch)
        batch_rgb = batch['rgb']
        rgbs = make_rgb_scales(batch_rgb, ratio_scales=(0.5, 4))

        loss_auto_encoder = None
        rgbs_pseudo = rgbs

        ### Get images and contexts

        device = rgbs[0][0].device
        batch_size = rgbs[0][0].shape[0]

        rgb, rgb_context = break_context(
            rgbs_pseudo, tgt=0, ctx=self.match_context, scl=0, stack=True)

        rgbs0 = {key: val[0] for key, val in rgbs.items()}

        ### Warp pose

        warp_context_pose = [idx for idx in self.warp_context if not is_str(idx)]
        if not self.use_gt_pose:
            pose_warp = self.compute_pose(
                rgbs0, self.networks['pose'],
                ctx=warp_context_pose, invert=True)
        else:
            pose_warp = {key: batch['pose'][key] for key in warp_context_pose}

        pose_warp = self.get_stereo_pose(batch, pose_warp, self.warp_context)
        pose_warp = self.pose_context(pose_warp, self.warp_context)

        ### Match pose

        if self.run_multi:
            match_context_pose = [idx for idx in self.match_context if not is_str(idx)]
            if not self.use_gt_pose:
                with torch.no_grad():
                    pose_match = self.compute_pose(
                        rgbs0, self.networks['pose'],
                        ctx=match_context_pose, invert=True)
            else:
                pose_match = {key: batch['pose'][key] for key in match_context_pose}
            pose_match = self.get_stereo_pose(batch, pose_match, self.match_context)
            pose_match = self.pose_context(pose_match, self.match_context)
        else:
            pose_match = None

        ### Augmentation Mask

        augmentation_mask = torch.zeros([batch_size, 1, 1, 1], device=device).float()
        if self.run_multi:
            if self.training and self.matching_augmentation:
                for batch_idx in range(batch_size):
                    rand_num = random.random()
                    if rand_num < 0.25:
                        rgb_context[batch_idx] = \
                            torch.stack([rgb[batch_idx] for _ in self.match_context], 0)
                        augmentation_mask[batch_idx] += 1
                    elif rand_num < 0.5:
                        pose_match[-1][batch_idx] *= 0
                        augmentation_mask[batch_idx] += 1

        ### Warp cameras

        intrinsics = batch['intrinsics']
        cams_warp = create_cameras(rgbs[0][0], intrinsics, pose_warp)

        ### Monocular depth

        if self.run_mono:

            if self.mono_type == 'multi':
                ctx = [ctx for ctx in self.match_context if curr_stereo(ctx)]
                pose_match2 = {key: val for key, val in pose_match.items() if curr_stereo(key)}
                cams_match2 = create_cameras(rgbs[0][0], intrinsics, pose_match2)
                rgb2, rgb_context2 = break_context(
                    rgbs, tgt=0, ctx=ctx, scl=0, stack=True)
                mono_depth_output = self.networks[mono_depth_string](
                    rgb=rgb2, rgb_context=rgb_context2, cams=cams_match2, intrinsics=intrinsics, mode='multi')
                predictions['depth_lowest_mono'] = {
                    tgt: [inv2depth(mono_depth_output['lowest_cost'].unsqueeze(1)).detach()]}
                predictions['volume_mono'] = {
                    tgt: mono_depth_output['cost_volume']
                }
                predictions['mask_confidence_mono'] = {
                    tgt: [mono_depth_output['confidence_mask'].unsqueeze(1)]
                }
            elif self.mono_type == 'mono':
                mono_depth_output = self.networks[mono_depth_string](
                    rgb=rgb, intrinsics=intrinsics)
            else:
                raise ValueError

            if self.use_gt_depth:
                depth_mono = [batch['depth'][0][:, 0]]
            else:
                depth_mono = mono_depth_output['depths']
                
            predictions['depth_mono'] = {tgt: depth_mono}
        else:
            mono_depth_output = depth_mono = None

        ### Multi-frame depth

        if self.run_multi:

            if self.multi_temporal_only:
                ctx = [ctx for ctx in self.match_context if not curr_stereo(ctx)]
                pose_match3 = {key: val for key, val in pose_match.items() if not is_str(key)}
                cams_match3 = create_cameras(rgbs[0][0], intrinsics[0], pose_match3)
                rgb3, rgb_context3 = break_context(
                    rgbs_pseudo, tgt=0, ctx=ctx, scl=0, stack=True)
                multi_depth_output = self.networks[multi_depth_string](
                    rgb=rgb3, rgb_context=rgb_context3, cams=cams_match3,
                    intrinsics=intrinsics, networks=self.networks,
                )
            else:
                cams_match = create_cameras(rgbs[0][0], intrinsics, pose_match)
                multi_depth_output = self.networks[multi_depth_string](
                    rgb=rgb, rgb_context=rgb_context, cams=cams_match,
                    intrinsics=intrinsics, networks=self.networks,
                )

            if self.use_gt_depth:
                depth_multi = [batch['depth'][0][:, 0]]
            else:
                depth_multi = multi_depth_output['depths']

            predictions['depth_multi'] = {tgt: depth_multi}
            predictions['volume_multi'] = {tgt: multi_depth_output['cost_volume']}
            predictions['depth_lowest_multi'] = {
                tgt: [inv2depth(d.unsqueeze(1)).detach() for d in multi_depth_output['lowest_cost']]}
            predictions['mask_confidence_multi'] = {
                tgt: [multi_depth_output['confidence_mask'].unsqueeze(1)]}

            if 'ssim_lowest_cost' in multi_depth_output:
                predictions['depth_lowest_ssim'] = {
                    tgt: [inv2depth(multi_depth_output['ssim_lowest_cost'].unsqueeze(1)).detach()]}

        else:

            multi_depth_output = depth_multi = None

        ### Confidence mask

        if self.run_multi:
            shape = rgbs0[0].shape[-2:]
            lowest_cost = self.interpolate_nearest(
                multi_depth_output['lowest_cost'][0].unsqueeze(1), size=shape).squeeze(1).to(device)
            confidence_mask = self.interpolate_nearest(
                multi_depth_output['confidence_mask'].unsqueeze(1), size=shape).to(device)
            if self.motion_masking and self.run_mono:

                if 'regression' in multi_depth_output:
                    inv_depth_low_res = multi_depth_output['regression']['disp_pred_low_res']
                    inv_depth_low_res = self.interpolate_nearest(
                        inv_depth_low_res.unsqueeze(1), size=shape).squeeze(1).to(device)
                    lowest_cost = inv_depth_low_res

                matching_depth = 1. / lowest_cost.unsqueeze(1).to(device)
                confidence_mask *= self.conf_mask(matching_depth, depth_mono[0])
                # predictions['mask_confidence'] = {0: [confidence_mask.unsqueeze(1)]}
            confidence_mask = confidence_mask * (1 - augmentation_mask)
        else:
            confidence_mask = None

        ########## LOSSES

        loss, metrics = [], {}
        mono_metrics, multi_metrics = {}, {}
        mono_visuals, multi_visuals = {}, {}

        valid_mask = get_from_dict(batch, 'mask')
        if valid_mask is not None:
            valid_mask = valid_mask[:, 0]

        predictions['output_mono'] = fix_predictions(mono_depth_output)
        predictions['output_multi'] = fix_predictions(multi_depth_output)

        if 'depth_regr' in multi_depth_output:
            predictions['depth_regr'] = {
                (0, 0): [d.unsqueeze(1) for d in multi_depth_output['depth_regr']]
            }

        if 'cal' in self.networks['multi_depth'].networks.keys():
            cal = self.networks['multi_depth'].networks['cal']
            depth1 = depth_multi[0]
            depth2 = predictions['depth_regr'][0][0]
            from knk_vision.vidar.vidar.utils.tensor import interpolate
            depth2 = interpolate(depth2, size=depth1.shape[-2:], scale_factor=None, mode='nearest')
            predictions['depth_regr'][0].insert(0, cal(depth1, depth2, rgb))

        if not self.training:
            return {
                'predictions': predictions,
            }

        gt_depth = None if 'depth' not in batch else batch['depth'][0]

        ### Temporal losses

        cams_warp_temp = {key: val for key, val in cams_warp.items()
                          if not curr_stereo(key) or key == 0}

        if len(cams_warp_temp) > 0 \
                and self.spatial_weight[0] < 1.0 \
                and self.spatio_temporal_weight[0] < 1.0:

            if self.run_mono:
                mono_loss_temp, mono_metrics_temp, mono_visuals_temp = \
                    self.compute_loss_and_metrics(
                        rgbs, depth_mono, cams_warp_temp, logvar=None, valid_mask=valid_mask
                    )
                loss.append((1 - self.spatial_weight[0]) *
                            (1 - self.spatio_temporal_weight[0]) * mono_loss_temp)
                mono_metrics.update(**mono_metrics_temp)
                mono_visuals.update(**{f'temp_{key}': val for key, val in mono_visuals_temp.items()})
                metrics.update({f'mono_temp_{key}': val for key, val in mono_metrics_temp.items()})

        if len(cams_warp_temp) > 0 \
                and self.spatial_weight[1] < 1.0 \
                and self.spatio_temporal_weight[1] < 1.0:

            if self.run_multi:
                multi_loss_temp, multi_metrics_temp, multi_visuals_temp = \
                    self.compute_loss_and_metrics(
                        rgbs, depth_multi, cams_warp_temp, logvar=None,
                        depths_consistency=depth_mono if self.apply_consistency else None,
                        confidence_mask=confidence_mask if self.apply_consistency else None,
                        valid_mask=valid_mask,
                    )
                loss.append((1 - self.spatial_weight[1]) *
                            (1 - self.spatio_temporal_weight[1]) * multi_loss_temp)
                multi_metrics.update(**multi_metrics_temp)
                multi_visuals.update(**{f'temp_{key}': val for key, val in multi_visuals_temp.items()})
                metrics.update({f'multi_temp_{key}': val for key, val in multi_metrics_temp.items()})

        ### Spatial Losses

        cams_warp_spat = {key: val for key, val in cams_warp.items()
                          if curr_stereo(key) or key == 0}

        if len(cams_warp_spat) > 0 \
                and self.spatial_weight[0] > 0.0 \
                and self.spatio_temporal_weight[0] < 1.0:

            if self.run_mono:
                mono_loss_spat, mono_metrics_spat, mono_visuals_spat = \
                    self.compute_loss_and_metrics(
                        rgbs, depth_mono, cams_warp_spat, logvar=None, valid_mask=valid_mask
                    )
                loss.append(self.spatial_weight[0] *
                            (1 - self.spatio_temporal_weight[0]) * mono_loss_spat)
                mono_metrics.update(**mono_metrics_spat)
                mono_visuals.update(**{f'spat_{key}': val for key, val in mono_visuals_spat.items()})
                metrics.update({f'mono_spat_{key}': val for key, val in mono_metrics_spat.items()})

        if len(cams_warp_spat) > 0 \
                and self.spatial_weight[1] > 0.0 \
                and self.spatio_temporal_weight[1] < 1.0:

            if self.run_multi:
                multi_loss_spat, multi_metrics_spat, multi_visuals_spat = \
                    self.compute_loss_and_metrics(
                        rgbs, depth_multi, cams_warp_spat, logvar=None,
                        depths_consistency=depth_mono if self.apply_consistency else None,
                        valid_mask=valid_mask,
                        confidence_mask=confidence_mask if self.apply_consistency else None
                    )
                loss.append(self.spatial_weight[1] *
                            (1 - self.spatio_temporal_weight[1]) * multi_loss_spat)
                multi_metrics.update(**multi_metrics_spat)
                multi_visuals.update(**{f'spat_{key}': val for key, val in multi_visuals_spat.items()})
                metrics.update({f'multi_spat_{key}': val for key, val in multi_metrics_spat.items()})

        ### Spatio-temporal Losses

        if self.spatio_temporal_weight[0] > 0.0:

            if self.run_mono:
                mono_loss_both, mono_metrics_both, mono_visuals_both = \
                    self.compute_loss_and_metrics(
                        rgbs, depth_mono, cams_warp, logvar=None, valid_mask=valid_mask
                    )
                loss.append(self.spatio_temporal_weight[0] * mono_loss_both)
                mono_metrics.update(**mono_metrics_both)
                mono_visuals.update(**{f'both_{key}': val for key, val in mono_visuals_both.items()})
                metrics.update({f'mono_both_{key}': val for key, val in mono_metrics_both.items()})

        if self.spatio_temporal_weight[1] > 0.0:

            if self.run_multi:
                multi_loss_both, multi_metrics_both, multi_visuals_both = \
                    self.compute_loss_and_metrics(
                        rgbs, depth_multi, cams_warp, logvar=None,
                        depths_consistency=depth_mono if self.apply_consistency else None,
                        confidence_mask=confidence_mask if self.apply_consistency else None,
                        valid_mask=valid_mask,
                    )
                loss.append(self.spatio_temporal_weight[1] * multi_loss_both)
                multi_metrics.update(**multi_metrics_both)
                multi_visuals.update(**{f'both_{key}': val for key, val in multi_visuals_both.items()})
                metrics.update({f'multi_both_{key}': val for key, val in multi_metrics_both.items()})

        ###

        if loss_auto_encoder is not None:
            loss.append(loss_auto_encoder)

        if 'depth_regr' in predictions:
            regr_loss, regr_metrics, regr_visuals = \
                self.compute_loss_and_metrics(
                    rgbs, predictions['depth_regr'][0], cams_warp, logvar=None, valid_mask=valid_mask
                )
            loss.append(regr_loss)

        depth_pred = [predictions['depth_regr'][0][0]]
        depth_gt = depth_mono[0].detach()
        supervision_output = self.losses['supervision'](depth_pred, depth_gt)
        loss.append(supervision_output['loss'])

        loss = sum(loss)

        metrics.update({
            'min_depth_bin': self.networks[multi_depth_string].networks['encoder'].min_depth_bin,
            'max_depth_bin': self.networks[multi_depth_string].networks['encoder'].max_depth_bin,
        })

        visuals = {
            **{f'mono_{key}': val for key, val in mono_visuals.items()},
            **{f'multi_{key}': val for key, val in multi_visuals.items()},
        }

        if self.run_mono and self.run_multi and \
                self.training and epoch < self.freeze_teacher_and_pose:
            self.networks[multi_depth_string].networks['encoder'].update_adaptive_depth_bins(depth_mono[0])
            if 'lowest_cost' in mono_depth_output:
                self.networks[mono_depth_string].networks['encoder'].min_depth_bin = \
                    self.networks[multi_depth_string].networks['encoder'].min_depth_bin
                self.networks[mono_depth_string].networks['encoder'].max_depth_bin = \
                    self.networks[multi_depth_string].networks['encoder'].max_depth_bin

        return {
            'loss': loss,
            'metrics': metrics,
            'visuals': visuals,
            'predictions': predictions,
        }

    def compute_loss_and_metrics(self, rgbs, depths, cams, depths_consistency=None,
                                 logvar=None, valid_mask=None, confidence_mask=None):
        """
        Compute model loss and metrics

        Parameters
        ----------
        rgbs : list[torch.Tensor]
            Input RGB images
        depths : list[torch.Tensor]
            Predicted depth maps
        cams : list[Camera]
            Image cameras
        depths_consistency : list[torch.Tensor]
            Depth maps used for consistency loss calculation
        logvar : list[torch.Tensor]
            Predicted log-variance for depth maps
        valid_mask : list[torch.Tensor]
            Valid mask for masking out pixels
        confidence_mask : list[torch.Tensor]
            Confidence mask for consistency calculation

        Returns
        -------
        loss : torch.Tensor
            Final loss
        metrics : Dict
            Dictionary with calculated metrics
        visuals : Dict
            Dictionary with calculated visualizations
        """
        num_scales = self.get_num_scales(depths)

        rgb_tgt = [rgbs[0][i] for i in range(num_scales)]
        rgb_ctx = [[rgbs[j][i] for j in cams.keys() if j != 0] for i in range(num_scales)]

        loss, metrics, visuals = [], {}, {}

        if 'reprojection' in self.losses:
            synth = self.view_synthesis(rgbs, depths, cams, return_masks=True)
            reprojection_mask = multiply_args(valid_mask, confidence_mask)
            reprojection_output = self.losses['reprojection'](
                rgb_tgt, rgb_ctx, synth['warps'], logvar=logvar,
                valid_mask=reprojection_mask, overlap_mask=synth['masks'])
            loss.append(reprojection_output['loss'])
            metrics.update(reprojection_output['metrics'])
            visuals['synth'] = synth
            visuals['reproj'] = reprojection_output

        if 'smoothness' in self.losses:
            smoothness_output = self.losses['smoothness'](rgb_tgt, depths)
            loss.append(smoothness_output['loss'])
            metrics.update(smoothness_output['metrics'])

        if 'consistency' in self.losses and depths_consistency is not None:
            consistency_output = self.losses['consistency'](
                depths_consistency, depths,
                confidence_mask=reprojection_output['mask'],
                valid_mask=valid_mask
            )
            loss.append(consistency_output['loss'])
            metrics.update(consistency_output['metrics'])

        loss = sum(loss)

        return loss, metrics, visuals
