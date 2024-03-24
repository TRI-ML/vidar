# Copyright 2023 Toyota Research Institute.  All rights reserved.

from typing import Any, Dict, List

import torch

from camviz.objects.camera import Camera as CVCam

from knk_vision.vidar.vidar.arch.models.releaseSesc.BaseFSM import BaseFSM
from knk_vision.vidar.vidar.arch.networks.layers.selffsm.dataset_interface_method import get_relative_poses_from_base_cam, \
    get_unbroken_extrinsic, run_break_if_not_yet
from knk_vision.vidar.vidar.geometry.camera import Camera
from knk_vision.vidar.vidar.utils.config import cfg_has
from knk_vision.vidar.vidar.utils.distributed import print0
from knk_vision.vidar.vidar.utils.flip import flip_lr
from knk_vision.vidar.vidar.utils.viz import viz_depth

VIZ_CAMERA_SCALE = 1.
VIZ_CAMERA_COLORS = ['red', 'yel', 'gre', 'blu', 'mag', 'cya'] * 100  # Full


class DictDotNotation(dict):
    """ Dictionary class with allowing dot access; link ... https://zenn.dev/kazuhito/articles/dbe6bbf8ce3ef2 """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class SelfCalibFSM(BaseFSM):
    """ SESC: Self-calibration models of Full Surrounding Monodepth (https://arxiv.org/pdf/2308.02153.pdf)

    Parameters
    ----------
    cfg : Config
        Configuration file for model generation
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        self.freeze_posenet = cfg_has(cfg.model, "freeze_posenet", False)
        print0("[INFO] Freeze PoseNet") if self.freeze_posenet else True

        self.freeze_depthnet = cfg_has(cfg.model, "freeze_depthnet", False)
        print0("[INFO] Freeze DepthNet") if self.freeze_depthnet else True

        self.set_attr(cfg.model, 'use_gt_intrinsics', True)  # False if SESC(I)
        print0("[INFO] Learn Intrinsics") if not self.use_gt_intrinsics else True

        self.mono_coeff = cfg_has(cfg.model, "mono_loss_coeff", 1.0)

    def forward(self, batch, epoch=0) -> dict:
        """Forward pass of the model, using a batch of data, finally output the loss and/or predictions"""

        if not self.training:
            if self.use_gt_intrinsics:
                intrinsics = batch['intrinsics'][self.tgt_key]
            else:
                scenarios = self.filename2independent_scenario(batch['filename'])
                cameras = self.filename2camera(batch["filename"])
                intrinsics = self.networks['intrinsics'](scenarios, device=batch['rgb'][self.tgt_key].device,
                                                         camera_list=cameras)
                pass
            ret_depth = self.get_mul_res_depth(rgb=batch['rgb'][self.tgt_key], intrinsics=intrinsics)
            output_new = {
                'predictions': {
                    'depth': {
                        self.tgt_key: [d for d in ret_depth]
                    },
                    'extrinsics_net': self.networks['extrinsics_net']
                }
            }
            return output_new  # End points of demo scripts

        self.set_flip_status(batch_arg=batch)
        losses = []

        out = self.forward_scfsm(batch, epoch)
        broken_keys = out.rgb.keys()

        # Create cams instance
        cams = {key: Camera(out.intrinsics[(0, key[1])], out.rgb[key], out.pred_from_main_cam[key]) for key in
                broken_keys}

        # Loss calculation via comparing
        if "pose_consistency" in self.losses.keys():
            pose_loss = self.losses['pose_consistency'](
                pose_dict=out.unbroken_posenet if not self.use_gt_pose else out.pose_gt,
                # may flipped
                main2other=out.ext_pred_forward,  # always NOT flipped
            )
            losses.append(pose_loss["loss"])

        if "depth_selfsup" in self.losses.keys():
            if self.mono_coeff > 0.:
                mono_loss = self.get_photometric_loss(broken_keys=broken_keys,
                                                      valid_mask_type="if_ddad",
                                                      ctx_generator=self.get_mono_pair,
                                                      broken_depth=out.pred_depth,
                                                      broken_rgb=out.rgb,
                                                      broken_cameras=cams,
                                                      with_smoothness=True)
                losses.append(self.mono_coeff + mono_loss)
                pass
            if self.stereo_coeff > 0.:
                stereo_loss = self.get_photometric_loss(broken_keys=broken_keys,
                                                        valid_mask_type="sky_ground",
                                                        ctx_generator=self.get_valid_stereo_pair,
                                                        broken_depth=out.pred_depth,
                                                        broken_rgb=out.rgb,
                                                        broken_cameras=cams,
                                                        with_smoothness=True,
                                                        use_default_automask=False,  # SfM hypothesis isn't established
                                                        )
                losses.append(stereo_loss * self.stereo_coeff)
                pass

        if self.display:
            self._draw_by_camviz(rgb=out.rgb,
                                 intrinsics=out.intrinsics,
                                 extrinsics_gt=out.extrinsics_gt,
                                 pred_depth=out.pred_depth,
                                 cameras=cams)

        out.clear()
        return {
            'metrics': {},
            'loss': sum(losses),
        }

    def forward_scfsm(self, batch, epoch=0) -> Dict[str, Dict[tuple, Any]]:
        """Forward pass of the model, using a batch of data"""

        rgb = batch['rgb']
        if self.use_gt_intrinsics:
            intrinsics = batch['intrinsics']
        else:
            scnearios = self.filename2scenario(batch['filename'])
            intrinsics = {0: self.networks['intrinsics'](scnearios, device=batch['rgb'][self.tgt_key].device)}
        pose_gt = batch['pose']

        if "extrinsics" in batch.keys():
            # (e.g.) DDAD, PD, ...
            if self.broken:
                bxcamx4x4 = get_unbroken_extrinsic(batch['extrinsics'])
            else:
                bxcamx4x4 = batch['extrinsics'][0]
            extrinsics_gt = get_relative_poses_from_base_cam(bxcamx4x4)  # [b,cam,4,4]
        else:
            # (e.g.) KITTI, ...
            if self.broken:
                unbroken_ext = {0: get_unbroken_extrinsic(batch['pose'])}  # [b,cam,4,4]
            else:
                unbroken_ext = batch['pose']
            extrinsics_gt = self.pose2extrinsics_from_maincam(unbroken_ext)  # [b,cam,4,4]
            pass

        # Before key arrangement, predict pose only for main CAM (if gt_to_scale_injection = True)
        front_cam_temporal_pose = {ctx: pose[:, 0, :, :] for ctx, pose in pose_gt.items()} \
            if not self.broken else {ctx: pose_gt[(ctx, 0)] for ctx in self.ctx_lst}
        computed_pose = self.get_broken_posenet(rgb, pose_freeze=self.freeze_posenet,
                                                gt_to_scale_injection=front_cam_temporal_pose)

        # Mount extrinsics_net to device
        ext_pred_forward = self.networks['extrinsics_net'](
            scenario_list=self.filename2scenario(batch["filename"]),
            device=extrinsics_gt.device)  # (b,cam,4,4), [0,:,:] is identity

        if ext_pred_forward.dtype == torch.float64:
            ext_pred_forward = ext_pred_forward.to(torch.float32)

        # Coordinate transformation; every observation "from" coord to pose[0][:,0]
        # The others are replaced by the PoseNet outputs
        pose_gt_broken = run_break_if_not_yet(pose_gt, 3)
        pred_from_main_cam = self.get_pred_all_extrinsics(
            broken_posenet_out=computed_pose if not self.use_gt_pose else pose_gt_broken,
            extrinsics=ext_pred_forward
        )
        unbroken_posenet = self.get_unbroken_key_pose(computed_pose)

        # Dict keys of which is {target_or_ref_ID: Tensor(b, cam, ch, h1, h2, ...)}
        # to the dict having tuple key like {(t_or_r_ID, camID): (b, ch, h1, h2, ...)}
        rgb = run_break_if_not_yet(rgb, 4)
        intrinsics = run_break_if_not_yet(intrinsics, 3)

        pred_depth = self.get_mul_depth_from_broken(broken_rgb=rgb, broken_intrinsics=intrinsics,
                                                    freeze_depth=self.freeze_depthnet)

        dot_dict = DictDotNotation()
        dot_dict.update(
            {
                "rgb": rgb,  # BrokenDict{ (ctx, cam_id): torch.Tensor(b, 1, h, w) }
                "intrinsics": intrinsics,  # BrokenDict{ (ctx, cam_id): torch.Tensor(b, 3, 3) }
                "pred_depth": pred_depth,  # BrokenDict{ (ctx, cam_id): ScaleList[ torch.Tensor(b, 1, h, w) ] }
                "ext_pred_forward": ext_pred_forward,  # Tensor(b,cam,4,4), parts of `pose_all_predictions`
                "unbroken_posenet": unbroken_posenet,  # Dict[ctx, Tensor(b,cam,4,4)], parts of `pose_all_predictions`
                "pred_from_main_cam": pred_from_main_cam,  # BrokenDict{ (ctx, cam_id): torch.Tensor(b, 4, 4) }
                "pose_gt_broken": pose_gt_broken,  # BrokenDict{ (ctx, cam_id): torch.Tensor(b, 4, 4) }
                "extrinsics_gt": extrinsics_gt,  # BrokenDict{ (ctx, cam_id): torch.Tensor(b, 4, 4) }
            }
        )
        return dot_dict

    def _draw_by_camviz(self,
                        rgb: Dict[tuple, torch.Tensor],
                        intrinsics: Dict[tuple, torch.Tensor],
                        extrinsics_gt: torch.Tensor,
                        pred_depth: Dict[tuple, List[torch.Tensor]],
                        cameras: Dict[tuple, Camera]
                        ):
        """Visualize the learning transition via CamViz"""
        if self.flip_status:
            rgb = flip_lr(rgb)
            pred_depth = flip_lr(pred_depth)
            pass
        points = {}
        for key in self.keys_for_visualization:
            points[key] = cameras[key].reconstruct_depth_map(pred_depth[key][0], to_world=True)

        for i, key in enumerate(self.keys_for_visualization):
            self.draw.addTexture('rgb_%d_%d' % key, rgb[key][0])
            self.draw.addTexture('dep_%d_%d' % key, viz_depth(pred_depth[key][0]))
            self.draw.addBufferf('pts_%d_%d' % key, points[key][0])
            self.draw.addBufferf('clr_%d_%d' % key, rgb[key][0])
        cvcams = {key: CVCam.from_vidar(val, b=0) for key, val in cameras.items()}

        # Only for draw
        cams_gt = {key: Camera(intrinsics[(0, key[1])], rgb[key], extrinsics_gt[:, key[1], :, :]) for key in
                   self.keys_for_visualization}
        cvcams_gt = {key: CVCam.from_vidar(val, b=0,
                                           scale=VIZ_CAMERA_SCALE) for key, val in cams_gt.items()}

        self.draw.clear()
        for i, key in enumerate(self.keys_for_visualization):
            self.draw['rgb_%d' % i].image('rgb_%d_%d' % key)
            self.draw['dep_%d' % i].image('dep_%d_%d' % key)
            # Edge coloring
            self.draw_colored_on_camviz2d(key_name='rgb_%d', index=i, draw_obj=self.draw,
                                          color=VIZ_CAMERA_COLORS[i])
            self.draw_colored_on_camviz2d(key_name='dep_%d', index=i, draw_obj=self.draw,
                                          color=VIZ_CAMERA_COLORS[i])
            self.draw['wld'].object(cvcams[key], tex='rgb_%d_%d' % key, color=VIZ_CAMERA_COLORS[i])
            self.draw['wld'].object(cvcams_gt[key], tex='rgb_%d_%d' % key)
            self.draw['wld'].size(2).color(VIZ_CAMERA_COLORS[i]).points('pts_%d_%d' % key,
                                                                        'clr_%d_%d' % key)
        self.draw.update(30)
        pass
