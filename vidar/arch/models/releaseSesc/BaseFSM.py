# Copyright 2023 Toyota Research Institute.  All rights reserved.

from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Dict

import numpy as np
import torch

from camviz import Draw
from knk_vision.vidar.vidar.arch.blocks.image.ViewSynthesis import ViewSynthesis
from knk_vision.vidar.vidar.arch.networks.layers.selffsm.metrics import cam_norm
from knk_vision.vidar.vidar.arch.networks.layers.selffsm.dataset_interface_method import \
    (get_scenario, get_camname, get_cam0_based_extrinsics_from_pose,
     get_scenario_independent_camera, get_ddad_self_occlusion_mask, get_str_from_broken)
from knk_vision.vidar.vidar.arch.models.BaseModel import BaseModel
from knk_vision.vidar.vidar.arch.models.utils import apply_rgb_mask
from knk_vision.vidar.vidar.geometry.pose_utils import get_scaled_translation, pose_vec2mat_homogeneous
from knk_vision.vidar.vidar.utils.config import Config, cfg_has
from knk_vision.vidar.vidar.utils.distributed import print0
from knk_vision.vidar.vidar.utils.flip import flip_lr, flip_pose_lr


def t_euler_2t_quat(xyz_rpy=None) -> np.ndarray:
    """Convert the List of camera position [x,y,z,roll,pitch,yaw] to [x,y,z,qx,qy,...,]"""
    if xyz_rpy is None:
        xyz_rpy = [0.3, -15.5, -1.0, -1.57, 0., 0., ]  # seq32, failuer case bird view
    return pose_vec2mat_homogeneous(torch.tensor(xyz_rpy).unsqueeze(0)).squeeze(
        0).detach().cpu().numpy()  # [x,y,z,qx,qy, ..., ]


class BaseFSM(BaseModel, ABC):
    """Generic model for Multi-camera monodepth learning

    Parameters
    ----------
    cfg : Config
        Configuration file for model generation
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        # Debug option
        self.display = False
        self.use_gt_pose = cfg_has(cfg.model, "use_gt_pose", False)
        self.window_scale = 1.
        if cfg_has(cfg.model, "debug", None) is not None:
            self.display = cfg_has(cfg.model.debug, "display", False)
            self.keys_for_visualization = cfg_has(cfg.model.debug, "keys_for_visualization",
                                                  [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])
            if type(self.keys_for_visualization[0]) != tuple:  # Make the key hashable
                self.keys_for_visualization = [tuple(element) for element in self.keys_for_visualization]
            print0("###### Visualization keys; {}".format(self.keys_for_visualization)) if self.display else True
            self.window_scale = cfg_has(cfg.model.debug, "window_scale", 1.)
            pass
        print0("[INFO] Use GT Pose") if self.use_gt_pose else True

        # For data argumentation
        self.flip_status = False

        # Load configuration
        self.camera_physical_layout_pairs = OrderedDict(
            [(key, val) for key, val in enumerate(cfg.model.camera_neighboring_pairs)])
        self.view_synthesis = ViewSynthesis(Config(upsample_depth=True))
        self.posenet_pair_key = \
            self._get_posenet_key_pairs(
                camera_num=len(cfg.model.cameras),
                contexts=cfg.model.context)
        self.ctx_lst = cfg.model.context
        self.stereo_coeff = cfg_has(cfg.model, "stereo_loss_coeff", 0.5)
        ddad_mask_path = cfg_has(cfg.model, "ddad_mask_path", "")
        self.self_occlusion_mask_tensor = None
        if ddad_mask_path != "":
            h_, w_ = cfg_has(cfg.model, "mask_h_w", [384, 640])
            self.self_occlusion_mask_tensor = get_ddad_self_occlusion_mask(ddad_mask_path, h=h_, w=w_)
            pass
        self.gt_to_scale_injection = cfg_has(cfg.model, "gt_to_scale_injection", False)

        # key configurations
        self.tgt_key = 0 if cfg.model.has('broken_key') and not cfg.model.broken_key else (0, 0)
        self.broken = cfg.model.has("broken_key", True)

        # Visualization (for debug)
        if self.display:
            self.hw = cfg_has(cfg.model.debug, "viz_size", [192, 320])
            wh = self.hw[::-1]
            self.window_wh = (1600, 1200)
            self.draw = Draw(self.window_wh)
            if self.window_scale < 1.0:
                self.draw.setSize((int(self.draw.wh[0] * self.window_scale), int(self.draw.wh[1] * self.window_scale)))
            self.draw.add3Dworld(
                'wld', (0.50, 0.00, 1.00, 1.00),  # ALL PCL area
                pose=t_euler_2t_quat([0.3, -20.5, -1.0, -1.57, 0., 0., ])
            )

            self.draw.add2Dimage('rgb_0', (0.00, 0.00, 0.25, 0.16), res=wh)
            self.draw.add2Dimage('rgb_1', (0.00, 0.16, 0.25, 0.33), res=wh)
            self.draw.add2Dimage('rgb_2', (0.00, 0.33, 0.25, 0.50), res=wh)
            self.draw.add2Dimage('rgb_3', (0.00, 0.50, 0.25, 0.67), res=wh)
            self.draw.add2Dimage('rgb_4', (0.00, 0.67, 0.25, 0.83), res=wh)
            self.draw.add2Dimage('rgb_5', (0.00, 0.83, 0.25, 1.00), res=wh)

            self.draw.add2Dimage('dep_0', (0.25, 0.00, 0.50, 0.16), res=wh)
            self.draw.add2Dimage('dep_1', (0.25, 0.16, 0.50, 0.33), res=wh)
            self.draw.add2Dimage('dep_2', (0.25, 0.33, 0.50, 0.50), res=wh)
            self.draw.add2Dimage('dep_3', (0.25, 0.50, 0.50, 0.67), res=wh)
            self.draw.add2Dimage('dep_4', (0.25, 0.67, 0.50, 0.83), res=wh)
            self.draw.add2Dimage('dep_5', (0.25, 0.83, 0.50, 1.00), res=wh)
        else:
            self.draw = None

        # Dataset specific method
        print0("[INFO] Mode: {}".format(cfg.model.dataset))
        if cfg.model.dataset == "ddad":
            self.filename2scenario = partial(get_scenario, id_where_scenario_is=0, id_where_cam_defined=2)
            self.filename2camera = partial(get_camname, id_where_cam_defined=2)
            self.filename2independent_scenario = partial(get_scenario_independent_camera, id_where_scenario_is=0)
            if self.broken:
                self.filename2scenario = partial(get_str_from_broken, id_where_dest_is=-4)
                self.filename2camera = partial(get_str_from_broken, id_where_dest_is=-2)
                self.filename2independent_scenario = partial(get_str_from_broken, id_where_dest_is=-4)
        elif cfg.model.dataset == "pd":
            self.filename2scenario = partial(get_scenario, id_where_scenario_is=0, id_where_cam_defined=2)
        elif cfg.model.dataset == "vkitti2":
            self.filename2scenario = partial(get_scenario, id_where_scenario_is=1, id_where_cam_defined=-2)
        elif cfg.model.dataset == "kitti":
            self.filename2scenario = partial(get_scenario, id_where_scenario_is=0, id_where_cam_defined=2)
            if self.broken:
                self.filename2scenario = partial(get_str_from_broken, id_where_dest_is=-5)
        else:
            raise NotImplementedError("{} was not implemented dataset".format(cfg.model.dataset))

    @abstractmethod
    def forward(self, batch, epoch=0) -> dict:
        return {}

    def set_flip_status(self, batch_arg: dict):
        """ Identify whether the batch input is flipped or not by batch['flipped'] """
        self.flip_status = True if 'flipped' in batch_arg.keys() else False
        pass

    def pose2extrinsics_from_maincam(self, unbroken_pose: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Get relative poses from canonical camera, that shapes (Bxcamx4x4), by converting the batch['pose']"""
        if self.flip_status:
            pose_est = flip_pose_lr(unbroken_pose)
            pass
        else:
            pose_est = unbroken_pose
            pass
        current_frames = pose_est[0]
        return get_cam0_based_extrinsics_from_pose(current_frames)

    def get_broken_posenet(self, rgb: dict, pose_freeze=False,
                           gt_to_scale_injection: Dict[int, torch.Tensor] = None) -> Dict[tuple, torch.Tensor]:
        """
        Get dictionary of PoseNet, the key of which is broken_key and value is NN's raw output

        Parameters
        ----------
        rgb : Dict[tuple, torch.Tensor] or Dict[int, torch.Tensor]
            The dictionary of images; the former shapes (Bx4x4) and the latter shapes (Bxcamx4x4)
        pose_freeze : bool
            Boolean identifier to determine if the present is frozen
        gt_to_scale_injection : Dict[int, torch.Tensor]
            Ground-truth ego-motion vectors such as {-1: (b,4,4)}, that will be used for scaling the translation.

        Returns
        -------
        Dict[tuple, torch.Tensor]
            Tuple-keyed dictionary; the key is (Context ID (!=0), Camera ID), and the value is Tensor(Bx4x4)

        """
        posenet_out_tgt_id_lst = self._get_tgt_camera_id()
        if pose_freeze:
            self.networks['pose'].eval()
        broken_key2pose = {}
        for cam_id in posenet_out_tgt_id_lst:  # Select camera ID from
            if not self.broken:
                rgb_to_feed_posenet: Dict[int: torch.Tensor] = {
                    k: rgb[k][:, cam_id, :, :, :] if not self.flip_status else flip_lr(
                        rgb[k][:, cam_id, :, :, :]) for k, _ in rgb.items()}
            else:
                rgb_to_feed_posenet: Dict[int: torch.Tensor] = {
                    k: rgb[(k, cam_id)] if not self.flip_status else flip_lr(
                        rgb[(k, cam_id)]) for k in [0] + sorted(self.ctx_lst)}
            # Specify "previous" to "current" frame everytime by invert=True
            posenet_out = self.compute_pose(  # self.compute_pose() => Only get "transformation"
                rgb_to_feed_posenet, self.networks['pose'], tgt=0, invert=True)  # {ctx_ID:(b, 4, 4)} }
            for ctx, tensor in posenet_out.items():
                if not self.gt_to_scale_injection:
                    broken_key2pose[(ctx, cam_id)] = tensor
                else:
                    posenet_norm = cam_norm(tensor)
                    scale_from_gt = cam_norm(gt_to_scale_injection[ctx])  # (B, )
                    broken_key2pose[(ctx, cam_id)] = get_scaled_translation(transformation=tensor,
                                                                            multiply=scale_from_gt / posenet_norm)
                    pass
                pass
            pass
        return broken_key2pose

    def _get_tgt_camera_id(self) -> tuple:
        """(0, 1, 2) if self.posenet_pair_key == [(-1, 0), (-1, 1), (-1, 2), (1, 0), (1, 1), (1, 2)]"""
        return tuple(set([item[1] for item in self.posenet_pair_key]))

    def get_unbroken_key_pose(self, broken_posenet_out: dict) -> dict:
        """
        Reshape PoseNet output to batch["pose"] style, such as {ctx: torch.Tensor(Bxcamx4x4)}
        from broken_key style like {(ctx, cam): torch.Tensor(b, 4, 4)
        """
        ret = {}
        posenet_settable_ids = list(dict.fromkeys([pair_key[0] for pair_key in self.posenet_pair_key]))
        for ctx in posenet_settable_ids:
            ret[ctx] = torch.stack([  # ctx = -1, 1, ...
                broken_posenet_out[(ctx, cam_id)] for cam_id
                in self._get_tgt_camera_id()]).transpose(0, 1)  # (cam, b, 4, 4) to (b, cam, 4, 4)
            pass
        return ret

    @staticmethod
    def _get_posenet_key_pairs(camera_num: int, contexts: list) -> list:
        """Generate pairs to be fed into PoseNet. PoseNet must be applied to all camera and all contexts"""
        return [(context, cam_id) for cam_id in range(camera_num) for context in contexts]

    @staticmethod
    def _get_projected_posenet_by_extrinsics(broken_posenet_out: dict,
                                             extrinsics: torch.Tensor,
                                             ) -> dict:
        """
        Rebase the coordinate of ego-motion by the extrinsics

        Parameters
        ----------
        broken_posenet_out : Dict[tuple, torch.Tensor]
           Tensor of the ego-motion (Bx4x4) with the key of tuple, (temporal_ID, camera_ID)
        extrinsics : torch.Tensor
           Relative poses from canonical camera, that shapes (Bxcamx4x4)

        Returns
        -------
        Dictionary : Dict[tuple, torch.Tensor]
            broken_posenet_out, the version of all base coordinates is shared
        """
        ret = {}
        for ctx, cam_id in broken_posenet_out.keys():
            ret[(ctx, cam_id)] = broken_posenet_out[(ctx, cam_id)] @ extrinsics[:, cam_id, :, :]
        return ret

    def get_pred_all_extrinsics(self, broken_posenet_out: dict, extrinsics: torch.Tensor, ) -> dict:
        """Get all relative pose from main camera, in a {(Temporal ID, Camera ID): Pose(b,4,4)} } style"""
        projected_out = self._get_projected_posenet_by_extrinsics(
            broken_posenet_out=broken_posenet_out,
            extrinsics=extrinsics  # (cam, 4, 4), [0,:,:] is identity
        )  # get from
        for cam_id in self._get_tgt_camera_id():
            projected_out.update({
                (0, cam_id): extrinsics[:, cam_id, :, :]}
            )  # extrinsics must be current
        return projected_out

    def get_mul_res_depth(self, rgb: torch.Tensor, intrinsics: torch.Tensor) -> list:
        """ forward the depth network with multi-resolution, that shapes ScaleList[torch.Tensor([Bx1xHxW])]"""
        return self.networks['depth'](rgb=rgb, intrinsics=intrinsics)['depths']

    def get_mul_depth_from_broken(self, broken_rgb, broken_intrinsics: dict,
                                  freeze_depth=False) -> Dict[tuple, list]:
        """
        Return the monodepth from the `current rgb frames` with keys like (0,0), (0,2), ...

        Parameters
        ----------
        broken_rgb : Dict[tuple, torch.Tensor]
            The tensor of input images (BxCxHxW) with the key of tuple, (temporal_ID, camera_ID)
        broken_intrinsics : Dict[tuple, torch.Tensor]
            The tensor of intrinsics (Bx3x3) with the key of tuple, (temporal_ID, camera_ID)
        freeze_depth : bool
            Identifier whether depth network is .eval() mode or not

        Returns
        -------
        Dict[tuple, List[torch.Tensor]]
            Depth prediction with multi-resolution; List[torch.Tensor] = [Tensor(Bx1xHxW), Tensor(Bx1xH/2xW/2), ...]
        """
        current_keys = sorted([key for key in broken_rgb.keys() if key[0] == 0])
        current_rgb = OrderedDict({key: broken_rgb[key] for key in current_keys})
        b = [val.shape[0] for val in current_rgb.values()]  # [b, ...], which length is (spatio-temporal camera pairs)
        all_rgb = torch.cat([val for val in current_rgb.values()], 0)  # create rgbs that shapes (b*p, ch, h, w)
        curr_intrinsics = OrderedDict({key: broken_intrinsics[key] for key in current_keys})
        all_intrinsics = torch.cat([val for val in curr_intrinsics.values()], 0)  # create (b*p, 4,4)
        if freeze_depth:
            self.networks['depth'].eval()
            pass
        out = self.networks['depth'](rgb=all_rgb, intrinsics=all_intrinsics)['depths']  # ScaleList[ (b*p, 1, h, w) ]
        out = [torch.split(d, b) for d in out]  # ScaleList[ PairsList[ Torch.Tensor(b, 1, H, W) ] ]
        out = [[d[i] for d in out] for i in range(len(out[0]))]  # PairsList[ ScaleList[Torch.Tensor(b, 1, H, W)] ]
        return OrderedDict({key: out[i] for i, key in enumerate(current_keys)})

    @staticmethod
    def _get_current_frame_keys(broken_key: list) -> list:
        """Return (ctx=0, cam) keys from broken_key, in order only to select current frames"""
        return [item for item in broken_key if item[0] == 0]

    def get_valid_stereo_pair(self, arg_ctx: list, arg_tgt: tuple, only_current=False) -> list:
        """Get the stereo pair indices for photometric loss considering the physical location from `arg_ctx`"""
        stereo_ctx = [key for key in arg_ctx if key[1] != arg_tgt[1]]
        if self.camera_physical_layout_pairs != {}:
            stereo_ctx = [key for key in stereo_ctx if key[1] in self.camera_physical_layout_pairs[arg_tgt[1]]]
        return stereo_ctx if not only_current else self._get_current_frame_keys(stereo_ctx)

    @staticmethod
    def get_mono_pair(arg_ctx: list, arg_tgt: tuple, **kwargs) -> list:
        """Mono-context keys by given arg_tgt, from `arg_ctx`"""
        return [key for key in arg_ctx if key[1] == arg_tgt[1]]  # Mono == Share camera ID

    def get_photometric_loss(self,
                             broken_keys: list,
                             ctx_generator,
                             valid_mask_type: str,
                             broken_depth: dict,
                             broken_rgb: dict,
                             broken_cameras: dict,
                             with_smoothness=False,
                             use_default_automask=True,
                             ) -> torch.Tensor:
        """
        Get photometric & smoothness loss, by the given configurations and tensor, and keys

        Parameters
        ----------
        broken_keys : List[tuple]
            Every key handled in the forward function
        ctx_generator
            Class method which gives corresponding context ID from target ID
        valid_mask_type : str
            Specify the masking type, such as "if_ddad" or "sky_ground"
        broken_depth : Dict[tuple, List[torch.Tensor]]
            Depth prediction with multi-resolution
        broken_rgb: Dict[tuple, torch.Tensor]
            Input images (BxCxHxW) with the key of tuple
        broken_cameras : Dict[tuple, Camera]
            Camera instances (moduled in vidar.geometry.camera) with the key of tuple
        with_smoothness : bool
            Flag to decide whether applying the smoothing loss for depth prediction
        use_default_automask : bool
            Flag to dynamically change whether applying the auto-mask or not, that is proposed in Monodepth2

        Returns
        -------
        torch.Tensor
            Loss of all summed.
        """
        loss_sum = 0
        tgt_candidate = self._get_current_frame_keys(broken_keys)
        for tgt in tgt_candidate:
            ctx_candidate = [key for key in broken_keys if key != tgt]
            ctx_keys = ctx_generator(arg_ctx=ctx_candidate, arg_tgt=tgt)
            depths_tgt = broken_depth[tgt]  # ScaleList[Torch.Tensor(b, 1, H, W)]
            num_scales = self.get_num_scales(depths_tgt)
            loss_sum += self._photometric_and_smooth_loss(tgt, ctx_keys, valid_mask_type,
                                                          broken_rgb, broken_cameras, depths_tgt,
                                                          num_scales, with_smoothness, use_default_automask,
                                                          )
        return loss_sum

    def _photometric_and_smooth_loss(self, tgt: tuple, ctx_keys: list, valid_mask_type: str,
                                     broken_rgb: dict, broken_cameras: dict, depths_tgt: list,
                                     num_scales, with_smoothness: bool, use_default_automask: bool,
                                     ):
        """ Core implementation of self.get_photometric_loss() """
        if self.flip_status:  # RGB is flipped, but PoseNet is not --> Make the original image
            broken_rgb = flip_lr(broken_rgb)
            depths_tgt = flip_lr(depths_tgt)
            pass
        cameras = {key: broken_cameras[key] for key in [tgt] + ctx_keys}
        synth = self.view_synthesis(
            rgb=broken_rgb,
            ctxs=ctx_keys,
            depths=depths_tgt,
            cams=cameras,
            return_masks=True, tgt=tgt)
        overlap_mask = synth['masks']
        if valid_mask_type == "if_ddad" and self.self_occlusion_mask_tensor is not None:
            mask_rgb = self.ddad_valid_mask(broken_tgt_key=tgt,
                                            batch_info_giver=broken_rgb[tgt],
                                            num_scales=num_scales) \
                if self.self_occlusion_mask_tensor is not None else None
            overlap_mask = apply_rgb_mask(synthesised_masks=synth['masks'], mask_rgb_tgt=mask_rgb)
        elif valid_mask_type == "sky_ground":
            mask_rgb = self.sky_and_ground_mask(
                ref_tensor=broken_rgb[tgt],
                num_scales=num_scales,
                top_cut=0.15,
                bottom_cut=0.75)
            mask_rgb = flip_lr(mask_rgb, flip=self.flip_status)
            overlap_mask = apply_rgb_mask(synthesised_masks=synth['masks'], mask_rgb_tgt=mask_rgb)
        else:
            pass
        loss = \
            self.losses['depth_selfsup'](
                broken_rgb[tgt], broken_rgb, synth['warps'], overlap_mask=overlap_mask,
                use_default_automask=use_default_automask)['loss']
        if (with_smoothness and "smoothness" in self.losses.keys()):
            loss += self.losses['smoothness'](rgb=broken_rgb[tgt], depth=depths_tgt)["loss"]
        return loss

    def draw_colored_on_camviz2d(self, draw_obj: Draw, key_name: str, index: int, width=9, pad=5, color='red'):
        """Draw a colored rectangle around the 2D visualizations on the CamViz window."""
        draw_obj[key_name % index].color(color).width(width).loop(np.array([
            [pad, pad],
            [pad, self.hw[0] - pad],
            [self.hw[1] - pad, self.hw[0] - pad],
            [self.hw[1] - pad, pad]
        ]))
        pass

    def ddad_valid_mask(self, broken_tgt_key: tuple, batch_info_giver: torch.Tensor, num_scales: int):
        """Get ScaleList[torch.Tensor(Bx1xHxW)] for self-occlusion mask if dataset is DDAD"""
        cam_id = broken_tgt_key[1]
        b_, dev_ = batch_info_giver.shape[0], batch_info_giver.device
        mask_tensor = self.self_occlusion_mask_tensor[cam_id].repeat([b_, 1, 1, 1]).to(dev_)
        return [mask_tensor for _ in range(num_scales)]

    @staticmethod
    def sky_and_ground_mask(ref_tensor: torch.Tensor, num_scales: int, top_cut: float,
                            bottom_cut: float) -> list:
        """
        Get a masking tensor for the top edge and bottom edge,  which shapes ScaleList[torch.Tensor].
        The`top_cut`x100 percent top area and `bottom_cut`x100 percent bottom area are removed, so replaced by 0

        Parameters
        ----------
        ref_tensor : torch.Tensor
            Reference giver of the type and batch_size
        num_scales : int
            Number of scales from depth predictions
        top_cut :
            Masking ratio to top-part
        bottom_cut
            Masking ratio to bottom-part

        Returns
        -------
        List[torch.Tensor]
            Masking tensor which shapes [Tensor(Bx1xHxW), Tensor(Bx1xH/2xW/2), ...]
        """
        size = ref_tensor.shape
        b_ch_h_w = torch.ones([size[0], 1, size[2], size[3]])  # (B, 1, h, w)
        h_st, h_fn = int(top_cut * size[2]), int(bottom_cut * size[2])
        b_ch_h_w[:, :, :h_st, :] = 0
        b_ch_h_w[:, :, h_fn:, :] = 0
        return [b_ch_h_w.to(ref_tensor.device) for _ in range(num_scales)]
