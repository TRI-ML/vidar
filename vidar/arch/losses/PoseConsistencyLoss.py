# Copyright 2023 Toyota Research Institute.  All rights reserved.

from abc import ABC
from typing import Dict

import torch
from torch import nn
from torch import linalg as LA

from vidar.utils.config import cfg_has
from vidar.arch.losses.BaseLoss import BaseLoss
from vidar.geometry.pose_utils import \
    (invert_multi_pose, pose_tensor2euler_tensor, pose_tensor2transl_vec, pose_tensor2rotmatrix, invert_pose)


class PoseConsistencyLoss(BaseLoss, ABC):
    """PoseConsistencyLoss class, to calculate the consistency error on ego-motion and extrinsics"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.w_transl = cfg.weight_translation
        self.w_rot = cfg.weight_rotation
        self.rot_loss_type = cfg_has(cfg, "rot_loss_type", "euclid")
        _implemented_loss = (
            "euclid", "geodesic")
        assert self.rot_loss_type in _implemented_loss, "rot_loss_type must be in {}".format(_implemented_loss)
        self.scale_balancing = True
        if self.rot_loss_type == "geodesic":
            self.geodesic = GeodesicLoss()
            pass

        self.l1_loss = nn.L1Loss()

    @staticmethod
    def _sum_squared_error(x1: torch.Tensor, x2: torch.Tensor, func, normalize=False) -> torch.Tensor:
        """Get the Euclidean norm between two vectors"""
        if not normalize:
            return torch.mean(torch.sum((func(x1) - func(x2)).pow(2), 1), 0)  # sum for axis=1, and reduce_mean on batch
        else:
            val = (func(x1) - func(x2)).pow(2)  # (b, 3)
            norm = LA.norm(val, dim=1).unsqueeze(1).clone().detach()  # (b,)
            return torch.mean(val / norm)  # sum for axis=1, and reduce_mean on batch

    @staticmethod
    def _get_coordinate_shifted(extrinsics: torch.Tensor, base_camera_id: int) -> torch.Tensor:
        """Rebase the coordinate of extrinsics, which shapes [b,cam,4,4] Tensor"""
        if base_camera_id == 0:
            return extrinsics
        else:
            base_ext = extrinsics[:, base_camera_id, :, :]  # (B, 4, 4)
            cam_num = extrinsics.shape[1]
            stack_target = [extrinsics[:, i, :, :] @ invert_pose(base_ext) for i in range(cam_num)]
            ret = torch.stack(stack_target).transpose(0, 1)  # (B, cam_id, 4, 4)
            return ret

    def forward(self, pose_dict: Dict[int, torch.Tensor], main2other: torch.Tensor,
                ) -> dict:
        """
        Forward the summation of translation and rotation errors.

        Parameters
        ----------
        pose_dict : Dict[int, torch.Tensor]
            Ego-motion tensor with the key of temporal ID, (Bxcamx4x4)
            Transformation is defined as `{prev_or_next_frame}^T_{current_frame}`, given a camera coordinate.
        main2other : torch.Tensor
            Extrinsic tensor with (Bxcamx4x4).
            Transformation is defined as `{target_camera_id}^T_{front_camera}`
        Returns
        -------
        dict
            Dictionary containing loss and metrics
        """

        pose_dict_after = pose_dict

        metrics = {}

        key_included_in_context = [pose_key for pose_key in list(pose_dict_after.keys()) if pose_key != 0][0]
        b, cam_total, _, __ = pose_dict_after[key_included_in_context].shape
        tag_device = pose_dict_after[key_included_in_context].device
        ctx_to_loop = list(set(pose_dict_after.keys()) - {0})

        rot_err_lst = []  # (cam, ctx)
        transl_err_lst = []  # (cam, ctx)

        for ctx in ctx_to_loop:  # (-2, -1, 1, 2, ...)
            rot_err_per_ctx = []
            transl_err_per_ctx = []

            extrinsics_from_base_cam = self._get_coordinate_shifted(extrinsics=main2other, base_camera_id=0)

            canonical_poses = invert_multi_pose(extrinsics_from_base_cam) @ pose_dict_after[
                ctx] @ extrinsics_from_base_cam  # (b, cam, 4, 4), canonical_poses[:,0] is main camera's one
            for cam_id in range(1, cam_total):
                if self.rot_loss_type == "euclid":
                    rot_err_per_ctx.append(
                        self._sum_squared_error(
                            canonical_poses[:, cam_id], canonical_poses[:, 0],
                            pose_tensor2euler_tensor,
                        )
                    )
                elif self.rot_loss_type == "geodesic":
                    rot_err_per_ctx.append(
                        self.geodesic(pose_tensor2rotmatrix(canonical_poses[:, cam_id]),
                                      pose_tensor2rotmatrix(canonical_poses[:, 0]))
                    )
                    pass
                else:
                    raise NotImplementedError()

                transl_err_per_ctx.append(
                    self._sum_squared_error(
                        canonical_poses[:, cam_id], canonical_poses[:, 0], pose_tensor2transl_vec,
                    )
                )
                pass
            rot_err_lst.append(torch.stack(rot_err_per_ctx, 0))  # (cam, )
            transl_err_lst.append(torch.stack(transl_err_per_ctx, 0))  # (cam, )
            pass

        # Error table which chapes [ctx, cam], from [cams_in_ctx1, cams_in_ctx2, ...]
        rot_err_table = torch.stack(rot_err_lst, 0)  # (ctx, cam)
        transl_err_table = torch.stack(transl_err_lst, 0)  # (ctx, cam)

        # For backprop
        loss = self.w_rot * torch.sum(rot_err_table) + self.w_transl * torch.sum(transl_err_table)

        # For visualize error per camera; sqrt( (e_x^2 + e_y^2 + e_z^2 + )/ (camera * ctx) ) -> [m] or [degree]
        metrics['pose_loss/t[m]'] = torch.sqrt(torch.mean(transl_err_table))
        metrics['pose_loss/rot[rad]'] = torch.sqrt(torch.mean(rot_err_table))

        return {
            'loss': loss.to(tag_device),
            'metrics': metrics,
        }


class GeodesicLoss(nn.Module):
    """ GeodesicLoss; Implementation is following https://github.com/thohemp/6DRepNet """

    def __init__(self, eps=1e-7, debug_mode=False):
        super().__init__()
        self.eps = eps
        self.debug_mode = debug_mode

    def forward(self, m1, m2):
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        theta = torch.acos(torch.clamp(cos, -1 + self.eps, 1 - self.eps))

        if self.debug_mode:
            print(theta)

        return torch.mean(theta)
