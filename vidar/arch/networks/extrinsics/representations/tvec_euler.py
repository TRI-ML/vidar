# Copyright 2023 Toyota Research Institute.  All rights reserved.

from typing import List

import torch
import torch.nn as nn

from knk_vision.vidar.vidar.arch.networks.extrinsics.utils import get_noised_init_vec
from knk_vision.vidar.vidar.geometry.pose_utils import pose_vec2mat_homogeneous
from knk_vision.vidar.vidar.utils.config import Config
from knk_vision.vidar.vidar.utils.distributed import print0


class TvecEulerRepresentation:
    """Representation class for parametrizing extrinsics

    Parameters
    ----------
    cfg : Config
         Configuration with parameters
    """

    __PI = 3.1415926
    __DIM = 6

    def __init__(self, cfg: Config):
        # Property always exist
        self._pose_dim = self.__DIM

        # common properties from argument
        self._back_cam_id = cfg.has('back_camera_id', [])
        self._detach_shared = cfg.has('fix_shared', False)
        self._detach_translation = cfg.has('detach_translation', False)
        self._shared_init_vars: Config = cfg.has('shared_init_params', None)
        self.multiply_to_transl_per_scenario = cfg.has('multiply_to_transl_per_scenario', None)
        self.dtype = self._str2torch_dtype(cfg.has('dtype', None))

        # print info
        print0("[INFO] Detach shared param") if self._detach_shared else True
        print0("[INFO] Camera {} looking the opposite from Main Camera".format(self._back_cam_id))
        print0("[INFO] Detach Translation Vector") if self._detach_translation else True
        print0("[WARN] Rotation initialization is NOT implemented") if self._shared_init_vars is not None else True
        print0("[WARN] Translation per scenario multiplied by {}".format(
            self.multiply_to_transl_per_scenario)) if self.multiply_to_transl_per_scenario is not None else True

        # returns
        self._eulers = None

        pass

    @staticmethod
    def _str2torch_dtype(arg_str):
        """Print warning message for rotation precision evaluation"""
        if arg_str is None:
            print0("[WARN] float32 will be use, it severely does the accuracy harm for Geodesic Error")
            return None
        else:
            ret = getattr(torch, arg_str)
            print0("[INFO] Specify dtype to extrinsics: {}".format(ret))
            return ret

    def __len__(self):
        """Return total parameter length per each element"""
        return self._pose_dim

    def detach_translaiton(self, pose_tensor: torch.Tensor):
        """ Detach translation from the 6D vector [translation, euler] """
        return torch.concat([pose_tensor[:3].detach().clone(), pose_tensor[3:]])

    def get_param(self, pose_defined_lst: List[float], noise_std: float, cam_id: int) -> nn.Parameter:
        """
        Convert yaml specified poses to torch.Tensor unless they are null list. If null, .zeros() is chosen

        Parameters
        ----------
        pose_defined_lst
            List to set initial parameter
        noise_std
             Noise to add for initial parameters
        cam_id
            Camera ID to set the desired initial pose for back propagation

        Returns
        -------
         nn.Parameter
            6-dim parameters, [translation (3), Euler angles(3)]
        """
        if pose_defined_lst:
            nn_params = \
                get_noised_init_vec(
                    translation=pose_defined_lst[:3],
                    ang_representation=pose_defined_lst[3:],
                    std=noise_std,
                )
        else:
            # If no initialization for extrinsics
            nn_params = \
                get_noised_init_vec(
                    translation=[0., 0., 0.],
                    ang_representation=[0., 0., 0.
                                        ] if not cam_id in self._back_cam_id else [self.__PI, 0., self.__PI],
                    std=noise_std,
                )
        return nn_params

    def get_1_from_init_param(self, cam_id: int, variable_key: str = "mean", coord: str = "x") -> float:
        """Get statistical values per coordinate for initialization."""
        cfg = getattr(self._shared_init_vars, variable_key)  # (e.g.) shared_init_params.mean
        return getattr(cfg, coord)[cam_id] if cfg.has(coord) else 0.  # (e.g.) if shared_init_params.mean.has("x")

    def get_shared_param(self, camera_id: int = 0, **kwargs) -> nn.Parameter:
        """Initialized the residual part of parameters"""
        if self._shared_init_vars is None:
            return nn.Parameter(torch.zeros(self._pose_dim), requires_grad=True)
        else:
            if self._shared_init_vars.has("mean"):
                t_mean = [self.get_1_from_init_param(
                    cam_id=camera_id, variable_key="mean", coord=coord) for coord in ("x", "y", "z")]
            else:
                t_mean = [0., 0., 0.]
            if self._shared_init_vars.has("std"):
                t_std = [self.get_1_from_init_param(
                    cam_id=camera_id, variable_key="std", coord=coord) for coord in ("x", "y", "z")]
            else:
                t_std = [0., 0., 0.]
            n_sigma = self._shared_init_vars.has("n_sigma", 0.)
            init_translation = torch.normal(torch.tensor(t_mean),
                                            n_sigma * torch.tensor(t_std))  # [x,y,z]
            return nn.Parameter(torch.concat([init_translation, torch.zeros(3)]),
                                requires_grad=True)  # since translation does not support

    def get_transformation(self,
                           keys: List[str],
                           src_dict: nn.ParameterDict,
                           ) -> torch.Tensor:
        """Return the homogeneous transformation matrix from nn.ParameterDict by keys"""
        scenario_key, shared_key = keys[0], keys[1]
        pose_base = src_dict[scenario_key]  # nn.Parameter shapes ([6,])
        if self.multiply_to_transl_per_scenario is not None:
            pose_base = pose_base * self.multiply_to_transl_per_scenario
            pass
        if shared_key is not None:
            offset_ = src_dict[shared_key]
            if self._detach_shared:
                pose_base = pose_base + offset_.detach().clone()
            else:
                pose_base = pose_base + offset_
            pass
        if self._detach_translation:
            pose_base = self.detach_translaiton(pose_base)
        pose_vec = pose_vec2mat_homogeneous(pose_base.unsqueeze(0), dtype=self.dtype)
        return pose_vec  # transformation matrix shapes Tensor[4,4]
