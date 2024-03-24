# Copyright 2023 Toyota Research Institute.  All rights reserved.

from __future__ import absolute_import, division, print_function
from abc import ABC
import itertools
import os
from typing import Dict, List, Any

import torch
import torch.nn as nn

from knk_vision.vidar.vidar.arch.networks.BaseNet import BaseNet
from knk_vision.vidar.vidar.arch.networks.extrinsics.representations.tvec_euler import TvecEulerRepresentation
from knk_vision.vidar.vidar.utils.config import Config, cfg_has


class RelativePoseNet(BaseNet, ABC):
    """ Class of extrinsics handler; forward extrinsics based on the dataset, camera, and sequence.

    Parameters
    ----------
    cfg
        Configuration with parameters
    """

    __CAM_TO_EXTRINMSICS_SEP_KEYWORDS = "_@@OfThe@@_"
    __EXTRISICS_IDENTIFICATOR = "extrinsicsID:"
    __OFFSET_PARAM_KEY = "shared_bias"
    __DEFINED_REPRE2CLASS = {
        "euler_translation": TvecEulerRepresentation,
    }

    def __init__(self, cfg):
        super().__init__(cfg)
        self.total_estimate_num = cfg_has(cfg, "total_estimate_num", 1)  # 1 if stereo camera, 5 if DDAD full learning

        # Bundle scenario and extrinsics pair
        scenario_ids = self._get_scenario_from_dirlist(
            cfg.search_scenario_from_path, cfg.has("remove_scenario", []))  # [["000150"], ["000151"], ..., ]
        self.extrinsics2scenario = {  # {'extrinsicsID:0': '000084',  ...}
            self.__EXTRISICS_IDENTIFICATOR + str(idx): scenario for idx, scenario in enumerate(scenario_ids)}
        self.scenario2extrinsics = self._get_scenario2extrinsics_table(self.extrinsics2scenario)
        self.__scenarios = list(self.scenario2extrinsics.keys())  # Default test scenario

        # Check parametrization method
        self.representation = self.__DEFINED_REPRE2CLASS[cfg.has('param_type', "euler_translation")](cfg)

        # Set Extrinsics to trainable tensor
        self.with_offset = cfg.has('add_shared_bias', False)
        init_std = cfg.has('init_std', 0)
        init_poses = self._get_init_pose_from_yaml(cfg, key='init_transl2euler')

        # Create nn.ParameterDict to storage all trainable param
        self._param_dict = nn.ParameterDict(
            self._create_pose_init(poses=init_poses, std=init_std, offset_per_cam=self.with_offset))

    def __len__(self):
        """Length of all parameters"""
        if self.extrinsics2scenario == {}:
            return len(self.representation) * self.total_estimate_num
        else:
            return len(self.representation) * self.total_estimate_num * len(self.extrinsics2scenario.keys())

    # For constructor process
    def _get_init_pose_from_yaml(self, cfg_: Config, key) -> List[List[Any]]:
        """Set the initial pose if it is set by cfg.key which must shape
            -  if self.extrinsics2scenario != {} --> ScenarioList[ CameraList[ Param[ float ] ] ]
            -  if self.extrinsics2scenario == {} --> CameraList[ Param[ float ] ]
        """
        if self.extrinsics2scenario != {}:
            # Extrinsics must vary from scenario to scenario; np.array(poses).shape == (scene_id, camera_id, 6 or 0)
            init_poses = cfg_has(cfg_, key,
                                 [[[] for _ in range(self.total_estimate_num)]
                                  for _ in range(len(self.extrinsics2scenario))]
                                 )
            assert (len(init_poses) == len(self.extrinsics2scenario) and len(
                init_poses[0]) == self.total_estimate_num), \
                "In consistency between init_transl2euler and scene_id detected"
            pass
        else:
            # Extrinsics consistent to all; np.array(poses).shape == (camera_id, 6 or 0)
            init_poses = cfg_has(cfg_, key, [[] for _ in range(self.total_estimate_num)])
            assert (init_poses and len(init_poses) == self.total_estimate_num), \
                "total_estimate_num must be same to len(init_transl2euler). " \
                "In order not to set the initial position, " \
                "use null list [] in the init_transl2euler. "
            pass
        return init_poses

    @staticmethod
    def _get_scenario2extrinsics_table(extrinsics2scenario: dict) -> dict:
        """ Revert extrinsics2scenario to cenario2extrinsics """
        ret = {}
        for ext_id, scenario_lst in extrinsics2scenario.items():
            for scenario in scenario_lst:
                ret[scenario] = ext_id
        return ret

    def _create_pose_init(self, poses: list, std: float, offset_per_cam: False) -> Dict[str, nn.Parameter]:
        """
        Initialize the extrinsic parameters based on the representation and the number of extrinsic pairs to learn.

        Parameters
        ----------
        poses : List[List[float]] or List[List[List[float]] ]
            The latter case is ScenarioList[ CameraList[ ParamList[ float ] ] ]
        std : float
            Noise to add for initialization
        offset_per_cam : bool
            Flag to decide whether having residual parameter applied for all sequence

        Returns
        -------
        Dict[str, nn.Parameter]
            Extrinsic parameters to feed into nn.ParameterDict()
        """
        trainable_poses = {}
        if self.extrinsics2scenario == {}:
            # len(np.array(poses).shape) must be 2, (camera_id, 6). 6 is dim relative pose from main sensor
            for idx, pose in enumerate(poses):
                cam_id = idx + 1  # idx=0 must be the camera for original point
                trainable = self.representation.get_param(pose_defined_lst=pose, noise_std=std, cam_id=cam_id)
                trainable_poses[str(cam_id)] = trainable
                pass
            pass
        else:
            for scene_num, ext_id in enumerate(self.extrinsics2scenario.keys()):
                # len(np.array(poses).shape) must be 3, (scene_id, camera_id, 6)
                for idx, pose in enumerate(poses[scene_num]):
                    # np.array(pose).shape must be (6,) or (0,)
                    cam_id = idx + 1  # idx=0 must be the camera for original point
                    trainable = self.representation.get_param(pose_defined_lst=pose, noise_std=std, cam_id=cam_id)
                    trainable_poses[str(cam_id) + self.__CAM_TO_EXTRINMSICS_SEP_KEYWORDS + ext_id] = trainable
                    pass
                pass
            pass
        if offset_per_cam:
            for cam_id in range(self.total_estimate_num):
                trainable_poses[
                    str(cam_id + 1) + self.__CAM_TO_EXTRINMSICS_SEP_KEYWORDS + self.__OFFSET_PARAM_KEY] = \
                    self.representation.get_shared_param(camera_id=cam_id)
        return trainable_poses

    @staticmethod
    def _get_scenario_from_dirlist(root_path: str, remove_scenario: list):
        """
        Get scenario directory name each of which gives different extrinsics
        [NOTICE]
        * If scenario is also set by scene_id, it is ignored
        * All scenario is supposed to different extrinsics.
        * If you share the extrinsics to two or more scenario, use scene_id

        Parameters
        ----------
        root_path : str
            The path where the directory name (corresponding to the sequence ID) is
        remove_scenario : List[str]
            Name of  directories that the corresponding extrinsics are not learned
        Returns
        -------
        List[List[str]]
            List of directories each of that has corresponding one extrinsics
            (e.g. If [[], ["000150", "000151"], ..., ], sequence "000150" and "000151" shares same the extrinsics)
        """
        scenario_double_list = [[elem] for elem in sorted(os.listdir(root_path)) if
                                os.path.isdir(os.path.join(root_path, elem))]  # [["000150"], ["000151"], ..., ]
        if remove_scenario:
            ret = []
            for ext_shared_scenario_lst in scenario_double_list:
                include_scenario_lst = [item for item in ext_shared_scenario_lst if item not in remove_scenario]
                if include_scenario_lst:
                    ret.append(include_scenario_lst)
                pass
            pass
        else:
            ret = scenario_double_list
        return ret

    # For private methods
    def _get_transformation(self, scenario_key: str) -> torch.Tensor:
        """
        Create a transformation matrix given the camera location and scenario ID, depending on the param_type

        Parameters
        ----------
        scenario_key : str
            Specify the camera id to extract the tensor from nn.ParameterDict()
            If it's multi scenario, `[cam_id]_@@OfThe@@_extrinsicsID:[scenario_id]` becomes key

        Returns
        -------
        torch.Tensor
            Extrinsics of (1x4x4) tensor
        """
        if self.with_offset:
            only_cam_id = scenario_key.split(self.__CAM_TO_EXTRINMSICS_SEP_KEYWORDS)[0]
            shared_key = only_cam_id + self.__CAM_TO_EXTRINMSICS_SEP_KEYWORDS + self.__OFFSET_PARAM_KEY
        else:
            shared_key = None
        transform = self.representation.get_transformation(keys=[scenario_key, shared_key], src_dict=self._param_dict)
        return transform.squeeze(0)

    def _get_extrinsics_from_scenarios(self, arg_scenario_list: list):
        """
        Given scenario list such as ['000084', '000131', ...], return the key for nn.ParamDict()
        like ['1_@@OfThe@@_extrinsicsID:0', '1_@@OfThe@@_extrinsicsID:1', ..., '2_@@OfThe@@_extrinsicsID:0', ...]
        """
        extrinsics_batch = [self.scenario2extrinsics[scenario] for scenario in arg_scenario_list]
        keys_for_moduledict = [cam + self.__CAM_TO_EXTRINMSICS_SEP_KEYWORDS + ext for cam, ext in
                               itertools.product([str(i) for i in range(1, self.total_estimate_num + 1)],
                                                 extrinsics_batch)]
        return keys_for_moduledict

    def _scenario2tensor(self, arg_scenario_list: list, device="cpu") -> torch.Tensor:
        """
        Given the scenario names as list, return the corresponding extrinsics for all camera

        Parameters
        ----------
        arg_scenario_list : List[str]
            Scenario names. Length is `B`
        device : str
            Device to mount

        Returns
        -------
        torch.Tensor
            Extrinsic tensor of shaping (camxBx4x4)
        """
        b_size = len(arg_scenario_list)
        keys_for_moduledict = self._get_extrinsics_from_scenarios(arg_scenario_list)
        tensors_list = [self._get_transformation(key).to(device)
                        for key in
                        keys_for_moduledict]  # [cam1_b1, cam1_b2, cam1_b3, cam2_b1, cam2_b2, cam2_b3, cam3_b1, ...]
        ret = torch.stack(tensors_list).view(-1, b_size, 4, 4)  # (cam, b, 4, 4)
        return ret

    @staticmethod
    def _insert_tensor(what: torch.Tensor, to: torch.Tensor, where: int) -> torch.Tensor:
        """ Insert the tensor(=what) of the first axis to tensor(=what) following the index(where)"""
        return torch.cat([to[:where], what, to[where:]], 0)

    # Public

    def forward(self, scenario_list=None, tgt_id=0, device="cpu") -> torch.Tensor:
        """
        Gives the homogeneous transformation matrix 4x4 based on the scenario_list.
        Note that extrinsics of the canonical camera is the identity matrix.

        Parameters
        ----------
        scenario_list : list
        tgt_id : int
            0 is only supported
        device : str

        Returns
        -------
        torch.Tensor
             Tensor([total_estimate_num+1, 4, 4]) if not scenario_list, else ([b, total_estimate_num+1, 4, 4])
        """
        if scenario_list is None:
            scenario_list = []
        if (scenario_list == [] and self.extrinsics2scenario == {}):
            # Extrinsics must consistent with all batch
            ret = torch.stack(
                [self._get_transformation(key).to(device) for key in
                 self._param_dict.keys()])  # (total_estimate_num, 4, 4)
            ret = self._insert_tensor(what=torch.eye(4).unsqueeze(0).to(device), to=ret, where=tgt_id)
        elif scenario_list != [] and self.extrinsics2scenario != {}:
            b_size = len(scenario_list)
            # TODO; this process doesn't read the folder name, but read the file ID.
            #  Do not sort the folder for the result consistency
            ret = self._scenario2tensor(scenario_list, device=device)  # (cam-1, b, 4, 4)
            ret = self._insert_tensor(what=torch.eye(4).repeat([1, b_size, 1, 1]).to(ret.device), to=ret,
                                      where=tgt_id).transpose(0, 1)
        else:
            raise NotImplementedError()
        return ret  # (cam, b, 4, 4)

    @property
    def scenarios(self) -> list:
        """Return all scenario which is parametrized"""
        return self.__scenarios

    @scenarios.setter
    def scenarios(self, arg_scenarios: list):
        """Update the scenario to be parametrized"""
        self.__scenarios = arg_scenarios
