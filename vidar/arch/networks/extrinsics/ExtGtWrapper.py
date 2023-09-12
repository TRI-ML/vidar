# Copyright 2023 Toyota Research Institute.  All rights reserved.

import pickle

import numpy as np
import torch

from vidar.arch.networks.layers.selffsm.implemented_dataset_cfg import IMPLEMENTED_DATASET2FRONT_CAM
from vidar.geometry.pose_utils import invert_pose
from vidar.utils.distributed import print0


class ExtGtLoader(object):
    """Ground-truth interface for extrinsics that is once generated and converted to .pkl"""

    def __init__(self, dumped_path_name: str, load_camera_id: tuple, mode="ddad"):
        # load pkl file
        f = open(dumped_path_name, 'rb')
        self.__scenario_and_camera2extrinsics = pickle.load(f)
        f.close()

        # load fundamental information based on the dataset mode
        print0('[INFO] Extrinsics Evaluation on : {}'.format(mode))
        self.main_cam = IMPLEMENTED_DATASET2FRONT_CAM[mode]
        self.camera_ids = load_camera_id
        self.__all_scenario = self._get_all_scenarios()

    def __call__(self, scenario_list: list, batch_info_giver=None):
        """Ground-truth of the extrinsic tensor with (Bxcamx4x4), that is sampled based on the `scenario_list`"""
        return self._get_multi_scenario_tensor(scenario_list, batch_info_giver)

    def _get_all_scenarios(self) -> list:
        """
        Return scenario index such as ['000150', '000151', '000152', ...] from generated extrinscs's keys like
        dict_keys([('000169', 'CAMERA_08'), ('000189', 'CAMERA_07'), ('000171', 'CAMERA_09'), ...])
        """
        return sorted(list(
            set([scenario_cam[0] for scenario_cam in list(self.__scenario_and_camera2extrinsics.keys()) if
                 scenario_cam[1] == self.main_cam])))

    def _get_multi_scenario_tensor(self, scenario_list: list, batch_info_giver: torch.Tensor):
        """Core implementation for __call__()"""
        out = torch.stack([self._scenario2multicam_tensor(scenario=key) for key in scenario_list], 0)
        return out if batch_info_giver is None else out.to(batch_info_giver.device)

    def _scenario2multicam_tensor(self, scenario: str) -> torch.Tensor:
        """
        Load extrinsic tensor from .pkl file given the scenario str, with Tensor[0, 4, 4] is identity
        and the others (Tensor[1:,4,4]) are rebased from the canonical camera, Tensor[0, 4, 4]
        """
        from_tensor = torch.tensor(
            # ndarray -> torch.Tensor is better since Creating a tensor from a list of ndarray is extremely slow.
            np.array([self.__scenario_and_camera2extrinsics[(scenario, self.camera_ids[0])]]))  # (1,4,4)
        to_tensors = [
            torch.tensor(self.__scenario_and_camera2extrinsics[(scenario, self.camera_ids[to_id])]) @ invert_pose(
                from_tensor) for to_id in range(1, len(self.camera_ids))]  # [(1,4,4), (1,4,4), ... ]
        return torch.concat([torch.eye(4).unsqueeze(0)] + to_tensors, 0)

    @property
    def scenarios(self):
        return self.__all_scenario
