# Copyright 2023 Toyota Research Institute.  All rights reserved.

import argparse
from collections import OrderedDict
import glob
import os

import torch

from vidar.core.wrapper import Wrapper
from vidar.utils.config import read_config

from vidar.arch.networks.layers.selffsm.metrics import *
from vidar.arch.networks.layers.selffsm.implemented_dataset_cfg import \
    (DATASET2FRONT_IMG_PATH_UNDER_SCENARIO, DATASET2IMG_EXT,
     DATASET2DEFAULT_HW, DATASET2DEFAULT_VIEW_KEY, DATASET2DEFAULT_SCENARIO
     )

from vidar.utils.config import cfg_has
from vidar.utils.data import make_batch


def parser(feed_by_lst=None):
    """Parser for config"""
    parser = argparse.ArgumentParser(description='Project trained model ')
    parser.add_argument('cfg', type=str, help='Data downloaded directory')
    if feed_by_lst is not None:
        args = parser.parse_args(feed_by_lst)
    else:
        args = parser.parse_args()
    return args


class SescInference:
    """
    Wrapper for SESC to visualize the calibrated extrinsics and point clouds

    Parameters
    ----------
    cfg : Config
        Configuration file for model generation
    """

    def __init__(self, cfg, **kwargs):
        super().__init__()

        # Same to the run.py
        os.environ['DIST_MODE'] = 'gpu' if torch.cuda.is_available() else 'cpu'
        cfg = read_config(cfg, **kwargs)
        wrapper = Wrapper(cfg, verbose=True)
        wrapper.eval()

        # Set appropriately
        dataset_mode = "train"
        searched_scenario_from_path = cfg.arch.networks.extrinsics_net.search_scenario_from_path
        rgb_stored_under_scenario = DATASET2FRONT_IMG_PATH_UNDER_SCENARIO[cfg.arch.model.dataset]
        img_extension = DATASET2IMG_EXT[cfg.arch.model.dataset]
        self.ctx = cfg.arch.model.context
        self.use_gt_pose = cfg_has(cfg.arch.model, "use_gt_pose", False)

        # Process
        self.dataset_with_mode = wrapper.datasets[dataset_mode]
        scenarios = self._get_scenario_from_dict(searched_scenario_from_path)

        self.scenario2image_num = self._get_scenario2image(searched_scenario_from_path,
                                                           scenarios, rgb_stored_under_scenario, img_extension
                                                           )
        self.scenario2serial_num = OrderedDict({key: i for i, key in enumerate(scenarios)})
        self.bacth_id_now = None
        self.set_scenario = None
        self.fsm = wrapper.arch  # arch must be inheritance of the vidar.arch.models.perceiver.BaseFSM

        # set variables to pass CamViz
        self.hw = DATASET2DEFAULT_HW[cfg.arch.model.dataset]
        self.keys = DATASET2DEFAULT_VIEW_KEY[cfg.arch.model.dataset]
        self.first_scenario = DATASET2DEFAULT_SCENARIO[cfg.arch.model.dataset]

    def forward_fsm(self, batch: dict) -> dict:
        """Forward pass of the model, using a batch of data"""
        out = self.fsm.forward_scfsm(batch)

        self._print_extrinsics_errors(
            gt_transformation=out.extrinsics_gt[:, 1:, :, :],
            pred_transformation=out.ext_pred_forward[:, 1:, :, :]
        )

        return {
            "rgb": out.rgb,  # BrokenDict{ (ctx, cam_id): torch.Tensor(b, 1, h, w) }
            "intrinsics": out.intrinsics,
            "pred_depth": out.pred_depth,  # BrokenDict{ (ctx, cam_id): ScaleList[ torch.Tensor(b, 1, h, w) ] }
            "pred_pose": out.pred_from_main_cam,  # BrokenDict{ (ctx, cam_id): torch.Tensor(b, 4, 4) }
            "gt_pose": out.pose_gt_broken,  # BrokenDict{ (ctx, cam_id): torch.Tensor(b, 4, 4) }
        }

    def reload_batch(self, scenario: str, to=0) -> dict:
        """ Get batch from the scenario ID such as "000000" and "000150", ... """
        self.set_scenario = self._step_scenario(now_scenario=scenario, to=to)
        self.bacth_id_now = self._scenario2first_batch_idx(scenario=self.set_scenario)
        return self.__reload_batch_from_serial_num(self.bacth_id_now)

    def reload_next_scenario(self):
        """ Return next scenario's batch (e.g. "000016" -> "000017") """
        return self.reload_batch(scenario=self.set_scenario, to=1)

    def reload_previous_scenario(self):
        """ Return next scenario's batch (e.g. "000016" -> "000015") """
        return self.reload_batch(scenario=self.set_scenario, to=-1)

    @staticmethod
    def _get_scenario_from_dict(root_path: str):
        """
        Get scenario directory name each of which gives different extrinsics
        [NOTICE]
        * If scenario is also set by scene_id, it is ignored
        * All scenario is supposed to different extrinsics.
        * If you share the extrinsics to two or more scenario, use scene_id

        Parameters
        ----------
        root_path : str
            Path to get the scenario names

        Returns
        -------
        List[str]
            scenario list like ["000150", "000151", ..., ]
        """
        return [elem for elem in sorted(os.listdir(root_path)) if
                os.path.isdir(os.path.join(root_path, elem))]

    @staticmethod
    def _get_scenario2image(scenario_root: str, scenario_lst: list, under_scenario_path: str, ext=".png"):
        """Sample the image from the scenario, to visualize colored point clouds"""
        all_png_file_num = [len(glob.glob(os.path.join(scenario_root, scenario, under_scenario_path, "*" + ext))) for
                            scenario in scenario_lst]
        return OrderedDict({scenario: rgb_num for scenario, rgb_num in zip(scenario_lst, all_png_file_num)})

    def _step_scenario(self, now_scenario: str, to: int) -> str:
        """ Get the scenario name based on the key input """
        now_scenario_serial = self.scenario2serial_num[now_scenario]
        new_int = min(max(now_scenario_serial + to, 0), len(self.scenario2serial_num) - 1)
        if (new_int == 0 or new_int == len(self.scenario2serial_num) - 1):
            print("Skip scenario slide because it is the last one")
        serial2scenario = {v: k for k, v in self.scenario2serial_num.items()}  # swap the key and val
        next_str = serial2scenario[new_int]
        return next_str

    def _scenario2first_batch_idx(self, scenario: str):
        """
        Get the ID of the first timestep in each scenarion from the serial number of batch, such as
        {'000000': 50, '000001': 100, '000002': 100, }
        """
        scenario_id = list(self.scenario2image_num.keys()).index(scenario)
        stack_image_nums = list(self.scenario2image_num.values())[0:scenario_id]
        scenario_start_id = sum(stack_image_nums) - len(self.ctx) * scenario_id
        return scenario_start_id

    def __reload_batch_from_serial_num(self, idx):
        """Get desired batch based on the serial ID of all batche"""
        return make_batch(self.dataset_with_mode[idx])

    @staticmethod
    def _print_extrinsics_errors(gt_transformation: torch.Tensor, pred_transformation: torch.Tensor):
        """Print extrinsic relating errors on the terminal"""
        gt_tgts = gt_transformation
        pred_tgts = pred_transformation
        print("%%% EXTRINSICS %%%")
        print("GT-norm\t\t", multicam_norm(gt_tgts)[0].data)
        print("Pred-norm\t", multicam_norm(pred_tgts)[0].data)
        print("t\t\t", define_metrics()["t"](gt=gt_tgts, pred=pred_tgts)[0].data)
        print("tn\t\t", define_metrics()["tn"](gt=gt_tgts, pred=pred_tgts)[0].data)
        print("ta\t\t", define_metrics()["ta"](gt=gt_tgts, pred=pred_tgts)[0].data)
        print("ts\t\t", define_metrics()["ts"](gt=gt_tgts, pred=pred_tgts)[0].data)
        print("rot\t\t", define_metrics()["rot"](gt=gt_tgts, pred=pred_tgts)[0].data)

        pass
