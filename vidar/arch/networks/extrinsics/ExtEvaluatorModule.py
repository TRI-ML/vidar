# Copyright 2023 Toyota Research Institute.  All rights reserved.

from collections import OrderedDict
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from vidar.arch.networks.extrinsics.ExtGtWrapper import ExtGtLoader
from vidar.arch.networks.layers.selffsm.implemented_dataset_cfg import CAMERA_NUMERIC2STR_KEY, IGNORE_SCENES
from vidar.arch.networks.layers.selffsm.metrics import define_metrics
from vidar.utils.distributed import print0


def get_extrinsics_eval_ignore_scenarios(dataset: str) -> list:
    """Based on the IGNORE_SCENES and dataset name, out-of-evaluation scenarios are chosen."""
    return IGNORE_SCENES[dataset] if dataset in IGNORE_SCENES.keys() else []


class ExtComparisonWithPkl:
    """Evaluation interface for extrinsic estimation by the comparison with pre-generated ground-truth (.pkl)"""

    def __init__(self,
                 dump_pkl_name: str,
                 dataset_name: str,
                 cameras: List[int],
                 metrics: list,
                 dst_dir_name="",
                 loop_batch_size=8,
                 fig_annotate=False,
                 follows_exclude_scenario: bool = True,
                 ):
        # Metrics check
        self.metrics2method = define_metrics()
        self.metrics = metrics  # (e.g.) ['t', "tn", "ta", 'rot']
        for arg_metrics in self.metrics:
            assert arg_metrics in self.metrics2method.keys(), "Set metrics from {}".format(self.metrics2method.keys())
        print0("[INFO] Evaluator method; {}".format(metrics))

        # Dataset check
        self._dataset_name = dataset_name
        self._load_camera_str = CAMERA_NUMERIC2STR_KEY[self._dataset_name](numeric_id=cameras)

        # Read pre-created .pkl for extrinsics GT
        self.gt_module = ExtGtLoader(
            dumped_path_name=dump_pkl_name,  # (e.g.) /data/vidar/save/extrinsics-ddad-full.pkl,
            load_camera_id=self._load_camera_str,  # (e.g.) ('CAMERA_01', 'CAMERA_05', 'CAMERA_06', ...)
            mode=self._dataset_name  # (e.g.) ddad
        )
        print0("[INFO] Evaluator load the pkl; {}".format(dump_pkl_name))

        # Data writing configuration
        plt.rcParams["figure.figsize"] = (20, 20)
        self.annotate = fig_annotate
        self.columns = self._gen_columns(task_list=self.metrics, camera_list=cameras)
        self.log_without_ext = dst_dir_name
        self.error_table = pd.DataFrame()

        # etc
        self.loop_batch_size = loop_batch_size
        if follows_exclude_scenario:
            self.exclude_scenario = get_extrinsics_eval_ignore_scenarios(dataset=dataset_name)
            print0("[INFO] Detect exclude_scenario for extrinsics evaluation") if self.exclude_scenario != [] else True
            pass
        else:
            self.exclude_scenario = []

    def compute(self, extrinsics_net_instance, epoch=0, device="cpu", scenario=None) -> pd.DataFrame:
        """
        Calculate the extrinsics error based on the tasks

        Parameters
        ----------
        extrinsics_net_instance : nn.Module
            `.forward()` which returns Tensor(Bxcamx4x4), and `.scenarios` gives list such as ['000000','000001', ....]

        epoch : int
            Ignored
        device : str
            Device to be used
        scenario : List[str]
            If None, all extrinsics are to be evaluated.

        Returns
        -------
        pd.DataFrame
            Error tables for extrinsics
        """
        return self._compute_error_table(extrinsics_net_instance, device, specify_scenario=scenario)

    def get_statics_per_camera(self, add_std=True, prefix="extrinsics") -> OrderedDict:
        """Extrinsic error per camera, like Dict["extrinsics | CAMERA_05_ave": nd.array(dim)]"""

        ret = OrderedDict()

        monitoring_cameras = self._load_camera_str[1:]  # self._load_camera_str[0] always identity, so doesn't track

        for i, camera_name in enumerate(monitoring_cameras):
            metrics_per_camera_columns = self.columns[i::len(monitoring_cameras)]  # [A1, B1, C1, ..., ]
            df_per_cam: pd.DataFrame = self.error_table[metrics_per_camera_columns]  # (B, len(self.metrics))
            ret[prefix + "|" + camera_name + "_ave_err"] = df_per_cam.mean().values  # shape = (CAM*len(self.metrics),)
            if add_std:
                ret[
                    prefix + "|" + camera_name + "_std_err"] = \
                    df_per_cam.std().values  # shape = (CAM*len(self.metrics),)
            pass

        return ret

    @staticmethod
    def _gen_columns(task_list: list, camera_list: list):
        """
        Generate columns to make error table composed of the "task_name" and "camera ID"

        Parameters
        ----------
        task_list : List[str]
            [taskA, taskB, ..]
        camera_list
            [0, 1, ...]

        Returns
        -------
        List[str]
            Flattened columns such as [taskA1, taskA2, taskA3, taskB1, ... ]
        """
        ret_columns = []
        for task_name in task_list:
            ret_columns += [task_name + str(idx) for idx in range(1, len(camera_list))]
            pass
        return ret_columns

    @staticmethod
    def _get_upper_limited_df(df: pd.DataFrame, uppper_threshold=1.) -> pd.DataFrame:
        """ Create the upper bound to the pd.Dataframes.values """
        return df.where(df <= uppper_threshold, uppper_threshold)

    def _compute_error_table(self, extrinsics_net, device_ref: str, specify_scenario: list = None) -> pd.DataFrame:
        """Core implementation for extrinsics error calculation, then generates self.error_table """
        if specify_scenario is None:
            target_scenarios = extrinsics_net.scenarios  # scenario list such as ["000150", "000151", .., ]
        else:
            target_scenarios = specify_scenario
        result_stack = []  # for all scenario

        for strt_idx in range(0, len(target_scenarios), self.loop_batch_size):
            col_stack = []  # for batch
            sample_scenario = target_scenarios[strt_idx:strt_idx + self.loop_batch_size]  # ScenarioList[0,1, .., b-1]

            pred = extrinsics_net(sample_scenario).to(device_ref)  #
            gt = self.gt_module(sample_scenario).to(device_ref)

            for metric in self.metrics:
                b_err = self.metrics2method[metric](gt=gt[:, 1:, :, :], pred=pred[:, 1:, :, :])  # (B, cam-1)
                col_stack.append(b_err)
                pass

            err_tables = torch.concat(col_stack, 1)  # (B, (cam-1) * len(self.metrics))

            result_stack.append(err_tables)
            pass
        data_ = torch.concat(result_stack,
                             0).detach().cpu().numpy()  # (len(target_scenarios), (cam-1) * len(self.metrics))
        index_ = ["s" + target for target in target_scenarios]

        tgt_df = pd.DataFrame(data=data_, index=index_, columns=self.columns)

        # Exclude ignorable scenarios
        exclude_candidates = ["s" + scene_ for scene_ in self.exclude_scenario]
        if len(set(index_) - set(exclude_candidates)) > 0 and len(set(index_) & set(exclude_candidates)) > 0:
            tgt_df = tgt_df.drop(index=list(set(index_) & set(exclude_candidates)))
            pass
        tgt_df = tgt_df.sort_index()

        self.error_table = tgt_df
        return tgt_df

    def to_csv(self, path: str = None):
        """Save result to csv files"""
        if path is None:
            self.error_table.to_csv(self.log_without_ext + ".csv")
        else:
            self.error_table.to_csv(path)
            pass
        pass

    def _to_seaborn_map(self, upperbound=1., annotate=False):
        """Format the error heatmap for extrinsic error visualization"""
        plt.figure()  # create new figure every time
        out_df = self._get_upper_limited_df(self.error_table, upperbound)
        return sns.heatmap(out_df, annot=annotate)

    def to_figure(self, upperbound=1., annotate=False):
        """Error heatmap map given the computed score table (self.error_table, the type is pd.DataFrame)"""
        return self._to_seaborn_map(upperbound, annotate).get_figure()  # return  matplotlib.figure.Figure

    @property
    def dataset(self):
        return self._dataset_name
