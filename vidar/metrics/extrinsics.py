# Copyright 2023 Toyota Research Institute.  All rights reserved.

import os
from typing import Dict, Any, List

import torch
import wandb

from knk_vision.vidar.vidar.arch.networks.extrinsics.ExtEvaluatorModule import ExtComparisonWithPkl
from knk_vision.vidar.vidar.metrics.base import BaseEvaluation
from knk_vision.vidar.vidar.utils.distributed import on_rank_0
from knk_vision.vidar.vidar.utils.logging import pcolor
from knk_vision.vidar.vidar.utils.distributed import print0


class ExtrinsicsEvaluation(BaseEvaluation):
    """
    Extrinsics evaluation class by comparing with the predicted extrinsics and pre-generated ground-truth (as .pkl)

    Parameters
    ----------
    cfg
        Configuration with parameters
    """

    __default_metrics = ['t', "tn", "ta", 'rot']
    __name = 'extrinsics'

    def __init__(self, cfg):
        super().__init__(cfg,
                         name=self.__name, task=self.__name,
                         metrics=tuple(cfg.has("metrics", self.__default_metrics)),
                         )
        self.extrinsics_exe = ExtComparisonWithPkl(
            dump_pkl_name=cfg.dump_pkl_path,
            dataset_name=cfg.dataset,
            cameras=cfg.cameras,
            metrics=cfg.has("metrics", self.__default_metrics),
            follows_exclude_scenario=cfg.has("follows_exclude_scenario", True)
        )

        # Receiver for extrinsics_net of the current epoch,  from models.forward()
        self.current_extrinsics_net_instance = None  # must be passed by ["predictions"]["extrinsics_net"]
        self.save_csv_dir = cfg.has("save_csv_dir", "")
        self.csv_serial_num = 0
        if self.save_csv_dir != "" and (not os.path.exists(self.save_csv_dir)):
            print0("Create folder to {}".format(self.save_csv_dir))
            os.makedirs(self.save_csv_dir)

    @staticmethod
    def reduce_fn(metrics: torch.Tensor, seen: torch.Tensor):  # dummy
        """Not used"""
        return 0

    def populate_metrics_dict(self, metrics, metrics_dict, prefix):
        """Populate metrics dictionary"""
        for metric in metrics:
            if metric.startswith(self.name):
                name, suffix = metric.split('|')
                for i, key in enumerate(self.metrics):
                    metrics_dict[f'{prefix}-{name}|{key}_{suffix}'] = \
                        metrics[metric][[i]].item()

    @on_rank_0
    def print(self, reduced_data: List[Dict[str, torch.Tensor]], prefixes: list):
        """
        Print evaluation results

        Parameters
        ----------
        reduced_data: List[ Dict[ "index", torch.tensor([dim]) ] ]
            List of the dictionary which indicate the task-name and the score.
            (e.g.) "index" == "EXTRINSICS|CAMERA_05_AVE_ERR" / torch.tensor([dim]) == np.ndarray([0.116, 0.675, ...])
        prefixes : List[str]
            Prefix for this evaluation method class (here, name of the dataset is chosen)
        """
        print()
        print(self.horz_line)
        print(self.metr_line.format(*((self.name.upper(),) + self.metrics)))
        for n, metrics in enumerate(reduced_data):
            if sum([self.name in key for key in metrics.keys()]) == 0:
                continue
            print(self.horz_line)
            print(self.wrap(pcolor('*** {:<59}'.format(prefixes[n]), **self.font1)))
            print(self.horz_line)
            for key, metric in sorted(metrics.items()):
                if self.name in key:
                    print(self.wrap(pcolor(self.outp_line.format(
                        *((key.upper(),) + tuple(metric.tolist()))), **self.font2)))
        print(self.horz_line)
        print()

    def reduce(self, output, dataloaders, prefixes, verbose=True) -> Dict[str, Any]:
        """
        Override the base function because evaluation of the extrinsics is "independent" to the validation split,
        So all process isn't applied to both the dataloader and predictions,  that comes from the "validation" split.

        Parameters
        ----------
        output : torch.Tensor
            Not used
        dataloaders : torch.Tensor
            Not used
        prefixes : List[str]
            Use only one keyword
        verbose : bool
            Always True

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with metrics
        """

        prefixes = [self.extrinsics_exe.dataset]

        # compute all error
        df = self.extrinsics_exe.compute(extrinsics_net_instance=self.current_extrinsics_net_instance)
        # reduced_data = List[ Dict["task", nd.array(dim)] ] --> to self.print()
        reduced_data = [self.extrinsics_exe.get_statics_per_camera(prefix=self.__name, add_std=True)]
        # metrics_dict = Dict["task", Any] --> to wandb logger
        metrics_dict = self.create_metrics_dict(reduced_data, prefixes)  # for wandb tracking
        metrics_dict.update({"calib_extrinsics": wandb.Image(self.extrinsics_exe.to_figure())})

        if verbose:
            self.print(reduced_data, prefixes)

        if self.save_csv_dir != "":
            self.extrinsics_exe.to_csv(
                os.path.join(self.save_csv_dir, "val_extrinsic_" + str(self.csv_serial_num).zfill(3)) + ".csv")
            self.csv_serial_num += 1
            pass

        # Refresh the current extrinsics-model
        self.current_extrinsics_net_instance = None
        return metrics_dict

    def evaluate(self, batch, output: Dict[str, Any], task, flipped_output) -> tuple:
        """
        Update the current status of extrinsics; memorize as self.current_extrinsics_net_instance.

        Parameters
        ----------
        batch : Dict[str, Any]
            Not used
        output : Dict[str, Any]
            Register output["extrinsics_net"] at the first time loop
            For this, the return of forward() implemented in the model class must have 'extrinsics_net',
            such that model.foward() return dict["predictions"]["extrinsics_net"]
        task : str
            Not used
        flipped_output : Any (or None)
            Not used

        Returns
        -------
        Tuple[Dict[str, torch.Tensor]]
            The tuple of dictionary, but not used for this evaluation class here
        """
        metrics, predictions = {}, {}
        if self.current_extrinsics_net_instance is None:
            self.current_extrinsics_net_instance = output["extrinsics_net"]  # Receive extrinsics_net
        else:
            pass
        return metrics, predictions  # (e.g.) Dict["metric": torch.tensor([b, dim_criterion])]
