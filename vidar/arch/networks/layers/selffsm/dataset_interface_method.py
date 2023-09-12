# Copyright 2023 Toyota Research Institute.  All rights reserved.

import glob
import os
from typing import Any, Dict, List

from PIL import Image
import torch
from torchvision.transforms import ToTensor

from vidar.geometry.pose_utils import invert_pose
from vidar.utils.data import break_key
from vidar.utils.distributed import print0
from vidar.utils.types import is_double_list


def get_unbroken_extrinsic(ext_dict: Dict[tuple, torch.Tensor]) -> torch.Tensor:
    """Return extrinsics from the pose tensor corresponding to the current frame"""
    cam_lst = sorted([item[1] for item in ext_dict.keys() if item[0] == 0])
    ret = torch.stack([  # ctx = -1, 1, ...
        ext_dict[(0, cam_id)] for cam_id
        in range(len(cam_lst))]).transpose(0, 1)  # (cam, b, 4, 4) to (b, cam, 4, 4)
    return ret


def get_relative_poses_from_base_cam(extrinsics_tensor: torch.Tensor,
                                     inverse_out=False,
                                     ) -> torch.Tensor:
    """
    Rebase the extrinsics to make the extrinsics of canonical camera an identity matrix

    Parameters
    ----------
    extrinsics_tensor : torch.Tensor
        Extrinsics of camera (Bxcamx4x4)
    inverse_out : bool
        Flag whether invert the matrix or not

    Returns
    -------
    torch.Tensor
        The rebased pose like {id|id=1,2,3...}^T_{0}, that shapes (Bxcamx4x4). Return[:,0,:,:] is an identity matrix
    """
    base_cam_id = 0
    b, cam_num, _, __ = extrinsics_tensor.shape
    base_ext = extrinsics_tensor[:, base_cam_id, :, :]  # (B, 4, 4)
    stack_target = [
        extrinsics_tensor[:, i, :, :] @ invert_pose(base_ext) if not inverse_out
        else base_ext @ invert_pose(extrinsics_tensor[:, i, :, :]) for i in range(cam_num)]
    stack_target[base_cam_id] = torch.eye(4).repeat(b, 1, 1).to(
        base_ext.device)  # Replace identity ro nealry-identity one
    ret = torch.stack(stack_target).transpose(0, 1)  # (B, cam_id, 4, 4)
    return ret


def get_cam0_based_extrinsics_from_pose(curren_pose_tensor: torch.Tensor):
    """Replace the canonical camera's pose with an identity matrix."""
    b = curren_pose_tensor.shape[0]
    canonical_pose = torch.eye(4).repeat([b, 1, 1]).unsqueeze(1).to(
        curren_pose_tensor.device)  # for main CAM, [b,1,4,4]
    the_other_pose = curren_pose_tensor[:, 1:, :, :]  # the other cam, [b, cam-1, 4, 4]
    return torch.concat([canonical_pose, the_other_pose], 1)


def get_scenario(batch_filename: dict, id_where_scenario_is: int = 0, id_where_cam_defined: int = 2) -> list:
    """
    Get parent directory name from the filename information

    Parameters
    ----------
    batch_filename : Dict[int, List[Any]]
        (e.g.) {0: [['000084/{}/CAMERA_01/1567566624218648', ..., ], ['000150/{}/CAMERA_01/1567566###', ..., ], ...]}
        If batch_filename[0] is not double-list, 2 options;
        - single batch with shared scenario case ( == multi-camera),
            such as {0: ['000084/{}/CAMERA_01/1567566624218648','000084/{}/CAMERA_05/1567566624218648', ..., ]
        - single camera case with different scenario,
            such as {0: ['000084/{}/CAMERA_01/1567566624218648','000150/{}/CAMERA_01/1567566624573594', ..., ]}
    id_where_scenario_is : int
        Specify where the scenario name located in the string
    id_where_cam_defined : int
        Specify where camera name located in the string

    Returns
    -------
    List[str]
        The list of the scenario names like ['000084', '000150', ... ]
    """
    if is_double_list(batch_filename[0]):
        tgt_frame_names = batch_filename[0]  # [[cam0/batch1, cam0/batch2, ...], [cam1/batch1, cam1/batch2, ...]]
        maincam_filenames = tgt_frame_names[0]  # index 0 must be target --> [cam0/batch1, cam0/batch2, ...]
        return [png_filepath.split('/')[id_where_scenario_is] for png_filepath in maincam_filenames]
    else:  # is_single_list == single batch or CAMERA
        cam_filenames = batch_filename[0]
        fed_scenario_cams = [cam_file.split('/')[id_where_cam_defined] for cam_file in cam_filenames]
        if len(set(fed_scenario_cams)) == len(
                cam_filenames):  # Single batch with shared scenario case --> return 1 scenario
            return [batch_filename[0][0].split('/')[id_where_scenario_is]]
        else:
            raise NotImplementedError('Non supported; set multi-camera in dataloader, like `cameras: [[0,1]]`')


def get_str_from_broken(batch_filename: Dict[tuple, List[str]],
                        id_where_dest_is: int = -4,
                        tgt_key=(0, 0)):
    """
    Get scenario ID of the other character from batch['filename']
    (e.g.) {(0, 0): ['000000/rgb/CAMERA_01/*.png', '000150/rgb/CAMERA_01/*.png'], ...} -> ['000000', '000150', ]

    Parameters
    ----------
    batch_filename: Dict[tuple, List[str]]
        batch['filename'] that shapes {(temporal ID,camera ID): batchList[path-to-batch-1, path-to-batch-2, ...] }
    id_where_dest_is : int
        Specify where the target character located in the string
    tgt_key : tuple
        The key to extract the character

    Returns
    -------
    List[str]
        The list of target characters
    """
    tgt_filenames = batch_filename[tgt_key]
    filenames = tgt_filenames if type(tgt_filenames) == list else [tgt_filenames]
    return [png_filepath.split('/')[id_where_dest_is] for png_filepath in filenames]


def get_scenario_independent_camera(batch_filename: dict, id_where_scenario_is: int = 0, tgt_id: Any = 0) -> list:
    """
    From single camera output {0: ['000150/rgb/CAMERA_07/15616458285936472', '000150/rgb/CAMERA_05/15616458288936462']}
    to scenarios such that  ['000150', '000150']

    MUST NOT USE for multi-camera dataloading case, such as [[0,1]] or [[1,5,6,7,8,9]]

    @param batch_filename:
    @param id_where_scenario_is:
    @return:
    """
    files = batch_filename[tgt_id]
    return [item.split('/')[id_where_scenario_is] for item in files]


def get_camname(batch_filename: dict, id_where_cam_defined: int = 2, tgt_idx: Any = 0) -> list:
    """
    Get parent directory name in DGP's dataloader output such as
    {0: ['000084/{}/CAMERA_01/1567566624218648', '000084/{}/CAMERA_05/1567566624218648' ..., ]}
    -> ['CAMERA_01', 'CAMERA_05', ... ]
    """
    tgt_frame_names = batch_filename[tgt_idx]
    if is_double_list(tgt_frame_names):  # [[cam0/batch1, cam0/batch2, ...], [cam1/batch1, cam1/batch2, ...]]
        # supposed that all batch has a same number of multi-cameras,
        b_size = len(tgt_frame_names[0])
        cam_name_used = [tgt_frame_names[i][0].split('/')[id_where_cam_defined] for i in range(len(tgt_frame_names))]
        return [cam_name_used] * b_size  # [[cam0, cam1, .., ], [cam0, cam1, .., ], ..., ]
    else:  # [cam1/batch1, cam6/batch2, ...]
        return [png_filepath.split('/')[id_where_cam_defined] for png_filepath in tgt_frame_names]  # [cam1, cam6, ...]


def numeric_cam_id2camera_dir_name(numeric_id: list, cam_prefix: str, zfill: int, add_const_to_id: int = 0) -> tuple:
    """Get Dataset-depending camera names such as ('CAMERA_01', 'CAMERA_05', ...) from numerical camera indices"""
    if type(numeric_id[0]) is not list:
        camera_dir_names = [cam_prefix + str(i + add_const_to_id).zfill(zfill) for i in numeric_id]
    else:
        camera_dir_names = numeric_cam_id2camera_dir_name(numeric_id[0], cam_prefix, zfill)
    return tuple(camera_dir_names)


def get_ddad_self_occlusion_mask(mask_path: str, h=384, w=640) -> torch.Tensor:
    """ Read the DDAD self-occlusion mask and convert to tensor;  0 means Mask (False) and 1 is NO Mask (True)"""
    print0("[INFO] DDAD Self-occlusion Mask")
    filenames = sorted(glob.glob(os.path.join(mask_path, "*")))
    img_pils = [Image.open(file).resize((w, h)) for file in filenames]
    img_tensor = torch.concat([ToTensor()(pil).unsqueeze_(0) for pil in img_pils], 0)
    return img_tensor


def run_break_if_not_yet(batch_item: Dict, n: int):
    """
    Make the data dictionary with the tuple key if it's never done.

    Parameters
    ----------
    batch_item  : Dict[Any, torch.Tensor]
        Output of the dataloader;
        - if batch['rgb'] ... that shapes {ctx: Tensor[b,cam,ch,h,w]}, requires n==4
        - if batch['pose'] ... that shapes {ctx: Tensor[b,cam,4,4]}, requires n==4
    n : int
        Specify the tensor rank

    Returns
    -------
    Dict[tuple, torch.Tensor]
        The data dictionary with a tuple (=broken) key
    """
    already_broken = True if (type(list(batch_item.keys())[0]) == tuple) else False
    if already_broken:
        return batch_item
    else:
        return break_key(batch_item, n)
