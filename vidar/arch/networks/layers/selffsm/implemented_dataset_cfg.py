# Copyright 2023 Toyota Research Institute.  All rights reserved.

from functools import partial

from knk_vision.vidar.vidar.arch.networks.layers.selffsm.dataset_interface_method import numeric_cam_id2camera_dir_name

# for evaluation

CAMERA_NUMERIC2STR_KEY = {
    "ddad": partial(numeric_cam_id2camera_dir_name, cam_prefix="CAMERA_", zfill=2),  # 1 -> CAMERA_01
    "pd": partial(numeric_cam_id2camera_dir_name, cam_prefix="camera_", zfill=2),  # 1 -> camera_01
    "vkitti2": partial(numeric_cam_id2camera_dir_name, cam_prefix="Camera_", zfill=0),  # 0 -> Camera_0
    "kitti": partial(numeric_cam_id2camera_dir_name, cam_prefix="image_", zfill=2, add_const_to_id=2),  # 0 -> image_02
}

IMPLEMENTED_DATASET2FRONT_CAM = {
    "ddad": 'CAMERA_01',
    "pd": 'camera_01',
    "vkitti2": 'Camera_0',
    "kitti": 'image_02',
    "nuscenes": 'CAM_FRONT',
}

# ignore scenes
IGNORE_SCENES = {
    "ddad": [str(x).zfill(6) for x in range(150, 200, 1)]  # ['000150', '000151', ..., '000199']
}

# for visualisation
DATASET2DEFAULT_HW = {
    "ddad": [384, 640],
    "vkitti2": [192, 640],
    "kitti": [192, 640],
}

DATASET2DEFAULT_VIEW_KEY = {
    "ddad": [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
    "vkitti2": [(0, 0), (0, 1)],
    "kitti": [(0, 0), (0, 1)]
}

DATASET2DEFAULT_SCENARIO = {
    "ddad": "000052",
    "vkitti2": "Scene02",
    "kitti": "2011_09_30",
}

DATASET2IMG_EXT = {
    "ddad": ".jpg",
    "pd": ".png",
    "vkitti2": '.jpg',
    "kitti": ".png",
}

DATASET2FRONT_IMG_PATH_UNDER_SCENARIO = {
    "ddad": "rgb_1216_1936/CAMERA_01/",
    "pd": "rgb/camera_01/",
    "vkitti2": 'clone/frames/rgb/Camera_0/',
    "kitti": ".png",
}

# etc

CAMERA_PAIR_REF_DDAD = {
    'CAMERA_01': 0,
    'CAMERA_05': 1,
    'CAMERA_06': 2,
    'CAMERA_07': 3,
    'CAMERA_08': 4,
    'CAMERA_09': 5,
}
