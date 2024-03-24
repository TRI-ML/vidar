
import os

import numpy as np

import cv2

from knk_vision.vidar.vidar.datasets.BaseDataset import BaseDataset
from knk_vision.vidar.vidar.datasets.utils.FolderTree import FolderTree
from knk_vision.vidar.vidar.datasets.utils.misc import stack_sample
from knk_vision.vidar.vidar.utils.read import read_image
from knk_vision.vidar.vidar.datasets.BaseDataset import BaseDataset
from knk_vision.vidar.vidar.datasets.utils.misc import invert_pose, stack_sample, make_relative_pose
from knk_vision.vidar.vidar.utils.read import read_image


class ScanNetTemporalDataset(BaseDataset):
    def __init__(self, tag=None, single_folder=False, split=None, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.tag = 'scannet_temporal' if tag is None else tag
        if split is None or split == '':
            split = ('', )
        self.rgb_tree = FolderTree(
            os.path.join(self.path, split),
            context=self.context, sub_folders=['color'], stride=stride,
            single_folder=single_folder, suffix='.jpg')

    def __len__(self):
        """Dataset length"""
        return len(self.rgb_tree)

    @staticmethod
    def load_intrinsics(filename):
        """Get intrinsics from filename"""
        filename_intrinsics = {key: '/'.join(val.split('/')[:-2]) + '/intrinsic/intrinsic_depth.txt'
                               for key, val in filename.items()}
        return {key: np.genfromtxt(val).astype(np.float32).reshape((4, 4))[:3, :3]
                for key, val in filename_intrinsics.items()}

    @staticmethod
    def load_depth(filename):
        """Get depth maps from filename"""
        try:
            filename_depth = {key: val.replace('color', 'depth').replace('.jpg', '.png')
                            for key, val in filename.items()}
            return {key: (cv2.imread(val, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0)
                    for key, val in filename_depth.items()}
        except:
            filename_depth = {key: val.replace('color', 'depth').replace('.jpg', '.npy')
                            for key, val in filename.items()}
            return {key: (np.load(val) / 1000.0).astype(np.float32)
                    for key, val in filename_depth.items()}

    @staticmethod
    def load_pose(filename):
        """Get poses from filename"""
        filename_pose = {key: val.replace('color', 'pose').replace('.jpg', '.txt')
                          for key, val in filename.items()}
        return {key: invert_pose(np.genfromtxt(val).astype(np.float32).reshape((4, 4)))
                for key, val in filename_pose.items()}

    def __getitem__(self, idx):
        """Get dataset sample given an index."""

        samples = []

        for _ in self.cameras:

            # Filename
            filename = self.rgb_tree.get_item(idx)

            # Base sample
            sample = {
                'idx': idx,
                'tag': self.tag,
                'filename': self.relative_path(filename),
                'splitname': '%010d' % idx
            }

            # Image
            sample['rgb'] = read_image(filename)

            # Intrinsics
            sample['intrinsics'] = self.load_intrinsics(filename)

            if self.with_depth:
                sample['depth'] = self.load_depth(filename)

            if self.with_pose:
                sample['pose'] = self.load_pose(filename)

            # If with context
            if self.with_context:
                filename_context = self.rgb_tree.get_context(idx)
                sample['rgb'].update(read_image(filename_context))
                if self.with_depth:
                    sample['depth'].update(self.load_depth(filename_context))
                if self.with_pose:
                    sample['pose'].update(self.load_pose(filename_context))

            # Stack sample
            samples.append(sample)

        # Make relative poses
        samples = make_relative_pose(samples)

        # Transform data
        if self.data_transform:
            samples = self.data_transform(samples)

        # Return stacked sample
        return stack_sample([samples])
