# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import numpy as np

from vidar.datasets.BaseDataset import BaseDataset
from vidar.datasets.utils.FolderTree import FolderTree
from vidar.datasets.utils.misc import stack_sample
from vidar.utils.read import read_image


class GenericDataset(BaseDataset):
    def __init__(self, tag=None, single_folder=False, split=None, extension='png', **kwargs):
        """
        Generic dataset, used to load information from folders

        Parameters
        ----------
        tag : String
            Dataset tag
        single_folder : Bool
            Whether the dataset is a single folder
        split : String
            Dataset split
        kwargs : Dict
            Additional arguments
        """
        super().__init__(**kwargs)
        self.tag = 'generic' if tag is None else tag
        if split is None or split == '':
            split = ('', )
        self.rgb_tree = FolderTree(
            self.path, context=self.context, sub_folders=split,
            single_folder=single_folder, suffix=f'.{extension}')

    def __len__(self):
        """Dataset length"""
        return len(self.rgb_tree)

    @staticmethod
    def get_intrinsics(rgb):
        """Return dummy intrinsics"""
        return np.array([[rgb.size[0] / 2., 0., rgb.size[0] / 2.],
                         [0., rgb.size[1], rgb.size[1] / 2.],
                         [0., 0., 1.]])

    def __getitem__(self, idx):
        """Get dataset sample"""
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
            sample['intrinsics'] = {
                0: self.get_intrinsics((sample['rgb'][0]))
            }

            # If with context
            if self.with_context:
                filename_context = self.rgb_tree.get_context(idx)
                sample['rgb'].update(read_image(filename_context))

            # Stack sample
            samples.append(sample)

        # Transform data
        if self.data_transform:
            samples = self.data_transform(samples)

        # Return stacked sample
        return stack_sample(samples)


