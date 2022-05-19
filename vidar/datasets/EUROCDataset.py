import re
from collections import defaultdict
import os
from abc import ABC

from PIL import Image
import numpy as np
from vidar.utils.read import read_image
from vidar.datasets.BaseDataset import BaseDataset
from vidar.datasets.utils.misc import stack_sample

########################################################################################################################
#### FUNCTIONS
########################################################################################################################

def dummy_calibration(image):
    w, h = [float(d) for d in image.size]
    return np.array([[1000. , 0.    , w / 2. - 0.5],
                     [0.    , 1000. , h / 2. - 0.5],
                     [0.    , 0.    , 1.          ]])

def get_idx(filename):
    return int(re.search(r'\d+', filename).group())

########################################################################################################################
#### DATASET
########################################################################################################################

class EUROCDataset(BaseDataset, ABC):
    def __init__(self, 
                split, 
                strides=(1,),
                spaceholder='{:19}',
                **kwargs):
        super().__init__(**kwargs)

        self.split = split
        self.spaceholder = spaceholder

        self.backward_context = strides[0]
        self.forward_context = strides[1]
        self.has_context = self.backward_context + self.forward_context > 0

        self.file_tree = defaultdict(list)
        self.read_files(self.path)

        self.files = []
        for k, v in self.file_tree.items():
            file_set = set(self.file_tree[k])
            files = [fname for fname in sorted(v) if self._has_context(fname, file_set)]
            # files = [fname for fname in sorted(v)]
            self.files.extend([[k, fname] for fname in files])

    def read_files(self, directory, ext=('.png', '.jpg', '.jpeg'), skip_empty=True):
        files = defaultdict(list)
        for entry in os.scandir(directory):
            relpath = os.path.relpath(entry.path, directory)
            if entry.is_dir():
                d_files = self.read_files(entry.path, ext=ext, skip_empty=skip_empty)
                if skip_empty and not len(d_files):
                    continue
                self.file_tree[entry.path] = d_files[entry.path]
            elif entry.is_file():
                if ext is None or entry.path.lower().endswith(tuple(ext)):
                    files[directory].append(relpath)
        return files

    def __len__(self):
        return len(self.files)

    def _change_idx(self, idx, filename):
        _, ext = os.path.splitext(os.path.basename(filename))
        return self.spaceholder.format(idx) + ext

    def _has_context(self, filename, file_set):
        context_paths = self._get_context_file_paths(filename, file_set)
        return len([f in file_set for f in context_paths]) >= len(self.context)

    def _get_context_file_paths(self, filename, file_set):
        fidx = get_idx(filename)
        # a hacky way to deal with two different strides in euroc dataset
        idxs = [-self.backward_context, -self.forward_context, self.backward_context, self.forward_context]
        potential_files = [self._change_idx(fidx + i, filename) for i in idxs]
        valid_paths = [fname for fname in potential_files if fname in file_set]
        return valid_paths

    def _read_rgb_context_files(self, session, filename):
        file_set = set(self.file_tree[session])
        context_paths = self._get_context_file_paths(filename, file_set)

        return [self._read_rgb_file(session, filename) for filename in context_paths]

    def _read_rgb_file(self, session, filename):
        gray_image = read_image(os.path.join(self.path, session, filename))
        gray_image_np = np.array(gray_image)
        rgb_image_np = np.stack([gray_image_np for _ in range(3)], axis=2)
        rgb_image = Image.fromarray(rgb_image_np)
        
        return rgb_image

    def _read_npy_depth(self, session, depth_filename):
        depth_file_path = os.path.join(self.path, session, '../../depth_maps', depth_filename)
        return np.load(depth_file_path)

    def _read_depth(self, session, depth_filename):
        """Get the depth map from a file."""
        return self._read_npy_depth(session, depth_filename)

    def _has_depth(self, session, depth_filename):
        depth_file_path = os.path.join(self.path, session, '../../depth_maps', depth_filename)

        return os.path.isfile(depth_file_path)

    def __getitem__(self, idx):

        samples = []

        session, filename = self.files[idx]
        image = self._read_rgb_file(session, filename)

        sample = {
            'idx': idx,
            'filename': '%s_%s' % (session, os.path.splitext(filename)[0]),
            'rgb': {0: image},
            'intrinsics': {0: dummy_calibration(image)},
        }

        if self.has_context:
            image_context = self._read_rgb_context_files(session, filename)
            sample['rgb'].update({
                    key: val for key, val in zip(self.context, image_context)
                })

        depth_filename = filename.split('.')[0] + 'depth.npy'
        if self.with_depth:
            if self._has_depth(session, depth_filename):
                sample['depth'] = {0: self._read_depth(session, depth_filename)}

        samples.append(sample)

        if self.data_transform:
            samples = self.data_transform(samples)

        return stack_sample(samples)

########################################################################################################################
if __name__ == "__main__":
    # data_dir = '/data/vidar/euroc/euroc_has_depth/V1_01_easy_has_depth/mav0'
    data_dir = '/data/vidar/euroc/euroc_cam/cam0'
    euroc_dataset = EUROCDataset(path=data_dir, 
                                # strides=[-249999872, 250000128],
                                # context=[0, 0],
                                strides=[49999872, 50000128],
                                context=[-1,1],
                                split='{:19}',
                                labels=['depth'],
                                cameras=[[0]],
                                )
    print(len(euroc_dataset))
    print('\nsample 0:')
    print(euroc_dataset[0].keys())
    print(euroc_dataset[0]['filename'])
    print(euroc_dataset[0]['rgb'])
    print(euroc_dataset[0]['intrinsics'])

    print('\nsample 1:')
    print(euroc_dataset[1].keys())
    print(euroc_dataset[1]['filename'])
    print(euroc_dataset[1]['rgb'])
    print(euroc_dataset[1]['intrinsics'])