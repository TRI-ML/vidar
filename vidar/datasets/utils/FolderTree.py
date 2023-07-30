
import os
from glob import glob

import numpy as np

from vidar.utils.data import make_list
from vidar.utils.types import is_list


class FolderTree:
    """
    Creates a dataset tree folder structure for file loading.

    Parameters
    ----------
    path : str
        Path where dataset is stored
    prefix : str
        Optional prefix for each filename
    sub_folders : list of str
        Optional list of sub_folders located inside each data folder where data is stored
    nested : bool
        If true, go one folder deeper to find scenes
    back_context : int
        How many previous images are required
    forward_context : int
        How many next images are required
    """
    def __init__(self, path, prefix='', suffix='', sub_folders=('',), deep=1,
                 start=None, finish=None, single_folder=False, nested=False, filter_nested=None,
                 keep_folders=None, remove_folders=None, stride=1, context=()):

        # Store context information
        self.context = list(context)
        if 0 not in self.context:
            self.context.append(0)
        self.num_context = 0 if len(self.context) == 0 else max(self.context) - min(self.context)
        self.with_context = self.num_context > 0
        self.min_context = 0 if not self.with_context else min(self.context)

        self.stride = stride
        self.pad_numbers = False

        # Initialize empty folder tree
        self.folder_tree = []

        # If we are providing a file list, treat each line as a scene
        if is_list(path):
            self.folder_tree = [[file] for file in path]
        # If we are providing a folder
        else:
            # Get folders
            string = '*' + '/*' * (deep - 1)
            folders = glob(os.path.join(path, string))
            folders.sort()

            # Remove and keep folders as needed
            if keep_folders is not None:
                folders = [f for f in folders if os.path.basename(f) in keep_folders]
            if remove_folders is not None:
                folders = [f for f in folders if os.path.basename(f) not in remove_folders]

            # If nesting, go one folder deeper in order to find the scenes
            if nested:
                upd_folders = []
                for folder in folders:
                    new_folders = glob(os.path.join(folder, '*'))
                    upd_folders.extend(new_folders)
                folders = upd_folders
                folders.sort()
                if filter_nested is not None:
                    folders = [f for f in folders if filter_nested in f]

            if single_folder:
                # Use current folder as the only one
                self.folder_tree.append(folders)
            else:
                # Populate folder tree
                for folder in folders:
                    # For each sub-folder
                    for sub_folder in make_list(sub_folders):
                        # Get and sort files in each folder
                        files = glob(os.path.join(folder, sub_folder, '{}*{}'.format(prefix, suffix)))
                        if self.pad_numbers:
                            for i in range(len(files)):
                                pref, suf = files[i].split('/')[:-1], files[i].split('/')[-1]
                                num, ext = suf.split('.')
                                files[i] = '/'.join(pref) + ('/%010d' % int(num)) + '.' + ext
                        # if len(remove) > 0:
                        #     for rem in remove:
                        #         files = [file for file in files if rem not in file]
                        files.sort()
                        if start is not None:
                            files = files[start:]
                        if finish is not None:
                            files = files[:finish]
                        if self.pad_numbers:
                            for i in range(len(files)):
                                pref, suf = files[i].split('/')[:-1], files[i].split('/')[-1]
                                num, ext = suf.split('.')
                                files[i] = '/'.join(pref) + ('/%d' % int(num)) + '.' + ext
                        if self.stride > 1:
                            files = files[::self.stride]
                        # Only store if there are more images than context
                        if len(files) > self.num_context:
                            self.folder_tree.append(files)

        # Get size of each folder
        self.slices = [len(folder) for folder in self.folder_tree]
        # Compensate for context size
        if self.with_context:
            self.slices = [s - self.num_context for s in self.slices]
        # Create cumulative size and get total
        self.slices = [0] + list(np.cumsum(self.slices))
        self.total = self.slices[-1]

    def __len__(self):
        """Dataset size"""
        return self.total

    def get_idxs(self, idx):
        """Get folder and file indexes given dataset index"""
        idx1 = np.searchsorted(self.slices, idx, side='right') - 1
        idx2 = idx - self.slices[idx1]
        return idx1, idx2

    def get_item(self, idx, return_loc=False):
        """Return filename item given index"""
        idx1, idx2 = self.get_idxs(idx)
        item = {0: self.folder_tree[idx1][idx2 - self.min_context]}
        if return_loc:
            return item, idx2 - self.min_context
        else:
            return item

    def get_context(self, idx):
        """Return forward context given index."""
        idx1, idx2 = self.get_idxs(idx)
        return {ctx: self.folder_tree[idx1][idx2 - self.min_context + ctx] for ctx in self.context}

    def get_random(self, idx, qty):
        idx1, idx2 = self.get_idxs(idx)

        n, m = len(self.folder_tree[idx1]), self.slices[idx1]
        rnd = np.random.permutation(n)
        rnd = [i for i in rnd if i != idx2]

        idxs = [idx]
        for i in range(qty - 1):
            idxs.append(rnd[i] + m)

        return idxs
