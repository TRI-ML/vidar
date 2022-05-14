# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import csv
import os

import cv2
import numpy as np

from vidar.datasets.BaseDataset import BaseDataset
from vidar.datasets.utils.FolderTree import FolderTree
from vidar.datasets.utils.misc import \
    convert_ontology, initialize_ontology, stack_sample, make_relative_pose
from vidar.utils.data import dict_remove_nones
from vidar.utils.decorators import iterate1
from vidar.utils.read import read_image


def make_tree(path, sub_folder, camera, mode, context):
    """
    Create a folder tree for a certain task

    Parameters
    ----------
    path : String
        Data path
    sub_folder : String
        Subfolder path
    camera : Int
        Camera index
    mode : String
        Which task we are using
    context : list[Int]
        Context samples

    Returns
    -------
    tree : FolderTree
        Folder tree containing task data
    """
    path = os.path.join(path, sub_folder)
    sub_folders = '{}/frames/{}/Camera_{}'.format(mode, sub_folder, camera)
    return FolderTree(path, sub_folders=sub_folders, context=context)


def semantic_color_to_id(semantic_color, ontology):
    """
    Convert semantic color to semantic ID

    Parameters
    ----------
    semantic_color : numpy.Array
        Matrix with semantic colors [H, W, 3]
    ontology : Dict
        Ontology dictionary, with {id: color}

    Returns
    -------
    semantic_id : numpy.Array
        Matrix with semantic IDs [H, W]
    """
    # Create semantic ID map
    semantic_id = np.zeros(semantic_color.shape[:2])
    # Loop over every ontology item and assign ID to color
    for key, val in ontology.items():
        idx = (semantic_color[:, :, 0] == val['color'][0]) & \
              (semantic_color[:, :, 1] == val['color'][1]) & \
              (semantic_color[:, :, 2] == val['color'][2])
        semantic_id[idx] = key
    # Return semantic ID map
    return semantic_id


class VKITTI2Dataset(BaseDataset):
    """
    VKITTI2 dataset class

    Parameters
    ----------
    path : String
        Path to the dataset
    split : String {'train', 'val', 'test'}
        Which dataset split to use
    ontology : String
        Which ontology should be used
    return_ontology : Bool
        Returns ontology information in the sample
    data_transform : Function
        Transformations applied to the sample
    """
    def __init__(self, split, tag=None, **kwargs):
        super().__init__(**kwargs)
        self.tag = 'vkitti2' if tag is None else tag

        # Store variables
        self.split = split
        self.mode = 'clone'

        # Initialize ontology
        if self.with_semantic:
            self.ontology, self.ontology_convert = initialize_ontology('vkitti2', self.ontology)
            self.return_ontology = self.return_ontology

        # Create RGB tree
        self.rgb_tree = make_tree(
            self.path, 'rgb', 0, self.mode, self.context)

        # Create semantic tree
        if self.with_semantic:
            self.semantic_tree = make_tree(
                self.path, 'classSegmentation', 0, self.mode, self.context)

        # Create instance tree
        if self.with_instance:
            self.instance_tree = make_tree(
                self.path, 'instanceSegmentation', 0, self.mode, self.context)

    def __len__(self):
        """Dataset length"""
        return len(self.rgb_tree)

    @staticmethod
    @iterate1
    def _get_depth(filename):
        """Get depth map from filename"""
        filename = filename.replace('rgb', 'depth').replace('jpg', 'png')
        return cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 100.

    @staticmethod
    @iterate1
    def _get_intrinsics(filename, camera, mode):
        """Get intrinsics from filename"""
        # Get sample number in the scene
        number = int(filename.split('/')[-1].replace('rgb_', '').replace('.jpg', ''))
        # Get intrinsic filename
        filename_idx = filename.rfind(mode) + len(mode)
        filename_intrinsics = os.path.join(filename[:filename_idx].replace(
            '/rgb/', '/textgt/'), 'intrinsic.txt')
        # Open intrinsic file
        with open(filename_intrinsics, 'r') as f:
            # Get intrinsic parameters
            lines = list(csv.reader(f, delimiter=' '))[1:]
            params = [float(p) for p in lines[number * 2 + camera][2:]]
            # Build intrinsics matrix
            intrinsics = np.array([[params[0], 0.0, params[2]],
                                   [0.0, params[1], params[3]],
                                   [0.0, 0.0, 1.0]]).astype(np.float32)
        # Return intrinsics
        return intrinsics

    @staticmethod
    @iterate1
    def _get_pose(filename, camera, mode):
        """Get pose from filename"""
        # Get sample number in the scene
        number = int(filename.split('/')[-1].replace('rgb_', '').replace('.jpg', ''))
        # Get intrinsic filename
        filename_idx = filename.rfind(mode) + len(mode)
        filename_pose = os.path.join(filename[:filename_idx].replace(
            '/rgb/', '/textgt/'), 'extrinsic.txt')
        # Open intrinsics file
        with open(filename_pose, 'r') as f:
            # Get pose parameters
            lines = list(csv.reader(f, delimiter=' '))[1:]
            pose = np.array([float(p) for p in lines[number * 2 + camera][2:]]).reshape(4, 4)
        # Return pose
        return pose

    @staticmethod
    def _get_ontology(filename, mode):
        """Get ontology from filename"""
        # Get ontology filename
        filename_idx = filename.rfind(mode) + len(mode)
        filename_ontology = os.path.join(filename[:filename_idx].replace(
            '/classSegmentation/', '/textgt/'), 'colors.txt')
        # Open ontology file
        with open(filename_ontology, 'r') as f:
            # Get ontology parameters
            lines = list(csv.reader(f, delimiter=' '))[1:]
            from collections import OrderedDict
            ontology = OrderedDict()
            for i, line in enumerate(lines):
                ontology[i] = {
                    'name': line[0],
                    'color': np.array([int(clr) for clr in line[1:]])
                }
        return ontology

    def _get_semantic(self, filename):
        """Get semantic from filename"""
        # Get semantic color map
        semantic_color = {key: np.array(val) for key, val in read_image(filename).items()}
        # Return semantic id map
        semantic_id = {key: semantic_color_to_id(val, self.ontology) for key, val in semantic_color.items()}
        return convert_ontology(semantic_id, self.ontology_convert)

    @staticmethod
    def _get_instance(filename):
        """Get instance from filename"""
        # Get instance id map
        return np.array(read_image(filename))

    @staticmethod
    def _get_bbox3d(filename):

        bboxes3d_dim = []
        bboxes3d_pos = []
        bboxes3d_rot = []
        bboxes3d_idx = []

        k = int(filename.split('/')[-1][4:-4])
        bb = '/'.join(filename.replace('/rgb/', '/textgt/').split('/')[:-4])
        bb += '/pose.txt'

        with open(bb, 'r') as file:
            for i, f in enumerate(file):
                if i == 0:
                    continue
                line = [float(a) for a in f.split(' ')]
                if line[0] == k and line[1] == 0:
                    bboxes3d_dim.append(np.array([line[6], line[5], line[4]]))
                    bboxes3d_pos.append(np.array(line[13:16]))
                    # bboxes3d_rot.append(np.array([line[18], line[17], line[16]]))
                    bboxes3d_rot.append(np.array([line[17], line[16], line[18]]))
                    bboxes3d_idx.append(np.array([line[2]]))

        return {
            'dim': np.stack(bboxes3d_dim, 0),
            'pos': np.stack(bboxes3d_pos, 0),
            'rot': np.stack(bboxes3d_rot, 0),
            'idx': np.stack(bboxes3d_idx, 0),
        }

    @staticmethod
    @iterate1
    def _get_optical_flow(filename, mode):
        """
        Get optical flow from filename. Code obtained here:
        https://europe.naverlabs.com/research/computer-vision-research-naver-labs-europe/proxy-virtual-worlds-vkitti-2/
        """
        # Get filename
        if mode == 'bwd':
            filename = filename.replace('rgb', 'backwardFlow')
        elif mode == 'fwd':
            filename = filename.replace('/rgb/', '/forwardFlow/').replace('rgb_', 'flow_')
        else:
            raise ValueError('Invalid optical flow mode')
        filename = filename.replace('jpg', 'png')
        # Return None if file does not exist
        if not os.path.exists(filename):
            return None
        else:
            # Get optical flow
            optical_flow = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            h, w = optical_flow.shape[:2]
            # Get invalid optical flow pixels
            invalid = optical_flow[..., 0] == 0
            # Normalize and scale optical flow values
            optical_flow = 2.0 / (2 ** 16 - 1.0) * optical_flow[..., 2:0:-1].astype('f4') - 1.
            optical_flow[..., 0] *= w - 1
            optical_flow[..., 1] *= h - 1
            # Remove invalid pixels
            optical_flow[invalid] = 0
            return optical_flow

    @staticmethod
    @iterate1
    def _get_scene_flow(filename, mode):
        """Get scene flow from filename. Code obtained here:
        https://europe.naverlabs.com/research/computer-vision-research-naver-labs-europe/proxy-virtual-worlds-vkitti-2/
        """
        # Get filename
        if mode == 'bwd':
            filename = filename.replace('rgb', 'backwardSceneFlow')
        elif mode == 'fwd':
            filename = filename.replace('/rgb/', '/forwardSceneFlow/').replace('rgb_', 'sceneFlow_')
        else:
            raise ValueError('Invalid scene flow mode')
        filename = filename.replace('jpg', 'png')
        # Return None if file does not exist
        if not os.path.exists(filename):
            return None
        else:
            # Get scene flow
            scene_flow = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            # Return normalized and scaled optical flow (-10m to 10m)
            return (scene_flow[:, :, ::-1] * 2. / 65535. - 1.) * 10.

    def __getitem__(self, idx):
        """Get dataset sample"""

        samples = []

        for camera in self.cameras:

            # Get filename
            filename = self.rgb_tree.get_item(idx)
            filename = {key: val.replace('Camera_0', 'Camera_{}'.format(camera))
                        for key, val in filename.items()}

            # Base sample
            sample = {
                'idx': idx,
                'tag': self.tag,
                'filename': self.relative_path(filename),
                'splitname': '%s_%010d' % (self.split, idx),
            }

            # Image and intrinsics
            sample.update({
                'rgb': read_image(filename),
                'intrinsics': self._get_intrinsics(filename, camera, self.mode),
            })

            # If returning pose
            if self.with_pose:
                sample['pose'] = self._get_pose(filename, camera, self.mode)

            # If returning depth
            if self.with_depth:
                sample['depth'] = self._get_depth(filename)

            # If returning input depth
            if self.with_input_depth:
                sample['input_depth'] = self._get_depth(filename)

            # If returning semantic
            if self.with_semantic:
                filename = self.semantic_tree.get_item(idx)
                sample.update({'semantic': self._get_semantic(filename)})
                # If returning ontology
                if self.return_ontology:
                    sample.update({'ontology': self._get_ontology(filename, self.mode)})

            # If returning instance
            if self.with_instance:
                filename = self.instance_tree.get_item(idx)
                sample.update({'instance': self._get_instance(filename)})

            # If returning 3D bounding boxes
            if self.with_bbox3d:
                filename = self.rgb_tree.get_item(idx)
                sample.update({
                    'bboxes3d': self._get_bbox3d(filename)
                })

            # If returning optical flow
            if self.with_optical_flow:
                sample['bwd_optical_flow'] = \
                    dict_remove_nones(self._get_optical_flow(filename, 'bwd'))
                sample['fwd_optical_flow'] = \
                    dict_remove_nones(self._get_optical_flow(filename, 'fwd'))

            # If returning scene flow
            if self.with_scene_flow:
                sample['bwd_scene_flow'] = \
                    dict_remove_nones(self._get_scene_flow(filename, 'bwd'))
                sample['fwd_scene_flow'] = \
                    dict_remove_nones(self._get_scene_flow(filename, 'fwd'))

            # If returning context information
            if self.with_context:

                # Get context filenames
                filename_context = self.rgb_tree.get_context(idx)
                filename_context = {key: val.replace('Camera_0', 'Camera_{}'.format(camera))
                            for key, val in filename_context.items()}

                # Get RGB context
                sample['rgb'].update(read_image(filename_context))

                # Get pose context
                if self.with_pose:
                    sample['pose'].update(self._get_pose(filename_context, camera, self.mode))

                # Get depth context
                if self.with_depth_context:
                    sample['depth'].update(self._get_depth(filename_context))

                # Get input depth context
                if self.with_input_depth_context:
                    sample['input_depth'].update(self._get_depth(filename_context))

                # Get semantic context
                if self.with_semantic_context:
                    sample['semantic'].update(self._get_semantic(self.semantic_tree.get_context(idx)))

                # Get optical flow context
                if self.with_optical_flow_context:
                    sample['bwd_optical_flow'].update(
                        dict_remove_nones(self._get_optical_flow(filename_context, 'bwd')))
                    sample['fwd_optical_flow'].update(
                        dict_remove_nones(self._get_optical_flow(filename_context, 'fwd')))

                # Get scene flow context
                if self.with_scene_flow_context:
                    sample['bwd_scene_flow'].update(
                        dict_remove_nones(self._get_scene_flow(filename_context, 'bwd')))
                    sample['fwd_scene_flow'].update(
                        dict_remove_nones(self._get_scene_flow(filename_context, 'fwd')))

            # Stack sample
            samples.append(sample)

        # Make relative poses
        samples = make_relative_pose(samples)

        # Transform data
        if self.data_transform:
            samples = self.data_transform(samples)

        # Return stacked sample
        return stack_sample(samples)

