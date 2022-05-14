# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import os
from abc import ABC

from torch.utils.data import Dataset

from vidar.utils.types import is_list


class BaseDataset(Dataset, ABC):
    """
    Base dataset class

    Parameters
    ----------
    path : String
        Dataset location
    context : Tuple
        Temporal context
    cameras : Tuple
        Camera names
    labels : Tuple
        Labels to be loaded
    labels_context :
        Context labels to be loaded
    data_transform : Function
        Transformations to be applied to sample
    ontology : String
        Which semantic ontology should be used
    return_ontology : Bool
        Whether the ontology should be returned
    virtual : Bool
        Whether the dataset is virtual or not
    kwargs : Dict
        Additional parameters
    """
    def __init__(self, path, context, cameras, labels=(), labels_context=(),
                 data_transform=None, ontology=None, return_ontology=False, virtual=False,
                 **kwargs):
        super().__init__()

        self.path = path
        self.labels = labels
        self.labels_context = labels_context
        self.cameras = cameras
        self.data_transform = data_transform

        self.num_cameras = len(cameras) if is_list(cameras) else cameras

        self.bwd_contexts = [ctx for ctx in context if ctx < 0]
        self.fwd_contexts = [ctx for ctx in context if ctx > 0]

        self.bwd_context = 0 if len(context) == 0 else - min(0, min(context))
        self.fwd_context = 0 if len(context) == 0 else max(0, max(context))

        self.context = [v for v in range(- self.bwd_context, 0)] + \
                       [v for v in range(1, self.fwd_context + 1)]

        self.num_context = self.bwd_context + self.fwd_context
        self.with_context = self.num_context > 0

        self.ontology = ontology
        self.return_ontology = return_ontology
        self.virtual = virtual

    def relative_path(self, filename):
        return {key: os.path.splitext(val.replace(self.path + '/', ''))[0]
                for key, val in filename.items()}

    # Label properties

    @property
    def with_depth(self):
        """If dataset contains depth"""
        return 'depth' in self.labels

    @property
    def with_input_depth(self):
        """If dataset contains input depth"""
        return 'input_depth' in self.labels

    @property
    def with_pose(self):
        """If dataset contains pose"""
        return 'pose' in self.labels

    @property
    def with_semantic(self):
        """If dataset contains semantic"""
        return 'semantic' in self.labels

    @property
    def with_instance(self):
        """If dataset contains instance"""
        return 'instance' in self.labels

    @property
    def with_optical_flow(self):
        """If dataset contains optical flow"""
        return 'optical_flow' in self.labels

    @property
    def with_scene_flow(self):
        """If dataset contains scene flow"""
        return 'scene_flow' in self.labels

    @property
    def with_bbox2d(self):
        """If dataset contains 2d bounding boxes"""
        return 'bbox2d' in self.labels

    @property
    def with_bbox3d(self):
        """If dataset contains 3d bounding boxes"""
        return 'bbox3d' in self.labels

    @property
    def with_lidar(self):
        """If dataset contains lidar"""
        return 'lidar' in self.labels

    @property
    def with_radar(self):
        """If dataset contains radar"""
        return 'radar' in self.labels

    @property
    def with_pointcache(self):
        """If dataset contains pointcaches"""
        return 'pointcache' in self.labels

    # Label context properties

    @property
    def with_depth_context(self):
        """If dataset contains context depth"""
        return 'depth' in self.labels_context

    @property
    def with_input_depth_context(self):
        """If dataset contains context input depth"""
        return 'input_depth' in self.labels_context

    @property
    def with_semantic_context(self):
        """If dataset contains context semantic"""
        return 'semantic' in self.labels_context

    @property
    def with_instance_context(self):
        """If dataset contains context instance"""
        return 'instance' in self.labels_context

    @property
    def with_optical_flow_context(self):
        """If dataset contains context optical flow"""
        return 'optical_flow' in self.labels_context

    @property
    def with_scene_flow_context(self):
        """If dataset contains context scene flow"""
        return 'scene_flow' in self.labels_context

    @property
    def with_bbox2d_context(self):
        """If dataset contains context 2d bounding boxes"""
        return 'bbox2d' in self.labels_context

    @property
    def with_bbox3d_context(self):
        """If dataset contains context 3d bounding boxes"""
        return 'bbox3d' in self.labels_context
