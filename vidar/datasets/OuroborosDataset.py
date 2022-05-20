# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import os
import pickle
from collections import OrderedDict

import numpy as np
from dgp.utils.camera import Camera
from dgp.utils.pose import Pose

from vidar.datasets.BaseDataset import BaseDataset
from vidar.datasets.utils.misc import \
    initialize_ontology, stack_sample, make_relative_pose
from vidar.utils.data import dict_remove_nones, make_list
from vidar.utils.read import read_image
from vidar.utils.types import is_str


def load_from_file(filename, key):
    """Load data cache from a file"""
    data = np.load(filename, allow_pickle=True)[key]
    if len(data.shape) == 0:
        data = None
    return data


def save_to_file(filename, key, value):
    """Save data to a cache file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez_compressed(filename, **{key: value})


def get_points_inside_bbox3d(points, bboxes3d):
    """
    Returns point indices inside bounding boxes

    Parameters
    ----------
    points : np.Array
        Points to be checked (unordered, usually from LiDAR or lifted monocular) [N,3]
    bboxes3d : list[np.Array]
        Corners of bounding boxes to be checked (corner of 3d bounding box property)

    Returns
    -------
    points : np.Array
        Points inside bounding boxes [N,3]
    """
    indices = []
    for bbox in bboxes3d:
        p1, p2, p3, p4 = np.array(bbox)[[0, 1, 3, 4]]
        d12, d14, d15 = p1 - p2, p1 - p3, p1 - p4

        u = np.expand_dims(np.cross(d14, d15), 1)
        v = np.expand_dims(np.cross(d12, d14), 1)
        w = np.expand_dims(np.cross(d12, d15), 1)

        p1 = np.expand_dims(p1, 0)
        p2 = np.expand_dims(p2, 0)
        p3 = np.expand_dims(p3, 0)
        p4 = np.expand_dims(p4, 0)

        pdotu = np.dot(points, u)
        pdotv = np.dot(points, v)
        pdotw = np.dot(points, w)

        idx = (pdotu < np.dot(p1, u)) & (pdotu > np.dot(p2, u)) & \
              (pdotv < np.dot(p1, v)) & (pdotv > np.dot(p4, v)) & \
              (pdotw > np.dot(p1, w)) & (pdotw < np.dot(p3, w))
        indices.append(idx.nonzero()[0])
    return np.concatenate([points[idx] for idx in indices], 0)


def _prepare_ouroboros_ontology(ontology):
    """Read and prepare ontology to return as a data field"""
    name = ontology.contiguous_id_to_name
    colormap = ontology.contiguous_id_colormap
    output = OrderedDict()
    for key in name.keys():
        output[key] = {'name': name[key], 'color': np.array(colormap[key])}
    return output


def generate_proj_maps(camera, Xw, shape, bwd_scene_flow=None, fwd_scene_flow=None):
    """Render pointcloud on image.

    Parameters
    ----------
    camera: Camera
        Camera object with appropriately set extrinsics wrt world.
    Xw: np.Array
        3D point cloud (x, y, z) in the world coordinate. [N,3]
    shape: np.Array
        Output depth image shape [H, W]
    bwd_scene_flow: np.Array
        Backward scene flow for projection [H,W,3]
    fwd_scene_flow: np.Array
        Forward scene flow for projection [H,W,3]

    Returns
    -------
    depth: np.Array
        Rendered depth image
    """
    assert len(shape) == 2, 'Shape needs to be 2-tuple.'
    # Move point cloud to the camera's (C) reference frame from the world (W)
    Xc = camera.p_cw * Xw
    if fwd_scene_flow is not None:
        fwd_scene_flow = camera.p_cw * (Xw + fwd_scene_flow) - Xc
    if bwd_scene_flow is not None:
        bwd_scene_flow = camera.p_cw * (Xw + bwd_scene_flow) - Xc
    # Project the points as if they were in the camera's frame of reference
    uv = Camera(K=camera.K).project(Xc).astype(int)
    # Colorize the point cloud based on depth
    z_c = Xc[:, 2]

    # Create an empty image to overlay
    H, W = shape
    proj_depth = np.zeros((H, W), dtype=np.float32)
    in_view = np.logical_and.reduce([(uv >= 0).all(axis=1), uv[:, 0] < W, uv[:, 1] < H, z_c > 0])
    uv, z_c = uv[in_view], z_c[in_view]
    proj_depth[uv[:, 1], uv[:, 0]] = z_c

    # Project scene flow into image plane
    proj_bwd_scene_flow = proj_fwd_scene_flow = None
    if bwd_scene_flow is not None:
        proj_bwd_scene_flow = np.zeros((H, W, 3), dtype=np.float32)
        proj_bwd_scene_flow[uv[:, 1], uv[:, 0]] = bwd_scene_flow[in_view]
    if fwd_scene_flow is not None:
        proj_fwd_scene_flow = np.zeros((H, W, 3), dtype=np.float32)
        proj_fwd_scene_flow[uv[:, 1], uv[:, 0]] = fwd_scene_flow[in_view]

    # Return projected maps
    return proj_depth, proj_bwd_scene_flow, proj_fwd_scene_flow


class OuroborosDataset(BaseDataset):
    """
    DGP dataset class

    Parameters
    ----------
    path : String
        Path to the dataset
    split : String {'train', 'val', 'test'}
        Which dataset split to use
    cameras : list[String]
        Which cameras to get information from
    depth_type : String
        Which lidar will be used to generate ground-truth information
    input_depth_type : String
        Which lidar will be used as input to the networks
    with_pose : Bool
        If enabled pose estimates are also returned
    with_semantic : Bool
        If enabled semantic images are also returned
    with_instance : Bool
        If enabled instance images are also returned
    with_optical_flow : Bool
        If enabled optical flow is also returned
    with_scene_flow : Bool
        If enabled scene flow is also returned
    with_bbox2d : Bool
        If enabled 2d bounding boxes are also returned
    bbox2d_depth : Bool
        If enabled 2d bounding boxes depth maps are also returned
    with_bbox3d : Bool
        If enabled 3d bounding boxes are also returned
    bbox3d_depth : Bool
        If enabled 3d bounding boxes depth maps are also returned
    with_pointcache : Bool
        If enabled pointcache pointclouds are also returned (6-dimensional: x, y, z, nx, ny, nz)
    with_extra_context : Bool
        If enabled extra context information (e.g. depth, semantic, instance) are also returned
    return_ontology : Bool
        Returns ontology information in the sample
    ontology : String
        Which ontology should be used
    back_context : Int
        Size of the backward context
    forward_context : Int
        Size of the forward context
    data_transform : Function
        Transformations applied to the sample
    dataset : String ['synchronized', 'parallel_domain']
        Which dataset will be used
    only_cache : Bool
        Only use cached pointcloud information, without loading the sensor
    """
    def __init__(self, split, tag=None,
                 depth_type=None, input_depth_type=None,
                 masks=None, **kwargs):
        super().__init__(**kwargs)
        self.tag = 'ouroboros' if tag is None else tag

        cameras = [c if is_str(c) else 'camera_%02d' % c for c in self.cameras]

        # Store variables
        self.split = split
        self.dataset_idx = 0
        self.sensors = list(cameras)

        # Store task information
        self.depth_type = depth_type
        self.input_depth_type = input_depth_type
        self.only_cache = False

        self.masks_path = masks

        # Initialize ontology
        if self.with_semantic:
            base_ontology = 'parallel_domain' if self.virtual else 'ddad'
            self.ontology, self.ontology_convert = initialize_ontology(base_ontology, self.ontology)

        # Add requested annotations
        requested_annotations = []
        if self.with_semantic:
            requested_annotations.append('semantic_segmentation_2d')
        if self.with_instance:
            requested_annotations.append('instance_segmentation_2d')
        if self.with_bbox2d:
            requested_annotations.append('bounding_box_2d')
        if self.with_bbox3d:
            requested_annotations.append('bounding_box_3d')

        # Add depth sensor
        if self.with_depth and not self.only_cache and \
                self.depth_type != 'zbuffer':
            self.sensors.append(depth_type)
        self.depth_idx = len(self.sensors) - 1

        # Add input depth sensor
        if self.with_input_depth and not self.only_cache and \
                self.input_depth_type != 'zbuffer' and \
                self.input_depth_type != self.depth_type:
            self.sensors.append(input_depth_type)
        self.input_depth_idx = len(self.sensors) - 1

        # Add radar sensor
        if self.with_radar:
            self.sensors.append('radar')
        self.radar_idx = len(self.sensors) - 1

        # Choose which dataset to use
        if not self.virtual:
            from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset
            dataset = SynchronizedSceneDataset
            extra_args = {}
        else:
            from dgp.datasets.pd_dataset import ParallelDomainSceneDataset
            dataset = ParallelDomainSceneDataset
            extra_args = {
                'use_virtual_camera_datums': False,
                # 'return_scene_flow': self.with_scene_flow,
            }

        # Initialize chosen dataset
        self.dataset = dataset(
            scene_dataset_json=self.path,
            split=split,
            datum_names=self.sensors,
            backward_context=self.bwd_context,
            forward_context=self.fwd_context,
            requested_annotations=requested_annotations,
            only_annotated_datums=False,
            **extra_args,
        )

    def depth_to_world_points(self, depth, datum_idx):
        """
        Unproject depth from a camera's perspective into a world-frame pointcloud

        Parameters
        ----------
        depth : np.Array
            Depth map to be lifted [H,W]
        datum_idx : Int
            Index of the camera

        Returns
        -------
        pointcloud : np.Array
            Lifted 3D pointcloud [Nx3]
        """
        # Access data
        intrinsics = self.get_current('intrinsics', datum_idx)
        pose = self.get_current('pose', datum_idx)
        # Create pixel grid for 3D unprojection
        h, w = depth.shape[:2]
        uv = np.mgrid[:w, :h].transpose(2, 1, 0).reshape(-1, 2).astype(np.float32)
        # Unproject grid to 3D in the camera frame of reference
        pcl = Camera(K=intrinsics).unproject(uv) * depth.reshape(-1, 1)
        # Return pointcloud in world frame of reference
        return pose * pcl

    def create_camera(self, datum_idx, context=None):
        """
        Create current camera

        Parameters
        ----------
        datum_idx : Int
            Index of the camera
        context : Int
            Context value for choosing current of reference information

        Returns
        -------
        camera : Camera
            DGP camera
        """
        camera_pose = self.get_current_or_context('pose', datum_idx, context)
        camera_intrinsics = self.get_current_or_context('intrinsics', datum_idx, context)
        return Camera(K=camera_intrinsics, p_cw=camera_pose.inverse())

    def get_optical_flow(self, filename, direction):
        """
        Get optical flow from a filename (only PD)

        Parameters
        ----------
        filename : String
            Optical flow filename
        direction : String
            Direction ['bwd', 'fwd']

        Returns
        -------
        optical_flow : np.Array
            Optical flow [H,W,2]
        """
        # Check if direction is valid
        assert direction in ['bwd', 'fwd']
        direction = 'back_motion_vectors_2d' if direction == 'bwd' else 'motion_vectors_2d'
        # Get filename path and load optical flow
        path = os.path.join(os.path.dirname(self.path),
                            filename.format(direction) + '.png')
        if not os.path.exists(path):
            return None
        else:
            optflow = np.array(read_image(path))
            # Convert to uv motion
            dx_i = optflow[..., 0] + optflow[..., 1] * 256
            dy_i = optflow[..., 2] + optflow[..., 3] * 256
            dx = ((dx_i / 65535.0) * 2.0 - 1.0) * optflow.shape[1]
            dy = ((dy_i / 65535.0) * 2.0 - 1.0) * optflow.shape[0]
            # Return stacked array
            return np.stack((dx, dy), 2)

    def get_fwd_optical_flow(self, filename):
        """Get forward optical flow"""
        return self.get_optical_flow(filename, 'fwd')

    def get_bwd_optical_flow(self, filename):
        """Get backwards optical flow"""
        return self.get_optical_flow(filename, 'bwd')

    def create_proj_maps(self, filename, camera_idx, depth_idx, depth_type,
                         world_points=None, context=None, with_scene_flow=False):
        """
        Creates the depth map for a camera by projecting LiDAR information.
        It also caches the depth map following DGP folder structure, so it's not recalculated

        Parameters
        ----------
        filename : String
            Filename used for loading / saving
        camera_idx : Int
            Camera sensor index
        depth_idx : Int
            Depth sensor index
        depth_type : String
            Which depth type will be loaded
        world_points : np.Array [Nx3]
            Points that will be projected (optional)
        context : Int
            Context value for choosing current of reference information
        with_scene_flow : Bool
            Return scene flow information as well or not

        Returns
        -------
        depth : np.Array
            Depth map for that datum in that sample [H,W]
        """
        # If we want the z-buffer (simulation)
        if depth_type == 'zbuffer':
            sensor_name = self.get_current('datum_name', camera_idx)
            filename = filename.replace(self.sensors[camera_idx], sensor_name)
            filename = '{}/{}.npz'.format(
                os.path.dirname(self.path), filename.format('depth'))
            return np.load(filename)['data'], None, None
        # Otherwise, we want projected information
        filename_depth = '{}/{}.npz'.format(
            os.path.dirname(self.path), filename.format('projected/depth/{}'.format(depth_type)))
        filename_bwd_scene_flow = '{}/{}.npz'.format(
            os.path.dirname(self.path), filename.format('projected/bwd_scene_flow/{}'.format(depth_type)))
        filename_fwd_scene_flow = '{}/{}.npz'.format(
            os.path.dirname(self.path), filename.format('projected/fwd_scene_flow/{}'.format(depth_type)))
        # Load and return if exists
        try:
            # Get cached depth map
            depth = load_from_file(filename_depth, 'depth')
            if not with_scene_flow:
                return depth, None, None
            else:
                # Get cached scene flow maps
                bwd_scene_flow = load_from_file(filename_bwd_scene_flow, 'scene_flow')
                fwd_scene_flow = load_from_file(filename_fwd_scene_flow, 'scene_flow')
                return depth, bwd_scene_flow, fwd_scene_flow
        except:
            pass
        # Initialize scene flow maps
        bwd_scene_flow = fwd_scene_flow = None
        # Calculate world points if needed
        if world_points is None:
            # Get lidar information
            lidar_pose = self.get_current_or_context('pose', depth_idx, context)
            lidar_points = self.get_current_or_context('point_cloud', depth_idx, context)
            world_points = lidar_pose * lidar_points
            # Calculate scene flow in world frame of reference
            if with_scene_flow:
                bwd_scene_flow = self.get_current_or_context('bwd_scene_flow', depth_idx, context)
                if bwd_scene_flow is not None:
                    bwd_scene_flow = lidar_pose * (lidar_points + bwd_scene_flow) - world_points
                fwd_scene_flow = self.get_current_or_context('fwd_scene_flow', depth_idx, context)
                if fwd_scene_flow is not None:
                    fwd_scene_flow = lidar_pose * (lidar_points + fwd_scene_flow) - world_points
        # Create camera
        camera = self.create_camera(camera_idx, context)
        image_shape = self.get_current_or_context('rgb', camera_idx, context).size[::-1]
        # Generate depth and scene flow maps
        depth, bwd_scene_flow, fwd_scene_flow = \
            generate_proj_maps(camera, world_points, image_shape, bwd_scene_flow, fwd_scene_flow)
        # Save depth map
        save_to_file(filename_depth, 'depth', depth)
        # Save scene flow
        if with_scene_flow:
            save_to_file(filename_bwd_scene_flow, 'scene_flow', bwd_scene_flow)
            save_to_file(filename_fwd_scene_flow, 'scene_flow', fwd_scene_flow)
        # Return depth and scene flow
        return depth, bwd_scene_flow, fwd_scene_flow

    def create_pointcache(self, filename, camera_idx, sample_idx, bbox3d):
        """
        Create pointcache from stored data

        Parameters
        ----------
        filename : String
            Pointcache filename
        camera_idx : Int
            Camera index
        sample_idx : Int
            Sample index
        bbox3d : List[BBox3D]
            List of 3D bounding boxes to consider

        Returns
        -------
        pointcache : np.Array
            Output pointcache, including 3D points and surface normals [N,6]
        """
        # Get cache path
        filename_pkl = '{}/{}.pkl'.format(
            os.path.dirname(self.path), filename.format('cached_pointcache'))
        # Load and return if exists
        try:
            return pickle.load(open(filename_pkl, 'rb'))
        except:
            pass
        # Get pointcache if not provided
        pointcache = {'points': [], 'instance_id': []}
        cam_pose = self.get_current('pose', camera_idx)
        for k, b in enumerate(bbox3d):
            if 'point_cache' in b.attributes.keys():
                scene_idx, sample_idx_in_scene, _ = self.dataset.dataset_item_index[sample_idx]
                scene_dir = self.dataset.scenes[scene_idx].directory
                full_pcl = []
                for item in eval(b.attributes['point_cache']):
                    filename = os.path.join(self.path, scene_dir, 'point_cache', item['sha']) + '.npz'
                    pcl_raw = np.load(filename)['data']
                    # Get points and normals
                    pcl = np.concatenate([pcl_raw['X'], pcl_raw['Y'], pcl_raw['Z']], 1)
                    nrm = np.concatenate([pcl_raw['NX'], pcl_raw['NY'], pcl_raw['NZ']], 1)
                    tvec, wxyz = item['pose']['translation'], item['pose']['rotation']
                    offset = Pose(tvec=np.float32([tvec['x'], tvec['y'], tvec['z']]),
                                  wxyz=np.float32([wxyz['qw'], wxyz['qx'], wxyz['qy'], wxyz['qz']]))
                    pcl = cam_pose * b.pose * offset * (pcl * item['size'])
                    # Concatenate points and normals
                    full_pcl.append(np.concatenate([pcl, nrm], 1))
                # Store information
                pointcache['points'].append(np.concatenate(full_pcl, 0))
                pointcache['instance_id'].append(b.instance_id)
        # Save pointcache
        os.makedirs(os.path.dirname(filename_pkl), exist_ok=True)
        with open(filename_pkl, "wb") as f:
            pickle.dump(pointcache, f)
        # Return pointcache
        return pointcache

    def get_keypoints(self, filename, rgb):
        """
        Get stored keypoints from filename

        Parameters
        ----------
        filename : String
            Keypoint filename
        rgb : PIL Image
            Sample image

        Returns
        -------
        keypoint_coord : np.Array
            Keypoint coordinates [N,2]
        keypoint_desc : np.Array
            Keypoint descriptor [N,128]
        """
        keypoint_path = ('%s/keypoints/%s.txt.npz' % (os.path.dirname(self.path), filename)).format('rgb')
        keypoints = np.load(keypoint_path)['data']
        keypoints_coord, keypoints_desc = keypoints[:, :2], keypoints[:, 2:]
        keypoints_coord[:, 0] *= rgb.size[0] / 320
        keypoints_coord[:, 1] *= rgb.size[1] / 240
        return keypoints_coord, keypoints_desc

    def get_current(self, key, sensor_idx, as_dict=False):
        """Return current timestep of a key from a sensor"""
        current = self.sample_dgp[self.bwd_context][sensor_idx][key]
        return current if not as_dict else {0: current}

    def get_backward(self, key, sensor_idx):
        """Return backward timesteps of a key from a sensor"""
        return [] if self.bwd_context == 0 else \
            [self.sample_dgp[i][sensor_idx][key] for i in range(0, self.bwd_context)]

    def get_forward(self, key, sensor_idx):
        """Return forward timesteps of a key from a sensor"""
        return [] if self.fwd_context == 0 else \
            [self.sample_dgp[i][sensor_idx][key]
             for i in range(self.bwd_context + 1,
                            self.bwd_context + self.fwd_context + 1)]

    def get_context(self, key, sensor_idx, as_dict=False):
        """Get both backward and forward contexts"""
        context = self.get_backward(key, sensor_idx) + self.get_forward(key, sensor_idx)
        if not as_dict:
            return context
        else:
            return {key: val for key, val in zip(self.context, context)}

    def get_current_or_context(self, key, sensor_idx, context=None, as_dict=False):
        """Return current or context information for a given key and sensor index"""
        if context is None:
            return self.get_current(key, sensor_idx, as_dict=as_dict)
        else:
            return self.get_context(key, sensor_idx, as_dict=as_dict)[context]

    def get_bbox3d(self, i):
        """Return dictionary with bounding box information"""
        bbox3d = self.get_current('bounding_box_3d', i)
        bbox3d = [b for b in bbox3d if b.num_points > 0]
        pose = self.get_current('pose', i)
        return bbox3d, {
            'pose': pose.matrix,
            'corners': np.stack([(pose * b).corners for b in bbox3d], 0),
            'class_id': np.stack([b.class_id for b in bbox3d], 0),
            'instance_id': np.stack([b.instance_id for b in bbox3d]),
        }

    def has_dgp_key(self, key, sensor_idx):
        """Returns True if the DGP sample contains a certain key"""
        return key in self.sample_dgp[self.bwd_context][sensor_idx].keys()

    def get_filename(self, sample_idx, datum_idx, context=0):
        """
        Returns the filename for an index, following DGP structure

        Parameters
        ----------
        sample_idx : Int
            Sample index
        datum_idx : Int
            Datum index
        context : Int
            Context offset for the sample

        Returns
        -------
        filename : String
            Filename for the datum in that sample
        """
        scene_idx, sample_idx_in_scene, _ = self.dataset.dataset_item_index[sample_idx]
        scene_dir = self.dataset.scenes[scene_idx].directory
        filename = self.dataset.get_datum(
            scene_idx, sample_idx_in_scene + context, self.sensors[datum_idx]).datum.image.filename
        return os.path.splitext(os.path.join(os.path.basename(scene_dir),
                                             filename.replace('rgb', '{}')))[0]

    def __len__(self):
        """Length of dataset"""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get dataset sample"""

        # Get DGP sample (if single sensor, make it a list)
        self.sample_dgp = self.dataset[idx]
        self.sample_dgp = [make_list(sample) for sample in self.sample_dgp]

        # Reorganize sensors to the right order
        sensor_names = [self.get_current('datum_name', i).lower() for i in range(len(self.sensors))]
        indexes = [sensor_names.index(v) for v in self.sensors]
        self.sample_dgp = [[s[idx] for idx in indexes] for s in self.sample_dgp]

        # Loop over all cameras
        samples = []
        for i in range(self.num_cameras):

            # Filename
            filename = self.get_filename(idx, i)

            # Base sample
            sample = {
                'idx': idx,
                'tag': self.tag,
                'filename': self.relative_path({0: filename}),
                'splitname': '%s_%010d' % (self.split, idx),
                'sensor_name': self.get_current('datum_name', i),
            }

            # Image and intrinsics
            sample.update({
                'rgb': self.get_current('rgb', i, as_dict=True),
                'intrinsics': self.get_current('intrinsics', i, as_dict=True),
            })

            # If masks are returned
            if self.masks_path is not None:
                sample.update({
                    'mask': read_image(os.path.join(
                        self.masks_path, '%02d.png' % self.cameras[i]))
                })

            # If depth is returned
            if self.with_depth:
                # Get depth and scene flow maps
                depth, bwd_scene_flow, fwd_scene_flow = self.create_proj_maps(
                    filename, i, self.depth_idx, self.depth_type,
                    with_scene_flow=self.with_scene_flow)
                # Include depth map
                sample.update({
                    'depth': {0: depth}
                })

            # If input depth is returned
            if self.with_input_depth:
                sample.update({
                    'input_depth': {0: self.create_proj_maps(
                        filename, i, self.input_depth_idx, self.input_depth_type)[0]}
                })

            # If radar depth is returned
            if self.with_radar:
                sample.update({
                    'depth_radar': {0: self.create_proj_maps(
                        filename, i, self.radar_idx, 'radar')[0]}
                })

            # If optical flow is returned
            if self.with_optical_flow:
                sample['bwd_optical_flow'] = dict_remove_nones(
                    {0: self.get_bwd_optical_flow(filename)})
                sample['fwd_optical_flow'] = dict_remove_nones(
                    {0: self.get_fwd_optical_flow(filename)})

            # If semantic is returned
            if self.with_semantic:
                sample['semantic'] = \
                    {0: self.get_current('semantic_segmentation_2d', i).label}

            # If bbox3d is returned and available
            dgp_key = 'bounding_box_3d'
            if self.with_bbox3d and self.has_dgp_key(dgp_key, i):
                bbox3d, sample['bbox3d'] = self.get_bbox3d(i)
                # If returning pointcache
                if self.with_pointcache:
                    sample['pointcache'] = self.create_pointcache(
                        filename, i, idx, bbox3d)

            # If pose is returned
            if self.with_pose:
                sample.update({
                    'extrinsics': {key: val.inverse().matrix for key, val in
                                   self.get_current('extrinsics', i, as_dict=True).items()},
                    'pose': {key: val.inverse().matrix for key, val in
                             self.get_current('pose', i, as_dict=True).items()},
                })

            # If context is returned
            if self.with_context:

                # Include context images
                sample['rgb'].update(self.get_context('rgb', i, as_dict=True))

                # Create contexts filenames if extra context is required
                filename_context = []
                for context in range(-self.bwd_context, 0):
                    filename_context.append(self.get_filename(idx, i, context))
                for context in range(1, self.fwd_context + 1):
                    filename_context.append(self.get_filename(idx, i, context))
                sample['filename_context'] = filename_context

                # If context pose is returned
                if self.with_pose:
                    # Get original values to calculate relative motion
                    inv_orig_extrinsics = Pose.from_matrix(sample['extrinsics'][0]).inverse()
                    inv_orig_pose = Pose.from_matrix(sample['pose'][0]).inverse()
                    sample['extrinsics'].update(
                        {key: (inv_orig_extrinsics * val.inverse()).matrix for key, val in zip(
                            self.context, self.get_context('extrinsics', i))})
                    sample['pose'].update(
                        {key: (val.inverse()).matrix for key, val in zip(
                            self.context, self.get_context('pose', i))})

                # If context depth is returned
                if self.with_depth_context:
                    depth_scene_flow = [
                        self.create_proj_maps(
                            filename, i, self.depth_idx, self.depth_type,
                            context=k, with_scene_flow=self.with_scene_flow)
                        for k, filename in enumerate(filename_context)]
                    sample['depth'].update(
                        {key: val for key, val in zip(
                            self.context, [dsf[0] for dsf in depth_scene_flow])})

                # If context optical flow is returned
                if self.with_optical_flow_context:
                        sample['fwd_optical_flow'].update(dict_remove_nones({
                            key: self.get_fwd_optical_flow(filename)
                            for key, filename in zip(self.context, filename_context)
                        }))
                        sample['bwd_optical_flow'].update(dict_remove_nones({
                            key: self.get_bwd_optical_flow(filename)
                            for key, filename in zip(self.context, filename_context)
                        }))

                # If context semantic is returned
                if self.with_semantic_context:
                    sample['semantic'].update({
                        key: self.get_context('semantic_segmentation_2d', key) for key in self.context
                    })

            samples.append(sample)

        # Make relative poses
        samples = make_relative_pose(samples)

        # Add LiDAR information

        lidar_sample = {}
        if self.with_lidar:

            # Include pointcloud information
            lidar_sample.update({
                'lidar_pointcloud': self.get_current('point_cloud', self.depth_idx),
            })

            # If pose is included
            if self.with_pose:
                lidar_sample.update({
                    'lidar_extrinsics': self.get_current('extrinsics', self.depth_idx).matrix,
                    'lidar_pose': self.get_current('pose', self.depth_idx).matrix,
                })

            # If scene flow is included
            if self.with_scene_flow:
                if self.with_bwd_context:
                    lidar_sample.update({
                        'lidar_bwd_scene_flow': self.get_current(
                            'bwd_scene_flow', self.depth_idx)
                    })
                if self.with_fwd_context:
                    lidar_sample.update({
                        'lidar_fwd_scene_flow': self.get_current(
                            'fwd_scene_flow', self.depth_idx)
                    })

            # If extra context is included
            if self.with_extra_context:
                lidar_sample['lidar_context'] = self.get_context('point_cloud', self.depth_idx)
                # If context pose is included
                if self.with_pose:
                    # Get original values to calculate relative motion
                    orig_extrinsics = Pose.from_matrix(lidar_sample['lidar_extrinsics'])
                    orig_pose = Pose.from_matrix(lidar_sample['lidar_pose'])
                    lidar_sample.update({
                        'lidar_extrinsics_context':
                            [(orig_extrinsics.inverse() * extrinsics).inverse().matrix
                             for extrinsics in self.get_context('extrinsics', self.depth_idx)],
                        'lidar_pose_context':
                            [(orig_pose.inverse() * pose).inverse().matrix
                             for pose in self.get_context('pose', self.depth_idx)],
                    })
                # If scene flow is included
                if self.with_scene_flow:
                    if self.with_bwd_context:
                        lidar_sample.update({
                            'lidar_fwd_scene_flow_context': self.get_context(
                                'fwd_scene_flow', self.depth_idx)
                        })
                    if self.with_fwd_context:
                        lidar_sample.update({
                            'lidar_bwd_scene_flow_context': self.get_context(
                                'bwd_scene_flow', self.depth_idx)
                        })

        # Add RADAR information

        radar_sample = {}
        if self.with_radar:

            # Include pointcloud information
            radar_sample.update({
                'radar_pointcloud': self.get_current('point_cloud', self.radar_idx),
            })

            # If pose is included
            if self.with_pose:
                radar_sample.update({
                    'radar_extrinsics': self.get_current('extrinsics', self.radar_idx).matrix,
                    'radar_pose': self.get_current('pose', self.radar_idx).matrix,
                })

        # # Apply same data transformations for all sensors
        # if self.data_transform:
        #     sample = [self.data_transform(smp) for smp in sample]
        #     # lidar_sample = self.data_transform(lidar_sample)
        #     # radar_sample = self.data_transform(radar_sample)

        # Apply same data transformations for all sensors
        if self.data_transform:
            samples = self.data_transform(samples)
            # lidar_sample = self.data_transform(lidar_sample)
            # radar_sample = self.data_transform(radar_sample)

        # Return sample (stacked if necessary)
        return stack_sample(samples, lidar_sample, radar_sample)
