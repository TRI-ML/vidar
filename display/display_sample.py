# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import json

import numpy as np
from camviz import BBox3D
from camviz import Camera as CameraCV
from camviz import Draw

from vidar.geometry.camera import Camera
from vidar.geometry.pose import Pose
from vidar.utils.data import make_batch, fold_batch, modrem
from vidar.utils.flip import flip_batch
from vidar.utils.viz import viz_depth, viz_optical_flow, viz_semantic


def change_key(dic, c, n):
    steps = sorted(dic.keys())
    return steps[(steps.index(c) + n) % len(steps)]


def display_sample(data, flip=False):

    tasks = ['rgb', 'depth', 'fwd_optical_flow', 'bwd_optical_flow','semantic']
    cam_colors = ['red', 'blu', 'gre', 'yel', 'mag', 'cya'] * 100

    data = make_batch(data)
    if flip:
        data = flip_batch(data)
    data = fold_batch(data)

    rgb = data['rgb']
    intrinsics = data['intrinsics']
    depth = data['depth']
    pose = data['pose']

    pose = Pose.from_dict(pose, to_global=True)
    cam = Camera.from_dict(intrinsics, rgb, pose)

    num_cams = rgb[0].shape[0]
    wh = rgb[0].shape[-2:][::-1]

    keys = [key for key in tasks if key in data.keys()]

    points = {}
    for key, val in cam.items():
        points[key] = cam[key].reconstruct_depth_map(
            depth[key], to_world=True).reshape(num_cams, 3, -1).permute(0, 2, 1)

    draw = Draw((wh[0] * 4, wh[1] * 3), width=2100)
    draw.add2DimageGrid('cam', (0.0, 0.0, 0.5, 1.0), n=(3, 2), res=wh)
    draw.add3Dworld('wld', (0.5, 0.0, 1.0, 1.0), pose=cam[0].Tcw.T[0])

    draw.addTexture('cam', n=num_cams)
    draw.addBuffer3f('lidar', 1000000, n=num_cams)
    draw.addBuffer3f('color', 1000000, n=num_cams)

    with_bbox3d = 'bbox3d' in data
    if with_bbox3d:
        bbox3d_corners = [[BBox3D(b) for b in bb] for bb in data['bbox3d']['corners']]

    with_pointcache = 'pointcache' in data
    if with_pointcache:
        pointcache = np.concatenate([np.concatenate(pp, 0) for pp in data['pointcache']['points']], 0)
        draw.addBufferf('pointcache', pointcache[:, :3])

    camcv = []
    for i in range(num_cams):
        camcv.append({key: CameraCV.from_vidar(val, i) for key, val in cam.items()})

    t, k = 0, 0
    key = keys[k]
    change = True
    color = True

    while draw.input():
        if draw.SPACE:
            color = not color
            change = True
        if draw.RIGHT:
            change = True
            k = (k + 1) % len(keys)
            while t not in data[keys[k]].keys():
                k = (k + 1) % len(keys)
            key = keys[k]
        if draw.LEFT:
            change = True
            k = (k - 1) % len(keys)
            while t not in data[keys[k]].keys():
                k = (k - 1) % len(keys)
            key = keys[k]
        if draw.UP:
            change = True
            t = change_key(data[key], t, 1)
            while t not in data[keys[k]].keys():
                t = change_key(data[key], t, 1)
        if draw.DOWN:
            change = True
            t = change_key(data[key], t, -1)
            while t not in data[keys[k]].keys():
                t = change_key(data[key], t, -1)
        if change:
            change = False
            for i in range(num_cams):
                img = data[key][t][i]
                if key == 'depth':
                    img = viz_depth(img, filter_zeros=True)
                elif key in ['fwd_optical_flow', 'bwd_optical_flow']:
                    img = viz_optical_flow(img)
                elif key == 'semantic':
                    ontology = json.load(open('vidar/datasets/ontologies/%s.json' % data['tag'][0]))
                    img = viz_semantic(img, ontology)
                draw.updTexture('cam%d' % i, img)
                draw.updBufferf('lidar%d' % i, points[t][i])
                draw.updBufferf('color%d' % i, data['rgb'][t][i])

        draw.clear()
        for i in range(num_cams):
            draw['cam%d%d' % modrem(i, 2)].image('cam%d' % i)
            draw['wld'].size(1).color(cam_colors[i]).points('lidar%d' % i, ('color%d' % i) if color else None)
            for cam_key, cam_val in camcv[i].items():
                clr = cam_colors[i] if cam_key == t else 'gra'
                tex = 'cam%d' % i if cam_key == t else None
                draw['wld'].object(cam_val, color=clr, tex=tex)
            if with_bbox3d:
                [[draw['wld'].object(b) for b in bb] for bb in bbox3d_corners]
            if with_pointcache:
                draw['wld'].color('whi').points('pointcache')

        draw.update(30)
