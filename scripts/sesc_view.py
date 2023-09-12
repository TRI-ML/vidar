# Copyright 2023 Toyota Research Institute.  All rights reserved.

import argparse
import os

from PIL import Image
import torch

from camviz import Draw
from camviz.objects.camera import Camera as CVCam

from vidar.arch.models.releaseSesc.SescInference import SescInference
from vidar.geometry.camera import Camera
from vidar.geometry.pose_utils import pose_vec2mat_homogeneous
from vidar.utils.viz import viz_depth

cam_colors = ['red', 'yel', 'gre', 'blu', 'mag', 'cya'] * 100  # Full
camera_scale = 1.
draw_scale = .6
cap_resize = (1024, 1024)
pcl_size = 3


def parser(feed_by_lst=None):
    """ Parser to handle the configuration file"""
    parser = argparse.ArgumentParser(description='Project trained model ')
    parser.add_argument('cfg', type=str, help='Data downloaded directory')
    if feed_by_lst is not None:
        args = parser.parse_args(feed_by_lst)
    else:
        args = parser.parse_args()
    return args


if __name__ == '__main__':
    arg = parser()
    assert os.path.exists(arg.cfg), "No such file or directory: {}".format(arg.pkl)

    predictor = SescInference(arg.cfg)

    scene_first = predictor.first_scenario
    hw = predictor.hw
    view_keys = predictor.keys

    reloaded = predictor.reload_batch(scene_first)

    print(" --> Read: {}".format(scene_first))

    wh = hw[::-1]
    draw = Draw((400, 400))
    draw.setSize((int(draw.wh[0] * draw_scale), int(draw.wh[1] * draw_scale)))
    draw.add3Dworld(
        'wld', (0.50, 0.00, 1.00, 1.00),
        # translation + quaternion
        pose=pose_vec2mat_homogeneous(
            torch.tensor([-1.5, -12., -20., -0.52, 0., 0., ]).unsqueeze(0)).squeeze(0).detach().cpu().numpy()
    )
    draw.add2Dimage('rgb_0', (0.00, 0.00, 0.25, 0.16), res=wh)
    draw.add2Dimage('rgb_1', (0.00, 0.16, 0.25, 0.33), res=wh)
    draw.add2Dimage('rgb_2', (0.00, 0.33, 0.25, 0.50), res=wh)
    draw.add2Dimage('rgb_3', (0.00, 0.50, 0.25, 0.67), res=wh)
    draw.add2Dimage('rgb_4', (0.00, 0.67, 0.25, 0.83), res=wh)
    draw.add2Dimage('rgb_5', (0.00, 0.83, 0.25, 1.00), res=wh)
    draw.add2Dimage('dep_0', (0.25, 0.00, 0.50, 0.16), res=wh)
    draw.add2Dimage('dep_1', (0.25, 0.16, 0.50, 0.33), res=wh)
    draw.add2Dimage('dep_2', (0.25, 0.33, 0.50, 0.50), res=wh)
    draw.add2Dimage('dep_3', (0.25, 0.50, 0.50, 0.67), res=wh)
    draw.add2Dimage('dep_4', (0.25, 0.67, 0.50, 0.83), res=wh)
    draw.add2Dimage('dep_5', (0.25, 0.83, 0.50, 1.00), res=wh)

    cnt = 0
    change = True
    color = True
    print('Enter `Q` to quit, `S` to search sequence ID')
    while draw.input():
        if draw.SPACE:
            print("\n Color shifting ...")
            color = not color
            change = True
        if draw.KEY_Q:
            break
        if draw.RIGHT:
            change = True
            print("\n Predicting NEXT scenario ...")
            reloaded = predictor.reload_next_scenario()
            pass
        if draw.LEFT:
            change = True
            print("\n Predicting PREVIOUS scenario ...")
            reloaded = predictor.reload_previous_scenario()
            pass
        if draw.KEY_S:
            change = True
            print("\n>>> Enter desired scenario (e.g. 000082))")
            scenario_overwrite = input()
            reloaded = predictor.reload_batch(scenario_overwrite)
            print("\n Inference s:{}...".format(scenario_overwrite))
            pass
        if draw.KEY_C:
            print('Captured!!')
            rgb_img = draw.to_image()[..., ::-1]
            pil_img = Image.fromarray(rgb_img)
            pil_img = pil_img.resize(cap_resize)
            pil_img.save('debug/camviz/self_fsm{}.jpg'.format(str(cnt)))
            cnt += 1  # Inference
        if change:
            change = False
            out = predictor.forward_fsm(batch=reloaded)
            print("Inference ID == `{}` done".format(predictor.set_scenario))
            intrinsics, rgb, pose_pred, pred_depth_dict, pose_gt \
                = [out[item] for item in
                   ["intrinsics", "rgb", "pred_pose", "pred_depth", "gt_pose"]
                   ]
            # Only for drawing the GP camera
            cams = {key: Camera(intrinsics[(0, key[1])], rgb[key], pose_pred[key]) for key in view_keys}
            points = {}
            for key in view_keys:
                points[key] = cams[key].reconstruct_depth_map(pred_depth_dict[key][0], to_world=True)
                pass
            for i, key in enumerate(view_keys):
                draw.addTexture('rgb_%d_%d' % key, rgb[key][0])
                draw.addTexture('dep_%d_%d' % key, viz_depth(pred_depth_dict[key][0]))
                draw.addBufferf('pts_%d_%d' % key, points[key][0])
                draw.addBufferf('clr_%d_%d' % key, rgb[key][0])
            cvcams = {key: CVCam.from_vidar(val, b=0, scale=camera_scale) for key, val in cams.items()}
            cams_gt_pose = {key: Camera(intrinsics[(0, key[1])], rgb[key], pose_gt[key]) for key in view_keys}
            cvcams_gt = {key: CVCam.from_vidar(val, b=0, scale=camera_scale) for key, val in cams_gt_pose.items()}
        else:
            pass
        # Clear window
        draw.clear()
        # Draw image textures on their respective screens
        for i, key in enumerate(view_keys):
            draw['rgb_%d' % i].image('rgb_%d_%d' % key)
            draw['dep_%d' % i].image('dep_%d_%d' % key)
            draw['wld'].object(cvcams[key],
                               tex=None,
                               # tex='rgb_%d_%d' % key
                               color=cam_colors[i]
                               )
            draw['wld'].object(cvcams_gt[key],
                               # tex='rgb_%d_%d' % key,
                               tex=None,
                               color='gra')
            draw['wld'].size(pcl_size).color(cam_colors[i]).points('pts_%d_%d' % key,
                                                                   ('clr_%d_%d' % key) if color else None)
            pass
        # Update window
        draw.update(30)
        pass
