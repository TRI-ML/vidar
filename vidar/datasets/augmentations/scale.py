# Copyright 2023 Toyota Research Institute.  All rights reserved.


def scale_sample(sample, params):
    """Scale intrinsics from a sample given parameters (scale ratio)"""
    for key in sample[0].keys():
        if key == 'intrinsics':
            for tgt in sample[0]['intrinsics'].keys():
                print(sample[0]['intrinsics'][tgt])
                sample[0]['intrinsics'][tgt][0, 0] *= params[0]
                sample[0]['intrinsics'][tgt][1, 1] *= params[0]
                print(sample[0]['intrinsics'][tgt])
        elif key == 'depth':
            for tgt in sample[0]['depth'].keys():
                print(sample[0]['depth'][tgt].min(), sample[0]['depth'][tgt].max())
                sample[0]['depth'][tgt] /= params[0]
                print(sample[0]['depth'][tgt].min(), sample[0]['depth'][tgt].max())
    return sample
