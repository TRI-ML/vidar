# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch
from pytorch3d.transforms.rotation_conversions import \
    matrix_to_euler_angles, euler_angles_to_matrix

from knk_vision.vidar.vidar.utils.data import keys_in
from knk_vision.vidar.vidar.utils.decorators import iterate1, iterate12
from knk_vision.vidar.vidar.utils.types import is_tensor, is_list, is_seq


def flip_lr_fn(tensor):
    """Function to flip horizontally"""
    return torch.flip(tensor, [-1])


def flip_flow_lr_fn(flow):
    """Function to flip a flow map horizontally"""
    flow_flip = torch.flip(flow, [3])
    flow_flip[:, :1, :, :] *= -1
    return flow_flip.contiguous()


def flip_intrinsics_lr_fn(K, shape):
    """Function to flip intrinsics horizontally"""
    K = K.clone()
    K[:, 0, 2] = shape[-1] - K[:, 0, 2]
    return K


def flip_pose_lr_fn(T):
    """Function to flip pose horizontally"""
    rot = T[:, :3, :3]
    axis = matrix_to_euler_angles(rot, convention='XYZ')
    axis[:, [1, 2]] = axis[:, [1, 2]] * -1
    rot = euler_angles_to_matrix(axis, convention='XYZ')
    T[:, :3, :3] = rot
    T[:, 0, -1] = - T[:, 0, -1]
    return T


@iterate1
def flip_lr(tensor, flip=True):
    """Flip a tensor horizontally (i.e. last dimension)"""
    # Not flipping option
    if not flip:
        return tensor
    # If it's a list, repeat
    if is_list(tensor):
        return [flip_lr(t) for t in tensor]
    # Return flipped tensor
    if tensor.dim() == 5:
        return torch.stack([flip_lr_fn(tensor[:, i])
                            for i in range(tensor.shape[1])], 1)
    else:
        return flip_lr_fn(tensor)


@iterate1
def flip_flow_lr(flow, flip=True):
    """Flip a flow map (optical flow or scene flow) horizontally"""
    # Not flipping option
    if not flip:
        return flow
    # If it's a list, repeat
    if is_list(flow):
        return [flip_flow_lr(f) for f in flow]
    # Flip flow and invert first dimension
    if flow.dim() == 5:
        return torch.stack([flip_flow_lr_fn(flow[:, i])
                            for i in range(flow.shape[1])], 1)
    else:
        return flip_flow_lr_fn(flow)


@iterate12
def flip_intrinsics_lr(K, shape, flip=True):
    """Flip camera intrinsics horizontally"""
    # Not flipping option
    if not flip:
        return K
    # If shape is a tensor, use it's dimensions
    if is_tensor(shape):
        shape = shape.shape
    # Flip horizontal information (first row)
    if K.dim() == 4:
        return torch.stack([flip_intrinsics_lr_fn(K[:, i], shape)
                            for i in range(K.shape[1])], 1)
    else:
        return flip_intrinsics_lr_fn(K, shape)


def flip_pose_lr(pose, flip=True):
    """Flip pose horizontally"""
    # Not flipping option
    if not flip:
        return pose
    # Repeat for all pose keys
    for key in pose.keys():
        # Get pose key
        if key == 0:
            if pose[key].dim() == 3:
                continue
            elif pose[key].dim() == 4:
                T = pose[key][:, 1:].clone()
            else:
                raise ValueError('Invalid pose dimension')
        else:
            T = pose[key].clone()
        # Flip pose
        if T.dim() == 4:
            T = torch.stack([flip_pose_lr_fn(T[:, i])
                             for i in range(T.shape[1])], 1)
        else:
            T = flip_pose_lr_fn(T)
        # Store flipped value back
        if key == 0:
            pose[key][:, 1:] = T
        else:
            pose[key] = T
    # Return flipped pose
    return pose


def flip_batch(batch, flip=True):
    """Flip batch"""
    # Not flipping option
    if not flip:
        return batch
    # If it's a list, repeat
    if is_seq(batch):
        return [flip_batch(b) for b in batch]
    # Flip batch
    flipped_batch = {"flipped": True}
    # Keys to not flip
    for key in keys_in(batch, ['idx', 'tag', 'filename', 'splitname', 'extrinsics']):
        flipped_batch[key] = batch[key]
    # Tensor flipping
    for key in keys_in(batch, ['rgb', 'mask', 'mask_rgb', 'input_depth', 'depth', 'semantic']):
        flipped_batch[key] = flip_lr(batch[key])
    # Intrinsics flipping
    for key in keys_in(batch, ['intrinsics']):
        flipped_batch[key] = flip_intrinsics_lr(batch[key], batch['rgb'])
    # Pose flipping
    for key in keys_in(batch, ['pose']):
        flipped_batch[key] = flip_pose_lr(batch[key])
    return flipped_batch


def flip_predictions(predictions, flip=True):
    """Flip predictions"""
    # Not flipping option
    if not flip:
        return predictions
    # Flip predictions
    flipped_predictions = {}
    for key in predictions.keys():
        if key.startswith('depth'):
            flipped_predictions[key] = flip_lr(predictions[key])
        if key.startswith('pose'):
            flipped_predictions[key] = flip_pose_lr(predictions[key])
    # Return flipped predictions
    return flipped_predictions


def flip_output(output, flip=True):
    """Flip output"""
    # Not flipping option
    if not flip:
        return output
    # If it's a list, repeat
    if is_seq(output):
        return [flip_output(b) for b in output]
    # Flip output
    flipped_output = {}
    # Do not flip loss and metrics
    for key in keys_in(output, ['loss', 'metrics']):
        flipped_output[key] = output[key]
    # Flip predictions
    flipped_output['predictions'] = flip_predictions(output['predictions'])
    # Return flipped output
    return flipped_output
