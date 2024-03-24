# Copyright 2023 Toyota Research Institute.  All rights reserved.

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as tf

from knk_vision.vidar.vidar.arch.losses.BaseLoss import BaseLoss
from knk_vision.vidar.vidar.geometry.camera import Camera
from knk_vision.vidar.vidar.utils.depth import inv2depth
from knk_vision.vidar.vidar.utils.tensor import match_scales


def view_synthesis(ref_image, depth, ref_cam, cam,
                   mode='bilinear', padding_mode='zeros', align_corners=True):
    """Grid sample for view synthesis"""
    assert depth.shape[1] == 1, 'Depth map should have C=1'
    # Reconstruct world points from target_camera
    world_points = cam.reconstruct(depth, frame='w')
    # Project world points onto reference camera
    ref_coords = ref_cam.project(world_points, frame='w')
    # View-synthesis given the projected reference points
    return tf.grid_sample(ref_image, ref_coords, mode=mode,
                          padding_mode=padding_mode, align_corners=align_corners)


def gradient_x(image):
    """ Calculates image gradient along x-axis"""
    return image[:, :, :, :-1] - image[:, :, :, 1:]


def gradient_y(image):
    """ Calculates image gradient along y-axis"""
    return image[:, :, :-1, :] - image[:, :, 1:, :]


def inv_depths_normalize(inv_depths):
    """ Normalizes inverse depth maps by their mean"""
    mean_inv_depths = [inv_depth.mean(2, True).mean(3, True) for inv_depth in inv_depths]
    return [inv_depth / mean_inv_depth.clamp(min=1e-6)
            for inv_depth, mean_inv_depth in zip(inv_depths, mean_inv_depths)]


def calc_smoothness(inv_depths, images, num_scales):
    """ Calculates smoothness loss for inverse depth maps"""
    inv_depths_norm = inv_depths_normalize(inv_depths)
    inv_depth_gradients_x = [gradient_x(d) for d in inv_depths_norm]
    inv_depth_gradients_y = [gradient_y(d) for d in inv_depths_norm]

    image_gradients_x = [gradient_x(image) for image in images]
    image_gradients_y = [gradient_y(image) for image in images]

    weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_x]
    weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_y]

    # Note: Fix gradient addition
    smoothness_x = [inv_depth_gradients_x[i] * weights_x[i] for i in range(num_scales)]
    smoothness_y = [inv_depth_gradients_y[i] * weights_y[i] for i in range(num_scales)]
    return smoothness_x, smoothness_y


def SSIM(x, y, C1=1e-4, C2=9e-4, kernel_size=3, stride=1):
    """
    Structural Similarity (SSIM) distance between two images.

    Parameters
    ----------
    x,y : torch.Tensor
        Input images [B,3,H,W]
    C1,C2 : float
        SSIM parameters
    kernel_size,stride : int
        Convolutional parameters

    Returns
    -------
    ssim : torch.Tensor
        SSIM distance [1]
    """
    pool2d = nn.AvgPool2d(kernel_size, stride=stride)
    refl = nn.ReflectionPad2d(1)

    x, y = refl(x), refl(y)
    mu_x = pool2d(x)
    mu_y = pool2d(y)

    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = pool2d(x.pow(2)) - mu_x_sq
    sigma_y = pool2d(y.pow(2)) - mu_y_sq
    sigma_xy = pool2d(x * y) - mu_x_mu_y
    v1 = 2 * sigma_xy + C2
    v2 = sigma_x + sigma_y + C2

    ssim_n = (2 * mu_x_mu_y + C1) * v1
    ssim_d = (mu_x_sq + mu_y_sq + C1) * v2
    ssim = ssim_n / ssim_d

    return ssim


class MultiViewPhotometricLoss(BaseLoss, ABC):
    """
    Self-Supervised multiview photometric loss.
    It takes two images, a depth map and a pose transformation to produce a
    reconstruction of one image from the perspective of the other, and calculates
    the difference between them

    Parameters
    ----------
    num_scales : int
        Number of inverse depth map scales to consider
    ssim_loss_weight : float
        Weight for the SSIM loss
    occ_reg_weight : float
        Weight for the occlusion regularization loss
    smooth_loss_weight : float
        Weight for the smoothness loss
    C1,C2 : float
        SSIM parameters
    photometric_reduce_op : str
        Method to reduce the photometric loss
    disp_norm : bool
        True if inverse depth is normalized for
    clip_loss : float
        Threshold for photometric loss clipping
    progressive_scaling : float
        Training percentage for progressive scaling (0.0 to disable)
    padding_mode : str
        Padding mode for view synthesis
    automask_loss : bool
        True if automasking is enabled for the photometric loss
    kwargs : dict
        Extra parameters
    """
    def __init__(self, num_scales=4, ssim_loss_weight=0.85, occ_reg_weight=0.1, smooth_loss_weight=0.1,
                 C1=1e-4, C2=9e-4, photometric_reduce_op='mean', disp_norm=True, clip_loss=0.5,
                 progressive_scaling=0.0, padding_mode='zeros', automask_loss=False, **kwargs):
        super().__init__()
        self.n = num_scales
        self.ssim_loss_weight = ssim_loss_weight
        self.occ_reg_weight = occ_reg_weight
        self.smooth_loss_weight = smooth_loss_weight
        self.C1 = C1
        self.C2 = C2
        self.photometric_reduce_op = photometric_reduce_op
        self.disp_norm = disp_norm
        self.clip_loss = clip_loss
        self.padding_mode = padding_mode
        self.automask_loss = automask_loss

        # Asserts
        if self.automask_loss:
            assert self.photometric_reduce_op == 'min', \
                'For automasking only the min photometric_reduce_op is supported.'

    @property
    def logs(self):
        """Returns class logs."""
        return {
            'num_scales': self.n,
        }

    def warp_ref_image(self, inv_depths, ref_image, K, ref_K, pose):
        """
        Warps a reference image to produce a reconstruction of the original one.

        Parameters
        ----------
        inv_depths : list[torch.Tensor]
            Inverse depth map of the original image [B,1,H,W]
        ref_image : torch.Tensor
            Reference RGB image [B,3,H,W]
        K : torch.Tensor
            Original camera intrinsics [B,3,3]
        ref_K : torch.Tensor
            Reference camera intrinsics [B,3,3]
        pose : Pose
            Original -> Reference camera transformation

        Returns
        -------
        ref_warped : torch.Tensor
            Warped reference image (reconstructing the original one) [B,3,H,W]
        """
        B, _, H, W = ref_image.shape
        device = ref_image.device
        # Generate cameras for all scales
        cams, ref_cams = [], []
        for i in range(self.n):
            _, _, DH, DW = inv_depths[i].shape
            scale_factor = DW / float(W)
            cams.append(Camera(K=K.float()).scaled(scale_factor).to(device))
            ref_cams.append(Camera(K=ref_K.float(), Tcw=pose).scaled(scale_factor).to(device))
        # View synthesis
        depths = [inv2depth(inv_depths[i]) for i in range(self.n)]
        ref_images = match_scales(ref_image, inv_depths, self.n)
        ref_warped = [view_synthesis(
            ref_images[i], depths[i], ref_cams[i], cams[i],
            padding_mode=self.padding_mode) for i in range(self.n)]
        # Return warped reference image
        return ref_warped

    def SSIM(self, x, y, kernel_size=3):
        """
        Calculates the SSIM (Structural Similarity) loss

        Parameters
        ----------
        x,y : torch.Tensor
            Input images [B,3,H,W]
        kernel_size : int
            Convolutional parameter

        Returns
        -------
        ssim : torch.Tensor
            SSIM loss [1]
        """
        ssim_value = SSIM(x, y, C1=self.C1, C2=self.C2, kernel_size=kernel_size)
        return torch.clamp((1. - ssim_value) / 2., 0., 1.)

    def calc_photometric_loss(self, t_est, images):
        """
        Calculates the photometric loss (L1 + SSIM)
        Parameters
        ----------
        t_est : list[torch.Tensor]
            List of warped reference images in multiple scales [B,3,H,W]
        images : list[torch.Tensor]
            List of original images in multiple scales [B,3,H,W]

        Returns
        -------
        photometric_loss : list[torch.Tensor]
            Photometric loss [B,1,H,W]
        """
        # L1 loss
        l1_loss = [torch.abs(t_est[i] - images[i])
                   for i in range(self.n)]
        # SSIM loss
        if self.ssim_loss_weight > 0.0:
            ssim_loss = [self.SSIM(t_est[i], images[i], kernel_size=3)
                         for i in range(self.n)]
            # Weighted Sum: alpha * ssim + (1 - alpha) * l1
            photometric_loss = [self.ssim_loss_weight * ssim_loss[i].mean(1, True) +
                                (1 - self.ssim_loss_weight) * l1_loss[i].mean(1, True)
                                for i in range(self.n)]
        else:
            photometric_loss = l1_loss
        # Clip loss
        if self.clip_loss > 0.0:
            for i in range(self.n):
                mean, std = photometric_loss[i].mean(), photometric_loss[i].std()
                photometric_loss[i] = torch.clamp(
                    photometric_loss[i], max=float(mean + self.clip_loss * std))
        # Return total photometric loss
        return photometric_loss

    def reduce_photometric_loss(self, photometric_losses):
        """
        Combine the photometric loss from all context images

        Parameters
        ----------
        photometric_losses : list[list[torch.Tensor]]
            Pixel-wise photometric losses from the entire context [B,3,H,W]

        Returns
        -------
        photometric_loss : torch.Tensor
            Reduced photometric loss [1]
        """
        # Reduce function
        def reduce_function(losses):
            if self.photometric_reduce_op == 'mean':
                return sum([l.mean() for l in losses]) / len(losses)
            elif self.photometric_reduce_op == 'min':
                return torch.cat(losses, 1).min(1, True)[0].mean()
            else:
                raise NotImplementedError(
                    'Unknown photometric_reduce_op: {}'.format(self.photometric_reduce_op))
        # Reduce photometric loss
        photometric_loss = sum([reduce_function(photometric_losses[i])
                                for i in range(self.n)]) / self.n
        # Store and return reduced photometric loss
        return photometric_loss

    def calc_smoothness_loss(self, inv_depths, images):
        """
        Calculates the smoothness loss for inverse depth maps.

        Parameters
        ----------
        inv_depths : list[torch.Tensor]
            Predicted inverse depth maps for all scales [B,1,H,W]
        images : list[torch.Tensor]
            Original images for all scales [B,3,H,W]

        Returns
        -------
        smoothness_loss : torch.Tensor
            Smoothness loss [1]
        """
        # Calculate smoothness gradients
        smoothness_x, smoothness_y = calc_smoothness(inv_depths, images, self.n)
        # Calculate smoothness loss
        smoothness_loss = sum([(smoothness_x[i].abs().mean() +
                                smoothness_y[i].abs().mean()) / 2 ** i
                               for i in range(self.n)]) / self.n
        # Apply smoothness loss weight
        smoothness_loss = self.smooth_loss_weight * smoothness_loss
        # Store and return smoothness loss
        return smoothness_loss



