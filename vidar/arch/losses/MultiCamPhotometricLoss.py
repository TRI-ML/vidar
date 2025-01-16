# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch

from knk_vision.vidar.vidar.arch.losses.MultiViewPhotometricLoss import MultiViewPhotometricLoss
from knk_vision.vidar.vidar.arch.networks.layers.fsm.utils import coords_from_motion, warp_from_coords, mask_from_coords
from knk_vision.vidar.vidar.utils.depth import inv2depth
from knk_vision.vidar.vidar.utils.tensor import match_scales
from knk_vision.vidar.vidar.utils.types import is_list, is_double_list


class MultiCamPhotometricLoss(MultiViewPhotometricLoss):
    """Photometric loss for multi-camera setups"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Large value for loss masking
        self.inf = 999999
        self.align_corners = True

    def warp(self, rgb_context, inv_depths, cam, cam_context,
             scene_flow=None, with_mask=True):
        """Warp reference images to target using inverse depth and camera motion"""
        # Initialize empty warp and mask list
        warps_context, masks_context = [], []
        # If mask is available, use it instead of calculating
        if is_list(with_mask):
            masks_context, with_mask = with_mask, False
        # Match inverse depth scales on reference images if necessary
        rgbs_context = rgb_context if is_double_list(rgb_context) else \
            [match_scales(rgb, inv_depths, self.n, align_corners=self.align_corners)
             for rgb in rgb_context]
        # Warp each reference image to target
        for j, (ref_rgbs, ref_cam) in enumerate(zip(rgbs_context, cam_context)):
            # Get warping coordinates
            ref_coords = [coords_from_motion(ref_cam, inv2depth(inv_depths[i]), cam) for i in range(self.n)]
            # Get warped images
            warps_context.append([warp_from_coords(
                ref_rgbs[i], ref_coords[i], align_corners=self.align_corners, padding_mode='zeros') #'reflection')
                for i in range(self.n)])
            # Get warped masks if requested
            if with_mask:
                masks_context.append([mask_from_coords(ref_coords[i]) for i in range(self.n)])
        # Return warped reference images
        return warps_context, masks_context

    def reduce_photometric_loss_min(self, photometric_losses,
                                    unwarped_photometric_losses=None):
        """
        Reduce photometric losses using minimum reprojection error

        Parameters
        ----------
        photometric_losses : list[Tensor]
            Photometric losses for each warped image [B,3,H,W]

        unwarped_photometric_losses : list[Tensor]
            Unwarped photometric losses for each reference image [B,3,H,W]

        Returns
        -------
        reduced_photometric_loss : Tensor
            Reduced loss value (single value)
        min_photometric_loss : Tensor
            Masked photometric loss [B,1,H,W]
        """
        # Calculate minimum photometric losses
        min_photometric_loss = [torch.cat(losses, 1).min(1, True)[0]
                                for losses in photometric_losses]
        # Get invalid minimum mask
        valid_mask = [warped < self.inf for warped in min_photometric_loss]
        # If unwarped photometric losses are provided
        if unwarped_photometric_losses is not None and \
                len(unwarped_photometric_losses[0]) > 0:
            # Calculate minimum unwarped photometric losses
            min_unwarped_photometric_loss = [torch.cat(losses, 1).min(1, True)[0]
                                             for losses in unwarped_photometric_losses]
            # Get minimum mask (warped < unwarped)
            minimum_mask = [warped < unwarped for warped, unwarped in
                            zip(min_photometric_loss, min_unwarped_photometric_loss)]
            # Update valid mask with minimum mask
            valid_mask = [minimum & valid for minimum, valid in
                          zip(minimum_mask, valid_mask)]
        # Get reduced photometric loss
        reduced_photometric_loss = sum(
            [loss[mask].mean() for mask, loss in
             zip(valid_mask, min_photometric_loss)]) / len(min_photometric_loss)
        # Mask min photometric loss for visualization
        for i in range(len(min_photometric_loss)):
            min_photometric_loss[i][~valid_mask[i]] = 0
        # Store and return reduced photometric loss
        return reduced_photometric_loss, min_photometric_loss

    def reduce_photometric_loss_mean(self, photometric_losses,
                                     unwarped_photometric_losses=None):
        """
        Reduce photometric losses using minimum reprojection error

        Parameters
        ----------
        photometric_losses : list[Tensor]
            Photometric losses for each warped image [B,3,H,W]

        unwarped_photometric_losses : list[Tensor]
            Unwarped photometric losses for each reference image [B,3,H,W]

        Returns
        -------
        reduced_photometric_loss : Tensor
            Reduced loss value (single value)
        min_photometric_loss : Tensor
            Masked photometric loss [B,1,H,W]
        """
        valid_mask = [[w < self.inf for w in warped] for warped in photometric_losses]
        if unwarped_photometric_losses is not None and \
                len(unwarped_photometric_losses[0]) > 0:
            # Get minimum mask (warped < unwarped)
            minimum_mask = [[w < u for w, u in zip(warped, unwarped)] for warped, unwarped in
                            zip(photometric_losses, unwarped_photometric_losses)]
            # Update valid mask with minimum mask
            valid_mask = [[m & v for m, v in zip(minimum, valid)]
                          for minimum, valid in zip(minimum_mask, valid_mask)]
        reduced_photometric_loss = []
        for i in range(len(photometric_losses)):
            for j in range(len(photometric_losses[i])):
                loss = photometric_losses[i][j][valid_mask[i][j]].mean()
                if not torch.isnan(loss):
                    reduced_photometric_loss.append(loss)
        reduced_photometric_loss = sum(reduced_photometric_loss) / len(reduced_photometric_loss)
        # Store and return reduced photometric loss
        return reduced_photometric_loss, [photometric_losses[0][0]]

    def forward(self, rgb, rgb_context, inv_depths,
                cam, cam_context, return_logs=False, progress=0.0,
                opt_flow=None, scene_flow=None, with_mask=False, automask=None):
        # Initialize photometric losses
        photometric_losses = [[] for _ in range(self.n)]
        unwarped_photometric_losses = [[] for _ in range(self.n)]
        # Create RGB scales
        rgbs = match_scales(rgb, inv_depths, self.n, align_corners=self.align_corners)
        rgbs_context = [match_scales(rgb, inv_depths, self.n, align_corners=self.align_corners)
                        for rgb in rgb_context]
        # Warp context to target
        warps_context, masks_context = self.warp(
            rgbs_context, inv_depths, cam, cam_context,
            scene_flow=scene_flow, with_mask=with_mask)
        for j in range(len(rgbs_context)):
            # Calculate and store image loss
            photometric_loss = self.calc_photometric_loss(warps_context[j], rgbs)
            for i in range(self.n):
                if with_mask:
                    # Apply mask if available
                    photometric_loss[i][~masks_context[j][i]] = self.inf
                # Stack photometric losses for each scale
                photometric_losses[i].append(photometric_loss[i])
            # If using automask, calculate and store unwarped losses
            if self.automask_loss and automask is not False:
                unwarped_image_loss = self.calc_photometric_loss(rgbs_context[j], rgbs)
                for i in range(self.n):
                    unwarped_photometric_losses[i].append(unwarped_image_loss[i])
        # Calculate reduced photometric loss
        reduced_loss, masked_loss = self.reduce_photometric_loss_min(
            photometric_losses, unwarped_photometric_losses)
        # Include smoothness loss if requested
        if self.smooth_loss_weight > 0.0:
            reduced_loss += self.calc_smoothness_loss(inv_depths, rgbs)
        # Remove masks from warps_context
        return {
            'loss': reduced_loss.unsqueeze(0),
            'metrics': {},
            'warp': warps_context,
            'masks': masks_context,
            'photo': masked_loss,
        }
