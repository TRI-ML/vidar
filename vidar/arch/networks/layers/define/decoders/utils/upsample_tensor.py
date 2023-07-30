
import torch
import torch.nn as nn
import torch.nn.functional as tfn


def upsample_tensor(tensor, mask, up=8):
    """Convex tensor upsampling 

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to be upsampled
    mask : torch.Tensor
        Upsampling mask
    up : int, optional
        Upsampling factor, by default 8

    Returns
    -------
    torch.Tensor
        Upsampled tensor
    """
    b, c, h, w = tensor.shape
    mask = mask.view(b, 1, 9, up, up, h, w)
    mask = torch.softmax(mask, dim=2)

    up_tensor = tfn.unfold(tensor, [3, 3], padding=1)
    up_tensor = up_tensor.view(b, -1, 9, 1, 1, h, w)

    up_tensor = torch.sum(mask * up_tensor, dim=2)
    up_tensor = up_tensor.permute(0, 1, 4, 2, 5, 3)
    return up_tensor.reshape(b, -1, up * h, up * w)
