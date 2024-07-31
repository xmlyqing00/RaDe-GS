import torch
from scene.lap_pyramid import pyr_up
from torch.nn.functional import interpolate


def upscale_render_pkg(
        render_pkg: dict, keywords: list, 
        upscale_shape: torch.Tensor, kernel: torch.Tensor, half_kernel_size: int
    ):

    upscaled_render_pkg = {}

    for keyname in keywords:
        upscaled_img = pyr_up(render_pkg[keyname], 4 * kernel, half_kernel_size)
        if upscaled_img.shape != upscale_shape:
            upscaled_img = interpolate(
                upscaled_img.unsqueeze(0), 
                (upscale_shape[-2], upscale_shape[-1]), 
                mode='bilinear', align_corners=True
            ).squeeze(0)
        upscaled_render_pkg[keyname] = upscaled_img.DETACH()
    
    return upscaled_render_pkg
