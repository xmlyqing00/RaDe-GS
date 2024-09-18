#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
import math
from torch.autograd import Variable
from math import exp
from scene.cameras import Camera
from utils.graphics_utils import patch_warp


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def _ncc(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq

    sigma1 = torch.sqrt(sigma1_sq + 1e-4)
    sigma2 = torch.sqrt(sigma2_sq + 1e-4)

    image1_norm = (img1 - mu1) / (sigma1 + 1e-8)
    image2_norm = (img2 - mu2) / (sigma2 + 1e-8)

    ncc = F.conv2d((image1_norm * image2_norm), window, padding=0, groups=channel)

    return torch.mean(ncc, dim=2)


# def _ncc(pred, gt, window, channel):
#     ntotpx, nviews, nc, h, w = pred.shape
#     flat_pred = pred.view(-1, nc, h, w)
#     mu1 = F.conv2d(flat_pred, window, padding=0, groups=channel).view(ntotpx, nviews, nc)
#     mu2 = F.conv2d(gt, window, padding=0, groups=channel).view(ntotpx, nc)

#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2).unsqueeze(1)  # (ntotpx, 1, nc)

#     sigma1_sq = F.conv2d(flat_pred * flat_pred, window, padding=0, groups=channel).view(ntotpx, nviews, nc) - mu1_sq
#     sigma2_sq = F.conv2d(gt * gt, window, padding=0, groups=channel).view(ntotpx, 1, 3) - mu2_sq

#     sigma1 = torch.sqrt(sigma1_sq + 1e-4)
#     sigma2 = torch.sqrt(sigma2_sq + 1e-4)

#     pred_norm = (pred - mu1[:, :, :, None, None]) / (sigma1[:, :, :, None, None] + 1e-8)  # [ntotpx, nviews, nc, h, w]
#     gt_norm = (gt[:, None, :, :, :] - mu2[:, None, :, None, None]) / (
#             sigma2[:, :, :, None, None] + 1e-8)  # ntotpx, nc, h, w

#     ncc = F.conv2d((pred_norm * gt_norm).view(-1, nc, h, w), window, padding=0, groups=channel).view(
#         ntotpx, nviews, nc)

#     return torch.mean(ncc, dim=2)



def loss_in_neighbor_view(
        view_cur: Camera, view_neighbor: Camera, 
        pts_data_cur: dict, pts_data_neighbor: dict,
        patch_template: torch.Tensor,
        depth_valid_threshold: float,
    ) -> dict:

    pts_homo_proj = pts_data_cur['pts_homo'] @ view_neighbor.world_view_transform
    proj_uvw = pts_homo_proj[:, :3] @ view_neighbor.intrins
    proj_depth = proj_uvw[:, 2:3]
    proj_uv = proj_uvw / proj_depth
    proj_uv[:, 0] = proj_uv[:, 0] / (view_neighbor.image_width-1) * 2 - 1
    proj_uv[:, 1] = proj_uv[:, 1] / (view_neighbor.image_height-1) * 2 - 1
    proj_uv_mask = \
        (proj_uv[:, 0] > -1) & (proj_uv[:, 0] < 1) & (proj_uv[:, 1] > -1) & (proj_uv[:, 1] < 1) & \
        (proj_depth.squeeze(1) > 0.1) & (proj_depth.squeeze(1) < 6)
    proj_uv = proj_uv[:, :2].reshape(1, -1, 1, 2)

    sampled_depth = F.grid_sample(
        pts_data_neighbor['depth'], 
        proj_uv, mode='bilinear', align_corners=True
    ).squeeze()

    diff_depth = (sampled_depth - proj_depth.squeeze(1)).abs()
    valid_mask = (diff_depth < depth_valid_threshold) & proj_uv_mask
    valid_mask = valid_mask & (pts_data_cur['homo_plane_depth'].view(-1) > 0)

    weights = (1.0 / torch.exp(diff_depth)).detach()
    weights[~valid_mask] = 0

    loss_geo = torch.mean(weights * diff_depth)
    total_patch_size = patch_template.shape[1]
    # pts_data_cur_normal = pts_data_cur['normal'].view(-1, 3)
    # valid_mask = valid_mask & (pts_data_cur_normal.norm(dim=-1) > 0)

    # valid_mask = torch.zeros_like(valid_mask, dtype=bool)
    # valid_mask[71187] = True
    # valid_mask[399859] = True

    with torch.no_grad():
        ori_pixels_patch = patch_template.clone()[valid_mask]

        H, W = view_cur.gt_gray_img.squeeze().shape
        pixels_patch = ori_pixels_patch.clone()
        pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
        pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
        ref_gray_val = F.grid_sample(view_cur.gt_gray_img.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
        ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

        ref_to_neareast_r = view_neighbor.world_view_transform[:3,:3].transpose(-1,-2) @ view_cur.world_view_transform[:3,:3]
        ref_to_neareast_t = -ref_to_neareast_r @ view_cur.world_view_transform[3,:3] + view_neighbor.world_view_transform[3,:3]
    
    # compute Homography
    ref_local_n = pts_data_cur['normal'].view(-1, 3)[valid_mask]
    ref_local_d = pts_data_cur['homo_plane_depth'].view(-1)[valid_mask]
    # print('min depth', ref_local_d.min())

    # ref_local_d[0] = 1.0506
    # ref_local_d[1] = 1.3158

    # ref_local_n[0] = torch.tensor([-0.3, 0.1, -0.65]).cuda()
    # ref_local_n[1] = torch.tensor([-0.2, -0.04, -0.64]).cuda()

    H_ref_to_neareast = ref_to_neareast_r[None] - \
        torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), 
                    ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
    # H_ref_to_neareast = ref_to_neareast_r[None] - ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1)
    H_ref_to_neareast = torch.matmul(view_neighbor.intrins.T[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
    H_ref_to_neareast = H_ref_to_neareast @ view_cur.intrins_inv.T
    
    ## compute neareast frame patch
    grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
    grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
    grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
    sampled_gray_val = F.grid_sample(
        view_neighbor.gt_gray_img.unsqueeze(0), 
        grid.reshape(1, -1, 1, 2), 
        align_corners=True
    )
    sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)

    ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
    ncc = ncc.reshape(-1) * weights[valid_mask]
    loss_ncc = ncc[ncc_mask.reshape(-1)].mean() 
    # if color_map_neighbor is not None:
    #     # loss_color = (sampled_color - pts_colors)[valid_mask].abs().mean()
    #     ncc, ncc_mask = lncc(sampled_color[valid_mask], pts_colors[valid_mask])
    #     loss_color = ncc[ncc_mask.squeeze()].mean()
    loss_dict = {
        'geo': loss_geo,
        'color': loss_ncc,
    }

    debug_dict = {
        'cur_patch': pixels_patch,
        'neighbor_patch': grid,
        'depth_err': diff_depth.reshape(view_cur.image_height, view_cur.image_width),
    }

    return loss_dict, debug_dict


def lncc(ref, nea):
    # ref_gray: [batch_size, total_patch_size]
    # nea_grays: [batch_size, total_patch_size]
    bs, tps = nea.shape
    patch_size = int(math.sqrt(tps))

    ref_nea = ref * nea
    ref_nea = ref_nea.view(bs, 1, patch_size, patch_size)
    ref = ref.view(bs, 1, patch_size, patch_size)
    nea = nea.view(bs, 1, patch_size, patch_size)
    ref2 = ref.pow(2)
    nea2 = nea.pow(2)

    # sum over kernel
    filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
    padding = patch_size // 2
    ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea_sum = F.conv2d(nea, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea2_sum = F.conv2d(nea2, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_nea_sum = F.conv2d(ref_nea, filters, stride=1, padding=padding)[:, :, padding, padding]

    # average over kernel
    ref_avg = ref_sum / tps
    nea_avg = nea_sum / tps

    cross = ref_nea_sum - nea_avg * ref_sum
    ref_var = ref2_sum - ref_avg * ref_sum
    nea_var = nea2_sum - nea_avg * nea_sum

    cc = cross * cross / (ref_var * nea_var + 1e-8)
    ncc = 1 - cc
    ncc = torch.clamp(ncc, 0.0, 2.0)
    ncc = torch.mean(ncc, dim=1, keepdim=True)
    mask = (ncc < 0.9)
    return ncc, mask