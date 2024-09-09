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

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"

import torch
import cv2
from torch.nn.functional import grid_sample, pad    
import trimesh
import datetime
import tinycudann as tcnn
from random import randint, shuffle, choice
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from copy import deepcopy
from scene import Scene, GaussianModel, lap_pyramid
from utils.general_utils import safe_state
import uuid
import imageio
import numpy as np
import torchvision
import pickle
from pathlib import Path
from tqdm import tqdm
from matplotlib import colormaps
from utils.vis_utils import apply_depth_colormap, apply_colormap, build_spheres
from utils.image_utils import psnr
from utils.graphics_utils import depth_double_to_normal, depth_to_points, depth_to_points_fast
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# TENSORBOARD_FOUND = False 

from scene.cameras import Camera


# function L1_loss_appearance is fork from GOF https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/train.py
def L1_loss_appearance(image, gt_image, gaussians, view_idx, return_transformed_image=False):
    appearance_embedding = gaussians.get_apperance_embedding(view_idx)
    # center crop the image
    origH, origW = image.shape[1:]
    H = origH // 32 * 32
    W = origW // 32 * 32
    left = origW // 2 - W // 2
    top = origH // 2 - H // 2
    crop_image = image[:, top:top+H, left:left+W]
    crop_gt_image = gt_image[:, top:top+H, left:left+W]
    
    # down sample the image
    crop_image_down = torch.nn.functional.interpolate(crop_image[None], size=(H//32, W//32), mode="bilinear", align_corners=True)[0]
    
    crop_image_down = torch.cat([crop_image_down, appearance_embedding[None].repeat(H//32, W//32, 1).permute(2, 0, 1)], dim=0)[None]
    mapping_image = gaussians.appearance_network(crop_image_down)
    transformed_image = mapping_image * crop_image
    if not return_transformed_image:
        return l1_loss(transformed_image, crop_gt_image)
    else:
        transformed_image = torch.nn.functional.interpolate(transformed_image, size=(origH, origW), mode="bilinear", align_corners=True)[0]
        return transformed_image

def get_residual_image(
        view, 
        gaussians: GaussianModel,
        base_material: tuple, 
        render_pkg_last: dict, 
        gt_image: torch.Tensor, 
        level_result_dir: Path,
        verbose: bool
    ):

    if render_pkg_last.get(view.image_name) is not None:
        return

    with torch.no_grad():

        pipe, background, scene_min, scene_range = base_material

        render_pkg = render(view, gaussians, pipe, background, image_height=gt_image.shape[1], image_width=gt_image.shape[2])
        render_pkg['depth'] = render_pkg['depth'] / render_pkg['mask']
        render_pkg['depth'] = torch.nan_to_num(render_pkg['depth'], 0, 0)
        render_pkg['render_loss'] = torch.abs(gt_image - render_pkg['render'])
        render_pkg['l1_loss'] = render_pkg['render_loss'].mean().double()
        render_pkg['psnr'] = psnr(render_pkg['render'], gt_image).mean().double()

        pts_xyz, rays_d, pts_radius = depth_to_points_fast(view, render_pkg['depth'])
        
        pts_mask = (render_pkg['depth'] > 0).view(-1)
        pts_xyz = pts_xyz[pts_mask]
        pts_xyz_view = torch.concat([pts_xyz, torch.ones_like(pts_xyz[:, 0:1])], dim=-1)
        pts_xyz_world = pts_xyz_view @ view.world_view_transform_inv
        pts_xyz_world = pts_xyz_world[:, :3]

        render_pkg['pts_depth'] = render_pkg['depth'].reshape(-1, 1)[pts_mask]
        render_pkg['rays_d'] = rays_d[pts_mask]
        render_pkg['pts_mask'] = pts_mask
        render_pkg['pts_xyz_world'] = pts_xyz_world 
        render_pkg['pts_xyz_world_normalized'] = (pts_xyz_world - scene_min) / scene_range
        render_pkg['residual_image'] = gt_image - render_pkg['render']
        pts_residual_color = render_pkg['residual_image'].permute(1, 2, 0).reshape(-1, 3)
        if pts_mask is not None:
            pts_residual_color = pts_residual_color[pts_mask]
        render_pkg['pts_residual_color'] = pts_residual_color
        pts_color = gt_image.permute(1, 2, 0).reshape(-1, 3)
        if pts_mask is not None:
            pts_color = pts_color[pts_mask]
        render_pkg['pts_color'] = pts_color
        
        render_pkg_last[view.image_name] = render_pkg

        if verbose:
            level_result_dir.mkdir(parents=True, exist_ok=True)
            vis_pcd = trimesh.PointCloud(
                pts_xyz_world.cpu().detach().numpy(), 
                pts_color.cpu().detach().numpy()
            )
            vis_pcd.export(level_result_dir / f'{view.image_name}_pcd_world_scale_sqrt2.ply')

            torchvision.utils.save_image(render_pkg['render_loss'], level_result_dir / f'{view.image_name}_l1loss.png')
            torchvision.utils.save_image(render_pkg['render'], level_result_dir / f'{view.image_name}_render.png')
            torchvision.utils.save_image(gt_image, level_result_dir / f'{view.image_name}_gt.png')
            torchvision.utils.save_image(render_pkg['residual_image'], level_result_dir / f'{view.image_name}_residual.png')


def get_pts_xyz_corrected_depth_offset(view, render_pkg, pos_encoding, pos_network, return_depth_map: bool = True):
    # ray view to world
    # rays_d_homo = torch.concat([render_pkg['rays_d'] , torch.ones_like(render_pkg['rays_d'][:, 0:1])], dim=-1)
    # rays_d_world = rays_d_homo @ view.world_view_transform_inv
    
    # compute depth offset
    pts_xyz_normalized = render_pkg['pts_xyz_world_normalized']
    pts_encoding = pos_encoding(pts_xyz_normalized)
    pts_input = torch.concat([pts_xyz_normalized, pts_encoding], dim=-1)
    d_offset = pos_network(pts_input).float()

    # depth to view
    depth_corrected = render_pkg['pts_depth'] + d_offset
    depth_corrected = torch.clamp(depth_corrected, 0, None)
    pts_xyz_corrected = render_pkg['rays_d'] * depth_corrected

    # view to world
    corrected_homo = torch.concat([pts_xyz_corrected, torch.ones_like(pts_xyz_corrected[:, 0:1])], dim=-1)
    pts_xyz_world = corrected_homo @ view.world_view_transform_inv

    

    # update the depth map
    if return_depth_map:
        depth_map = render_pkg['depth'].clone().reshape(-1, 1)
        depth_map[render_pkg['pts_mask']] = depth_corrected
        depth_map = depth_map.reshape(*render_pkg['depth'].shape)
        
        # patch xyz
        # offsets = [(i, j) for i in range(-2, 3) for j in range(-2, 3)]
        # padded_depth = pad(depth_map, (2, 2, 2, 2), mode='reflect')  # shape (3, H+4, W+4)
        # H, W = depth_map.shape[1], depth_map.shape[2]
        # features = []
        # for dy, dx in offsets:
        #     # Shift the image to access neighbors
        #     neighbor = padded_depth[:, 2+dy:H+2+dy, 2+dx:W+2+dx]
        #     features.append(neighbor)
        # depth_map_patch = torch.stack(features, dim=-1)

        return pts_xyz_world, depth_map
    else:
        return pts_xyz_world, None


def loss_in_neighbor_view(
        pts_homo: torch.Tensor, view_neighbor: Camera, 
        depth_map_neighbor: torch.Tensor, 
        color_map_neighbor: torch.Tensor,
        pts_colors: torch.Tensor,
        depth_valid_threshold: float,
    ):
    pts_homo_proj = pts_homo @ view_neighbor.world_view_transform
    proj_uvw = pts_homo_proj[:, :3] @ view_neighbor.intrins
    proj_depth = proj_uvw[:, 2:3]
    proj_uv = proj_uvw / proj_depth
    proj_uv[:, 0] = proj_uv[:, 0] / (view_neighbor.image_width-1) * 2 - 1
    proj_uv[:, 1] = proj_uv[:, 1] / (view_neighbor.image_height-1) * 2 - 1
    proj_uv = proj_uv[:, :2].reshape(1, -1, 1, 2)

    sampled_depth = grid_sample(
        depth_map_neighbor.unsqueeze(1), 
        proj_uv, mode='bilinear', align_corners=True
    ).squeeze()

    if color_map_neighbor is not None:
        sampled_color = grid_sample(
            color_map_neighbor, 
            proj_uv, mode='bilinear', align_corners=True
        ).squeeze().permute(1, 0)

    diff_depth = sampled_depth - proj_depth.squeeze(1)
    valid_mask = torch.abs(diff_depth) < depth_valid_threshold
    loss = diff_depth[valid_mask].abs().mean()

    if color_map_neighbor is not None:
        loss_color = (sampled_color - pts_colors)[valid_mask].abs().mean()

    return loss, loss_color


def render_image_with_residual(
        view, 
        pos_encoding, pos_network, 
        residual_encoding, residual_network, 
        render_pkg: dict, 
    ):

    pts_xyz_world, _ = get_pts_xyz_corrected_depth_offset(view, render_pkg, pos_encoding, pos_network, return_depth_map=False)
    pts_xyz = pts_xyz_world[:, :3]
    pts_encoding = residual_encoding(pts_xyz).float()
    pts_input = torch.concat([pts_xyz, pts_encoding], dim=-1)
    residual_color_pts = residual_network(pts_input).float()
    final_image = render_pkg['render'].clone().view(3, -1)
    final_image[:, render_pkg['pts_mask']] += residual_color_pts.permute(1, 0)
    final_image = final_image.view(3, *render_pkg['render'].shape[1:])

    return final_image


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    tb_writer = prepare_output_and_logger(dataset, opt.verbose)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, gap = pipe.interval)
    # scene.init_gaussian_from_pcd()

    trainCameras = scene.getTrainCameras().copy()
    gaussians.training_setup(opt)
    gaussians.compute_3D_filter(cameras=trainCameras)

    # if checkpoint:
        # (model_params, first_iter) = torch.load(checkpoint)
        # gaussians.restore(model_params, opt)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    if opt.verbose:
        out_dir = Path(scene.model_path) / 'render'
        out_dir.mkdir(parents=True, exist_ok=True)
    
    total_iter = 0
        
    config_tcnn = {
        'encoding': {
            'otype': 'HashGrid',
            'n_levels': 16,  # 16
            'n_features_per_level': 2,
            'log2_hashmap_size': 19, # 19
            'base_resolution': 16,
            'per_level_scale': 1.4, # 2.0
            'interpolation': 'Smoothstep'
        },
        'network': {
            'otype': 'FullyFusedMLP',
            'activation': 'LeakyReLU',
            'output_activation': 'None',
            'n_neurons': dataset.tcnn_num_neurons,
            'n_hidden_layers': dataset.tcnn_num_layers,
        }
    }

    n_input_dims = 3
    n_output_dims = 3
    viewpoint_stack = None

    cur_lap_level = dataset.lap_pyramid_level - 1

    ## Base model: 3DGS
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        if checkpoint.exists():
            print(f"Loading checkpoint {checkpoint}")
            saved_dict = pickle.load(open(checkpoint / 'base_material.pkl', "rb"))
            base_material = saved_dict['base_materical']
            render_pkg_last_train = saved_dict['render_pkg_last_train']
            render_pkg_last_test = saved_dict['render_pkg_last_test']
            train_view_pairs = saved_dict['train_view_pairs']
            (model_params, first_iter) = torch.load(checkpoint / 'chkpnt3000.pth')
            gaussians.restore(model_params, opt)
            gaussians.compute_3D_filter(cameras=trainCameras)
            skip_base_training = True
        else:
            print(f"Checkpoint {checkpoint} not found. Starting from scratch.")
            skip_base_training = False
    else:
        skip_base_training = False
    
    if not skip_base_training:
        if testing_iterations[-1] < opt.base_iterations:
            testing_iterations.append(opt.base_iterations)

        ema_loss_for_log, ema_depth_loss_for_log, ema_mask_loss_for_log, ema_normal_loss_for_log = 0.0, 0.0, 0.0, 0.0
        progress_bar = tqdm(range(1, opt.base_iterations + 1), desc="Training progress")

        for iteration in range(1, opt.base_iterations + 1):

            total_iter += 1
            
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

            iter_start.record()

            gaussians.update_learning_rate(total_iter)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if total_iter % 800 == 0:
                gaussians.oneupSHdegree()

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
                shuffle(viewpoint_stack)
            
            # if cur_lap_level == dataset.lap_pyramid_level - 1:
            viewpoint_cam = viewpoint_stack.pop()
            # else:
                # viewpoint_cam: Camera = viewpoint_stack[0]
            # gt_image = viewpoint_cam.original_image
            gt_image = viewpoint_cam.gauss_pyramid[cur_lap_level].squeeze(0)

            render_pkg = render(viewpoint_cam, gaussians, pipe, background, image_height=gt_image.shape[1], image_width=gt_image.shape[2])
            rendered_image: torch.Tensor
            rendered_image, viewspace_point_tensor, visibility_filter, radii = (
                                                                        render_pkg["render"], 
                                                                        render_pkg["viewspace_points"], 
                                                                        render_pkg["visibility_filter"], 
                                                                        render_pkg["radii"])

            final_image = rendered_image

            if dataset.use_decoupled_appearance:
                Ll1_render = L1_loss_appearance(final_image, gt_image, gaussians, viewpoint_cam.uid)
            else:
                Ll1_render = l1_loss(final_image, gt_image)

            if opt.verbose and (iteration % 50 == 0 or iteration < 100):
                loss_map = torch.abs(final_image - gt_image).sum(dim=0)
                rendered_image_color = apply_colormap(
                    torch.clip(torch.max(rendered_image, dim=0, keepdim=True)[0].permute(1, 2, 0), min=0, max=1)
                ).permute(2, 0, 1)

                # torchvision.utils.save_image(rendered_alpha_t, out_dir / f'{total_iter:05d}_alpha_t.png')
                torchvision.utils.save_image(loss_map, out_dir / f'{total_iter:05d}_l1loss.png')
                torchvision.utils.save_image(rendered_image, out_dir / f'{total_iter:05d}_rendered.png')
                torchvision.utils.save_image(rendered_image_color, out_dir / f'{total_iter:05d}_rendered_color.png')
                torchvision.utils.save_image(final_image, out_dir / f'{total_iter:05d}_final.png')
                torchvision.utils.save_image(gt_image, out_dir / f'{total_iter:05d}_gt.png')


            if iteration >= opt.depth_opt_from_iter:
                rendered_mask: torch.Tensor = render_pkg["mask"]
                rendered_depth: torch.Tensor = render_pkg["depth"]
                rendered_middepth: torch.Tensor = render_pkg["middepth"]
                rendered_normal: torch.Tensor = render_pkg["normal"]
                depth_distortion: torch.Tensor = render_pkg["depth_distortion"]
                rendered_depth = torch.nan_to_num(rendered_depth / rendered_mask, 0, 0)

                # depth distortion loss
                lambda_distortion = opt.lambda_distortion
                depth_distortion = torch.where(rendered_mask>0,depth_distortion/(rendered_mask * rendered_mask).detach(),0)
                distortion_map = depth_distortion[0] * viewpoint_cam.edge_pyramid[cur_lap_level].squeeze(0, 1)
                distortion_loss = distortion_map.mean()

                # normal consistency loss
                depth_middepth_normal, _ = depth_double_to_normal(viewpoint_cam, rendered_depth, rendered_middepth)
                depth_ratio = 0.6
                rendered_normal = torch.nn.functional.normalize(rendered_normal, p=2, dim=0)
                rendered_normal = rendered_normal.permute(1,2,0)
                normal_error_map = (1 - (rendered_normal.unsqueeze(0) * depth_middepth_normal).sum(dim=-1))
                depth_normal_loss = (1-depth_ratio) * normal_error_map[0].mean() + depth_ratio * normal_error_map[1].mean()
                lambda_depth_normal = opt.lambda_depth_normal

                if opt.verbose and iteration % 50 == 0:

                    depth_img = apply_depth_colormap(rendered_depth.permute(1, 2, 0)).cpu().numpy() * 255
                    imageio.imwrite(out_dir / f'{total_iter:05d}_depth.png', depth_img.astype(np.uint8))

                    rendered_normal_img = np.clip(np.rint(rendered_normal.detach().cpu().numpy() * 255), 0, 255).astype(np.uint8)
                    imageio.imwrite(out_dir / f'{total_iter:05d}_normal.png', rendered_normal_img)

                    depth_middepth_normal_map = np.clip(np.rint(depth_middepth_normal[0].detach().cpu().numpy() * 255), 0, 255).astype(np.uint8)
                    imageio.imwrite(out_dir / f'{total_iter:05d}_normal_from_depth0.png', depth_middepth_normal_map)

                    depth_middepth_normal_map = np.clip(np.rint(depth_middepth_normal[1].detach().cpu().numpy() * 255), 0, 255).astype(np.uint8)
                    imageio.imwrite(out_dir / f'{total_iter:05d}_normal_from_depth1.png', depth_middepth_normal_map)

                    normal_error_img = np.clip(np.rint(normal_error_map[0].detach().cpu().numpy() * 255), 0, 255).astype(np.uint8)
                    imageio.imwrite(out_dir / f'{total_iter:05d}_normal_err0.png', normal_error_img)

                    normal_error_img = np.clip(np.rint(normal_error_map[1].detach().cpu().numpy() * 255), 0, 255).astype(np.uint8)
                    imageio.imwrite(out_dir / f'{total_iter:05d}_normal_err1.png', normal_error_img)

            else:
                lambda_distortion = 0
                lambda_depth_normal = 0
                distortion_loss = torch.tensor([0],dtype=torch.float32,device="cuda")
                depth_normal_loss = torch.tensor([0],dtype=torch.float32,device="cuda")
                
            rgb_loss = (1.0 - opt.lambda_dssim) * Ll1_render + opt.lambda_dssim * (1.0 - ssim(final_image, gt_image.unsqueeze(0)))
            
            loss = rgb_loss + depth_normal_loss * lambda_depth_normal + distortion_loss * lambda_distortion
            loss.backward()

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_depth_loss_for_log = 0.4 * distortion_loss.item() + 0.6 * ema_depth_loss_for_log
                ema_normal_loss_for_log = 0.4 * depth_normal_loss.item() + 0.6 * ema_normal_loss_for_log

                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{4}f}", "loss_dep": f"{ema_depth_loss_for_log:.{4}f}", "loss_normal": f"{ema_normal_loss_for_log:.{4}f}"})
                    progress_bar.update(10)
                if iteration == opt.base_iterations:
                    progress_bar.close()
                
                report_loss = (Ll1_render, loss, distortion_loss, depth_normal_loss)
                # Log and save
                training_report(
                    tb_writer, 
                    iteration, total_iter, cur_lap_level,
                    report_loss, 
                    l1_loss, 
                    iter_start.elapsed_time(iter_end), 
                    testing_iterations, 
                    scene, 
                    None, (pipe, background)
                )
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(total_iter))
                    scene.save(total_iter)

                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold, 0)
                        gaussians.compute_3D_filter(cameras=trainCameras)

                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        print("\n[ITER {}] Reset Gaussians opacity".format(total_iter))
                        gaussians.reset_opacity()
                    
                if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                    if iteration < opt.lap_level_upgrade_interval - 100:
                        # don't update in the end of training
                        gaussians.compute_3D_filter(cameras=trainCameras)

                # Optimizer step
                if iteration < opt.lap_level_upgrade_interval:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                if iteration in checkpoint_iterations or iteration == opt.base_iterations:
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        
        base_material = (pipe, background, dataset.scene_min, dataset.scene_range)
            
        # Residual Training
        for cur_lap_level in range(max(0, dataset.lap_pyramid_level - 2), -1, -1):
            print('Getting residual image from train cameras')
            # Get residual image
            training_views = scene.getTrainCameras().copy()
            if training_views and len(training_views) > 0:
                render_pkg_last_train = {}
                for view in training_views:
                    get_residual_image(
                        view, 
                        gaussians, 
                        base_material, 
                        render_pkg_last_train, 
                        view.gauss_pyramid[cur_lap_level].squeeze(0), 
                        Path(scene.model_path) / f'level_{cur_lap_level}',
                        opt.verbose
                    )
            else:
                render_pkg_last_train = None
            print('Getting residual image from test cameras')
            testing_views = scene.getTestCameras().copy()
            if testing_views and len(testing_views) > 0:
                render_pkg_last_test = {}
                for view in testing_views:
                    get_residual_image(
                        view, 
                        gaussians, 
                        base_material, 
                        render_pkg_last_test, 
                        view.gauss_pyramid[cur_lap_level].squeeze(0), 
                        Path(scene.model_path) / f'level_{cur_lap_level}',
                        opt.verbose
                    )
            else:
                render_pkg_last_test = None

            # collect neighboring cameras for training
            print('Collecting neighboring cameras for training')
            viewpoint_stack = scene.getTrainCameras().copy()
            view_num = len(viewpoint_stack)
            view_dist = np.zeros((view_num, view_num))

            with torch.no_grad():
                for view_idx1 in range(view_num):
                    
                    view1 = viewpoint_stack[view_idx1]

                    for view_idx2 in range(view_idx1 + 1, view_num):
                        
                        view2 = viewpoint_stack[view_idx2]
                        rot_dist = np.linalg.norm(view1.rot_vec - view2.rot_vec)
                        trans_dist = np.linalg.norm(view1.T - view2.T)
                        view_dist[view_idx1, view_idx2] = rot_dist + trans_dist
                        view_dist[view_idx2, view_idx1] = view_dist[view_idx1, view_idx2]

            train_view_pairs = []
            neighbor_num = 3
            for view_idx1 in range(view_num):

                neighbor_idx = np.argsort(view_dist[view_idx1])
                train_view_pairs.append((view_idx1, *neighbor_idx[1:1 + neighbor_num]))
            
            # print(train_view_pairs)
            # Record result, prepare next level 
            pickle.dump(
                {
                    'base_materical': base_material, 
                    'render_pkg_last_train': render_pkg_last_train, 
                    'render_pkg_last_test': render_pkg_last_test,
                    'train_view_pairs': train_view_pairs
                }, 
                open(scene.model_path + "/base_material.pkl", "wb")
            )

    pos_encoding = tcnn.Encoding(3, config_tcnn['encoding'])
    pos_network = tcnn.Network(3+pos_encoding.n_output_dims, 1, config_tcnn['network'])

    residual_opt = torch.optim.AdamW([
        {'params': pos_encoding.parameters()},
        {'params': pos_network.parameters()}
    ], lr=opt.residual_lr)
    residual_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        residual_opt, T_max=opt.lap_level_upgrade_interval
    )
    residual_opt.zero_grad(set_to_none = True)
    
    level_result_dir = Path(scene.model_path) / f'depth_{cur_lap_level}'
    level_result_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training depth {cur_lap_level}")
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(1, opt.lap_level_upgrade_interval + 1), desc="Training progress")

    train_view_pairs_stack = None
    training_views = scene.getTrainCameras().copy()

    if opt.verbose:
        train_depth_dir = Path(scene.model_path) / 'train_depth_verbose'
        train_depth_dir.mkdir(parents=True, exist_ok=True)

    for iteration in range(1, opt.lap_level_upgrade_interval + 1):

        total_iter += 1

        if not train_view_pairs_stack:
            train_view_pairs_stack = train_view_pairs.copy()
            # shuffle(train_view_pairs_stack)
        view_pair = train_view_pairs_stack.pop()
        # view_pair = train_view_pairs_stack[5]

        view_cur = training_views[view_pair[0]]
        view_neighbor = training_views[choice(view_pair[1:])]
        # view_neighbor = training_views[view_pair[1]]

        W, H = view_cur.image_width, view_cur.image_height
        pts_xyz_corrected_cur, depth_map_cur = get_pts_xyz_corrected_depth_offset(view_cur, render_pkg_last_train[view_cur.image_name], pos_encoding, pos_network)
        pts_xyz_corrected_neighbor, depth_map_neighbor = get_pts_xyz_corrected_depth_offset(view_neighbor, render_pkg_last_train[view_neighbor.image_name], pos_encoding, pos_network)
        # pts_xyz_color = render_pkg_last_train[view_cur.image_name]['pts_color']
        pts_xyz_color_cur = view_cur.gray_patch_feat.reshape(25, -1).permute(1, 0)[render_pkg_last_train[view_cur.image_name]['pts_mask']]
        pts_xyz_color_neighbor = view_neighbor.gray_patch_feat.reshape(25, -1).permute(1, 0)[render_pkg_last_train[view_neighbor.image_name]['pts_mask']]
        # pts_xyz_residual_color = render_pkg_last_train[view.image_name]['pts_residual_color']

        loss_depth, loss_color = loss_in_neighbor_view(
            pts_xyz_corrected_cur, view_neighbor, 
            depth_map_neighbor, 
            view_neighbor.gray_patch_feat, pts_xyz_color_cur,
            opt.depth_valid_threshold,
        )

        loss_depth2, loss_color2 = loss_in_neighbor_view(
            pts_xyz_corrected_neighbor, view_cur, 
            depth_map_cur, 
            view_cur.gray_patch_feat, pts_xyz_color_neighbor,
            opt.depth_valid_threshold,
        )
        # loss2 = depth_loss_in_neighbor_view(pts_xyz_corrected_neighbor, view_cur, depth_map_cur, opt.depth_valid_threshold)
        loss = loss_depth + loss_color + loss_depth2 + loss_color2
        # loss = loss_depth2 + loss_depth

        # if iteration > 1000 and loss_depth.item() > 0.05:
            # print(iteration, view_cur.image_name, view_neighbor.image_name)

        # if False:
        #     torchvision.utils.save_image(
        #         render_pkg_last_train[view_neighbor.image_name]['depth'][0] / 6, 
        #         level_result_dir / f'{view.image_name}_{view_neighbor.image_name}_view_depth.png'
        #     )
        #     proj_depth_img = np.zeros((H, W), dtype=np.uint8)
        #     sampled_depth_img = np.zeros((H, W), dtype=np.uint8)
        #     diff_depth_img = np.zeros((H, W), dtype=np.uint8)
        #     for i in range(proj_uv.shape[1]):
        #         u, v = proj_uv[0, i, 0].tolist()
        #         u = int((u + 1) / 2 * W)
        #         v = int((v + 1) / 2 * H)
        #         if 0 <= u < W and 0 <= v < H:
        #             proj_depth_img[v, u] = int(proj_depth[i].item() / 6.0 * 255)
        #             sampled_depth_img[v, u] = int(sampled_depth[0, 0, i].item() / 6.0 * 255)
        #             diff_depth_img[v, u] = np.clip(int(abs(proj_depth[i].item() - sampled_depth[0, 0, i].item()) * 255), 0, 255)

        #     cv2.imwrite(level_result_dir / f'{view.image_name}_{view_neighbor.image_name}_proj_depth.png', proj_depth_img)
        #     cv2.imwrite(level_result_dir / f'{view.image_name}_{view_neighbor.image_name}_sampled_depth.png', sampled_depth_img)
        #     cv2.imwrite(level_result_dir / f'{view.image_name}_{view_neighbor.image_name}_diff_depth.png', diff_depth_img)
        
        # sampled_color = grid_sample(
        #     view_neighbor.gauss_pyramid[cur_lap_level],
        #     proj_uv, mode='bilinear', align_corners=True
        # )
        # sampled_residual_color = grid_sample(
        #     render_pkg_last_train[view_neighbor.image_name]['residual_image'].unsqueeze(0), 
        #     proj_uv, mode='bilinear', align_corners=True
        # )

        # valid_mask = abs(sampled_depth - proj_depth).squeeze() < 0.3
        # loss_neighbor_color += l1_loss(sampled_color.squeeze().permute(1, 0)[valid_mask], pts_xyz_color[valid_mask])
        # loss_neighbor_residual_color += l1_loss(sampled_residual_color.squeeze().permute(1, 0)[valid_mask], pts_xyz_residual_color[valid_mask])

        

        # loss = loss_neighbor_color + loss_neighbor_residual_color
        # loss = loss_neighbor_color
        loss.backward()

        # residual_opt.step()
        residual_opt.zero_grad(set_to_none = True)
        residual_scheduler.step()

        with torch.no_grad():
            if iteration % 10 == 0:
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                progress_bar.set_postfix({
                    "loss": f"{ema_loss_for_log:.{4}f}",
                    # 'color': f"{loss_neighbor_color.item():.{4}f}",
                    # 'res_color': f"{loss_neighbor_residual_color.item():.{4}f}"
                })
                progress_bar.update(10)

                if opt.verbose:
                    torchvision.utils.save_image(view_cur.original_image, train_depth_dir / f'{iteration}_cur_{view_cur.image_name}_gt.png')
                    torchvision.utils.save_image(view_neighbor.original_image, train_depth_dir / f'{iteration}_neighbor_{view_neighbor.image_name}_gt.png')
                    torchvision.utils.save_image((depth_map_cur - 3) / 3, train_depth_dir / f'{iteration}_cur_{view_cur.image_name}_depth.png')
                    torchvision.utils.save_image((depth_map_neighbor - 3) / 3, train_depth_dir / f'{iteration}_neighbor_{view_neighbor.image_name}_depth.png')


                if tb_writer:
                    tb_writer.add_scalar('train_depth/total_loss', loss.item(), total_iter)
                    # print(loss_depth.item(), loss_color.item())
                    tb_writer.add_scalar('train_depth/depth_loss', loss_depth.item(), total_iter)
                    tb_writer.add_scalar('train_depth/color_loss', loss_color.item(), total_iter)
                    tb_writer.add_scalar('train_depth/depth_loss2', loss_depth2.item(), total_iter)
                    tb_writer.add_scalar('train_depth/color_loss2', loss_color2.item(), total_iter)
                    # tb_writer.add_scalar('train_depth/neighbor_color_loss', loss_neighbor_color.item(), total_iter)
                    # tb_writer.add_scalar('train_depth/neighbor_residual_color_loss', loss_neighbor_residual_color.item(), total_iter)
                    tb_writer.add_scalar('scene/lr_tcnn_depth', residual_opt.param_groups[0]['lr'], total_iter)

            if iteration in testing_iterations or iteration == opt.lap_level_upgrade_interval or iteration == 1:
                print('\n[ITER {}] Depth Evaluating'.format(total_iter))

                validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras(), 'render_pkg': render_pkg_last_test}, 
                                    {'name': 'train', 'cameras' : scene.getTrainCameras(), 'render_pkg': render_pkg_last_train})

                for config in validation_configs:
                    test_result_dir = Path(scene.model_path) / f'test_depth_{total_iter:05d}_{config["name"]}'
                    test_result_dir.mkdir(parents=True, exist_ok=True)
                    if not config['cameras'] or len(config['cameras']) == 0:
                        continue

                    for idx, view in enumerate(config['cameras']):
                        # rays_d = config['render_pkg'][view.image_name]['rays_d']
                        # rays_d_homo = torch.concat([rays_d, torch.ones_like(rays_d[:, 0:1])], dim=-1)
                        # rays_d_world = rays_d_homo @ view.world_view_transform_inv

                        pts_xyz_normalized = config['render_pkg'][view.image_name]['pts_xyz_world_normalized']
                        pts_encoding = pos_encoding(pts_xyz_normalized)
                        pts_input = torch.concat([pts_xyz_normalized, pts_encoding], dim=-1)
                        d_offset = pos_network(pts_input).float()

                        # depth to view
                        depth_corrected = config['render_pkg'][view.image_name]['pts_depth'] + d_offset
                        depth_corrected = torch.clamp(depth_corrected, 0, None)
                        # print(idx, view.image_name)

                        if opt.verbose:
                            depth_img = torch.zeros((H * W), dtype=torch.float32, device="cuda")
                            depth_img[config['render_pkg'][view.image_name]['pts_mask']] = (d_offset.squeeze() + 0.5) * 255
                            depth_img = torch.clamp(depth_img.long(), 0, 255)
                            colormap = colormaps.get_cmap('viridis')
                            colormap = torch.tensor(colormap.colors).to(depth_img.device)  # type: ignore
                            depth_color_img = colormap[depth_img]
                            depth_color_img = depth_color_img.reshape(H, W, -1).permute(2, 0, 1)
                            torchvision.utils.save_image(depth_color_img, test_result_dir / f'{view.image_name}_offset.png')

                            depth_img = torch.zeros((H * W), dtype=torch.float32, device="cuda")
                            depth_img[config['render_pkg'][view.image_name]['pts_mask']] = (depth_corrected.squeeze() - 3) / 3
                            depth_img = depth_img.reshape(1, H, W)
                            torchvision.utils.save_image(depth_img, test_result_dir / f'{view.image_name}_depth_corrected.png')

                    
    progress_bar.close()
    
    return
    # Train color model
    pos_encoding.eval()
    pos_network.eval()

    residual_encoding = tcnn.Encoding(n_input_dims, config_tcnn['encoding'])
    residual_network = tcnn.Network(3+residual_encoding.n_output_dims, n_output_dims, config_tcnn['network'])

    residual_opt = torch.optim.AdamW([
        {'params': residual_encoding.parameters()},
        {'params': residual_network.parameters()}
    ], lr=opt.residual_lr)
    residual_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        residual_opt, T_max=opt.lap_level_upgrade_interval
    )
    residual_opt.zero_grad(set_to_none = True)
    
    level_result_dir = Path(scene.model_path) / f'residual_{cur_lap_level}'
    level_result_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training residual {cur_lap_level}")
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(1, opt.lap_level_upgrade_interval + 1), desc="Training residual")

    train_view_pairs_stack = None

    for iteration in range(1, opt.lap_level_upgrade_interval + 1):

        total_iter += 1

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            shuffle(viewpoint_stack)
        view = viewpoint_stack.pop()
        
        final_image = render_image_with_residual(
            view, 
            pos_encoding, pos_network, 
            residual_encoding, residual_network, 
            render_pkg_last_train[view.image_name]
        )
        gt_image = view.gauss_pyramid[cur_lap_level].squeeze(0)

        if dataset.use_decoupled_appearance:
            Ll1_render = L1_loss_appearance(final_image, gt_image, gaussians, viewpoint_cam.uid)
        else:
            Ll1_render = l1_loss(final_image, gt_image)

        ssim_loss = ssim(final_image, gt_image.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1_render + opt.lambda_dssim * (1.0 - ssim_loss)
        loss.backward()

        residual_opt.step()
        residual_opt.zero_grad(set_to_none = True)
        residual_scheduler.step()

        with torch.no_grad():
            psnr_val = psnr(final_image, gt_image).mean()

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{4}f}"})
                progress_bar.update(10)

            if tb_writer:
                tb_writer.add_scalar('train_residual/l1_loss', Ll1_render.item(), total_iter)
                tb_writer.add_scalar('train_residual/total_loss', loss.item(), total_iter)
                tb_writer.add_scalar('train_residual/psnr', psnr_val.item(), total_iter)
                tb_writer.add_scalar('train_residual/ssim', ssim_loss.item(), total_iter)
                tb_writer.add_scalar('scene/lr_tcnn_residual', residual_opt.param_groups[0]['lr'], total_iter)

            if iteration % 10 == 0 and opt.verbose:
                base_color = render_pkg_last_train[view.image_name]['render']
                residual_color = final_image - base_color
                loss_map = torch.abs(final_image - gt_image).sum(dim=0)
                torchvision.utils.save_image(loss_map, level_result_dir / f'{total_iter}_l1loss.png')
                torchvision.utils.save_image(residual_color, level_result_dir / f'{total_iter}_residual_color.png')
                torchvision.utils.save_image(final_image, level_result_dir / f'{total_iter}_final.png')
                torchvision.utils.save_image(gt_image, level_result_dir / f'{total_iter}_gt.png')
                torchvision.utils.save_image(base_color, level_result_dir / f'{total_iter}_base.png')

            if iteration in testing_iterations or iteration == opt.lap_level_upgrade_interval:
                print('\n[ITER {}] Residual Evaluating'.format(total_iter))
                residual_encoding.eval()
                residual_network.eval()

                validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras(), 'render_pkg': render_pkg_last_test}, 
                                    {'name': 'train', 'cameras' : scene.getTrainCameras(), 'render_pkg': render_pkg_last_train})

                for config in validation_configs:
                    test_result_dir = Path(scene.model_path) / f'test_residual_{total_iter:05d}_{config["name"]}'
                    test_result_dir.mkdir(parents=True, exist_ok=True)
                    if not config['cameras'] or len(config['cameras']) == 0:
                        continue

                    l1_test = 0.0
                    psnr_test = 0.0
                    for idx, test_view in enumerate(config['cameras']):
                        # print(config['name'], test_view.image_name)
                        test_gt_image = torch.clamp(test_view.gauss_pyramid[cur_lap_level], 0.0, 1.0)[0]
                        render_result = render_image_with_residual(
                            test_view, 
                            pos_encoding, pos_network, 
                            residual_encoding, residual_network, 
                            config['render_pkg'][test_view.image_name]
                        )
                        l1_test += l1_loss(render_result, test_gt_image).mean().double()
                        psnr_test += psnr(render_result, test_gt_image).mean().double()

                        if opt.verbose:
                            pts_xyz_world = get_pts_xyz_corrected_depth_offset(
                                test_view,
                                config['render_pkg'][test_view.image_name],
                                pos_encoding, pos_network,
                                False
                            )
                            base_color = config['render_pkg'][test_view.image_name]['render']
                            residual_color = render_result - base_color
                            torchvision.utils.save_image(residual_color, test_result_dir / f'{test_view.image_name}_residual.png')
                            torchvision.utils.save_image(render_result, test_result_dir / f'{test_view.image_name}_render.png')
                            torchvision.utils.save_image(test_gt_image, test_result_dir / f'{test_view.image_name}_gt.png')
                            torchvision.utils.save_image(base_color, test_result_dir / f'{test_view.image_name}_base.png')
                    psnr_test /= len(config['cameras'])
                    l1_test /= len(config['cameras'])          
                    print("\n[ITER {}] Residual Evaluating {}: L1 {} PSNR {}".format(total_iter, config['name'], l1_test, psnr_test))
                    if config["name"] == "test":
                        with open(scene.model_path + "/chkpnt" + str(total_iter) + ".txt", "w") as file_object:
                            print("\n[ITER {}] Residual Evaluating {}: L1 {} PSNR {}".format(total_iter, config['name'], l1_test, psnr_test), file=file_object)
                    if tb_writer:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, total_iter)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, total_iter)
            
                torch.cuda.empty_cache()
                residual_encoding.train()
                residual_network.train()
    
    progress_bar.close()


def prepare_output_and_logger(args, verbose):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    else:
        current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        args.model_path = os.path.join(args.model_path, current_time)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        if not TENSORBOARD_FOUND:
            print("Tensorboard not available: not logging progress")
        if not verbose:
            print("Verbose mode off: not logging progress")
    return tb_writer


def training_report(tb_writer, iter, total_iter, lap_level, report_loss, l1_loss, elapsed, testing_iterations, scene : Scene, last_pc: GaussianModel, renderArgs):
  
    Ll1, loss, depth_loss, normal_loss = report_loss
    if tb_writer:
        tb_writer.add_scalar('train/l1_loss', Ll1.item(), total_iter)
        tb_writer.add_scalar('train/depth_loss', depth_loss.item(), total_iter)
        tb_writer.add_scalar('train/normal_loss', normal_loss.item(), total_iter)
        tb_writer.add_scalar('train/total_loss', loss.item(), total_iter)
        tb_writer.add_scalar('train/gauss_num', scene.gaussians.get_xyz.shape[0], total_iter)
        tb_writer.add_scalar('iter_time', elapsed, total_iter)
        tb_writer.add_scalar('grad/feature_dc', scene.gaussians._features_dc.grad.abs().max(), total_iter)
        tb_writer.add_scalar('grad/feature_rest', scene.gaussians._features_rest.grad.abs().max(), total_iter)
        tb_writer.add_scalar('grad/opacity', scene.gaussians._opacity.grad.abs().max(), total_iter)
        tb_writer.add_scalar('grad/scaling', scene.gaussians._scaling.grad.abs().max(), total_iter)

        for param_group in scene.gaussians.optimizer.param_groups:
            if param_group["name"] == "xyz":
                tb_writer.add_scalar('scene/lr_xyz', param_group['lr'], total_iter)

    # Report test and samples of training set
    if iter in testing_iterations:
        print('original lap_level', lap_level)
        lap_level = 0
        print('new lap_level', lap_level)
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            # out_dir = Path(scene.model_path) / f'report_{total_iter:05d}_{config["name"]}'
            # out_dir.mkdir(parents=True, exist_ok=True)
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                l1_depth = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    gt_image = torch.clamp(viewpoint.gauss_pyramid[lap_level], 0.0, 1.0)[0]
                    render_result = render(
                        viewpoint, 
                        scene.gaussians, 
                        *renderArgs, 
                        image_height=gt_image.shape[1], image_width=gt_image.shape[2]
                    )
                    rendered_image = render_result['render']
                     
                    # depth = render_result["depth"]
                    if last_pc is not None:
                        last_render_result = render(
                            viewpoint, 
                            last_pc, 
                            *renderArgs, 
                            image_height=gt_image.shape[1], image_width=gt_image.shape[2]
                        )
                        last_render_image = last_render_result['render']
                        image = torch.clamp(rendered_image + last_render_image * render_result['alpha_t'], 0.0, 1.0)
                    else:
                        image = torch.clamp(rendered_image, 0.0, 1.0)

                    # torchvision.utils.save_image(image, out_dir / f'{viewpoint.image_name}.png')
                    # torchvision.utils.save_image(gt_image, out_dir / f'{viewpoint.image_name}_gt.png')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=total_iter)
                        if iter == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=total_iter)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                # l1_depth /= len(config['cameras'])  
                l1_depth = 0        
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} depth {}".format(total_iter, config['name'], l1_test, psnr_test, l1_depth))
                print("\n[ITER {}] Gaussian Number: {}".format(total_iter, scene.gaussians.get_xyz.shape[0]))
                if config["name"] == "test":
                    with open(scene.model_path + "/chkpnt" + str(total_iter) + ".txt", "w") as file_object:
                        print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(total_iter, config['name'], l1_test, psnr_test), file=file_object)
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, total_iter)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, total_iter)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_depth', l1_depth, total_iter)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, total_iter)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], total_iter)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training GS & TCNN parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5000, 10000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5000, 10000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[15000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    # torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(dataset=lp.extract(args), 
             opt=op.extract(args), 
             pipe=pp.extract(args), 
             testing_iterations=args.test_iterations, 
             saving_iterations=args.save_iterations, 
             checkpoint_iterations=args.checkpoint_iterations, 
             checkpoint=args.start_checkpoint, 
        )

    # All done
    print("\nTraining complete.")
