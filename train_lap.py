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
import torch
import trimesh
import datetime
from random import randint, shuffle
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
from pathlib import Path
from tqdm import tqdm
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


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
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
    
    pc_pyramid = {}
    total_iter = 0
    render_pkg_last = {}
    old_gaussian_num = 0

    for cur_lap_level in range(dataset.lap_pyramid_level - 1, -1, -1):

        print('Current Lap level:', f'{cur_lap_level} / {dataset.lap_pyramid_level - 1}')

        viewpoint_stack = None
        ema_loss_for_log, ema_depth_loss_for_log, ema_mask_loss_for_log, ema_normal_loss_for_log = 0.0, 0.0, 0.0, 0.0
        progress_bar = tqdm(range(1, opt.lap_level_upgrade_interval + 1), desc="Training progress")

        for iteration in range(1, opt.lap_level_upgrade_interval + 1):

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
            if total_iter % 1000 == 0:
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
            
            rendered_mask: torch.Tensor = render_pkg["mask"]
            rendered_alpha_t: torch.Tensor = render_pkg["alpha_t"]
            rendered_depth: torch.Tensor = render_pkg["depth"]
            rendered_middepth: torch.Tensor = render_pkg["middepth"]
            rendered_normal: torch.Tensor = render_pkg["normal"]
            depth_distortion: torch.Tensor = render_pkg["depth_distortion"]
            rendered_depth = rendered_depth / rendered_mask
            rendered_depth = torch.nan_to_num(rendered_depth, 0, 0)

 
            if cur_lap_level == dataset.lap_pyramid_level - 1:
                final_image = rendered_image
            else:
                # pass
                view_pkg_last = render_pkg_last[viewpoint_cam.image_name]
                render_fix_front = render_pkg['render'] + view_pkg_last['render'] * rendered_alpha_t # auto broadcast .repeat(3, 1, 1)
                # render_fix_back = view_pkg_last['render'] + render_pkg['render'] * view_pkg_last['alpha_t'] # auto broadcast .repeat(3, 1, 1)
                # final_image = torch.where(view_pkg_last['depth'] > rendered_depth, render_fix_front, render_fix_back)
                final_image = render_fix_front
                # x = rendered_depth.view(-1) == 0
                # rendered_image = rendered_image.view(3, -1)[:, x]

            if dataset.use_decoupled_appearance:
                Ll1_render = L1_loss_appearance(final_image, gt_image, gaussians, viewpoint_cam.uid)
            else:
                Ll1_render = l1_loss(final_image, gt_image)

            if opt.verbose and (iteration % 50 == 0 or iteration < 100):
                loss_map = torch.abs(final_image - gt_image).sum(dim=0)
                rendered_image_color = apply_colormap(
                    torch.clip(torch.max(rendered_image, dim=0, keepdim=True)[0].permute(1, 2, 0), min=0, max=1)
                ).permute(2, 0, 1)

                torchvision.utils.save_image(rendered_alpha_t, out_dir / f'{total_iter:05d}_alpha_t.png')
                torchvision.utils.save_image(loss_map, out_dir / f'{total_iter:05d}_l1loss.png')
                torchvision.utils.save_image(rendered_image, out_dir / f'{total_iter:05d}_rendered.png')
                torchvision.utils.save_image(rendered_image_color, out_dir / f'{total_iter:05d}_rendered_color.png')
                torchvision.utils.save_image(final_image, out_dir / f'{total_iter:05d}_final.png')
                torchvision.utils.save_image(gt_image, out_dir / f'{total_iter:05d}_gt.png')

                if cur_lap_level != dataset.lap_pyramid_level - 1:
                    view_pkg_last = render_pkg_last[viewpoint_cam.image_name]
                    # torchvision.utils.save_image(view_pkg_last['render'], out_dir / f'{total_iter:05d}_last_rendered.png')
                    # torchvision.utils.save_image(view_pkg_last['alpha_t'], out_dir / f'{total_iter:05d}_last_alpha_t.png')
                    # torchvision.utils.save_image(render_fix_front, out_dir / f'{total_iter:05d}_render_fix_front.png')
                    # torchvision.utils.save_image(render_fix_back, out_dir / f'{total_iter:05d}_render_fix_back.png')
                    # torchvision.utils.save_image((view_pkg_last['depth'] > rendered_depth).float(), out_dir / f'{total_iter:05d}_depth_test.png')
                    
                    # depth_diff = view_pkg_last['depth'] - rendered_depth
                    # depth_diff = torch.clip(depth_diff + 0.5, 0, 1).permute(1, 2, 0)
                    # depth_diff_img = apply_colormap(depth_diff).permute(2, 0, 1)
                    # torchvision.utils.save_image(depth_diff_img, out_dir / f'{total_iter:05d}_depth_diff.png')

                depth_img = apply_depth_colormap(rendered_depth.permute(1, 2, 0)).cpu().numpy() * 255
                imageio.imwrite(out_dir / f'{total_iter:05d}_depth.png', depth_img.astype(np.uint8))

                tmp = torch.nn.functional.normalize(rendered_normal, p=2, dim=0)
                tmp = tmp.permute(1,2,0)
                rendered_normal_img = np.clip(np.rint(tmp.detach().cpu().numpy() * 255), 0, 255).astype(np.uint8)
                imageio.imwrite(out_dir / f'{total_iter:05d}_normal.png', rendered_normal_img)


            if iteration >= opt.depth_opt_from_iter:
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

            # if cur_lap_level < dataset.lap_pyramid_level - 1:
            #     # freeze old gaussians
            #     for param_group in gaussians.optimizer.param_groups:
            #         if param_group['name'] in ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation']:
            #             param_group['params'][0].grad[:old_gaussian_num] = 0

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_depth_loss_for_log = 0.4 * distortion_loss.item() + 0.6 * ema_depth_loss_for_log
                ema_normal_loss_for_log = 0.4 * depth_normal_loss.item() + 0.6 * ema_normal_loss_for_log

                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{4}f}", "loss_dep": f"{ema_depth_loss_for_log:.{4}f}", "loss_normal": f"{ema_normal_loss_for_log:.{4}f}"})
                    progress_bar.update(10)
                if iteration == opt.lap_level_upgrade_interval:
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
                    pc_pyramid.get(cur_lap_level + 1, None), (pipe, background)
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
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold, old_gaussian_num)
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

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        
        # Record result, prepare next level 
        if cur_lap_level > 0:
            # old_gaussian_num = gaussians.get_xyz.shape[0]

            viewpoint_stack = scene.getTrainCameras().copy()
            if opt.verbose:
                upscale_loss_dir = out_dir / f'upscale_{cur_lap_level}'
                upscale_loss_dir.mkdir(parents=True, exist_ok=True)

            depth_disturbs = [1]
            
            with torch.no_grad():
                l1_test = 0.0
                psnr_test = 0.0
                step = len(viewpoint_stack) // opt.new_gauss_view_num

                pts_xyz_collection = torch.empty(0, 3, device='cuda')
                pts_radius_collection = torch.empty(0, 1, device='cuda')
                pts_color_collection = torch.empty(0, 3, device='cuda')

                for view_idx, view in tqdm(enumerate(viewpoint_stack), desc='Save last rendering results'):

                    next_gt_image = view.gauss_pyramid[cur_lap_level - 1].squeeze(0)

                    render_pkg = render(view, gaussians, pipe, background, image_height=next_gt_image.shape[1], image_width=next_gt_image.shape[2])
                    render_pkg['depth'] = render_pkg['depth'] / render_pkg['mask']
                    render_pkg['depth'] = torch.nan_to_num(render_pkg['depth'], 0, 0)
                    render_pkg_last[view.image_name] = render_pkg
                    render_pkg_last[view.image_name]['render_loss'] = torch.abs(next_gt_image - render_pkg['render'])

                    l1_res = render_pkg_last[view.image_name]['render_loss'].mean().double()
                    l1_test += l1_res
                    psnr_test += psnr(render_pkg['render'], next_gt_image).mean().double()

                    if opt.verbose:
                        torchvision.utils.save_image(render_pkg_last[view.image_name]['render_loss'], upscale_loss_dir / f'{view.image_name}_l1loss.png')
                        torchvision.utils.save_image(render_pkg['render'], upscale_loss_dir / f'{view.image_name}_render.png')
                        torchvision.utils.save_image(next_gt_image, upscale_loss_dir / f'{view.image_name}_gt.png')

                    if view_idx % step == 0:
                        
                        pts_bad = render_pkg_last[view.image_name]['render_loss'].sum(dim=0) > opt.new_gauss_l1loss_thres
                        if opt.verbose:
                            torchvision.utils.save_image(pts_bad.float(), upscale_loss_dir / f'{view.image_name}_bad.png')

                        for depth_disturb in depth_disturbs:
                            print('Add new Gaussians at', view_idx, 'with depth_disturb', depth_disturb)
                            pts_xyz, pts_radius = depth_to_points_fast(view, render_pkg['depth'] * depth_disturb)
                            pts_color = next_gt_image.permute(1, 2, 0).reshape(-1, 3)

                            pts_mask = pts_radius.squeeze(1) > 0
                            pts_mask = pts_mask & pts_bad.reshape(-1)
                            pts_xyz = pts_xyz[pts_mask]
                            pts_radius = pts_radius[pts_mask]
                            pts_color = pts_color[pts_mask]
                            pts_xyz_view = torch.concat([pts_xyz, torch.ones_like(pts_radius)], dim=-1)
                            pts_xyz_world = pts_xyz_view @ view.world_view_transform_inv
                            pts_xyz_world = pts_xyz_world[:, :3]

                            pts_xyz_collection = torch.cat([pts_xyz_collection, pts_xyz_world], dim=0)
                            pts_radius_collection = torch.cat([pts_radius_collection, pts_radius], dim=0)
                            pts_color_collection = torch.cat([pts_color_collection, pts_color], dim=0)

                            if opt.verbose:
                                vis_spheres: trimesh.Trimesh = build_spheres(
                                    pts_xyz_world.cpu().detach().numpy(), 
                                    pts_radius.cpu().detach().numpy(), 
                                    pts_color.cpu().detach().numpy()
                                )
                                vis_spheres.export(upscale_loss_dir / f'{view.image_name}_spheres_world_scale_sqrt2_{depth_disturb:.3f}.ply')
                        
                l1_test /= len(viewpoint_stack)
                psnr_test /= len(viewpoint_stack)

                print(f"\n[ITER {total_iter} Upscale] Evaluating Train: L1 {l1_test} PSNR {psnr_test}")

                gaussians.__init__(dataset.sh_degree)
                scene.add_gaussian_from_pts(pts_xyz_collection, pts_radius_collection, pts_color_collection, init=True)
                gaussians.training_setup(opt)
                gaussians.compute_3D_filter(cameras=trainCameras)
                
                if opt.verbose:
                    render_new_gauss_pkg = render(view, gaussians, pipe, background, image_height=next_gt_image.shape[1], image_width=next_gt_image.shape[2])
                    torchvision.utils.save_image(render_new_gauss_pkg['render'], upscale_loss_dir / f'new_gauss_render.png')



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
    if False and TENSORBOARD_FOUND:
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
                    render_image = render_result['render']
                     
                    # depth = render_result["depth"]
                    if last_pc is not None:
                        last_render_result = render(
                            viewpoint, 
                            last_pc, 
                            *renderArgs, 
                            image_height=gt_image.shape[1], image_width=gt_image.shape[2]
                        )
                        last_render_image = last_render_result['render']
                        image = torch.clamp(render_image + last_render_image * render_result['alpha_t'], 0.0, 1.0)
                    else:
                        image = torch.clamp(render_image, 0.0, 1.0)

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
    parser = ArgumentParser(description="Training script parameters")
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
             debug_from=args.debug_from)

    # All done
    print("\nTraining complete.")
