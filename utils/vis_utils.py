# copy from https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/utils/vis_utils.py
# copy from nerfstudio and 2DGS
import torch
from matplotlib import cm
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from trimesh import primitives, util


def vis_patch(debug_dict: dict, cur_img, neighbor_img, ids: list):

    H, W = cur_img.shape[1:]
    gt_img_cur = (cur_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    gt_img_neighbor = (neighbor_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    img_cur = np.zeros((H, W, 3), dtype=np.uint8)
    img_neighbor = np.zeros((H, W, 3), dtype=np.uint8)
    p_num = debug_dict['cur_patch'].shape[1]
    
    for patch_id in ids:
        c = np.random.randint(50, 255, 3)
        for pid in range(p_num):
            x, y = debug_dict['cur_patch'][patch_id, pid]
            x, y = int((x.item() + 1) * (W - 1) / 2), int((y.item() + 1) * (H - 1) / 2)
            if x >= 0 and x < W and y >= 0 and y < H:
                img_cur[y, x] = c
                gt_img_cur[y, x] = c
            else:
                print('outside cur patch_id:', patch_id, 'pid:', pid, 'x:', x, 'y:', y)
            x, y = debug_dict['neighbor_patch'][patch_id, pid]
            x, y = int((x.item() + 1) * (W - 1) / 2), int((y.item() + 1) * (H - 1) / 2)
            if x >= 0 and x < W and y >= 0 and y < H:
                img_neighbor[y, x] = c
                gt_img_neighbor[y, x] = c
            else:
                print('outside neighbor patch_id:', patch_id, 'pid:', pid, 'x:', x, 'y:', y)
    
    return {
        'gt_img_cur': gt_img_cur,  
        'gt_img_neighbor': gt_img_neighbor,
        'img_patch_cur': img_cur,
        'img_patch_neighbor': img_neighbor
    }


def apply_colormap(image, cmap="viridis"):
    colormap = cm.get_cmap(cmap)
    colormap = torch.tensor(colormap.colors).to(image.device)  # type: ignore
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return colormap[image_long[..., 0]]


def apply_depth_colormap(
    depth,
    accumulation = None,
    near_plane = 2.0,
    far_plane = 4.0,
    cmap="turbo",
):
    # near_plane = near_plane or float(torch.min(depth))
    # far_plane = far_plane or float(torch.max(depth))
    near_plane = near_plane
    far_plane = far_plane

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)
    # depth = torch.nan_to_num(depth, nan=0.0) # TODO(ethan): remove this

    colored_image = apply_colormap(depth, cmap=cmap)

    if accumulation is not None:
        colored_image = colored_image * accumulation + (1 - accumulation)

    return colored_image

def save_points(path_save, pts, colors=None, normals=None, BRG2RGB=False):
    """save points to point cloud using open3d"""
    assert len(pts) > 0
    if colors is not None:
        assert colors.shape[1] == 3
    assert pts.shape[1] == 3

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        # Open3D assumes the color values are of float type and in range [0, 1]
        if np.max(colors) > 1:
            colors = colors / np.max(colors)
        if BRG2RGB:
            colors = np.stack([colors[:, 2], colors[:, 1], colors[:, 0]], axis=-1)
        cloud.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals)

    o3d.io.write_point_cloud(path_save, cloud)
    

def colormap(img, cmap='jet'):
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H/dpi, W/dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data / 255.).float().permute(2,0,1)
    plt.close()
    if img.shape[1:] != (H, W):
        img = torch.nn.functional.interpolate(img[None], (W, H), mode='bilinear', align_corners=False)[0]
    return img


def build_spheres(pt: np.array, radius: np.array, color: np.array):
    
    if pt.ndim > 1:
        meshes = []
        for i in range(pt.shape[0]):
            r = radius[i] if radius.ndim > 1 else radius
            c = color[i] if color.ndim > 1 else color
            meshes.append(build_spheres(pt[i], r, c))
        return util.concatenate(meshes)

    sphere = primitives.Sphere(radius=radius, center=pt, subdivisions=2)
    sphere.visual.vertex_colors = color

    return sphere
