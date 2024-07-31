import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from pathlib import Path


def pyr_down(image: torch.Tensor, kernel: torch.Tensor, half_kernel_size: int):
    """
    Perform pyrDown (downsampling) operation in PyTorch.
    :param image: Input image tensor of shape (1, 3, H, W)
    :param kernel_size: Size of the Gaussian kernel
    :return: Downsampled image tensor
    """
    # Apply Gaussian blur
    if image.shape[1] == 1:
        image = F.conv2d(image, kernel[0:1], stride=1, padding=half_kernel_size, groups=1)
    else:
        image = F.conv2d(image, kernel, stride=1, padding=half_kernel_size, groups=3)
    # Downsample by taking every second pixel
    return image[..., ::2, ::2]


def pyr_up(image: torch.Tensor, kernel: torch.Tensor, half_kernel_size: int):
    """
    Perform pyrUp (upsampling) operation in PyTorch.
    :param image: Input image tensor of shape (1, 3, H, W)
    :param kernel_size: Size of the Gaussian kernel
    :return: Upsampled image tensor
    """
    # Upsample by inserting zeros
    # upsampled_image = torch.zeros(image.shape[0], image.shape[1], image.shape[2]*2, image.shape[3]*2, device=image.device)
    # upsampled_image[:, :, ::2, ::2] = image
    
    # Apply Gaussian blur
    # return F.conv2d(upsampled_image, kernel, stride=1, padding=half_kernel_size, groups=3)
    out_image = F.conv_transpose2d(image, kernel, stride=2, padding=half_kernel_size, groups=3, output_padding=1)
    return out_image


def get_gaussian_kernel(kernel_size, sigma=1.0):
    """
    Create a Gaussian kernel to be used in convolution.
    :param kernel_size: Size of the Gaussian kernel
    :param sigma: Standard deviation of the Gaussian
    :return: Gaussian kernel tensor
    """
    k = kernel_size // 2
    x = torch.arange(-k, k+1, dtype=torch.float32)
    gauss = torch.exp(-0.5 * (x**2) / sigma**2)
    gauss = gauss / gauss.sum()
    
    kernel = gauss[:, None] * gauss[None, :]
    kernel = kernel[None, None, :, :]  # Reshape to 4D tensor
    kernel = kernel.repeat(3, 1, 1, 1)  # Repeat for each channel
    kernel.requires_grad = False
    kernel = kernel.to(torch.float32)
    return kernel


def build_gaussian_pyramid(image: torch.Tensor, pyramid_level: int, kernel: torch.Tensor, half_kernel_size: int):
    """
    Build a Gaussian pyramid for an image.
    :param image: Input image
    :param levels: Number of levels in the pyramid
    :return: List of Gaussian pyramid images
    """
    gaussian_pyramid = [image]
    for i in range(1, pyramid_level):
        image = pyr_down(image, kernel, half_kernel_size)
        gaussian_pyramid.append(image)
    return gaussian_pyramid


def reconstruct_image(laplacian_pyramid, kernel: torch.Tensor, half_kernel_size: int):
    """
    Reconstruct the original image from the Laplacian pyramid.
    :param laplacian_pyramid: List of Laplacian pyramid images
    :return: Reconstructed image
    """
    levels = len(laplacian_pyramid)
    image = laplacian_pyramid[-1]
    
    for i in range(levels - 1, 0, -1):
        image = pyr_up(image, 4 * kernel, half_kernel_size) 
        if image.shape != laplacian_pyramid[i - 1].shape:
            image = F.interpolate(image, (laplacian_pyramid[i - 1].shape[-2], laplacian_pyramid[i - 1].shape[-1]), mode='bilinear', align_corners=True)
        image = image + laplacian_pyramid[i - 1]
    
    return image[0]


def compute_psnr(original_image, reconstructed_image):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between the original image and the reconstructed image.
    :param original_image: Original image
    :param reconstructed_image: Reconstructed image from the Laplacian pyramid
    :return: PSNR value
    """
    # print(original_image.max())
    mse = torch.mean((original_image - reconstructed_image) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr


def display_reconstructed(image: torch.Tensor, reconstructed_image: torch.Tensor, out_path: str = None):

    plt.figure(figsize=(10, 5))
        
    plt.subplot(1, 2, 1)
    plt.imshow(to_pil_image(image))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(to_pil_image(reconstructed_image))
    plt.title('Reconstructed Image')
    plt.axis('off')

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path)

    
def display_pyramids(gaussian_pyramid: list, laplacian_pyramid: list, out_path: str = None):
    """
    Display the Gaussian and Laplacian pyramids.
    :param gaussian_pyramid: List of Gaussian pyramid images
    :param laplacian_pyramid: List of Laplacian pyramid images
    """
    levels = len(gaussian_pyramid)
    
    plt.figure(figsize=(12, 6))
    
    for i in range(levels):
        plt.subplot(2, levels, i + 1)
        plt.imshow(to_pil_image(gaussian_pyramid[i].squeeze(0)))
        plt.title(f'Gaussian Level {i}')
        plt.axis('off')
        
        plt.subplot(2, levels, i + levels + 1)
        # laplacian_image = cv2.normalize(laplacian_pyramid[i], None, 0, 255, cv2.NORM_MINMAX)
        plt.imshow(to_pil_image(abs(laplacian_pyramid[i]).squeeze(0)))
        plt.title(f'Laplacian Level {i}')
        plt.axis('off')
    
    plt.tight_layout()
    # plt.show()
    if out_path:
        plt.savefig(out_path)


def build_lap_pyramid(
        cam_list: list, 
        pyramid_level: int, 
        device: torch.device,
        debug: bool = False,
        out_dir: str = None,
        prefix: str = None
    ):

    kernel_size = 5
    half_kernel_size = kernel_size // 2
    kernel = get_gaussian_kernel(5).to(device)
    new_cam_list = []

    if debug:
        assert out_dir is not None
        out_dir = Path(out_dir) / f'{prefix}_lap_pyramid'
        out_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(enumerate(cam_list), total=len(cam_list), desc=f'Lap Pyramid {prefix}')

    for cam_idx, cam in enumerate(cam_list):

        gaussian_pyramid = build_gaussian_pyramid(cam.original_image.unsqueeze(0), pyramid_level, kernel, half_kernel_size)
        laplacian_pyramid = [gaussian_pyramid[-1]]
        edge_pyramid = build_gaussian_pyramid(cam.edge.unsqueeze(0).unsqueeze(1), pyramid_level, kernel, half_kernel_size)

        # torchvision.utils.save_image(gaussian_pyramid[pyramid_level-1], f'{out_dir}/gauss_{cam_idx}_{pyramid_level-1}.jpg')
        
        for i in range(pyramid_level - 1, 0, -1):
            gaussian_expanded = pyr_up(gaussian_pyramid[i], 4 * kernel, half_kernel_size)
            if gaussian_expanded.shape != gaussian_pyramid[i - 1].shape:
                gaussian_expanded = F.interpolate(gaussian_expanded, (gaussian_pyramid[i - 1].shape[-2], gaussian_pyramid[i - 1].shape[-1]), mode='bilinear', align_corners=True)
            laplacian = gaussian_pyramid[i - 1] - gaussian_expanded
            laplacian_pyramid.append(laplacian)

            # torchvision.utils.save_image(gaussian_expanded, f'{out_dir}/gauss_{cam_idx}_{i-1}.jpg')

        laplacian_pyramid = laplacian_pyramid[::-1]
        
        cam.lap_pyramid = laplacian_pyramid
        cam.gauss_pyramid = gaussian_pyramid
        cam.edge_pyramid = edge_pyramid
        new_cam_list.append(cam)

        if debug:

            reconstructed_image = reconstruct_image(laplacian_pyramid, kernel, half_kernel_size)
            psnr = compute_psnr(cam.original_image, reconstructed_image)
            pbar.update(1)
            pbar.set_postfix({'PSNR': f'{psnr:.2f}'})
            # print(f"PSNR for camera {cam_idx}: {psnr:.2f}")

            out_path = f'{out_dir}/reconstructed_{cam_idx}.jpg'
            display_reconstructed(cam.original_image, reconstructed_image, out_path)

            out_path = f'{out_dir}/pyramids_{cam_idx}.jpg'
            display_pyramids(gaussian_pyramid, laplacian_pyramid, out_path)

            plt.close('all')

    return new_cam_list

