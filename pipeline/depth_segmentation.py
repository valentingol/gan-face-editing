import os
import os.path as osp

import cv2
from cv2 import cvtColor
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt as dist_edt
import torch
import torchvision.transforms as transforms

from pipeline.utils.depth_segmentation.model import DPTDepthModel

from sklearn.cluster import KMeans


def depth_estimation_mix(data_dir, input_path, output_path, model_path,
                         configs):
    """
    Performs background correction of original
    images with using depth estimation model and saves
    the result in output_path.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load depth estimation model. Input needs to be (1, 3, 384, 384)
    net = DPTDepthModel(
        path=model_path,
        backbone="vitl16_384",
        non_negative=True,
        enable_attention_hooks=False,
    )

    net.to(device)
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    with torch.no_grad():
        # Original images
        depth_original, original_imgs = {}, {}
        for image_name in os.listdir(data_dir):
            base_image_name = image_name.split('.')[0]
            if not osp.exists(osp.join(output_path, base_image_name)):
                os.makedirs(osp.join(output_path, base_image_name))
            image = Image.open(osp.join(data_dir, image_name))
            image = image.resize((512, 512), Image.BILINEAR)
            original_imgs[base_image_name] = image
            # resizing to network required input size
            image = image.resize((384, 384), Image.BILINEAR)
            image_tsr = to_tensor(image)
            depth = depth_estimation(image_tsr, net, device)
            depth = cv2.resize(depth, (512, 512),
                               interpolation=cv2.INTER_NEAREST)
            depth_original[base_image_name] = depth
        print('Original images depth estimation done')
        n_images = len(os.listdir(input_path))
        for i, image_dir in enumerate(os.listdir(input_path)):
            depth_org = depth_original[image_dir]
            img_org = original_imgs[image_dir]

            for file_name in os.listdir(osp.join(input_path, image_dir)):
                carac_name = file_name.split('.')[0]

                image = Image.open(osp.join(input_path, image_dir, file_name))
                image = image.resize((512, 512), Image.BILINEAR)
                if carac_name == 'bald' or carac_name.startswith("Se"):
                    image = np.array(image)
                    image = image.astype(np.uint8)
                    image = cvtColor(image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(osp.join(output_path, image_dir, file_name),
                                image)
                    print(f'image {i+1}/{n_images} done  ', end='\r')
                    continue

                # Get depth estimation of the edited image
                image_resized = image.resize((384, 384), Image.BILINEAR)
                image_tsr = to_tensor(image_resized)
                depth = depth_estimation(image_tsr, net, device)
                depth = cv2.resize(depth, (512, 512),
                                   interpolation=cv2.INTER_NEAREST)
                image = np.array(image)

                # Add Foreground of the original image
                img_final = fix_background(
                    depth_org, depth, img_org, image,
                    foreground_coef=configs['foreground_coef'])
                img_final = img_final.astype(np.uint8)
                img_final = cvtColor(img_final, cv2.COLOR_RGB2BGR)
                # Save image
                cv2.imwrite(osp.join(output_path, image_dir, file_name),
                            img_final)
            print(f'image {i+1}/{n_images} done  ', end='\r')
    print()


def get_foreground(depth):
    """Get foreground pixels from depth map using K-means clustering.

    Parameters
    ----------
        depth : np.array
            Depth map array

    Returns
    -------
    np.array, shape (n_foreground_pixels, 1):
        Values of foreground pixels
    """
    Z = depth.flatten().reshape(-1, 1)
    km = KMeans(n_clusters=2, random_state=0).fit(Z)
    label = km.labels_.reshape(-1, 1)

    return Z[label.flatten() == 0]


def get_foreground_mask(depth):
    """Get foreground mask from depth map

    Parameters
    ----------
    depth : np.array
        Depth map array

    Returns
    -------
    mask : np.array
        foreground mask array of shape depth.shape
    """
    # Get foreground pixels values
    ground = get_foreground(depth)
    mask = np.ones_like(depth)
    # Take pixels with depth value greater than the mean of
    # foreground pixels
    mask[np.where(depth < ground[:, 0].mean())] = 0
    return mask


def fix_background(depth_org, depth_img, img_org, img, foreground_coef):
    """Return target image with base_image background
    using monocular depth estimation

    Parameters
    ----------
    depth_org : np.array
        Depth map of original image
    depth np.array:
        Depth map of target image
    img_org : np.array
        Original image
    image : np.array
        Target image
    foreground_coef: float
        Coefficient of foreground exponential weighting:
        exp(-coef * mask)

    Returns
    -------
    np.array:
        Target image with img_org background
    """
    # Foreground of the transformed image
    mask_target = get_foreground_mask(depth_img)
    mask_base = get_foreground_mask(depth_org)

    if np.sum(mask_target) > np.sum(mask_base):
        # If the target image has more foreground pixels than the
        # original image take the original image foregroud mask
        mask = mask_base
    else:
        # Else take the target image foreground mask
        mask = mask_target

    # Smooth foreground mask
    mask_background = dist_edt(1-mask)
    mask_background = mask_background/mask_background.max()
    mask_background = np.multiply(mask_background, 1-mask)

    # Foreground coefficients
    mask_foreground_smooth = np.exp(-foreground_coef
                                    * mask_background[:, :, None])

    # Background coefficients
    mask_background_smooth = mask_background[:, :, None]

    mask_total = mask_foreground_smooth + mask_background_smooth

    # Normalized foreground and background coefficients
    mask_foreground_smooth_final = mask_foreground_smooth / mask_total
    mask_background_smooth_final = mask_background_smooth / mask_total

    # Take the foreground of image, and the background of the img_org
    image = np.multiply(mask_foreground_smooth_final, img) + \
        np.multiply(mask_background_smooth_final, img_org)
    return image


def depth_estimation(image, net, device):
    """Compute depth map from image using monocular depth estimation

    Parameters
    ----------
    image : torch.Tensor
        Image tensor of shape (1, 3, H, W)
    net : torch.nn.Module
        Depth estimation model
    device : torch.device
        Device to run the model

    Returns
    -------
    depth_map : np.array
        Depth map array of shape (H, W). On CPU device.
    """
    if image.ndim == 3:
        image = torch.unsqueeze(image, 0)
    image = image.to(device)
    net.eval()
    with torch.no_grad():
        depth_map = net(image)[0]
    depth_map = depth_map.cpu().numpy()
    return depth_map


if __name__ == "__main__":
    print('Applying depth estimation mixup...')
    # Path to the original images
    data_dir = 'data/face-challenge'
    # Path to the edited images
    input_path = 'res/run1/images_post_segmentation'
    # Path to the segmented edited images
    output_path = 'res/run1/images_post_depth_segmentation'
    # Path to the model
    model_path = ('postprocess/depth_segmentation/model/'
                  'dpt_large-midas-2f21e586.pt')
    configs = {
        'foreground_coef': 8.0,
        }

    depth_estimation_mix(data_dir=data_dir, input_path=input_path,
                         output_path=output_path, model_path=model_path,
                         configs=configs)
