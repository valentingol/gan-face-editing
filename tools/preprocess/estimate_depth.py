import os
import os.path as osp

import cv2
from cv2 import cvtColor
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt as dist_edt
import torch
import torchvision.transforms as transforms

from preprocess.depth_estimation.architecture.model import DPTDepthModel

from sklearn.cluster import KMeans


def depth_estimation_mix(edited_path, original_path, output_path, model_path):
    """Performs background correction of original 
    images with using depth estimation model and saves 
    the result in output_path

    Args:
        edited_path (str): path to the edited images
        original_path (str): path to the original images
        output_path (str): path to saved the output images
        model_path (str): path to the trained depth estimation model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # load depth estimation model. Input needs to be (1, 3, 384, 384)
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
        for image_name in os.listdir(original_path):
            base_image_name = image_name.split('.')[0]
            if not osp.exists(osp.join(output_path, base_image_name)):
                os.makedirs(osp.join(output_path, base_image_name))
            image = Image.open(osp.join(original_path, image_name))
            image = image.resize((512, 512), Image.BILINEAR)
            original_imgs[base_image_name] = image
            # resizing to network required input size
            image = image.resize((384, 384), Image.BILINEAR)
            image_tsr = to_tensor(image)
            depth = depth_estimation(image_tsr, net, device)
            depth = cv2.resize(depth, (512, 512),  
                               interpolation = cv2.INTER_NEAREST)
            depth_original[base_image_name] = depth
        print('Original images depth estimation done')
        n_images = len(os.listdir(edited_path))
        for i, image_dir in enumerate(os.listdir(edited_path)):
            depth_org = depth_original[image_dir]
            img_org = original_imgs[image_dir]

            for file_name in os.listdir(osp.join(edited_path, image_dir)):
                carac_name = file_name.split('.')[0]
                
                image = Image.open(osp.join(edited_path, image_dir, file_name))
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
                                   interpolation = cv2.INTER_NEAREST)
                image = np.array(image)
                
                # Add Foreground of the original image
                img_final = fix_background(depth_org, depth, img_org, image)
                img_final = img_final.astype(np.uint8)
                img_final = cvtColor(img_final, cv2.COLOR_RGB2BGR)
                # Save image
                cv2.imwrite(osp.join(output_path, image_dir, file_name),
                            img_final)
                print(f'image {i+1}/{n_images} done  ', end='\r')
    print()


def get_foreground(depth):
    """Get foreground pixels from depth map 
    using K-means clustering

    Args:
        depth (np.array): depth map array

    Returns:
        np.array: np.array of shape (n_foreground_pixels, 1) 
                  containing the values of foreground pixels
    """
    Z = depth.flatten().reshape(-1,1)
    km = KMeans(n_clusters=2, random_state=0).fit(Z)
    label = km.labels_.reshape(-1,1)

    return Z[label.flatten() == 0]

def get_foreground_mask(depth):
    """Get foreground mask from depth map

    Args:
        depth (np.array): depth map array

    Returns:
        np.array: foreground mask array of shape depth.shape
    """
    # get foreground pixels values
    ground = get_foreground(depth)
    mask = np.ones_like(depth)
    # take pixels with depth value greater than the mean of foreground pixels
    mask[np.where(depth < ground[:, 0].mean())] = 0
    return mask


def fix_background(depth_org, depth_img, img_org, img):
    """Return target image with base_image background 
    using monocular depth estimation

    Args:
        depth_org (np.array): depth map of original image
        depth (np.array): depth map of target image
        img_org (np.array): original image
        image (np.array): target image       

    Returns:
        np.array: target image with img_org background
    """
    #foreground of the transformed image
    mask_target = get_foreground_mask(depth_img)
    mask_base = get_foreground_mask(depth_org)
    
    
    if np.sum(mask_target) > np.sum(mask_base): 
        #if the target image has more foreground pixels than the original image
        #take the original image foregroud mask
        mask = mask_base
    else:
        #else take the target image foreground mask
        mask = mask_target

    # smooth foreground mask
    mask_background = dist_edt(1-mask)
    mask_background = mask_background/mask_background.max()
    mask_background = np.multiply(mask_background, 1-mask)
    
    # foreground coefficients
    mask_foreground_smooth = np.exp(-8*mask_background[:,:,None])
    
    #background coefficients
    mask_background_smooth = mask_background[:,:,None]
    
    mask_total = mask_foreground_smooth + mask_background_smooth
    
    # normalized foreground and background coefficients
    mask_foreground_smooth_final = mask_foreground_smooth / mask_total
    mask_background_smooth_final = mask_background_smooth / mask_total
    
    # take the foreground of image, and the background of the img_org
    image = np.multiply(mask_foreground_smooth_final, img) + \
        np.multiply(mask_background_smooth_final, img_org)
    return image

def depth_estimation(image, net, device):
    """Compute depth map from image using monocular depth estimation

    Args:
        image (torch.Tensor): image tensor of shape (1, 3, H, W)
        net (torch.nn.Module): depth estimation model
        device (torch.device): device to run the model

    Returns:
        np.array: depth map array of shape (H, W)
    """
    if image.ndim == 3:
        image = torch.unsqueeze(image, 0)
    image = image.to(device)
    net.eval()
    with torch.no_grad():
        pred = net(image)[0]
    pred = pred.cpu().numpy()
    return pred


if __name__ == "__main__":
    
    print('Depth estimation processing')
    # Path to the original images
    original_path ='data/input_images'
    # Path to the edited images
    edited_path = 'preprocess/segmentation/edited_images_postsegmentation'
    # Path to the segmented edited images
    output_path = 'preprocess/depth_estimation/edited_images_postdepth'
    # Path to the model
    model_path = "preprocess/depth_estimation/cp/dpt_large-midas-2f21e586.pt"

    depth_estimation_mix(edited_path=edited_path, original_path=original_path,
                         output_path=output_path, model_path=model_path)
