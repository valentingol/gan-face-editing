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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

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
            depth = cv2.resize(depth, (512, 512),  interpolation = cv2.INTER_NEAREST)
            depth_original[base_image_name] = depth
        print('Original images depth estimation done')
        n_images = len(os.listdir(edited_path))
        print(os.listdir(edited_path))
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
                depth = cv2.resize(depth, (512, 512),  interpolation = cv2.INTER_NEAREST)
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
    Z = depth.flatten().reshape(-1,1)
    km = KMeans(n_clusters=2, random_state=0).fit(Z)
    label = km.labels_.reshape(-1,1)

    return Z[label.flatten() == 0]

def get_foreground_mask(depth):
    ground = get_foreground(depth)
    mask = np.ones_like(depth)
    mask[np.where(depth < 9*ground[:, 0].mean()/10)] = 0
    return mask


def fix_background(depth_base, depth_target, base_image, target_image):
    
    #foreground of the transformed image
    mask_target = get_foreground_mask(depth_target)
    mask_base = get_foreground_mask(depth_base)
    
    
    if np.sum(mask_target) > np.sum(mask_base):
        mask_target = mask_base
    #transistion coefficient between the two images (foreground from target and background from base)
    mask_background_transition_edt = dist_edt(1-mask_target)
    mask_background_transition_edt = mask_background_transition_edt/mask_background_transition_edt.max()
    mask_background_transition_edt = np.multiply(mask_background_transition_edt, 1-mask_target)
    
    # foreground coefficients
    mask_target_smooth = np.exp(-8*mask_background_transition_edt[:,:,None])
    
    #background coefficients
    mask_background_smooth = mask_background_transition_edt[:,:,None]
    
    mask_target_smooth_final = mask_target_smooth/(mask_target_smooth+mask_background_smooth)
    mask_background_smooth_final = mask_background_smooth/(mask_target_smooth+mask_background_smooth)
    
    image = np.multiply(mask_target_smooth_final, target_image) + \
        np.multiply(mask_background_smooth_final, base_image)
    return image

def depth_estimation(image, net, device):
    """
    Classes Legend:
    ##### From Model #######
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
    edited_path = 'preprocess/mixup/edited_images_postmixup'
    # Path to the segmented edited images
    output_path = 'preprocess/segmentation/edited_images_postsegmentation'
    # Path to the model
    model_path = "preprocess/depth_estimation/cp/dpt_large-midas-2f21e586.pt"

    depth_estimation_mix(edited_path=edited_path, original_path=original_path,
                         output_path=output_path, model_path=model_path)
