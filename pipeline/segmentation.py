# Code adapted from https://github.com/zllrunning/face-parsing.PyTorch
"""Segmentation mixup using BiSeNet."""

import os
import os.path as osp

import cv2
import numpy as np
from cv2 import cvtColor
from PIL import Image
from scipy.ndimage import distance_transform_edt as dist_edt
from torchvision import transforms

from pipeline.utils.segmentation.tools import (get_model, init_segmentation,
                                               segmentation)


def alpha_from_dist(dist, margin):
    """Compute alpha from distance matrix.

    Parameters
    ----------
    dist : numpy.ndarray
        Distance 2D matrix.
    margin : float
        Margin for the alpha computation (smoothness).
    """
    return np.where(dist < margin, dist / margin, 1.0)


def segmentation_mix(data_dir, input_path, output_path, model_path, configs):
    """Apply segmentation mixup.

    Parameters
    ----------
    data_dir : str
        Path to the directory with original images
    input_path : str
        Path to the directory with input (edited) images
    output_path : str
        Path to the output directory
    model_path : str
        Path to the BiSeNet model weights.
    configs : dict or GlobalConfig
        Configurations for the domain mixup.
    """
    avoid_transformations_list = configs['avoid_transformations']

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    net, device = get_model(model_path)

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    for image_name in os.listdir(data_dir):
        base_image_name = image_name.split('.')[0]
        if not osp.exists(osp.join(output_path, base_image_name)):
            os.makedirs(osp.join(output_path, base_image_name))

    # Original images
    if data_dir[-1] == '/':
        org_seg_dir = data_dir[:-1] + '_segmented_'
    else:
        org_seg_dir = data_dir + '_segmented_'
    if not osp.exists(org_seg_dir) or not os.listdir(org_seg_dir):
        print('Apply segmentation on original images...', end=' ')
        init_segmentation(data_dir, model_path=model_path)
        print('done')

    seg_original, original_imgs = {}, {}
    for image_name in os.listdir(data_dir):
        base_image_name = image_name.split('.')[0]
        image = Image.open(osp.join(data_dir, image_name))
        image = image.resize((512, 512), Image.BILINEAR)
        original_imgs[base_image_name] = image
        seg_name = image_name.split('.')[0] + '.npy'
        seg_image = np.load(osp.join(org_seg_dir, seg_name))
        seg_original[base_image_name] = seg_image

    n_images = len(os.listdir(input_path))
    for i, image_dir in enumerate(os.listdir(input_path)):
        seg_org = seg_original[image_dir]
        img_org = original_imgs[image_dir]

        for file_name in os.listdir(osp.join(input_path, image_dir)):
            charac_name = file_name.split('.')[0]
            image = Image.open(osp.join(input_path, image_dir, file_name))
            image = image.resize((512, 512), Image.BILINEAR)

            if charac_name in avoid_transformations_list:  # Not handled
                image = add_top_foreground(image, img_org, seg_org,
                                           margin=configs['foreground_margin'])
                cv2.imwrite(osp.join(output_path, image_dir, file_name), image)
                print(f'image {i+1}/{n_images} done  ', end='\r')
                continue

            charac_name = charac_name.split('_')[0]
            # Get segmentation of the edited image
            image_tsr = to_tensor(image)
            seg = segmentation(image_tsr, net, device)
            # Get binary mask relevant for the current characteristic
            seg_charac = process_segmentation(seg, charac_name)

            if seg_charac is None:  # Not handled
                image = add_top_foreground(image, img_org, seg_org,
                                           margin=configs['foreground_margin'])
                cv2.imwrite(osp.join(output_path, image_dir, file_name), image)
                print(f'image {i+1}/{n_images} done  ', end='\r')
                continue

            # Get binary mask relevant for the current characteristic
            seg_org_charac = process_segmentation(seg_org, charac_name)
            # Merge the images continuously
            img_final = merge_images(img_org, seg_org_charac, image,
                                     seg_charac, margin=configs['margin'])
            # Add Foreground of the original image
            img_final = add_top_foreground(img_final, img_org, seg_org,
                                           margin=configs['foreground_margin'])
            # Save image
            cv2.imwrite(osp.join(output_path, image_dir, file_name), img_final)
        print(f'image {i+1}/{n_images} done  ', end='\r')
    print()


def add_top_foreground(img, img_org, seg_org, margin):
    """Add foreground on the top half of the original image to the image."""
    foreground = np.where(seg_org == 5, 1, 0)
    foreground[foreground.shape[0] // 2:, :] = 0
    dist = dist_edt(foreground)
    alpha = alpha_from_dist(dist, margin=margin)[..., None]
    img = alpha*img_org + (1-alpha) * img
    img = img.astype(np.uint8)
    img = cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def merge_images(img_org, seg_org, img, seg, margin):
    """Merge the images."""
    seg_both = seg_org * seg
    dist = dist_edt(seg_both)
    alpha = alpha_from_dist(dist, margin=margin)[..., None]
    new_img = img_org*alpha + img * (1.0-alpha)
    return new_img


def process_segmentation(seg, charac_name):
    """Return segmentation depending on the current characteristic.

    Parameters
    ----------
    seg: np.ndarray
        Segmentation of the image (labels 0: background, 1: eyes,
        2: nose, 3: mouth, 4: hair, 5: hat, 6: other)
    charac_name: str
        Ident of the characteristic ('Sk', 'A', 'Se', 'bald', ...)

    Returns
    -------
    seg: np.ndarray, optional
        New segmentation with 1 = pixel to keep in the original image
        and 0 = pixel to change with edited image. Return None if the
        characteristic is not handled.

    Labels:
    0: background
    1: eyes
    2: nose
    3: mouth
    4: hair
    5: foreground
    6: other
    """
    if charac_name in {'Sk', 'A', 'Se', 'Ch'}:
        return np.where(seg != 0, 0, 1)
    if charac_name in {'B', 'Hc', 'Hs'}:
        return np.where(seg == 4, 0, 1)
    if charac_name in {'Pn', 'Bn'}:
        return np.where(seg == 2, 0, 1)
    if charac_name == 'N':
        eyes = np.where(seg == 1, 1, 0).astype(np.uint8)
        kernel = np.ones((5, 5), 'uint8')
        eyes = cv2.dilate(eyes, kernel=kernel, iterations=1)
        return np.where(eyes == 1, 0, 1)
    if charac_name == 'Be':
        eyes = np.where(seg == 1, 1, 0).astype(np.uint8)
        kernel = np.ones((5, 5), 'uint8')
        eyes = cv2.dilate(eyes, kernel=kernel, iterations=4)
        # Under eye starts 10 pixels below the top of the eye
        under_eyes = np.where(eyes[:-30, :] == 1, 1, 0)
        rest = np.zeros((30, 512))
        under_eyes = np.concatenate((rest, under_eyes), axis=0)
        return np.where(under_eyes == 1, 0, 1)
    if charac_name == 'Bp':
        return np.where(seg == 3, 0, 1)
    return None  # Not handled


if __name__ == "__main__":
    print('Applying segmentation mixup...')
    # Path to the original images
    DATA_DIR = 'data/face_challenge'
    # Path to the edited images
    INPUT_PATH = 'res/run1/output_images'
    # Path to the segmented edited images
    OUTPUT_PATH = 'res/run1/output_images'
    # Path to the model
    MODEL_PATH = 'postprocess/segmentation/model/79999_iter.pth'

    CONFIGS = {'margin': 21, 'foreground_margin': 6}

    segmentation_mix(data_dir=DATA_DIR, input_path=INPUT_PATH,
                     output_path=OUTPUT_PATH, model_path=MODEL_PATH,
                     configs=CONFIGS)
