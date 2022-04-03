import os
import os.path as osp

import cv2
from cv2 import cvtColor
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt as dist_edt
import torch
import torchvision.transforms as transforms

from preprocess.segmentation.utils.model import BiSeNet


def alpha_from_dist(dist, margin=21):
        return np.where(dist < margin, dist / margin, 1.0)


def segmentation_mix(edited_path, original_path, output_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)

    net.to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()


    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        # Original images
        seg_original, original_imgs = {}, {}
        for image_name in os.listdir(original_path):
            base_image_name = image_name.split('.')[0]
            if not osp.exists(osp.join(output_path, base_image_name)):
                os.makedirs(osp.join(output_path, base_image_name))
            image = Image.open(osp.join(original_path, image_name))
            image = image.resize((512, 512), Image.BILINEAR)
            original_imgs[base_image_name] = image
            image_tsr = to_tensor(image)
            seg = segmentation(image_tsr, net, device)
            seg_original[base_image_name] = seg
        print('Original images segmentation done')
        n_images = len(os.listdir(edited_path))
        for i, image_dir in enumerate(os.listdir(edited_path)):
            seg_org = seg_original[image_dir]
            img_org = original_imgs[image_dir]

            for file_name in os.listdir(osp.join(edited_path, image_dir)):
                carac_name = file_name.split('.')[0]
                image = Image.open(osp.join(edited_path, image_dir, file_name))
                image = image.resize((512, 512), Image.BILINEAR)
                if carac_name == 'N_max':  # Not handled
                    image = add_foreground(image, img_org, seg_org)
                    cv2.imwrite(osp.join(output_path, image_dir, file_name),
                                image)
                    print(f'image {i+1}/{n_images} done  ', end='\r')
                    continue

                carac_name = carac_name.split('_')[0]
                # Get segmentation of the edited image
                image_tsr = to_tensor(image)
                seg = segmentation(image_tsr, net, device)
                # Get binary mask relevant for the current caracteristic
                seg_carac = process_segmentation(seg, carac_name)

                if seg_carac is None:  # Not handled
                    image = add_foreground(image, img_org, seg_org)
                    cv2.imwrite(osp.join(output_path, image_dir, file_name),
                                image)
                    print(f'image {i+1}/{n_images} done  ', end='\r')
                    continue

                # Get binary mask relevant for the current caracteristic
                seg_org_carac = process_segmentation(seg_org, carac_name)
                # Merge the images continuously
                img_final = merge_by_segmentation(img_org, seg_org_carac,
                                                  image, seg_carac)
                # Add Foreground of the original image
                img_final = add_foreground(img_final, img_org, seg_org)
                # Save image
                cv2.imwrite(osp.join(output_path, image_dir, file_name),
                            img_final)
                print(f'image {i+1}/{n_images} done  ', end='\r')
    print()


def add_foreground(img, img_org, seg_org):
    foreground = np.where(seg_org == 5, 1, 0)
    dist = dist_edt(foreground)
    alpha = alpha_from_dist(dist, margin=6)[..., None]
    img = alpha * img_org + (1 - alpha) * img
    img = img.astype(np.uint8)
    img = cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def merge_by_segmentation(img_org, seg_org, img, seg):
    seg_both = seg_org * seg
    dist = dist_edt(seg_both)
    alpha = alpha_from_dist(dist)[..., None]
    new_img = img_org * alpha + img * (1.0 - alpha)
    return new_img


def process_segmentation(seg, carac_name):
    """ Return a relevant segmentation depending on the current caracteristic.

    Prameters:
    ----------
    seg: np.ndarray
        Segmentation of the image (labels 0: background, 1: eyes,
        2: nose, 3: mouth, 4: hair, 5: hat, 6: other)
    carac_name: str
        Ident of the caracteristic ('Sk', 'A', 'Se', 'bald', ...)

    Returns:
    --------
    seg: np.ndarray or None
        New segmentation with 1 = pixel to keep in the original image
        and 0 = pixel to change with edited image. Return None if the
        caracteristic is not handled.
    """


    """
    Labels:
    0: background
    1: eyes
    2: nose
    3: mouth
    4: hair
    5: foreground
    6: other
    """
    if carac_name in {'Sk', 'A', 'Se', 'Ch'}:
        return np.where(seg != 0, 0, 1)
    elif carac_name in {'B', 'Hc', 'Hs'}:
        return np.where(seg == 4, 0, 1)
    elif carac_name in {'Pn', 'Bn'}:
        return np.where(seg == 2, 0, 1)
    elif carac_name == 'N':
        eyes = np.where(seg == 1, 1, 0).astype(np.uint8)
        kernel = np.ones((5, 5), 'uint8')
        eyes = cv2.dilate(eyes, kernel=kernel, iterations=1)
        return np.where(eyes == 1, 0, 1)
    elif carac_name == 'Be':
        eyes = np.where(seg == 1, 1, 0).astype(np.uint8)
        kernel = np.ones((5, 5), 'uint8')
        eyes = cv2.dilate(eyes, kernel=kernel, iterations=4)
        # under eye starts 10 pixels below the top of the eye
        under_eyes = np.where(eyes[:-30, :] == 1, 1, 0)
        rest = np.zeros((30, 512))
        under_eyes = np.concatenate((rest, under_eyes), axis=0)
        return np.where(under_eyes == 1, 0, 1)
    elif carac_name == 'Bp':
        return np.where(seg == 3, 0, 1)
    else:
        return None


def segmentation(images, net, device):
    """
    Classes Legend:
    ##### From Model #######
    ----- Background ------
    0: background

    --------- Eye ---------
    4: in the eye 1 (no if glasses)
    5: in the eye 2 (no if glasses)
    6: glasses

    -------- Nose ---------
    10: nose

    -------- Mouth --------
    11: inside of mouth
    12: upper lip
    13: lower lip

    -------- Hair ---------
    17: hair

    -------- Foreground ----
    16: clothes
    18: hat

    -------- Other --------
    1: back of face
    2: eyebrow 1
    3: eyebrow 2


    7: ear 1
    8: ear 2
    9: earrings

    14: neck
    15: necklace

    ###### NEW #######
    0: background
    1: eyes
    2: nose
    3: mouth
    4: hair
    5: foreground
    6: other
    """
    if images.ndim == 3:
        images = torch.unsqueeze(images, 0)
    images = images.to(device)
    pred = net(images)[0]
    pred = pred.squeeze(0).cpu().numpy().argmax(0)
    # Set class label to negative value temporarily to avoid conflict
    # between old and new classes (background remains 0)
    lor = np.logical_or
    pred = np.where(lor(lor(pred == 4, pred == 5), pred == 6), -1, pred)
    pred = np.where(pred == 10, -2, pred)
    pred = np.where(lor(lor(pred == 11, pred == 12), pred == 13), -3, pred)
    pred = np.where(pred == 17, -4, pred)
    pred = np.where(lor(pred == 16, pred == 18), -5, pred)
    pred = np.where(pred > 0, -6, pred)  # set remaining classes to 'other'
    pred = - pred
    return pred


if __name__ == "__main__":
    print('Segmentation processing')
    # Path to the original images
    original_path ='data/input_images'
    # Path to the edited images
    edited_path = 'preprocess/mixup/edited_images_postmixup'
    # Path to the segmented edited images
    output_path = 'preprocess/segmentation/edited_images_postsegmentation'
    # Path to the model
    model_path = 'preprocess/segmentation/cp/79999_iter.pth'

    segmentation_mix(edited_path=edited_path, original_path=original_path,
                     output_path=output_path, model_path=model_path)
