"""Apply face parsing segmentation."""

import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from pipeline.utils.segmentation.model import BiSeNet


def get_model():
    """Get segmentation model (and device)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = BiSeNet(n_classes=19)

    net.to(device)
    net.load_state_dict(
        torch.load('postprocess/segmentation/model/'
                   '79999_iter.pth'))
    net.eval()
    return net, device


def init_segmentation(data_dir):
    """Apply segmentation (face parsing) on all input image."""
    if data_dir[-1] == '/':
        data_dir = data_dir[:-1]
    output_path = data_dir + '_segmented_'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        if os.listdir(output_path):
            print(f'WARNING: A non-empty folder at {output_path} that may '
                  'already contain face parsing was found. Images inside '
                  'will be overwritten.')

    net, device = get_model()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        for image_name in os.listdir(data_dir):
            basename = image_name.split('.')[0]
            image = Image.open(osp.join(data_dir, image_name))
            image = image.resize((512, 512), Image.BILINEAR)
            image_tsr = to_tensor(image)
            seg = segmentation(image_tsr, net, device)
            seg = seg.astype(np.uint8)
            np.save(osp.join(output_path, basename + '.npy'), seg)


def segmentation(images, net, device):
    """Apply segmentation on the images with interesting labels.

    Classes Legend:
    ##### Original (from model) #######
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
    with torch.no_grad():
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
        pred = -pred
    return pred


if __name__ == '__main__':
    print('Segment images (face parsing)...')
    DATA_DIR = 'data/face_challenge'
    MODEL_PATH = 'postprocess/segmentation/model/79999_iter.pth'
    init_segmentation(data_dir=DATA_DIR)
