"""GFP GAN."""

import os
import warnings

import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from PIL import Image
from scipy.ndimage import distance_transform_edt as dist_edt
from torchvision import transforms

from pipeline.utils.segmentation.tools import get_model, segmentation

if not torch.cuda.is_available():  # CPU
    GPU_ENABLED = False
else:
    from realesrgan import RealESRGANer
    GPU_ENABLED = True


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


def gfp_gan_mix(input_path, output_path, model_path, configs):
    """Mix original image and restored image from GFP GAN."""
    margin = configs['margin']
    transformations_list = configs['transformations']

    # NOTE: Real-ESRGAN is useless as we mask the background
    realesrgan = False

    os.makedirs(output_path, exist_ok=True)
    n_images = len(os.listdir(input_path))

    net, device = get_model()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Set up background up-sampler
    if realesrgan:
        if not GPU_ENABLED:
            bg_upsampler = None
            warnings.warn(
                'The unoptimized RealESRGAN is slow on CPU. We do not '
                'use it. If you really want to use it, please modify the '
                'corresponding codes.'
                )
        else:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2, model_path='https://github.com/xinntao/Real-'
                'ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model, tile=configs['bg_tile'], tile_pad=10, pre_pad=0,
                half=True)
    else:
        bg_upsampler = None

    if not os.path.isfile(model_path):
        raise ValueError(f'GFP GAN model not found at {model_path}.')

    restorer = GFPGANer(model_path=model_path, upscale=configs["upscale"],
                        arch='clean', channel_multiplier=2,
                        bg_upsampler=bg_upsampler)

    for i, image_dir in enumerate(os.listdir(input_path)):
        for img_name in os.listdir(os.path.join(input_path, image_dir)):
            transfo_name = img_name.split('.')[0]

            if (transfo_name in transformations_list
                    or transformations_list == ['all']):  # Apply GFP GAN
                # Read image
                img_path = os.path.join(input_path, image_dir, img_name)
                image = Image.open(img_path)
                image = image.resize((512, 512), Image.BILINEAR)
                image_tsr = to_tensor(image)
                seg = segmentation(image_tsr, net, device)
                # mask: background or foreground (no face)
                # bg_mask : background
                mask = np.logical_or(seg == 0, seg == 5)
                bg_mask = (seg == 0)

                # Restore face
                input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                input_img = cv2.resize(input_img, (512, 512),
                                       interpolation=cv2.INTER_LINEAR)
                _, restored_images, _ = restorer.enhance(
                    input_img, has_aligned=True, only_center_face=False,
                    paste_back=True
                    )
                assert len(restored_images) == 1, (
                    'Only one face is expected as long as has_aligned=True.'
                    )
                restored_img = restored_images[0]
                dist = dist_edt(mask)
                alpha = alpha_from_dist(dist, margin=margin)
                # Take all the background from the restored image
                alpha = np.maximum(alpha, bg_mask)
                alpha = alpha[..., None]
                # Mix
                output_img = input_img*alpha + restored_img * (1.0-alpha)
                save_path = os.path.join(output_path, image_dir, img_name)
                cv2.imwrite(save_path, output_img)

            else:  # Do nothing on the image
                img_path = os.path.join(input_path, image_dir, img_name)
                save_path = os.path.join(output_path, image_dir, img_name)
                Image.open(img_path).save(save_path)

        print(f'image {i+1}/{n_images} done  ', end='\r')
    print()


if __name__ == '__main__':
    print('Applying GFP GAN restoration...')
    DATA_DIR = 'data/face_challenge'
    INPUT_PATH = 'res/run1/images_post_translation'
    OUTPUT_PATH = 'res/run1/output_images'
    MODEL_PATH = 'postprocess/gfp_gan/model/GFPGANv1.3.pth'
    CONFIGS = {'upscale': 2, 'bg_tile': 400, 'margin': 12,
               'transformations': ['all']}
    gfp_gan_mix(input_path=INPUT_PATH, output_path=OUTPUT_PATH,
                model_path=MODEL_PATH, configs=CONFIGS)
