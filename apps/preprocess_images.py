"""Preprocess images for face editing."""
import os

import cv2

from anycostgan.tools.align_face import align_face
from pipeline.utils.global_config import GlobalConfig


def align_dir(data_dir, output_dir=None):
    """Align faces in a directory."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    n_images = len(os.listdir(data_dir))
    for i, img_name in enumerate(os.listdir(data_dir)):
        if (img_name.endswith('.jpg') or img_name.endswith('.png')
                or img_name.endswith('.jpeg')):
            img_path = os.path.join(data_dir, img_name)
            if output_dir is None:
                output_path = img_path
            else:
                output_path = os.path.join(output_dir, img_name)
            align_face(img_path).save(output_path)
            print(f'{i+1}/{n_images} done', end='\r')
    print()


def resize_dir(data_dir, output_dir=None, size=(512, 512)):
    """Resize all images in the data directory to the given size."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    n_images = len(os.listdir(data_dir))
    for i, img_name in enumerate(os.listdir(data_dir)):
        if (img_name.endswith('.jpg') or img_name.endswith('.png')
                or img_name.endswith('.jpeg')):
            img_path = os.path.join(data_dir, img_name)
            if output_dir is None:
                output_path = img_path
            else:
                output_path = os.path.join(output_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(output_path, img)
            print(f'{i+1}/{n_images} done', end='\r')
    print()


if __name__ == '__main__':
    # NOTE: images are transformed in place by default
    config = GlobalConfig.build_from_argv(fallback='configs/exp/base.yaml')
    DATA_DIR = config.data_dir
    print('Align images...')
    align_dir(DATA_DIR)
    print('Resize images...')
    resize_dir(DATA_DIR, size=(512, 512))
    print('All images processed!')
