"""Compute distances to domains."""

import os

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt as dist_edt


def compute_dist(domains_img_path, domains_dist_path):
    """Compute distances to domains."""
    if not os.path.exists(domains_img_path):
        raise ValueError('Path to the directory with images of domains '
                         f'does not exist: {domains_img_path}')
    if not os.path.exists(domains_dist_path):
        os.makedirs(domains_dist_path)
    for domain_name in os.listdir(domains_img_path):
        if domain_name.endswith('.png') or domain_name.endswith('.jpg'):
            domain_img = cv2.imread(os.path.join(domains_img_path,
                                                 domain_name))
            # domain: coords of pixel corresponding to black zones
            domain_y, domain_x = np.where(domain_img[..., 0] <= 100)
            domain_y = np.expand_dims(domain_y, axis=-1)
            domain_x = np.expand_dims(domain_x, axis=-1)
            domain = np.concatenate([domain_x, domain_y], axis=-1)
            domain = np.squeeze(domain)
            domain_basename = os.path.splitext(domain_name)[0]
            dist = dist_edt(domain)
            np.save(os.path.join(domains_dist_path,
                                 domain_basename + '_dist.npy'), dist)
        print(f'distance to {domain_basename} computed')


if __name__ == '__main__':
    # domains img: black and white image (domain is black pixels)
    IMG_PATH = 'postprocess/domain_mixup/images'
    DIST_PATH = 'postprocess/domain_mixup/distances'
    compute_dist(IMG_PATH, DIST_PATH)
