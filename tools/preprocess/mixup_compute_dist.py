import os

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt as dist_edt

if __name__ == '__main__':
    # domains img: balck and white image (domain is black pixels)
    domains_img_path = './preprocess/mixup/domains/images'
    domains_dist_path = './preprocess/mixup/domains/distances'

    for domain_name in os.listdir(domains_img_path):
        if domain_name.endswith('.png'):
            domain_img = cv2.imread(os.path.join(domains_img_path, domain_name))
            # domain: coords of pixel corresponding to black zones
            domain_y, domain_x = np.where(domain_img[..., 0] <= 100)
            domain_y = np.expand_dims(domain_y, axis=-1)
            domain_x = np.expand_dims(domain_x, axis=-1)
            domain = np.concatenate([domain_x, domain_y], axis=-1)
            domain = np.squeeze(domain)
            domain_basename = os.path.splitext(domain_name)[0]
            dist = dist_edt(domain)
            np.save(os.path.join(domains_dist_path, domain_basename + '_dist.npy'), dist)
        print(f'distance to {domain_basename} computed')
