import os

import cv2
import numpy as np
from scipy.spatial.distance import cdist

if __name__ == '__main__':
    # domains img: balck and white image (domain is black pixels)
    domains_img_path = './preprocess/mixup/domains/images'
    domains_dist_path = './preprocess/mixup/domains/distances'
    # Create matrix of coords
    y, x = np.meshgrid(np.arange(512), np.arange(512))
    y, x = np.expand_dims(y, axis=-1), np.expand_dims(x, axis=-1)
    coords = np.concatenate([x, y], axis=-1)  # coords[i, j] = [i, j]

    for domain_name in os.listdir(domains_img_path):
        if domain_name.endswith('.png'):
            domain_img = cv2.imread(os.path.join(domains_img_path, domain_name))
            # domain: coords of pixel corresponding to black zones
            domain_y, domain_x = np.where(domain_img[..., 0] <= 100)
            domain_y = np.expand_dims(domain_y, axis=-1)
            domain_x = np.expand_dims(domain_x, axis=-1)
            domain = np.concatenate([domain_x, domain_y], axis=-1)
            domain_basename = os.path.splitext(domain_name)[0]

            # Compute distance between each pixel and the doamin
            # NOTE; lines are processed per batch to avoid memory error
            batch_len = 4
            dist = np.zeros((512, 512))
            for i in range(0, len(coords), batch_len):
                coords_i = coords[i: i + batch_len].reshape(-1, 2)
                dist_i = cdist(coords_i, domain)
                dist_i = np.min(dist_i, axis=-1)
                dist[i: i + batch_len] = dist_i.reshape((batch_len, -1))
            dist = dist.T
            np.save(os.path.join(domains_dist_path, domain_basename + '_dist.npy'), dist)
        print(f'distance to {domain_basename} computed')
