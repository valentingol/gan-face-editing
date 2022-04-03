import os

import cv2
import numpy as np
from scipy.spatial.distance import cdist

def alpha_from_dist(dist, margin=42):
    return np.where(dist < margin, dist / margin, 1.0)


if __name__ == '__main__':
    print('Mixup processing')
    input_images_path = './data/input_images'
    edited_images_path = './data/anycost/edited_images'
    new_edited_images_path = './preprocess/mixup/edited_images_postmixup'
    # distances to domains (computed with `compute_dist.py`)
    domains_dist_path = './preprocess/mixup/domains/distances'

    dists = {}
    for domain_name in os.listdir(domains_dist_path):
        if domain_name.endswith('.npy'):
            dist = np.load(os.path.join(domains_dist_path, domain_name))
            carac_name = domain_name.split('.')[0]
            carac_name = carac_name.split('_')[0]
            dists[carac_name] = dist
    n_images = len(os.listdir(edited_images_path))
    for idx, img_name in enumerate(os.listdir(edited_images_path)):
        original_image = cv2.imread(os.path.join(input_images_path,
                                                 img_name + '.png'))
        for edited_name in os.listdir(os.path.join(edited_images_path, img_name)):
            carac_name = edited_name.split('.')[0]
            carac_name = carac_name.split('_')[0]
            edited_img = cv2.imread(os.path.join(edited_images_path, img_name,
                                                    edited_name))
            if carac_name in dists:
                # Distance between each pixel and the doamin
                dist = dists[carac_name]

                # Mixup
                alpha = alpha_from_dist(dist)
                np.save('alpha.npy', alpha)
                alpha = np.expand_dims(alpha, axis=-1)
                new_edited_img = alpha * original_image + (1 - alpha) * edited_img

                # Save new edited image
                outdir = os.path.join(new_edited_images_path, img_name)
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                cv2.imwrite(os.path.join(outdir, edited_name), new_edited_img)

            else: # no modification
                cv2.imwrite(os.path.join(outdir, edited_name), edited_img)

        print(f'image {idx + 1}/{n_images} done', end='\r')
    print()