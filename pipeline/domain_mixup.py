"""Apply domain mixup."""

import os

import cv2
import numpy as np

from pipeline.utils.domain_mixup.dist import compute_dist


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


def domain_mix(
        data_dir, input_path, output_path, domains_dist_path, domains_img_path,
        configs
        ):
    """Apply domain mixup.

    Parameters
    ----------
    data_dir : str
        Path to the directory with original images
    input_path : str
        Path to the directory with input (edited) images
    output_path : str
        Path to the output directory
    domains_dist_path : str
        Path to the distance matrix between domains (if exists).
    domains_img_path : str
        Path to the directory with images of domains
    configs : dict or GlobalConfig
        Configurations for the domain mixup.
    """
    if not os.path.exists(domains_dist_path) \
            or len(os.listdir(domains_dist_path)) == 0 \
            or configs['recompute_dist']:
        # Compute distance to domains
        print('Computing distance to domains...')
        compute_dist(domains_img_path, domains_dist_path)

    dists = {}
    for domain_name in os.listdir(domains_dist_path):
        if domain_name.endswith('.npy'):
            dist = np.load(os.path.join(domains_dist_path, domain_name))
            carac_name = domain_name.split('.')[0]
            carac_name = carac_name.split('_')[0]
            dists[carac_name] = dist
    n_images = len(os.listdir(input_path))
    for idx, img_name in enumerate(os.listdir(input_path)):
        original_image = cv2.imread(os.path.join(data_dir, img_name + '.png'))
        for edited_name in os.listdir(os.path.join(input_path, img_name)):
            carac_name = edited_name.split('.')[0]
            carac_name = carac_name.split('_')[0]
            edited_img = cv2.imread(
                    os.path.join(input_path, img_name, edited_name)
                    )
            if carac_name in dists:
                # Distance between each pixel and the doamin
                dist = dists[carac_name]

                # Mixup
                alpha = alpha_from_dist(dist, margin=configs['margin'])
                alpha = np.expand_dims(alpha, axis=-1)
                new_edited_img = alpha * original_image \
                    + (1 - alpha) * edited_img

                # Save new edited image
                outdir = os.path.join(output_path, img_name)
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                cv2.imwrite(os.path.join(outdir, edited_name), new_edited_img)

            else:  # no modification
                cv2.imwrite(os.path.join(outdir, edited_name), edited_img)

        print(f'image {idx + 1}/{n_images} done', end='\r')
    print()


if __name__ == '__main__':
    print('Applying domain mixup...')
    DATA_DIR = 'data/face_challenge'
    INPUT_PATH = 'res/run1/images_post_translation'
    OUTPUT_PATH = 'res/run1/images_post_domain_mixup'
    # Distances to domains (computed with `utils/domain_mixup/dist.py`)
    DOMAINS_DIST_PATH = 'postprocess/domain_mixup/distances'
    # Images representing the domains (black and white)
    DOMAINS_IMG_PATH = 'postprocess/domain_mixup/images'

    CONFIGS = {'margin': 42, 'recompute_dist': False}

    domain_mix(
            DATA_DIR, INPUT_PATH, OUTPUT_PATH, DOMAINS_DIST_PATH,
            DOMAINS_IMG_PATH, CONFIGS
            )
