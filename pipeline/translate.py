import os

import torch
import numpy as np
import cv2

from anycostgan.models import get_pretrained
from pipeline.utils.translation.get_translations import get_translations


def generate_image(generator, **input_kwargs):
    def image_to_np(x):
        assert x.shape[0] == 1
        x = x.squeeze(0).permute(1, 2, 0)
        x = (x + 1) * 0.5  # 0-1
        x = (x * 255).cpu().numpy().astype('uint8')
        return x

    with torch.no_grad():
        out = generator(**input_kwargs)[0].clamp(-1, 1)
        out = image_to_np(out)
        out = np.ascontiguousarray(out)
        return out


def apply_translations(projection_dir, output_path, anycost_config, configs):
    use_caracs_in_img = configs['use_caracs_in_img']
    use_precomputed = configs['use_precomputed']

    latent_dir = os.path.join(projection_dir, 'projected_latents')
    translation_dir = os.path.join(projection_dir, 'translations_vect')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = get_pretrained('generator', anycost_config).to(device)
    input_kwargs = {'styles': None, 'noise': None, 'randomize_noise': False,
                    'input_is_style': True}
    n_images = len(os.listdir(latent_dir))

    for i, fname in enumerate(os.listdir(latent_dir)):
        basename = fname.split('.')[0]
        if use_caracs_in_img:
            try:
                carac_list = list(map(int, basename.split('_')))
            except ValueError:
                raise ValueError(
                    'When "use_caracs_in_img" is set to True, the name of '
                    'images should be like: "%d_%d_%d_%d_%d_%d_%d_%d_%d.png/'
                    '.jpg" where "%d" are integer cararacteristics '
                    '(see https://transfer-learning.org/rules for details).'
                    'Otherwise you can set "use_caracs_in_img" to '
                    'False to get all translation for all images.')
        else:
            carac_list = None
        # Get intial projected latent code
        base_code = np.load(os.path.join(latent_dir,
                                         basename + '.npy'))
        base_code = torch.tensor(base_code).float().to(device)
        # Get translations for the image
        translations = get_translations(translation_dir=translation_dir,
                                        carac_list=carac_list,
                                        use_precomputed=use_precomputed)
        translations = {k: v.to(device) for k, v in translations.items()}

        if not os.path.exists(os.path.join(output_path, basename)):
            os.makedirs(os.path.join(output_path, basename))

        # Apply all translations and save the results
        for ident, translation in translations.items():
            input_kwargs['styles'] = base_code + translation
            with torch.no_grad():
                image = generate_image(generator, **input_kwargs)
            image = cv2.resize(image, (512, 512))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            path = os.path.join(output_path, basename, ident + '.png')
            cv2.imwrite(path, image)
        print(f'image {i + 1}/{n_images} done', end='\r')
    print()


if __name__ == '__main__':
    projection_dir = 'projection/run1'
    output_path = 'res/run1/images_post_translation'
    flexible_config = False

    anycost_config = 'anycost-ffhq-config-f-flexible' if flexible_config \
        else 'anycost-ffhq-config-f'
    print('Applying translations in latent space...')

    configs = {'use_caracs_in_img': True, 'use_precomputed': True}

    apply_translations(projection_dir, output_path, anycost_config, configs)
