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


def apply_translations(latent_path, output_path, flexible_config):
    config = 'anycost-ffhq-config-f-flexible' \
        if flexible_config else 'anycost-ffhq-config-f'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = get_pretrained('generator', config).to(device)
    input_kwargs = {'styles': None, 'noise': None, 'randomize_noise': False,
                    'input_is_style': True}
    n_images = len(os.listdir(latent_path))
    for i, fname in enumerate(os.listdir(latent_path)):
        basename = fname.split('.')[0]
        caract = list(map(int, basename.split('_')))
        # Get intial projected latent code
        base_code = np.load(os.path.join(latent_path,
                                         basename + '.npy'))
        base_code = torch.tensor(base_code).float().to(device)
        # Get translations for the image
        translations = get_translations(caract, flexible_config=flexible_config)
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
    latent_path = 'data/projected_latents'
    output_path = 'res/images_post_translation'
    flexible_config = False

    print('Translate latent spaces')
    apply_translations(latent_path, output_path, flexible_config)
