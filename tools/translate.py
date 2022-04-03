import os

import torch
import numpy as np
import cv2

from models import get_pretrained
from utils.translation import get_translations

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


if __name__ == '__main__':
    print('Translate latent spaces')
    flexible_config = False

    latent_dir = 'anycost-flex' if flexible_config else 'anycost'
    config = 'anycost-ffhq-config-f-flexible' if flexible_config else 'anycost-ffhq-config-f'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = get_pretrained('generator', config).to(device)
    input_kwargs = {'styles': None, 'noise': None, 'randomize_noise': False,
                    'input_is_style': True}

    n_images = len(os.listdir(f'data/{latent_dir}/projected_latents'))
    for i, fname in enumerate(os.listdir(f'data/{latent_dir}/projected_latents')):
        basename = fname.split('.')[0]
        caract = list(map(int, basename.split('_')))
        # Get intial projected latent code
        base_code = np.load(f'data/{latent_dir}/projected_latents/{basename}.npy')
        base_code = torch.tensor(base_code).float().to(device)
        # Get translations for the image
        translations = get_translations(caract, flexible_config=flexible_config)
        translations = {k: v.to(device) for k, v in translations.items()}

        if not os.path.exists(f'data/{latent_dir}/edited_images/{basename}'):
            os.makedirs(f'data/{latent_dir}/edited_images/{basename}')

        # Apply all translations and save the results
        for ident, translation in translations.items():
            input_kwargs['styles'] = base_code + translation
            with torch.no_grad():
                image = generate_image(generator, **input_kwargs)
            image = cv2.resize(image, (512, 512))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            path = f'data/{latent_dir}/edited_images/{basename}/{ident}.png'
            cv2.imwrite(path, image)
        print(f'image {i + 1}/{n_images} done', end='\r')
    print()
