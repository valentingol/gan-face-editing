# Code from https://github.com/mit-han-lab/anycost-gan
"""Get models."""

import torch
from torchvision import models

from anycostgan.models.anycost_gan import Generator
from anycostgan.models.encoder import ResNet50Encoder
from anycostgan.thirdparty.inception import InceptionV3
from anycostgan.utils.torch_utils import safe_load_state_dict_from_url

URL_TEMPLATE = 'https://hanlab.mit.edu/projects/anycost-gan/files/{}_{}.pt'


def load_state_dict_from_url(url, key=None):
    """Load state dict from url."""
    if url.startswith('http'):
        state_dict = safe_load_state_dict_from_url(url, map_location='cpu',
                                                   progress=True)
    else:
        state_dict = torch.load(url, map_location='cpu')
    if key is not None:
        return state_dict[key]
    return state_dict


def get_url(model, config):
    """Get url of model state."""
    if model in ['attribute-predictor', 'inception']:
        assert config is None
        # Not used for inception
        url = URL_TEMPLATE.format('attribute', 'predictor')
    else:
        assert config is not None
        url = URL_TEMPLATE.format(model, config)
    return url


def get_pretrained(model, config=None):
    """Get pre trained model."""
    url = get_url(model, config)

    if model == 'generator':
        if config in [
                'anycost-ffhq-config-f', 'anycost-ffhq-config-f-flexible',
                'stylegan2-ffhq-config-f'
        ]:
            resolution = 1024
            channel_multiplier = 2
        elif config == 'anycost-car-config-f':
            resolution = 512
            channel_multiplier = 2
        else:
            raise NotImplementedError(f'Unknown config: {config}')
        model = Generator(resolution, channel_multiplier=channel_multiplier)
        model.load_state_dict(load_state_dict_from_url(url, 'g_ema'))
        return model

    if model == 'encoder':
        # NOTICE: the encoders are trained with VGG LPIPS loss
        # to keep consistent with optimization-based projection
        # the numbers in the papers are reported with encoders
        # trained with AlexNet LPIPS loss
        if config in [
                'anycost-ffhq-config-f', 'anycost-ffhq-config-f-flexible',
                'stylegan2-ffhq-config-f'
        ]:
            n_style = 18
            style_dim = 512
        else:
            raise NotImplementedError(f'Unknown config: {config}')

        model = ResNet50Encoder(n_style=n_style, style_dim=style_dim)
        model.load_state_dict(load_state_dict_from_url(url, 'state_dict'))
        return model

    if model == 'attribute-predictor':  # attribute predictor is general
        predictor = models.resnet50()
        predictor.fc = torch.nn.Linear(predictor.fc.in_features, 40 * 2)
        predictor.load_state_dict(load_state_dict_from_url(url, 'state_dict'))
        return predictor

    if model == 'inception':  # inception models
        return InceptionV3([3], normalize_input=False, resize_input=True)

    if model == 'boundary':
        if config in [
                'anycost-ffhq-config-f', 'anycost-ffhq-config-f-flexible',
                'stylegan2-ffhq-config-f'
        ]:
            return load_state_dict_from_url(url)
        raise NotImplementedError(f'Unknown config: {config}')

    raise NotImplementedError(f'Unknown model: {model}')
