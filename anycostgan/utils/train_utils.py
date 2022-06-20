# Code from https://github.com/mit-han-lab/anycost-gan
"""Utility functions for training the GAN."""

import random

import numpy as np
import torch
import torch.nn.functional as F
from models.dynamic_channel import CHANNEL_CONFIGS, sample_random_sub_channel

__all__ = [
        'requires_grad', 'accumulate', 'get_mixing_z', 'get_g_arch',
        'adaptive_downsample256', 'get_teacher_multi_res', 'get_random_g_arch',
        'partially_load_d_for_multi_res', 'partially_load_d_for_ada_ch'
        ]


def requires_grad(model, flag=True):
    """Change the requires_grad behaviour of all parameters."""
    for param in model.parameters():
        param.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    """Smooth mixup of two models parameters."""
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())

    for k, param1 in params1.keys():
        param1.data.mul_(decay).add_((1-decay) * params2[k].data)


def get_mixing_z(batch_size, latent_dim, prob, device):
    """Get a random batch of noise vectors."""
    if prob > 0 and random.random() < prob:
        return torch.randn(batch_size, 2, latent_dim, device=device)

    return torch.randn(batch_size, 1, latent_dim, device=device)


def get_g_arch(ratios, device='cuda'):
    """Get the architecture of the generator."""
    out = []
    for ratio in ratios:
        one_hot = [0] * len(CHANNEL_CONFIGS)
        one_hot[CHANNEL_CONFIGS.index(ratio)] = 1
        out += one_hot
    return torch.from_numpy(np.array(out)).float().to(device)


def adaptive_downsample256(img):
    """Adaptive downsample to 256x256."""
    img = img.clamp(-1, 1)
    if img.shape[-1] > 256:
        return F.interpolate(
                img, size=(256, 256), mode='bilinear', align_corners=True
                )
    return img


def get_teacher_multi_res(teacher_out, n_res):
    """Get the teacher for multi-resolution."""
    teacher_rgbs = [teacher_out]
    cur_res = teacher_out.shape[-1] // 2
    for _ in range(n_res - 1):
        # for simplicity, we use F.interpolate. Be sure to always use this.
        teacher_rgbs.insert(
                0,
                F.interpolate(
                        teacher_out, size=cur_res, mode='bilinear',
                        align_corners=True
                        )
                )
        cur_res = cur_res // 2
    return teacher_rgbs


def get_random_g_arch(
        generator, min_channel, divided_by, dynamic_channel_mode, seed=None
        ):
    """Get a random architecture for the generator."""
    rand_ratio = sample_random_sub_channel(
            generator, min_channel=min_channel, divided_by=divided_by,
            seed=seed, mode=dynamic_channel_mode, set_channels=False
            )
    return get_g_arch(rand_ratio)


def partially_load_d_for_multi_res(d, sd, n_res=4):
    """Load the iscriminator for multi-resolution."""
    new_sd = {}
    for key, value in sd.items():
        if key.startswith('convs.') and not key.startswith('convs.0.'):
            k_sp = key.split('.')
            k_sp[0] = 'blocks'
            k_sp[1] = str(int(k_sp[1]) - 1)
            new_sd['.'.join(k_sp)] = value
        else:
            new_sd[key] = value
    for i_res in range(1, n_res):  # just retain the weights
        new_sd[f'convs.{i_res}.0.weight'] = d.state_dict(
        )[f'convs.{i_res}.0.weight']
        new_sd[f'convs.{i_res}.1.bias'] = d.state_dict(
        )[f'convs.{i_res}.1.bias']
    d.load_state_dict(new_sd)


def partially_load_d_for_ada_ch(d, sd):
    """Handle the new modulation FC."""
    blocks_with_mapping = []
    for key, value in d.state_dict().items():
        if '_mapping.' in key:
            sd[key] = value
            blocks_with_mapping.append('.'.join(key.split('.')[:2]))
    blocks_with_mapping = list(set(blocks_with_mapping))
    for blk in blocks_with_mapping:
        sd[blk + '.conv1.2.bias'] = sd.pop(blk + '.conv1.1.bias')
        sd[blk + '.conv2.3.bias'] = sd.pop(blk + '.conv2.2.bias')
    d.load_state_dict(sd)
