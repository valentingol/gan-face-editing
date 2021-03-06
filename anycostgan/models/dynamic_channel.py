# Code from https://github.com/mit-han-lab/anycost-gan
"""Dynamic channel utility functions."""

import math
import random

import torch

from anycostgan.models.anycost_gan import G_CHANNEL_CONFIG
from anycostgan.models.ops import (ConstantInput, ModulatedConv2d, StyledConv,
                                   ToRGB)

CHANNEL_CONFIGS = [0.25, 0.5, 0.75, 1.0]


def get_full_channel_configs(model):
    """Get the full channel configuration of the model."""
    full_channels = []
    for m in model.modules():
        if isinstance(m, ConstantInput):
            full_channels.append(m.input.shape[1])
        elif isinstance(m, ModulatedConv2d):
            if m.weight.shape[1] == 3 and m.weight.shape[-1] == 1:
                continue
            full_channels.append(m.weight.shape[1])  # get the output channels
    return full_channels


def set_sub_channel_config(model, sub_channels):
    """Set the sub channel configuration of the model."""
    ptr = 0
    for m in model.modules():
        if isinstance(m, ConstantInput):
            m.first_k_oup = sub_channels[ptr]
            ptr += 1
        elif isinstance(m, ModulatedConv2d):
            if m.weight.shape[1] == 3 and m.weight.shape[-1] == 1:
                continue
            m.first_k_oup = sub_channels[ptr]
            ptr += 1
    assert ptr == len(sub_channels), (ptr, len(sub_channels))  # all used


def set_uniform_channel_ratio(model, ratio):
    """Set the channel ratio of the model."""
    full_channels = get_full_channel_configs(model)
    resolution = model.resolution
    org_channel_mult = full_channels[-1] * 1. / G_CHANNEL_CONFIG[resolution]

    channel_max = model.channel_max
    channels = {
        k: min(channel_max, int(v * ratio * org_channel_mult))
        for k, v in G_CHANNEL_CONFIG.items()
    }
    channel_config = [v for k, v in channels.items() if k <= resolution]
    channel_config2 = []  # duplicate the config
    for channel in channel_config:
        channel_config2.append(channel)
        channel_config2.append(channel)
    channel_config = channel_config2

    set_sub_channel_config(model, channel_config)


def remove_sub_channel_config(model):
    """Remove the sub channel configuration of the model."""
    for m in model.modules():
        if hasattr(m, 'first_k_oup'):
            del m.first_k_oup


def reset_generator(model):
    """Reset the generator."""
    remove_sub_channel_config(model)
    if hasattr(model, 'target_res'):
        del model.target_res


def get_current_channel_config(model):
    """Get the current channel configuration of the model."""
    channels = []
    for m in model.modules():
        if hasattr(m, 'first_k_oup'):
            channels.append(m.first_k_oup)
    return channels


def _get_offical_sub_channel_config(ratio, org_channel_mult):
    """Get the sub channel configuration of the model."""
    channel_max = 512
    # NOTE: in Python 3.6 onwards,
    # the order of dictionary insertion is preserved
    channel_config = [
        min(channel_max, int(v * ratio * org_channel_mult))
        for _, v in G_CHANNEL_CONFIG.items()
    ]
    channel_config2 = []  # duplicate the config
    for channel in channel_config:
        channel_config2.append(channel)
        channel_config2.append(channel)
    return channel_config2


def get_random_channel_config(full_channels, org_channel_mult, min_channel=8,
                              divided_by=1):
    """Get the random channel configuration of the model."""
    # Use the official config as the smallest number here
    # (so that we can better compare the computation)
    bottom_line = _get_offical_sub_channel_config(CHANNEL_CONFIGS[0],
                                                  org_channel_mult)
    bottom_line = bottom_line[:len(full_channels)]

    new_channels = []
    ratios = []
    for full_c, bottom in zip(full_channels, bottom_line):
        valid_channel_configs = [
            a for a in CHANNEL_CONFIGS if a * full_c >= bottom
        ]
        # (if too small, discard the ratio)
        ratio = random.choice(valid_channel_configs)
        ratios.append(ratio)
        channel = int(ratio * full_c)
        channel = min(max(channel, min_channel), full_c)
        channel = math.ceil(channel * 1. / divided_by) * divided_by
        new_channels.append(channel)
    return new_channels, ratios


def sample_random_sub_channel(model, min_channel=8, divided_by=1, seed=None,
                              mode='uniform', set_channels=True):
    """Sample the random sub channel configuration of the model."""
    if seed is not None:  # whether to sync between workers
        random.seed(seed)

    if mode == 'uniform':
        # Case 1: sample a uniform channel config
        rand_ratio = random.choice(CHANNEL_CONFIGS)
        if set_channels:
            set_uniform_channel_ratio(model, rand_ratio)
        return [rand_ratio] * len(get_full_channel_configs(model))
    if mode == 'flexible':
        # Case 2: sample flexible per-channel ratio
        full_channels = get_full_channel_configs(model)
        org_channel_mult = full_channels[-1] \
            / G_CHANNEL_CONFIG[model.resolution]
        rand_channels, rand_ratios = get_random_channel_config(
            full_channels, org_channel_mult, min_channel, divided_by)
        if set_channels:
            set_sub_channel_config(model, rand_channels)
        return rand_ratios
    if mode == 'sandwich':
        # case 3: sandwich sampling for flexible ratio setting
        rrr = random.random()
        if rrr < 0.25:  # largest
            if set_channels:
                # Use the largest channel
                remove_sub_channel_config(model)
            return [CHANNEL_CONFIGS[-1]] * len(get_full_channel_configs(model))
        if rrr < 0.5:  # smallest
            if set_channels:
                set_uniform_channel_ratio(model, CHANNEL_CONFIGS[0])
            return [CHANNEL_CONFIGS[0]] * len(get_full_channel_configs(model))
        full_channels = get_full_channel_configs(model)
        org_channel_mult = full_channels[-1] \
            / G_CHANNEL_CONFIG[model.resolution]
        rand_channels, rand_ratios = get_random_channel_config(
            full_channels, org_channel_mult, min_channel, divided_by)
        if set_channels:
            set_sub_channel_config(model, rand_channels)
        return rand_ratios
    raise NotImplementedError(f"Unknown mode: {mode}, expected one of "
                              "'uniform', 'flexible', 'sandwich'.")


def sort_channel(g):
    """Sort the channel configuration of the model."""

    def _get_sorted_input_idx(style_conv, sample_latents):
        assert isinstance(style_conv, (StyledConv, ToRGB)), type(style_conv)
        importance = torch.sum(torch.abs(style_conv.conv.weight.data),
                               dim=(0, 1, 3, 4))
        # We consider the modulated weights
        style = style_conv.conv.modulation(sample_latents).abs().mean(0)

        assert style.shape == importance.shape
        importance = importance * style
        return torch.sort(importance, dim=0, descending=True)[1]

    def _reorg_input_channel(style_conv, idx):
        """Reorganize the input channel of the style conv."""
        assert idx.numel() == style_conv.conv.weight.data.shape[2]
        style_conv.conv.weight.data = torch.index_select(
            style_conv.conv.weight.data, 2, idx)  # inp
        style_conv.conv.modulation.weight.data = torch.index_select(
            style_conv.conv.modulation.weight.data, 0, idx)
        style_conv_bias_idx = style_conv.conv.modulation.bias.data[idx]
        style_conv.conv.modulation.bias.data = style_conv_bias_idx

    def _reorg_output_channel(style_conv, idx):
        """Reorganize the output channel of the style conv."""
        assert idx.numel() == style_conv.conv.weight.data.shape[1]
        style_conv.conv.weight.data = torch.index_select(
            style_conv.conv.weight.data, 1, idx)  # oup
        style_conv.activate.bias.data = style_conv.activate.bias.data[idx]

    # NOTE:
    # 1. MLP does not need to be changed
    # 2. noise has only 1 channel, no need to change
    sorted_idx = None
    latent_in = torch.randn(100000, 512, device=next(g.parameters()).device)
    latents = g.style(latent_in)  # get the input latents

    for conv1, conv2, to_rgb in zip(g.convs[::2][::-1], g.convs[1::2][::-1],
                                    g.to_rgbs[::-1]):
        # Modulate conv weight shape: [1, oup, inp, h, w]
        # Modulation linear shape: [style_dim, inp]
        if sorted_idx is None:
            sorted_idx = _get_sorted_input_idx(to_rgb, latents)
        # to_rgb
        _reorg_input_channel(to_rgb, sorted_idx)
        # conv2
        _reorg_output_channel(conv2, sorted_idx)
        sorted_idx = _get_sorted_input_idx(conv2, latents)
        _reorg_input_channel(conv2, sorted_idx)
        # conv1
        _reorg_output_channel(conv1, sorted_idx)
        sorted_idx = _get_sorted_input_idx(conv1, latents)
        _reorg_input_channel(conv1, sorted_idx)

    # sort to_rgb1
    _reorg_input_channel(g.to_rgb1, sorted_idx)
    # sort conv1
    _reorg_output_channel(g.conv1, sorted_idx)
    sorted_idx = _get_sorted_input_idx(g.conv1, latents)
    _reorg_input_channel(g.conv1, sorted_idx)

    # sort fixed input
    g.input.input.data = torch.index_select(g.input.input.data, 1, sorted_idx)
