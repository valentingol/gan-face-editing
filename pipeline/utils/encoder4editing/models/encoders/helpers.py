# ArcFace implementation from https://github.com/TreB1eN/InsightFace_Pytorch
"""Utilities for encoders."""

from collections import namedtuple

import torch
import torch.nn.functional as F
from torch.nn import (AdaptiveAvgPool2d, BatchNorm2d, Conv2d, MaxPool2d,
                      Module, PReLU, ReLU, Sequential, Sigmoid)


class Flatten(Module):
    """Flatten module."""

    def forward(self, inputs):
        """Pass forward."""
        return inputs.view(inputs.size(0), -1)


def l2_norm(inputs, axis=1):
    """Apply L2 normalization."""
    norm = torch.norm(inputs, 2, axis, True)
    output = torch.div(inputs, norm)
    return output


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    """A named tuple describing a ResNet block."""


def get_block(in_channel, depth, num_units, stride=2):
    """Get bottleneck block."""
    return [Bottleneck(in_channel, depth, stride)
            ] + [Bottleneck(depth, depth, 1) for _ in range(num_units - 1)]


def get_blocks(num_layers):
    """Get multi-blocks of bottleneck.."""
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    else:
        raise ValueError(f"Invalid number of layers: {num_layers}. "
                         "Must be one of [50, 100, 152]")
    return blocks


class SEModule(Module):
    """SE module."""

    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1,
                          padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1,
                          padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        """Forward pass."""
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class bottleneck_IR(Module):
    """Bottleneck IR module."""

    def __init__(self, in_channel, depth, stride):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth), Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth))

    def forward(self, x):
        """Forward pass."""
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class bottleneck_IR_SE(Module):
    """Bottleneck IR SE module."""

    def __init__(self, in_channel, depth, stride):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth), Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth), SEModule(depth, 16))

    def forward(self, x):
        """Forward pass."""
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


def _upsample_add(x, y):
    """Upsample and add two feature maps.

    Parameters
    ----------
    x: torch.tensor
        Top feature map to be upsampled.
    y: torch.tensor
        Lateral feature map.

    Returns
    -------
    torch tensor
        Added feature map.

    Note
    ----
    Note in PyTorch, when input size is odd, the upsampled feature map
    with `F.upsample(..., scale_factor=2, mode='nearest')`
    maybe not equal to the lateral feature map size.
    e.g.
    original input size: [N,_,15,15] ->
    conv2d feature map size: [N,_,8,8] ->
    upsampled feature map size: [N,_,16,16]
    So we choose bilinear upsample which supports arbitrary output sizes.
    """
    _, _, height, weight = y.size()
    return F.interpolate(x, size=(height, weight), mode='bilinear',
                         align_corners=True) + y
