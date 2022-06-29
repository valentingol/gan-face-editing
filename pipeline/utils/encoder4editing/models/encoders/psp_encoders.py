"""PSP encoders."""

import math
from enum import Enum

import numpy as np
import torch
from torch import nn
from torch.nn import BatchNorm2d, Conv2d, Module, PReLU, Sequential

from pipeline.utils.encoder4editing.models.encoders.helpers import (
    _upsample_add, bottleneck_IR, bottleneck_IR_SE, get_blocks)
from pipeline.utils.encoder4editing.models.stylegan2.model import EqualLinear


class ProgressiveStage(Enum):
    """Progressive stage."""

    WTraining = 0
    Delta1Training = 1
    Delta2Training = 2
    Delta3Training = 3
    Delta4Training = 4
    Delta5Training = 5
    Delta6Training = 6
    Delta7Training = 7
    Delta8Training = 8
    Delta9Training = 9
    Delta10Training = 10
    Delta11Training = 11
    Delta12Training = 12
    Delta13Training = 13
    Delta14Training = 14
    Delta15Training = 15
    Delta16Training = 16
    Delta17Training = 17
    Inference = 18


class GradualStyleBlock(Module):
    """Gradual style block."""

    def __init__(self, in_c, out_c, spatial):
        super().__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [
            Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        ]
        for _ in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        """Forward pass."""
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class GradualStyleEncoder(Module):
    """Gradual style encoder."""

    def __init__(self, num_layers, mode='ir', opts=None):
        super().__init__()
        assert num_layers in [50, 100, 152], ('num_layers should be 50,100, '
                                              'or 152')
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64), PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        log_size = int(math.log(opts.stylegan_size, 2))
        self.style_count = 2*log_size - 2
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1,
                                   padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1,
                                   padding=0)

    def forward(self, x):
        """Forward pass."""
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, layer in enumerate(modulelist):
            x = layer(x)
            if i == 6:
                c_1 = x
            elif i == 20:
                c_2 = x
            elif i == 23:
                c_3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c_3))

        p_2 = _upsample_add(c_3, self.latlayer1(c_2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p_2))

        p_1 = _upsample_add(p_2, self.latlayer2(c_1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p_1))

        out = torch.stack(latents, dim=1)
        return out


class Encoder4Editing(Module):
    """Encoder for editing."""

    def __init__(self, num_layers, mode='ir', opts=None):
        super().__init__()
        assert num_layers in [50, 100, 152], ('num_layers should be 50,100, '
                                              'or 152')
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64), PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        log_size = int(math.log(opts.stylegan_size, 2))
        self.style_count = 2*log_size - 2
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1,
                                   padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1,
                                   padding=0)

        self.progressive_stage = ProgressiveStage.Inference

    def get_deltas_starting_dimensions(self):
        """Get a list of dimensions for the deltas.

        Get a list of the initial dimension of every delta
        from which it is applied.
        """
        # Each dimension has a delta applied to it
        return list(range(self.style_count))

    def set_progressive_stage(self, new_stage: ProgressiveStage):
        """Set the progressive stage."""
        self.progressive_stage = new_stage
        print('Changed progressive stage to: ', new_stage)

    def forward(self, x):
        """Forward pass."""
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, layer in enumerate(modulelist):
            x = layer(x)
            if i == 6:
                c_1 = x
            elif i == 20:
                c_2 = x
            elif i == 23:
                c_3 = x

        # Infer main W and duplicate it
        w_0 = self.styles[0](c_3)
        w_latent = w_0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        stage = self.progressive_stage.value
        features = c_3
        for i in range(1, min(stage + 1, self.style_count)):
            # Infer additional deltas
            if i == self.coarse_ind:
                # FPN's middle features
                p_2 = _upsample_add(c_3, self.latlayer1(c_2))
                features = p_2
            elif i == self.middle_ind:
                # FPN's fine features
                p_1 = _upsample_add(p_2, self.latlayer2(c_1))
                features = p_1
            delta_i = self.styles[i](features)
            w_latent[:, i] += delta_i
        return w_latent


class BackboneEncoderUsingLastLayerIntoW(Module):
    """Backbone encoder using last layer into W space."""

    def __init__(self, num_layers, mode='ir', opts=None):
        super().__init__()
        print('Using BackboneEncoderUsingLastLayerIntoW')
        assert num_layers in [50, 100, 152], ('num_layers should be 50,100, '
                                              'or 152')
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64), PReLU(64))
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)
        log_size = int(math.log(opts.stylegan_size, 2))
        self.style_count = 2*log_size - 2

    def forward(self, x):
        """Forward pass."""
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x.repeat(self.style_count, 1, 1).permute(1, 0, 2)
