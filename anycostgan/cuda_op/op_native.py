# Code from https://github.com/mit-han-lab/anycost-gan
"""Native PyTorch versions of the custom operations."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import leaky_relu


def fused_leaky_relu(input_, bias, negative_slope=0.2, scale=2**0.5):
    """Fused LeakyReLU activation function."""
    return scale * leaky_relu(
            input_ + bias[:input_.shape[1]], negative_slope, inplace=True
            )


class FusedLeakyReLU(nn.Module):
    """Fused Leaky ReLU module."""

    def __init__(self, channel, negative_slope=0.2, scale=2**0.5):
        """Initialize FusedLeakyReLU module."""
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, x):
        """Forward pass of FusedLeakyReLU module."""
        return self.scale * leaky_relu(
                x + self.bias.reshape((1, -1, 1, 1))[:, :x.shape[1]],
                self.negative_slope, inplace=True
                )


def upfirdn2d(X, kernel, up=1, down=1, pad=(0, 0)):
    """Upsample and FIRDN (Fractional Interpolated Directed Neighbor)."""
    out = upfirdn2d_native(
            X, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
            )
    return out


def upfirdn2d_native(
        X, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
        ):
    """Upsample and FIRDN (Fractional Interpolated Directed Neighbor)."""
    _, chanels, _, _ = X.shape
    kernel_h, kernel_w = kernel.shape

    assert up_y == up_x and up_y in [1, 2]

    if up_y == 2:
        w = X.new_zeros(2, 2)
        w[0, 0] = 1
        out = F.conv_transpose2d(
                X,
                w.view(1, 1, 2, 2).repeat(chanels, 1, 1, 1), groups=chanels,
                stride=2
                )
    else:
        out = X

    out = F.pad(out, [pad_x0, pad_x1, pad_y0, pad_y1])
    out = F.conv2d(
            out,
            kernel.view(1, 1, kernel_h, kernel_w).repeat(chanels, 1, 1, 1),
            groups=chanels
            )

    return out[:, :, ::down_y, ::down_x]
