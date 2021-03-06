"""Native CUDA implementation of the Fused Leaky ReLU activation function."""

import os

import torch
from torch import nn
from torch.autograd import Function
from torch.utils.cpp_extension import load

module_path = os.path.dirname(__file__)
fused = load(
    'fused', sources=[
        os.path.join(module_path, 'fused_bias_act.cpp'),
        os.path.join(module_path, 'fused_bias_act_kernel.cu'),
    ],
)


class FusedLeakyReLUFunctionBackward(Function):
    """Backward version of FusedLeakyReLUFunction."""

    @staticmethod
    def forward(ctx, grad_output, out, negative_slope, scale):
        """Forward pass FusedLeakyReLUFunctionBackward."""
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        empty = grad_output.new_empty(0)

        grad_input = fused.fused_bias_act(grad_output, empty, out, 3, 1,
                                          negative_slope, scale)

        dim = [0]

        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        grad_bias = grad_input.sum(dim).detach()

        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        """Backward pass FusedLeakyReLUFunctionBackward."""
        out, = ctx.saved_tensors
        gradgrad_out = fused.fused_bias_act(gradgrad_input, gradgrad_bias, out,
                                            3, 1, ctx.negative_slope,
                                            ctx.scale)

        return gradgrad_out, None, None, None


class FusedLeakyReLUFunction(Function):
    """Forward version of FusedLeakyReLUFunction."""

    @staticmethod
    def forward(ctx, X, bias, negative_slope, scale):
        """Forward pass of FusedLeakyReLUFunction."""
        empty = X.new_empty(0)
        out = fused.fused_bias_act(X, bias, empty, 3, 0, negative_slope, scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of FusedLeakyReLUFunction."""
        out, = ctx.saved_tensors

        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(
            grad_output, out, ctx.negative_slope, ctx.scale)

        return grad_input, grad_bias, None, None


class FusedLeakyReLU(nn.Module):
    """Fused Leaky ReLU module."""

    def __init__(self, channel, negative_slope=0.2, scale=2**0.5):
        """Initialize FusedLeakyReLU module."""
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, X):
        """Forward pass of FusedLeakyReLU module."""
        return fused_leaky_relu(X, self.bias[:X.shape[1]], self.negative_slope,
                                self.scale)


def fused_leaky_relu(X, bias, negative_slope=0.2, scale=2**0.5):
    """Apply Fused Leaky ReLU."""
    return FusedLeakyReLUFunction.apply(X, bias[:X.shape[1]], negative_slope,
                                        scale)
