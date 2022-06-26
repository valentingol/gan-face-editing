# Code from https://github.com/mit-han-lab/anycost-gan
"""Loss functions for the GAN."""

import math

import torch
import torch.nn.functional as F
from torch import autograd


def d_logistic_loss(real_pred, fake_pred):
    """Logistic loss for the discriminator."""
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    """Gradient penalty for the discriminator."""
    grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
            )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    """Non-saturating loss for the generator."""
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    """Path regularization loss for the generator."""
    noise = torch.randn_like(fake_img) / math.sqrt(
            fake_img.shape[2] * fake_img.shape[3]
            )
    grad, = autograd.grad(
            outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
            )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))
    path_mean = mean_path_length + decay * (
            path_lengths.mean().item() - mean_path_length
            )
    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean, path_lengths
