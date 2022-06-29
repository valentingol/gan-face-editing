# Code from https://github.com/mit-han-lab/anycost-gan
"""Datasets utilities."""

import random

import torchvision.transforms.functional as F
from PIL import Image
from torchvision import datasets, transforms


class NativeDataset(datasets.ImageFolder):
    """Base Dataset."""

    def __getitem__(self, index):
        """Get item."""
        # Only return the image
        return super().__getitem__(index)[0]


class MultiResize:
    """Resize the image to multi resolutions."""

    def __init__(self, highest_res, n_res=4, interpolation=Image.BILINEAR):
        """Initialize the transform."""
        all_res = []
        for _ in range(n_res):
            all_res.append(highest_res)
            highest_res = highest_res // 2
        all_res = sorted(all_res)  # always low to high
        self.transforms = [
            transforms.Resize(r, interpolation) for r in all_res
        ]

    def __call__(self, img):
        """Apply the transform."""
        return [t(img) for t in self.transforms]


class GroupRandomHorizontalFlip:
    """Randomly flip the image horizontally."""

    def __init__(self, p=0.5):
        """Initialize the transform."""
        self.proba = p

    def __call__(self, img):
        """Call the transform."""
        if random.random() < self.proba:
            return [F.hflip(i) for i in img]
        return img

    def __repr__(self):
        """Representation."""
        return self.__class__.__name__ + f'(p={self.proba})'


class GroupTransformWrapper:
    """Grouped Transformation Wrapper.

    Applying the same transform (no randomness) to each of the
    images in a list.
    """

    def __init__(self, transform):
        """Initialize the transform."""
        self.transform = transform

    def __call__(self, img):
        """Call the transform."""
        return [self.transform(i) for i in img]

    def __repr__(self):
        """Representation."""
        return self.__class__.__name__
