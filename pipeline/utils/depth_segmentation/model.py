# Code from https://github.com/isl-org/DPT
"""Depth estimation models."""

import torch
from torch import nn

from pipeline.utils.depth_segmentation.base_model import BaseModel
from pipeline.utils.depth_segmentation.blocks import (
    FeatureFusionBlock_custom, Interpolate, _make_encoder)
from pipeline.utils.depth_segmentation.vit import forward_vit


def _make_fusion_block(features, use_bn):
    """Return a fusion block."""
    return FeatureFusionBlock_custom(features, nn.ReLU(False), deconv=False,
                                     bn=use_bn, expand=False,
                                     align_corners=True,
                                     )


class DPT(BaseModel):
    """Dense Prediction transformer."""

    def __init__(self, head, features=256, backbone="vitb_rn50_384",
                 readout="project", channels_last=False, use_bn=False,
                 enable_attention_hooks=False):
        """Initialize model."""
        super().__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone, features,
            False,  # True: from scratch, False: from pretrained
            groups=1, expand=False, exportable=False, hooks=hooks[backbone],
            use_readout=readout, enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head

    def forward(self, x):
        """Forward pass."""
        if self.channels_last:
            x.contiguous(memory_format=torch.channels_last)
        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return out


class DPTDepthModel(DPT):
    """Depth estimation model."""

    def __init__(self, path=None, non_negative=True, scale=1.0, shift=0.0,
                 invert=False, **kwargs):
        """Initialize model."""
        features = kwargs["features"] if "features" in kwargs else 256

        self.scale = scale
        self.shift = shift
        self.invert = invert

        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1,
                      padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True), nn.Conv2d(32, 1, kernel_size=1, stride=1,
                                     padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(), nn.Identity(),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)

    def forward(self, x):
        """Forward pass."""
        inv_depth = super().forward(x).squeeze(dim=1)

        if self.invert:
            depth = self.scale * inv_depth + self.shift
            depth[depth < 1e-8] = 1e-8
            depth = 1.0 / depth
            return depth
        return inv_depth
