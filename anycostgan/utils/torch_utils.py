# Code from https://github.com/mit-han-lab/anycost-gan
"""Pytorch utilities."""

import horovod.torch as hvd
import torch
import torch.nn.functional as F


def safe_load_state_dict_from_url(
        url, model_dir=None, map_location=None, progress=True,
        check_hash=False, file_name=None
        ):
    """Load state dict from a url.

    A safe version of torch.hub.load_state_dict_from_url in
    distributed environment the main idea is to only download the
    file on worker 0.
    """
    try:
        hvd.init()
        world_size = hvd.size()
    except ImportError:  # load horovod failed, just normal environment
        return torch.hub.load_state_dict_from_url(
                url, model_dir, map_location, progress, check_hash, file_name
                )

    if world_size == 1:
        return torch.hub.load_state_dict_from_url(
                url, model_dir, map_location, progress, check_hash, file_name
                )
    # Here world size > 1
    # Possible download... let it only run on worker 0 to prevent conflict
    if hvd.rank() == 0:
        _ = torch.hub.load_state_dict_from_url(
                url, model_dir, map_location, progress, check_hash, file_name
                )
    hvd.broadcast(torch.tensor(0), root_rank=0, name='dummy')
    return torch.hub.load_state_dict_from_url(
            url, model_dir, map_location, progress, check_hash, file_name
            )


def adaptive_resize(img, target_res):
    """Resize an image to target resolution."""
    assert img.shape[-1] == img.shape[-2]
    if img.shape[-1] != target_res:
        return F.interpolate(
                img, size=target_res, mode='bilinear', align_corners=True
                )
    return img


class AverageMeter:
    """Average Meter.

    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/
    imagenet/main.py
    """

    def __init__(self):
        """Initialize the average meter."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset the average meter."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update the average meter."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DistributedMeter:
    """Distributed Meter."""

    def __init__(self, name, dim=None):
        """Initialize the meter."""
        self.name = name
        if dim is None:
            self.sum = torch.tensor(0.)
        else:
            self.sum = torch.zeros(dim)
        self.n = torch.tensor(0.)

    def update(self, val):
        """Update the meter."""
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        """Get the average."""
        return self.sum / self.n
