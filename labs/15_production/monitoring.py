"""Controlled monitoring streams and PSI drift detection.

`histogram_counts` and `brightness_feature` are given (scaffold). `psi` is
your task -- see README.md section 6.
"""

import torch


def histogram_counts(values: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
    """Count values in fixed bins, keeping underflow and overflow.

    Values outside the reference range belong to the first or last bin.
    Dropping them would hide exactly the distribution shift this histogram
    is intended to measure and can leave PSI with an all-zero distribution.
    """
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if values.numel() == 0:
        raise ValueError("values must not be empty")
    if not torch.isfinite(values).all():
        raise ValueError("values must be finite")
    if bin_edges.ndim != 1 or bin_edges.numel() < 2:
        raise ValueError("bin_edges must contain at least two values")
    if not torch.isfinite(bin_edges).all():
        raise ValueError("bin_edges must be finite")
    if not torch.all(bin_edges[1:] > bin_edges[:-1]):
        raise ValueError("bin_edges must be strictly increasing")
    if values.device != bin_edges.device:
        raise ValueError("values and bin_edges must be on the same device")

    bin_ids = torch.bucketize(values, bin_edges[1:-1], right=True)
    return torch.bincount(bin_ids, minlength=bin_edges.numel() - 1).to(torch.float32)


def psi(ref_counts: torch.Tensor, cur_counts: torch.Tensor, eps: float = 1e-4) -> float:
    # TODO:
    # 1. ref_props = ref_counts / ref_counts.sum(); same for cur_props
    # 2. clamp BOTH to a minimum of `eps` (avoids log(0) for empty bins --
    #    see lecture.md for what happens if you skip this)
    # 3. return sum((cur_props - ref_props) * log(cur_props / ref_props))
    raise NotImplementedError("TODO: psi")


def brightness_feature(images: torch.Tensor) -> torch.Tensor:
    """Mean pixel value per image -- the monitored feature, no labels
    needed."""
    return images.mean(dim=(1, 2, 3))
