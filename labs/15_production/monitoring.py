"""Controlled monitoring streams and PSI drift detection.

`histogram_counts` and `brightness_feature` are given (scaffold). `psi` is
your task -- see README.md section 6.
"""

import torch


def histogram_counts(values: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
    counts = torch.zeros(len(bin_edges) - 1)
    for i in range(len(bin_edges) - 1):
        if i < len(bin_edges) - 2:
            mask = (values >= bin_edges[i]) & (values < bin_edges[i + 1])
        else:
            mask = (values >= bin_edges[i]) & (values <= bin_edges[i + 1])
        counts[i] = mask.sum()
    return counts


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
