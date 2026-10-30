from collections.abc import Sequence
from typing import Unpack

import torch
from torch import Tensor

from ..types import ProxyLossOptions
from .subcenter_arcface_loss import SubCenterArcFaceLoss


def _class_margins(
    class_counts: Tensor | Sequence[int],
    margin_bounds: tuple[float, float],
    power: float,
) -> Tensor:
    counts = torch.as_tensor(class_counts).detach().cpu()
    if counts.ndim != 1 or len(counts) == 0:
        raise ValueError("class_counts must be one-dimensional and non-empty")
    if counts.is_floating_point() or counts.dtype == torch.bool:
        raise TypeError("class_counts must be integers")
    if not bool((counts > 0).all()):
        raise ValueError("class_counts must be positive")
    if int(counts.min()) == int(counts.max()):
        raise ValueError(
            "class_counts must contain two distinct sizes; "
            "use ArcFaceLoss for a constant margin"
        )
    lower, upper = margin_bounds
    if not 0 <= lower < upper:
        raise ValueError("margin_bounds must satisfy 0 <= lower < upper")
    if not power > 0:
        raise ValueError("power must be positive")
    decay = counts.to(torch.float64) ** -power
    largest, smallest = decay.max(), decay.min()
    return lower + (upper - lower) * (decay - smallest) / (largest - smallest)


class DynamicArcFaceLoss(SubCenterArcFaceLoss):
    class_margins: Tensor

    def __init__(
        self,
        class_counts: Tensor | Sequence[int],
        embedding_size: int,
        margin_bounds: tuple[float, float] = (0.05, 0.5),
        power: float = 0.25,
        scale: float = 64,
        sub_centers: int = 3,
        **options: Unpack[ProxyLossOptions],
    ) -> None:
        margins = _class_margins(class_counts, margin_bounds, power)
        super().__init__(
            len(margins),
            embedding_size,
            margin_bounds[1],
            scale,
            sub_centers,
            **options,
        )
        self.register_buffer(
            "class_margins", margins.to(self.weight.dtype), persistent=False
        )

    def target_margins(self, cosine: Tensor, labels: Tensor) -> Tensor:
        return self.class_margins[labels].to(cosine.dtype)
