import torch
from torch import Tensor

from .base_reducer import BaseReducer


class ThresholdReducer(BaseReducer):
    def __init__(self, low: float | None = None, high: float | None = None) -> None:
        super().__init__()
        if low is None and high is None:
            raise ValueError("At least one of low or high must be specified")
        self.low = low
        self.high = high

    def element_reduction(
        self,
        losses: Tensor,
        loss_indices: Tensor,
        embeddings: Tensor,
        labels: Tensor | None,
    ) -> Tensor:
        keep = torch.ones_like(losses, dtype=torch.bool)
        if self.low is not None:
            keep &= losses > self.low
        if self.high is not None:
            keep &= losses < self.high
        kept = torch.where(keep, losses, torch.zeros_like(losses))
        return kept.sum() / keep.sum().clamp_min(1)
