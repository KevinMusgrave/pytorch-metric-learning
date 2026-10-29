import torch
from torch import Tensor

from .base_reducer import BaseReducer


class SumReducer(BaseReducer):
    def element_reduction(
        self,
        losses: Tensor,
        loss_indices: Tensor,
        embeddings: Tensor,
        labels: Tensor | None,
    ) -> Tensor:
        return torch.sum(losses)
