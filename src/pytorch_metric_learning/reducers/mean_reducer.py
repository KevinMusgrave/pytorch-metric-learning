import torch
from torch import Tensor

from .base_reducer import BaseReducer


class MeanReducer(BaseReducer):
    def element_reduction(
        self,
        losses: Tensor,
        anchors: Tensor,
        embeddings: Tensor,
        labels: Tensor | None,
    ) -> Tensor:
        return torch.mean(losses)
