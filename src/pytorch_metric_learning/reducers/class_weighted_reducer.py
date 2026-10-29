import torch
from torch import Tensor

from .base_reducer import BaseReducer


class ClassWeightedReducer(BaseReducer):
    weights: Tensor

    def __init__(self, weights: Tensor) -> None:
        super().__init__()
        self.register_buffer("weights", torch.as_tensor(weights))

    def element_reduction(
        self,
        losses: Tensor,
        loss_indices: Tensor,
        embeddings: Tensor,
        labels: Tensor | None,
    ) -> Tensor:
        if labels is None:
            raise ValueError("ClassWeightedReducer requires labels")
        weights = self.weights.to(device=losses.device, dtype=losses.dtype)
        return torch.mean(losses * weights[labels[loss_indices]])
