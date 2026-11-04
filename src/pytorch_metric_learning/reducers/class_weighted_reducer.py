import torch
from torch import Tensor

from ..utils import _common
from .base_reducer import BaseReducer


class ClassWeightedReducer(BaseReducer):
    weights: Tensor

    def __init__(self, weights: Tensor) -> None:
        super().__init__()
        self.register_buffer("weights", torch.as_tensor(weights))

    def element_reduction(
        self,
        losses: Tensor,
        anchors: Tensor,
        embeddings: Tensor,
        labels: Tensor | None,
    ) -> Tensor:
        if labels is None:
            raise ValueError("ClassWeightedReducer requires labels")
        if self.weights.device != losses.device:
            raise ValueError(
                f"ClassWeightedReducer weights are on {self.weights.device}, but "
                f"losses are on {losses.device}. Move the reducer with .to(device)."
            )
        dtype = _common.accumulation_dtype(losses.dtype)
        weights = self.weights.to(dtype=dtype)
        return torch.mean(losses.to(dtype) * weights[labels[anchors]]).to(losses.dtype)
