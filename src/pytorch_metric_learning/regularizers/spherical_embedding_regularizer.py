import torch
from torch import Tensor

from ..reducers import BaseReducer
from ..types import LossDict
from ..utils._common import accumulation_dtype, arange_like
from .base_regularizer import BaseRegularizer


class SphericalEmbeddingRegularizer(BaseRegularizer):
    def __init__(self, *, reducer: BaseReducer | None = None) -> None:
        super().__init__(reducer=reducer)

    def compute_loss(self, x: Tensor) -> LossDict:
        norms = torch.linalg.vector_norm(x.to(accumulation_dtype(x.dtype)), dim=1)
        return {
            "loss": {
                "losses": (norms - norms.mean()) ** 2,
                "indices": arange_like(x),
            }
        }

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x).to(x.dtype)
