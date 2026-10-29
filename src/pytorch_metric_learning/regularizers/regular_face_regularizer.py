from typing import Unpack

import torch
from torch import Tensor

from ..distances import BaseDistance, CosineSimilarity
from ..types import LossDict, RegularizerOptions
from ..utils import _common
from .base_regularizer import BaseRegularizer


class RegularFaceRegularizer(BaseRegularizer):
    def __init__(self, **options: Unpack[RegularizerOptions]) -> None:
        super().__init__(**options)
        if not self.distance.is_inverted:
            raise ValueError("distance must be a similarity with is_inverted=True")

    def compute_loss(self, x: Tensor) -> LossDict:
        cos = self.distance(x)
        with torch.no_grad():
            others = cos.detach().clone().fill_diagonal_(_common.neg_inf(x.dtype))
            nearest = others.argmax(dim=1, keepdim=True)
        losses = cos.gather(1, nearest).squeeze(1)
        return {"loss": {"losses": losses, "indices": _common.arange_like(x)}}

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()
