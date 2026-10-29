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
        num_classes = x.size(0)
        cos = self.distance(x)
        with torch.no_grad():
            cos1 = cos.clone()
            cos1.fill_diagonal_(_common.neg_inf(x.dtype))
            _, indices = self.distance.smallest_dist(cos1, dim=1)
            mask = torch.zeros_like(cos)
            mask[torch.arange(num_classes, device=x.device), indices] = 1
        return {
            "loss": {
                "losses": torch.sum(cos * mask, dim=1),
                "indices": _common.arange_like(x),
                "reduction_type": "element",
            }
        }

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()
