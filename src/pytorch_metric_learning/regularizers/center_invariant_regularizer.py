from typing import Unpack

import torch
from torch import Tensor

from ..distances import BaseDistance, LpDistance
from ..types import LossDict, RegularizerOptions
from ..utils import _common
from .base_regularizer import BaseRegularizer


class CenterInvariantRegularizer(BaseRegularizer):
    def __init__(self, **options: Unpack[RegularizerOptions]) -> None:
        super().__init__(**options)
        _common.check_distance_type(
            self, LpDistance, power=1, normalize_embeddings=False
        )

    def compute_loss(self, x: Tensor) -> LossDict:
        squared_weight_norms = self.distance.get_norm(x) ** 2
        deviations_from_mean = squared_weight_norms - torch.mean(squared_weight_norms)
        return {
            "loss": {
                "losses": (deviations_from_mean**2) / 4,
                "indices": _common.arange_like(x),
                "reduction_type": "element",
            }
        }

    def default_distance(self) -> BaseDistance:
        return LpDistance(power=1, normalize_embeddings=False)
