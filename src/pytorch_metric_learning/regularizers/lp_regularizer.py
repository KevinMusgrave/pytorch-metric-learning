from typing import Unpack

import torch
from torch import Tensor

from ..types import LossDict, RegularizerOptions
from ..utils._common import arange_like
from .base_regularizer import BaseRegularizer


class LpRegularizer(BaseRegularizer):
    def __init__(
        self, p: float = 2, power: float = 1, **options: Unpack[RegularizerOptions]
    ) -> None:
        super().__init__(**options)
        self.p = p
        self.power = power

    def compute_loss(self, x: Tensor) -> LossDict:
        reg = torch.norm(x, p=self.p, dim=1)
        if self.power != 1:
            reg = reg**self.power
        return {
            "loss": {
                "losses": reg,
                "indices": arange_like(x),
                "reduction_type": "element",
            }
        }
