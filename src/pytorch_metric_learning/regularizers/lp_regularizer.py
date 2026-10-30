import torch
from torch import Tensor

from ..reducers import BaseReducer
from ..types import LossDict
from ..utils._common import arange_like
from .base_regularizer import BaseRegularizer


class LpRegularizer(BaseRegularizer):
    def __init__(
        self, p: float = 2, power: float = 1, *, reducer: BaseReducer | None = None
    ) -> None:
        super().__init__(reducer=reducer)
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
            }
        }
