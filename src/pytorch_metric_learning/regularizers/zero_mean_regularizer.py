import torch
from torch import Tensor

from ..types import LossDict
from ..utils._common import arange_like
from .base_regularizer import BaseRegularizer


class ZeroMeanRegularizer(BaseRegularizer):
    def compute_loss(self, x: Tensor) -> LossDict:
        return {
            "loss": {
                "losses": torch.abs(torch.sum(x, dim=1)),
                "indices": arange_like(x),
                "reduction_type": "element",
            }
        }
