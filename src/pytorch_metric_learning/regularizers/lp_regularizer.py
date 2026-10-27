import torch

from ..utils import common_functions as c_f
from .base_regularizer import BaseRegularizer


class LpRegularizer(BaseRegularizer):
    def __init__(self, p=2, power=1, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.power = power

    def compute_loss(self, x):
        reg = torch.norm(x, p=self.p, dim=1)
        if self.power != 1:
            reg = reg**self.power
        return {
            "loss": {
                "losses": reg,
                "indices": c_f.torch_arange_from_size(x),
                "reduction_type": "element",
            }
        }
