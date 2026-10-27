import torch

from ..utils import common_functions as c_f
from .base_regularizer import BaseRegularizer


class ZeroMeanRegularizer(BaseRegularizer):
    def compute_loss(self, x):
        return {
            "loss": {
                "losses": torch.abs(torch.sum(x, dim=1)),
                "indices": c_f.torch_arange_from_size(x),
                "reduction_type": "element",
            }
        }
