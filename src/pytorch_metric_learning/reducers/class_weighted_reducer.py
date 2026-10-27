import torch

from ..utils import common_functions as c_f
from .base_reducer import BaseReducer


class ClassWeightedReducer(BaseReducer):
    def __init__(self, weights, **kwargs):
        super().__init__(**kwargs)
        self.weights = weights

    def element_reduction(self, losses, loss_indices, embeddings, labels, **kwargs):
        weights = c_f.to_device(self.weights, losses, dtype=losses.dtype)
        return torch.mean(losses * weights[labels[loss_indices]])
