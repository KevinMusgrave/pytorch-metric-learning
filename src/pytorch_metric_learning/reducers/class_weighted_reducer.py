import torch

from .base_reducer import BaseReducer


class ClassWeightedReducer(BaseReducer):
    def __init__(self, weights, **kwargs):
        super().__init__(**kwargs)
        self.weights = weights

    def element_reduction(self, losses, loss_indices, embeddings, labels, **kwargs):
        weights = self.weights.to(device=losses.device, dtype=losses.dtype)
        return torch.mean(losses * weights[labels[loss_indices]])
