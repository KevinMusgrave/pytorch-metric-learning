import torch

from .base_reducer import BaseReducer


class MeanReducer(BaseReducer):
    def element_reduction(self, losses, loss_indices, embeddings, labels, **kwargs):
        return torch.mean(losses)
