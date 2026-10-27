import torch

from .mean_reducer import MeanReducer


class SumReducer(MeanReducer):
    def element_reduction(self, losses, loss_indices, embeddings, labels, **kwargs):
        return torch.sum(losses)
