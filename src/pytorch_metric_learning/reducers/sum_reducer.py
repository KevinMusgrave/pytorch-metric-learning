import torch

from pytorch_metric_learning.reducers import MeanReducer


class SumReducer(MeanReducer):
    def element_reduction(self, losses, *_):
        return torch.sum(losses)
