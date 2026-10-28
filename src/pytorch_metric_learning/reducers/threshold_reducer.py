import torch

from .base_reducer import BaseReducer


class ThresholdReducer(BaseReducer):
    def __init__(self, low=None, high=None, **kwargs):
        super().__init__(**kwargs)
        if not (low is not None or high is not None):
            raise ValueError("At least one of low or high must be specified")
        self.low = low
        self.high = high

    def element_reduction(self, losses, loss_indices, embeddings, labels, **kwargs):
        if self.low is not None:
            losses = losses[losses > self.low]
        if self.high is not None:
            losses = losses[losses < self.high]
        if len(losses) >= 1:
            return torch.mean(losses)
        return self.zero_loss(embeddings)
