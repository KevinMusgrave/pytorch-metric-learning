import torch

from .base_reducer import BaseReducer


class ThresholdReducer(BaseReducer):
    def __init__(self, low=None, high=None, **kwargs):
        super().__init__(**kwargs)
        assert (low is not None) or (high is not None), (
            "At least one of low or high must be specified"
        )
        self.low = low
        self.high = high

    def element_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction_helper(losses, embeddings)

    def pos_pair_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction_helper(losses, embeddings)

    def neg_pair_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction_helper(losses, embeddings)

    def triplet_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction_helper(losses, embeddings)

    def element_reduction_helper(self, losses, embeddings):
        if self.low is not None:
            losses = losses[losses > self.low]
        if self.high is not None:
            losses = losses[losses < self.high]
        if len(losses) >= 1:
            return torch.mean(losses)
        return self.zero_loss(embeddings)
