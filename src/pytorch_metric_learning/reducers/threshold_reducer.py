import torch
import numpy as np
from .base_reducer import BaseReducer


class ThresholdReducer(BaseReducer):
    def __init__(self, low=-np.inf, high=np.inf, **kwargs):
        super().__init__(**kwargs)
        self.low = low if low is not None else -np.inf       # Since there is no None default value it could be better to exclude testing for low=None in test_treshold_reducer
        self.high = high if high is not None else np.inf     # Since there is no None default value it could be better to exclude testing for high=None in test_treshold_reducer
        self.add_to_recordable_attributes(list_of_names=["low", "high"], is_stat=False)
        self.add_to_recordable_attributes(
            list_of_names=["num_past_filter", "num_above_low", "num_below_high"],
            is_stat=True,
        )

    def element_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction_helper(losses, embeddings)

    def pos_pair_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction(losses, loss_indices[0], embeddings, labels)

    def neg_pair_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction(losses, loss_indices[0], embeddings, labels)

    def triplet_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction(losses, loss_indices[0], embeddings, labels)

    def element_reduction_helper(self, losses, embeddings):
        low_condition = losses > self.low
        high_condition = losses < self.high
        losses = losses[low_condition & high_condition]
        
        num_past_filter = len(losses)
        if num_past_filter >= 1:
            loss = torch.mean(losses)
        else:
            loss = self.zero_loss(embeddings)
        self.set_stats(low_condition, high_condition, num_past_filter)
        return loss

    @torch.no_grad()
    def set_stats(self, low_condition, high_condition, num_past_filter):
        if self.collect_stats:
            self.num_past_filter = num_past_filter
            if np.isfinite(self.low):       # Why record this only if it was not None?
                self.num_above_low = torch.sum(low_condition).item()
            if np.isfinite(self.high):      # Why record this only if it was not None?
                self.num_above_high = torch.sum(high_condition).item()
