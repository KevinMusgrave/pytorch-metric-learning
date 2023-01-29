import torch

from .base_reducer import BaseReducer


class ThresholdReducer(BaseReducer):
    def __init__(self, low=None, high=None, **kwargs):
        super().__init__(**kwargs)
        assert (low is not None) or (
            high is not None
        ), "At least one of low or high must be specified"
        self.low = low
        self.high = high
        self.add_to_recordable_attributes(list_of_names=["low", "high"], is_stat=False)
        self.add_to_recordable_attributes(
            list_of_names=["num_past_filter", "num_above_low", "num_below_high"],
            is_stat=True,
        )

    def element_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction_helper(losses, embeddings)

    def pos_pair_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction_helper(losses, embeddings)

    def neg_pair_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction_helper(losses, embeddings)

    def triplet_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction_helper(losses, embeddings)

    def element_reduction_helper(self, losses, embeddings):
        low_condition, high_condition = None, None
        if self.low is not None:
            low_condition = losses > self.low
            losses = losses[low_condition]
        if self.high is not None:
            high_condition = losses < self.high
            losses = losses[high_condition]
        num_past_filter = len(losses)
        if num_past_filter >= 1:
            loss = torch.mean(losses)
        else:
            loss = self.zero_loss(embeddings)
        self.set_stats(low_condition, high_condition, num_past_filter)
        return loss

    def set_stats(self, low_condition, high_condition, num_past_filter):
        if self.collect_stats:
            self.num_past_filter = num_past_filter
            with torch.no_grad():
                if self.low is not None:
                    self.num_above_low = torch.sum(low_condition).item()
                if self.high is not None:
                    self.num_above_high = torch.sum(high_condition).item()
