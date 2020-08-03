from .base_reducer import BaseReducer
import torch


class ThresholdReducer(BaseReducer):
    def __init__(self, low=None, high=None, **kwargs):
        super().__init__(**kwargs)
        assert (low is not None) or (high is not None), "At least one of low or high must be specified"
        self.low = low
        self.high = high
        self.add_to_recordable_attributes(list_of_names=["low", "high"], is_stat=False)

    def element_reduction(self, losses, *_):
        return self.element_reduction_helper(losses, "elements_above_threshold")
    
    def pos_pair_reduction(self, losses, *args):
        return self.element_reduction_helper(losses, "pos_pairs_above_threshold")

    def neg_pair_reduction(self, losses, *args):
        return self.element_reduction_helper(losses, "neg_pairs_above_threshold")

    def triplet_reduction(self, losses, *args):
        return self.element_reduction_helper(losses, "triplets_above_threshold")

    def element_reduction_helper(self, losses, attr_name):
        low_condition = torch.ones_like(losses, dtype=torch.bool)
        high_condition = torch.ones_like(losses, dtype=torch.bool)
        if self.low is not None:
            low_condition &= losses > self.low
        if self.high is not None:
            high_condition &= losses < self.high
        threshold_condition = low_condition & high_condition 
        num_above_threshold = torch.sum(threshold_condition)
        if num_above_threshold >= 1:
            loss = torch.mean(losses[threshold_condition])
        else:
            loss = torch.mean(losses)*0 # set loss to 0
        self.add_to_recordable_attributes(name=attr_name, is_stat=True)
        setattr(self, attr_name, num_above_threshold)
        return loss