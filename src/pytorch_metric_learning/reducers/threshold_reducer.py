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
        if self.low is not None:
            self.add_to_recordable_attributes(list_of_names=["low"], is_stat=False)
        if self.high is not None:
            self.add_to_recordable_attributes(list_of_names=["high"], is_stat=False)

    def element_reduction(self, losses, *_):
        return self.element_reduction_helper(losses, "elements")

    def pos_pair_reduction(self, losses, *args):
        return self.element_reduction_helper(losses, "pos_pairs")

    def neg_pair_reduction(self, losses, *args):
        return self.element_reduction_helper(losses, "neg_pairs")

    def triplet_reduction(self, losses, *args):
        return self.element_reduction_helper(losses, "triplets")

    def element_reduction_helper(self, losses, attr_name):
        low_condition = losses > self.low if self.low is not None else True
        high_condition = losses < self.high if self.high is not None else True
        threshold_condition = low_condition & high_condition
        num_past_filter = torch.sum(threshold_condition)
        if num_past_filter >= 1:
            loss = torch.mean(losses[threshold_condition])
        else:
            loss = torch.mean(losses) * 0  # set loss to 0
        self.set_stats(low_condition, high_condition, num_past_filter, attr_name)
        return loss

    def set_stats(self, low_condition, high_condition, num_past_filter, attr_name):
        if self.collect_stats:
            curr_attr_name = "{}_past_filter".format(attr_name)
            self.add_to_recordable_attributes(name=curr_attr_name, is_stat=True)
            setattr(self, curr_attr_name, num_past_filter.item())
            with torch.no_grad():
                if self.low is not None:
                    curr_attr_name = "{}_above_low".format(attr_name)
                    self.add_to_recordable_attributes(name=curr_attr_name, is_stat=True)
                    setattr(self, curr_attr_name, torch.sum(low_condition).item())
                if self.high is not None:
                    curr_attr_name = "{}_below_high".format(attr_name)
                    self.add_to_recordable_attributes(name=curr_attr_name, is_stat=True)
                    setattr(self, curr_attr_name, torch.sum(high_condition).item())
