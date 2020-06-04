from .base_reducer import BaseReducer
import torch


class AvgNonZeroReducer(BaseReducer):
    def __init__(self, reduction_type):
        super().__init__()
        assert reduction_type in ["per_element", "per_pair", "per_triplet"]
        if reduction_type == "per_element":
            self.add_to_recordable_attributes(name="num_non_zero_elements")
        elif reduction_type == "per_pair":
            self.add_to_recordable_attributes(list_of_names=["num_non_zero_pos_pairs", "num_non_zero_neg_pairs"])
        elif reduction_type == "per_triplet":
            self.add_to_recordable_attributes(list_of_names=["num_non_zero_triplets"])

    def per_element_reduction(self, losses, *_):
        loss, self.num_non_zero_elements = self.per_element_reduction_helper(losses)
        return loss
    
    def per_pair_reduction(self, losses, *args):
        pos_loss, self.num_non_zero_pos_pairs = self.per_element_reduction_helper(losses[0])
        neg_loss, self.num_non_zero_neg_pairs = self.per_element_reduction_helper(losses[1])
        return pos_loss + neg_loss

    def per_triplet_reduction(self, losses, *args):
        loss, self.num_non_zero_triplets = self.per_element_reduction_helper(losses)
        return loss

    def per_element_reduction_helper(self, losses):
        num_non_zero = len((losses > 0).nonzero())
        if num_non_zero >= 1:
            loss = torch.sum(losses) / num_non_zero
        else:
            loss = torch.mean(losses)
            assert loss == 0 # mean must be zero, otherwise something is wrong
        return loss, num_non_zero