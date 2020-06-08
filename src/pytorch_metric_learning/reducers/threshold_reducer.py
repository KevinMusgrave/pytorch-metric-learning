from .base_reducer import BaseReducer
import torch


class ThresholdReducer(BaseReducer):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
        self.add_to_recordable_attributes(name="threshold")

    def element_reduction(self, losses, *_):
        self.add_to_recordable_attributes(name="elements_above_threshold")
        loss, self.elements_above_threshold = self.element_reduction_helper(losses)
        return loss
    
    def pos_pair_reduction(self, losses, *args):
        self.add_to_recordable_attributes(name="pos_pairs_above_threshold")
        loss, self.pos_pairs_above_threshold = self.element_reduction_helper(losses)
        return loss

    def neg_pair_reduction(self, losses, *args):
        self.add_to_recordable_attributes(name="neg_pairs_above_threshold")
        loss, self.neg_pairs_above_threshold = self.element_reduction_helper(losses)
        return loss

    def triplet_reduction(self, losses, *args):
        self.add_to_recordable_attributes(name="triplets_above_threshold")
        loss, self.triplets_above_threshold = self.element_reduction_helper(losses)
        return loss

    def element_reduction_helper(self, losses):
        num_above_threshold = len((losses > self.threshold).nonzero())
        if num_above_threshold >= 1:
            loss = torch.sum(losses) / num_above_threshold
        else:
            loss = torch.mean(losses)*0 # set loss to 0
        return loss, num_above_threshold