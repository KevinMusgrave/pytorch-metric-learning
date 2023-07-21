import torch

from ..utils import common_functions as c_f
from .base_reducer import BaseReducer


class DivisorReducer(BaseReducer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.divisor = 1
        self.add_to_recordable_attributes(name="divisor", is_stat=True)

    def unpack_loss_info(self, loss_info):
        if loss_info["reduction_type"] != "already_reduced":
            self.divisor = loss_info["divisor"]
        return super().unpack_loss_info(loss_info)

    def sum_and_divide(self, losses, embeddings):
        if self.divisor != 0:
            output = torch.sum(losses.float()) / self.divisor
            output = c_f.to_dtype(output, tensor=losses)
            return output
        return self.zero_loss(embeddings)

    def element_reduction(self, losses, loss_indices, embeddings, labels):
        return self.sum_and_divide(losses, embeddings)

    def pos_pair_reduction(self, *args, **kwargs):
        return self.element_reduction(*args, **kwargs)

    def neg_pair_reduction(self, *args, **kwargs):
        return self.element_reduction(*args, **kwargs)

    def triplet_reduction(self, *args, **kwargs):
        return self.element_reduction(*args, **kwargs)
