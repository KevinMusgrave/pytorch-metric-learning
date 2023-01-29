import torch

from ..utils import common_functions as c_f
from .base_reducer import BaseReducer


class DivisorReducer(BaseReducer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_to_recordable_attributes(name="divisor", is_stat=True)

    def unpack_loss_info(self, loss_info):
        losses, loss_indices, reduction_type, kwargs = super().unpack_loss_info(
            loss_info
        )
        if reduction_type != "already_reduced":
            kwargs = {"divisor": loss_info["divisor"]}
            self.divisor = kwargs["divisor"]
        return losses, loss_indices, reduction_type, kwargs

    def sum_and_divide(self, losses, embeddings, divisor):
        if divisor != 0:
            output = torch.sum(losses) / divisor
            if torch.isnan(output) and losses.dtype == torch.float16:
                output = torch.sum(c_f.to_dtype(losses, dtype=torch.float32)) / divisor
                output = c_f.to_dtype(output, dtype=torch.float16)
            return output
        return self.zero_loss(embeddings)

    def element_reduction(self, losses, loss_indices, embeddings, labels, divisor=1):
        return self.sum_and_divide(losses, embeddings, divisor)

    def pos_pair_reduction(self, *args, **kwargs):
        return self.element_reduction(*args, **kwargs)

    def neg_pair_reduction(self, *args, **kwargs):
        return self.element_reduction(*args, **kwargs)

    def triplet_reduction(self, *args, **kwargs):
        return self.element_reduction(*args, **kwargs)
