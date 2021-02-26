import torch

from .base_reducer import BaseReducer


class DivisorReducer(BaseReducer):
    def unpack_loss_info(self, loss_info):
        losses, loss_indices, reduction_type, kwargs = super().unpack_loss_info(
            loss_info
        )
        if reduction_type != "already_reduced":
            kwargs = {"divisor": loss_info["divisor"]}
            self.divisor = kwargs["divisor"]
            self.add_to_recordable_attributes(name="divisor", is_stat=True)
        return losses, loss_indices, reduction_type, kwargs

    def sum_and_divide(self, losses, embeddings, divisor):
        if divisor != 0:
            return torch.sum(losses) / divisor
        return self.zero_loss(embeddings)

    def element_reduction(self, losses, loss_indices, embeddings, labels, divisor=1):
        return self.sum_and_divide(losses, embeddings, divisor)

    def pos_pair_reduction(self, *args, **kwargs):
        return self.element_reduction(*args, **kwargs)

    def neg_pair_reduction(self, *args, **kwargs):
        return self.element_reduction(*args, **kwargs)

    def triplet_reduction(self, *args, **kwargs):
        return self.element_reduction(*args, **kwargs)
