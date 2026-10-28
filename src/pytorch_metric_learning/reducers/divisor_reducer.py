import torch

from .base_reducer import BaseReducer


class DivisorReducer(BaseReducer):
    def unpack_loss_info(self, loss_info):
        losses, loss_indices, reduction_type, kwargs = super().unpack_loss_info(
            loss_info
        )
        if reduction_type != "already_reduced":
            kwargs = {"divisor": loss_info["divisor"]}
        return losses, loss_indices, reduction_type, kwargs

    def sum_and_divide(self, losses, embeddings, divisor):
        if divisor != 0:
            output = torch.sum(losses) / divisor
            if torch.isnan(output) and losses.dtype == torch.float16:
                output = torch.sum(losses.to(dtype=torch.float32)) / divisor
                output = output.to(dtype=torch.float16)
            return output
        return self.zero_loss(embeddings)

    def element_reduction(self, losses, loss_indices, embeddings, labels, **kwargs):
        return self.sum_and_divide(losses, embeddings, kwargs.get("divisor", 1))
