from .base_reducer import BaseReducer
import torch


class DivisorReducer(BaseReducer):
    def unpack_loss_info(self, loss_info):
        losses, loss_indices, reduction_type = super().unpack_loss_info(loss_info)
        self.add_to_recordable_attributes(name="total_divisor", is_stat=True)
        self.total_divisor = 0
        if (reduction_type == "already_reduced") and (
            "divisor_summands" not in loss_info
        ):
            self.total_divisor = 1
            return losses, loss_indices, reduction_type
        divisor_summands = loss_info["divisor_summands"]
        for name, value in divisor_summands.items():
            self.total_divisor += value
            if torch.is_tensor(value):
                value = value.item()
            self.add_to_recordable_attributes(name=name, is_stat=True)
            setattr(self, name, value)
        return losses, loss_indices, reduction_type

    def sum_and_divide(self, losses):
        if self.total_divisor != 0:
            output = torch.sum(losses) / self.total_divisor
            if torch.is_tensor(self.total_divisor):
                # remove from autograd graph if necessary
                self.total_divisor = self.total_divisor.item()
            return output
        return torch.sum(losses * 0)

    def already_reduced_reduction(self, losses, *args):
        losses = super().already_reduced_reduction(losses, *args)
        return self.sum_and_divide(losses)

    def element_reduction(self, losses, *_):
        return self.sum_and_divide(losses)

    def pos_pair_reduction(self, losses, *args):
        return self.sum_and_divide(losses)

    def neg_pair_reduction(self, losses, *args):
        return self.sum_and_divide(losses)

    def triplet_reduction(self, losses, *args):
        return self.sum_and_divide(losses)
