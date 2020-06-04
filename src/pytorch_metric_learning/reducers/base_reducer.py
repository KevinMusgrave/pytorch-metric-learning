import torch
from ..utils import common_functions as c_f


class BaseReducer(torch.nn.Module):
    
    def forward(self, losses, loss_indices, labels):
        reduction_type = self.assert_losses_size(losses, loss_indices)
        reduction_func = self.get_reduction_func(reduction_type)
        return reduction_func(losses, loss_indices, labels)

    def per_element_reduction(self, losses, loss_indices, labels):
        raise NotImplementedError

    def per_pair_reduction(self, losses, loss_indices, labels):
        raise NotImplementedError
    
    def per_triplet_reduction(self, losses, loss_indices, labels):
        raise NotImplementedError

    def get_reduction_func(self, reduction_type):
        return self.get_reduction_func_dict(reduction_type)

    def get_reduction_func_dict(self):
        return {"per_element": self.per_element_reduction,
                "per_pair": self.per_pair_reduction,
                "per_triplet": self.per_triplet_reduction}

    def assert_losses_size(self, losses, loss_indices):
        # element indices
        if torch.is_tensor(loss_indices):
            assert torch.is_tensor(losses)
            assert len(losses) > 0
            assert len(losses) == len(loss_indices)
            return "per_element"
        elif c_f.is_list_or_tuple(loss_indices):
            # pair indices
            if len(loss_indices) == 4:
                assert c_f.is_list_or_tuple(losses)
                assert len(losses) == 2
                assert all(torch.is_tensor(x) for x in losses)
                assert all(len(x) > 0 for x in losses)
                assert len(losses[0]) == len(loss_indices[0]) == len(loss_indices[1])
                assert len(losses[1]) == len(loss_indices[2]) == len(loss_indices[3])
                return "per_pair"
            # triplet indices
            elif len(loss_indices) == 3:
                assert torch.is_tensor(losses)
                loss_length = len(losses)
                assert all(len(x) == loss_length for x in loss_indices)
                return "per_triplet"
        raise TypeError("losses and loss_indices should be type torch.tensor or list or tuple")
