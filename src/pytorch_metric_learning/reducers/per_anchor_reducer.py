import torch

from ..utils import common_functions as c_f
from .base_reducer import BaseReducer
from .mean_reducer import MeanReducer


def aggregation_func(x, num_per_row):
    zero_denom = num_per_row == 0
    x = torch.sum(x, dim=1) / num_per_row
    x[zero_denom] = 0
    return x


class PerAnchorReducer(BaseReducer):
    def __init__(self, reducer=None, aggregation_func=aggregation_func, **kwargs):
        super().__init__(**kwargs)
        self.reducer = reducer if reducer is not None else MeanReducer()
        self.aggregation_func = aggregation_func

    def element_reduction(self, losses, loss_indices, embeddings, labels, **kwargs):
        loss_dict = {
            "loss": {
                "losses": losses,
                "indices": loss_indices,
                "reduction_type": "element",
            }
        }
        return self.reducer(loss_dict, embeddings, labels)

    def pair_reduction(self, losses, loss_indices, embeddings, labels):
        batch_size = embeddings.shape[0]
        device, dtype = losses.device, losses.dtype
        new_array = torch.zeros(batch_size, batch_size, device=device, dtype=dtype)
        pos_inf = c_f.pos_inf(dtype)
        new_array += pos_inf

        anchors, others = loss_indices
        new_array[anchors, others] = losses
        pos_inf_mask = new_array == pos_inf
        num_inf = torch.sum(pos_inf_mask, dim=1)

        new_array[pos_inf_mask] = 0
        num_per_row = batch_size - num_inf
        output = self.aggregation_func(new_array, num_per_row)

        loss_dict = {
            "loss": {
                "losses": output,
                "indices": c_f.torch_arange_from_size(embeddings),
                "reduction_type": "element",
            }
        }
        return self.reducer(loss_dict, embeddings, labels)

    def pos_pair_reduction(self, losses, loss_indices, embeddings, labels, **kwargs):
        return self.pair_reduction(losses, loss_indices, embeddings, labels)

    def neg_pair_reduction(self, losses, loss_indices, embeddings, labels, **kwargs):
        return self.pair_reduction(losses, loss_indices, embeddings, labels)

    def triplet_reduction(self, losses, loss_indices, embeddings, labels, **kwargs):
        raise NotImplementedError("PerAnchorReducer does not support triplets")
