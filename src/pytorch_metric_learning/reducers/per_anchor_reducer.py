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

    def element_reduction(self, losses, loss_indices, embeddings, labels):
        loss_dict = {
            "loss": {
                "losses": losses,
                "indices": loss_indices,
                "reduction_type": "element",
            }
        }
        return self.reducer(loss_dict, embeddings, labels)

    def tuple_reduction_helper(self, losses, loss_indices, embeddings, labels):
        batch_size = embeddings.shape[0]
        anchors, others = loss_indices

        # Prepare tensors for results
        anchors = c_f.to_device(anchors, tensor=losses)
        others = c_f.to_device(others, tensor=losses)
        output = c_f.to_device(torch.zeros(batch_size, batch_size), tensor=losses, dtype=losses.dtype)
        num_per_row = c_f.to_device(torch.zeros(batch_size), tensor=losses, dtype=torch.long)     # Remember to fuse in an unique call to to_device when to_device will accept list inputs

        # Insert loss values in corresponence of anchor-embedding
        output[anchors, others] = losses

        # Calculate the count of 'others' for each unique anchor
        # Equivalent to:    'num_per_row[anchors[i]] += 1'     for every i
        num_per_row = num_per_row.scatter_add_(0, anchors, torch.ones_like(anchors, device=anchors.device))

        # Aggregate results
        output = self.aggregation_func(output, num_per_row)

        loss_dict = {
            "loss": {
                "losses": output,
                "indices": c_f.torch_arange_from_size(embeddings),
                "reduction_type": "element",
            }
        }
        return self.reducer(loss_dict, embeddings, labels)

    def pos_pair_reduction(self, *args, **kwargs):
        return self.tuple_reduction_helper(*args, **kwargs)

    def neg_pair_reduction(self, *args, **kwargs):
        return self.tuple_reduction_helper(*args, **kwargs)

    def triplet_reduction(self, *_):        # Explicitly indicate hyperparameters are ignored
        raise NotImplementedError("Triplet reduction not supported")
