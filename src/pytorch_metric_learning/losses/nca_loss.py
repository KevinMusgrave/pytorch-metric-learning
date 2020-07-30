#! /usr/bin/env python3

from .base_metric_loss_function import BaseMetricLossFunction
from ..utils import loss_and_miner_utils as lmu, common_functions as c_f
import torch

class NCALoss(BaseMetricLossFunction):
    def __init__(self, softmax_scale=1, **kwargs):
        super().__init__(**kwargs)
        self.softmax_scale = softmax_scale

    # https://www.cs.toronto.edu/~hinton/absps/nca.pdf
    def compute_loss(self, embeddings, labels, indices_tuple):
        if len(embeddings) <= 1:
            return self.zero_losses()
        return self.nca_computation(embeddings, embeddings, labels, labels, indices_tuple)

    def nca_computation(self, query, reference, query_labels, reference_labels, indices_tuple):
        miner_weights = lmu.convert_to_weights(indices_tuple, query_labels, dtype=query.dtype)
        x = -lmu.dist_mat(query, reference, squared=True)
        if query is reference:
            diag_idx = torch.arange(query.size(0))
            x[diag_idx, diag_idx] = c_f.neg_inf(query.dtype)
        same_labels = (query_labels.unsqueeze(1) == reference_labels.unsqueeze(0)).type(query.dtype)
        exp = torch.nn.functional.softmax(self.softmax_scale*x, dim=1)
        exp = torch.sum(exp * same_labels, dim=1)
        non_zero = exp!=0
        loss = -torch.log(exp[non_zero])*miner_weights[non_zero]
        return {"loss": {"losses": loss, "indices": c_f.torch_arange_from_size(query)[non_zero], "reduction_type": "element"}}