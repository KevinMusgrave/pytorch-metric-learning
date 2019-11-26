#! /usr/bin/env python3

from .base_metric_loss_function import BaseMetricLossFunction
from ..utils import loss_and_miner_utils as lmu
import torch

class NCALoss(BaseMetricLossFunction):
    # https://www.cs.toronto.edu/~hinton/absps/nca.pdf
    def compute_loss(self, embeddings, labels, *_):
        return self.nca_computation(embeddings, embeddings, labels, labels)

    def nca_computation(self, query, reference, query_labels, reference_labels):
        x = -lmu.dist_mat(query, reference, squared=True)
        if query is reference:
            diag_idx = torch.arange(query.size(0))
            x[diag_idx, diag_idx] = float('-inf')
        same_labels = (query_labels.unsqueeze(1) == reference_labels.unsqueeze(0)).float()
        exp = torch.nn.functional.softmax(x, dim=1)
        exp = torch.sum(exp * same_labels, dim=1)
        non_zero_prob = torch.masked_select(exp, exp != 0)
        return -torch.mean(torch.log(non_zero_prob))