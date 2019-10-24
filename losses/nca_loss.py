#! /usr/bin/env python3

from . import base_metric_loss_function as bmlf
from ..utils import loss_and_miner_utils as lmu
import torch


class NCALoss(bmlf.BaseMetricLossFunction):
    # modified from https://github.com/microsoft/snca.pytorch/blob/master/lib/NCA.py
    # https://www.cs.toronto.edu/~hinton/absps/nca.pdf
    def compute_loss(self, embeddings, labels, *_):
        return self.nca_computation(embeddings, embeddings, labels, labels)

    def nca_computation(self, query, reference, query_labels, reference_labels):
        query_batch_size = len(query)
        reference_batch_size = len(reference)
        x = lmu.dist_mat(query, reference, squared=True)
        exp = torch.exp(-x)
        
        if query is reference:
            exp = exp - torch.diag(exp.diag())
        repeated_labels = query_labels.view(query_batch_size, 1).repeat(1, reference_batch_size)
        same_labels = (repeated_labels == reference_labels).float()
        
        # sum over all positive neighbors of each anchor
        p = torch.sum(exp * same_labels, dim=1)
        # sum over all neighbors of each anchor (excluding the anchor)
        Z = torch.sum(exp * (1 - same_labels), dim=1)
        prob = p / Z
        non_zero_prob = torch.masked_select(prob, prob != 0)
        
        return -torch.mean(torch.log(non_zero_prob))