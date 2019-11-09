#! /usr/bin/env python3

from .base_metric_loss_function import BaseMetricLossFunction
import torch
from ..utils import loss_and_miner_utils as lmu, common_functions as c_f


class MarginLoss(BaseMetricLossFunction):

    def __init__(self, margin, nu, beta, **kwargs):
        self.margin = margin
        self.nu = nu
        self.beta = beta
        self.num_pos_pairs = 0
        self.num_neg_pairs = 0
        self.record_these = ["num_pos_pairs", "num_neg_pairs"]
        super().__init__(**kwargs)

    def compute_loss(self, embeddings, labels, indices_tuple):
        anchor_idx, positive_idx, negative_idx = lmu.convert_to_triplets(indices_tuple, labels)
        if len(anchor_idx) == 0:
            self.num_pos_pairs = 0
            self.num_neg_pairs = 0
            return 0
        anchors, positives, negatives = embeddings[anchor_idx], embeddings[positive_idx], embeddings[negative_idx]
        beta = self.maybe_mask_param(self.beta, labels[anchor_idx])
        beta_reg_loss = self.compute_reg_loss(beta)

        d_ap = torch.nn.functional.pairwise_distance(positives, anchors, p=2)
        d_an = torch.nn.functional.pairwise_distance(negatives, anchors, p=2)

        pos_loss = torch.nn.functional.relu(d_ap - beta + self.margin)
        neg_loss = torch.nn.functional.relu(beta - d_an + self.margin)

        self.num_pos_pairs = (pos_loss > 0.0).nonzero().size(0)
        self.num_neg_pairs = (neg_loss > 0.0).nonzero().size(0)

        pair_count = self.num_pos_pairs + self.num_neg_pairs 

        return (torch.sum(pos_loss + neg_loss) + beta_reg_loss) / (pair_count + 1e-16)

    def compute_reg_loss(self, beta):
        if self.nu > 0:
            beta_mean = c_f.try_torch_operation(torch.mean, beta)
            return beta_mean * self.nu
        return 0