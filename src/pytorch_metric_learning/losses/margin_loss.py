#! /usr/bin/env python3

from .base_metric_loss_function import BaseMetricLossFunction
import torch
from ..utils import loss_and_miner_utils as lmu, common_functions as c_f


class MarginLoss(BaseMetricLossFunction):

    def __init__(self, margin, nu, beta, triplets_per_anchor="all", **kwargs):
        self.margin = margin
        self.nu = nu
        self.beta = beta
        self.triplets_per_anchor = triplets_per_anchor
        self.add_to_recordable_attributes(list_of_names=["num_pos_pairs", "num_neg_pairs", "margin_loss", "beta_reg_loss"])
        super().__init__(**kwargs)

    def compute_loss(self, embeddings, labels, indices_tuple):
        anchor_idx, positive_idx, negative_idx = lmu.convert_to_triplets(indices_tuple, labels, self.triplets_per_anchor)
        if len(anchor_idx) == 0:
            self.num_pos_pairs = 0
            self.num_neg_pairs = 0
            return 0
        anchors, positives, negatives = embeddings[anchor_idx], embeddings[positive_idx], embeddings[negative_idx]
        beta = self.maybe_mask_param(self.beta, labels[anchor_idx])
        self.beta_reg_loss = self.compute_reg_loss(beta)

        d_ap = torch.nn.functional.pairwise_distance(positives, anchors, p=2)
        d_an = torch.nn.functional.pairwise_distance(negatives, anchors, p=2)

        pos_loss = torch.nn.functional.relu(d_ap - beta + self.margin)
        neg_loss = torch.nn.functional.relu(beta - d_an + self.margin)

        self.num_pos_pairs = (pos_loss > 0.0).nonzero().size(0)
        self.num_neg_pairs = (neg_loss > 0.0).nonzero().size(0)

        pair_count = self.num_pos_pairs + self.num_neg_pairs 

        if pair_count > 0:
            self.margin_loss = torch.sum(pos_loss + neg_loss) / pair_count
            self.beta_reg_loss = self.beta_reg_loss / pair_count
        else:
            self.margin_loss, self.beta_reg_loss = 0, 0
            
        return self.margin_loss + self.beta_reg_loss

    def compute_reg_loss(self, beta):
        if self.nu > 0:
            beta_sum = c_f.try_torch_operation(torch.sum, beta)
            return beta_sum * self.nu
        return 0