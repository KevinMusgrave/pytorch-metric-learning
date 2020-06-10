#! /usr/bin/env python3

from .base_metric_loss_function import BaseMetricLossFunction
import torch
from ..utils import loss_and_miner_utils as lmu, common_functions as c_f
from ..reducers import DivisorReducer

class MarginLoss(BaseMetricLossFunction):

    def __init__(self, margin, nu, beta, triplets_per_anchor="all", learn_beta=False, num_classes=None, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.nu = nu
        self.initialize_beta(beta, learn_beta, num_classes)
        self.triplets_per_anchor = triplets_per_anchor
        self.add_to_recordable_attributes(list_of_names=["beta"])
        
    def compute_loss(self, embeddings, labels, indices_tuple):
        indices_tuple = lmu.convert_to_triplets(indices_tuple, labels, self.triplets_per_anchor)
        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            return self.zero_losses()
        anchors, positives, negatives = embeddings[anchor_idx], embeddings[positive_idx], embeddings[negative_idx]

        beta = self.beta[labels[anchor_idx]] if len(self.beta) > 1 else self.beta
        beta_reg_loss = self.compute_reg_loss(beta)

        d_ap = torch.nn.functional.pairwise_distance(positives, anchors, p=2)
        d_an = torch.nn.functional.pairwise_distance(negatives, anchors, p=2)

        pos_loss = torch.nn.functional.relu(d_ap - beta + self.margin)
        neg_loss = torch.nn.functional.relu(beta - d_an + self.margin)

        num_pos_pairs = (pos_loss > 0.0).nonzero().size(0)
        num_neg_pairs = (neg_loss > 0.0).nonzero().size(0)

        divisor_summands = {"num_pos_pairs": num_pos_pairs, "num_neg_pairs": num_neg_pairs}

        margin_loss = pos_loss + neg_loss

        if len(beta) > 1:
            beta_idx = anchor_idx
            beta_reduction_type = "element"
        else:
            beta_idx = None
            beta_reduction_type = "already_reduced"

        loss_dict = {"margin_loss": {"losses": margin_loss, "indices": indices_tuple, "reduction_type": "triplet", "divisor_summands": divisor_summands}, 
                    "beta_reg_loss": {"losses": beta_reg_loss, "indices": beta_idx, "reduction_type": beta_reduction_type, "divisor_summands": divisor_summands}}

        return loss_dict

    def compute_reg_loss(self, beta):
        if self.nu > 0:
            return beta * self.nu
        return 0

    def sub_loss_names(self):
        return ["margin_loss", "beta_reg_loss"]

    def get_default_reducer(self):
        return DivisorReducer()

    def initialize_beta(self, beta, learn_beta, num_classes):
        if not torch.is_tensor(beta):
            self.beta = torch.tensor(beta)
        if self.beta.dim() == 0:
            self.beta = torch.tensor([self.beta])
        if num_classes:
            self.beta = torch.ones(num_classes) * self.beta
        if learn_beta:
            self.beta = torch.nn.Parameter(self.beta)
        self.beta = self.beta.float()
