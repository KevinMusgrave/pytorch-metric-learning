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
        self.learn_beta = learn_beta
        self.initialize_beta(beta, num_classes)
        self.triplets_per_anchor = triplets_per_anchor
        self.add_to_recordable_attributes(list_of_names=["beta"])
        
    def compute_loss(self, embeddings, labels, indices_tuple):
        indices_tuple = lmu.convert_to_triplets(indices_tuple, labels, self.triplets_per_anchor)
        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            return self.zero_losses()
        anchors, positives, negatives = embeddings[anchor_idx], embeddings[positive_idx], embeddings[negative_idx]

        beta = self.beta if len(self.beta) == 1 else self.beta[labels[anchor_idx]]
        beta = beta.to(embeddings.device)
        
        d_ap = torch.nn.functional.pairwise_distance(positives, anchors, p=2)
        d_an = torch.nn.functional.pairwise_distance(negatives, anchors, p=2)

        pos_loss = torch.nn.functional.relu(d_ap - beta + self.margin)
        neg_loss = torch.nn.functional.relu(beta - d_an + self.margin)

        num_pos_pairs = (pos_loss > 0.0).nonzero().size(0)
        num_neg_pairs = (neg_loss > 0.0).nonzero().size(0)

        divisor_summands = {"num_pos_pairs": num_pos_pairs, "num_neg_pairs": num_neg_pairs}

        margin_loss = pos_loss + neg_loss

        loss_dict = {"margin_loss": {"losses": margin_loss, "indices": indices_tuple, "reduction_type": "triplet", "divisor_summands": divisor_summands}, 
                    "beta_reg_loss": self.compute_reg_loss(beta, anchor_idx, divisor_summands)}

        return loss_dict

    def compute_reg_loss(self, beta, anchor_idx, divisor_summands):
        if self.learn_beta:
            loss = beta * self.nu
            if len(self.beta) == 1:
                return {"losses": loss, "indices": None, "reduction_type": "already_reduced"}
            else:
                return {"losses": loss, "indices": anchor_idx, "reduction_type": "element", "divisor_summands": divisor_summands}
        return self.zero_loss()

    def sub_loss_names(self):
        return ["margin_loss", "beta_reg_loss"]

    def get_default_reducer(self):
        return DivisorReducer()

    def initialize_beta(self, beta, num_classes):
        self.beta = torch.tensor([float(beta)])
        if num_classes:
            self.beta = torch.ones(num_classes) * self.beta
        if self.learn_beta:
            self.beta = torch.nn.Parameter(self.beta)