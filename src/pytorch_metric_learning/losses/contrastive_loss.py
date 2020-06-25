#! /usr/bin/env python3

import torch
from .generic_pair_loss import GenericPairLoss
from ..utils import loss_and_miner_utils as lmu
from ..reducers import AvgNonZeroReducer


class ContrastiveLoss(GenericPairLoss):
    """
    Contrastive loss using either distance or similarity.
    Args:
        pos_margin: The distance (or similarity) over (under) which positive pairs will contribute to the loss.
        neg_margin: The distance (or similarity) under (over) which negative pairs will contribute to the loss.  
    """
    def __init__(self, pos_margin=0, neg_margin=1, **kwargs):
        super().__init__(mat_based_loss=False, **kwargs)
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        
    def _compute_loss(self, pos_pair_dist, neg_pair_dist, indices_tuple):
        pos_loss, neg_loss = 0, 0
        if len(pos_pair_dist) > 0:
            pos_loss = self.get_per_pair_loss(pos_pair_dist, "pos")
        if len(neg_pair_dist) > 0:
            neg_loss = self.get_per_pair_loss(neg_pair_dist, "neg")
        pos_pairs = lmu.pos_pairs_from_tuple(indices_tuple)
        neg_pairs = lmu.neg_pairs_from_tuple(indices_tuple)
        return {"pos_loss": {"losses": pos_loss, "indices": pos_pairs, "reduction_type": "pos_pair"}, 
                "neg_loss": {"losses": neg_loss, "indices": neg_pairs, "reduction_type": "neg_pair"}}

    def get_per_pair_loss(self, pair_dists, pos_or_neg):
        loss_calc_func = self.pos_calc if pos_or_neg == "pos" else self.neg_calc
        margin = self.pos_margin if pos_or_neg == "pos" else self.neg_margin
        per_pair_loss = loss_calc_func(pair_dists, margin)
        return per_pair_loss

    def pos_calc(self, pos_pair_dist, margin):
        # how much bigger is pos_pair_dist than the margin
        return torch.nn.functional.relu(self.distance.pos_neg_margin(margin, pos_pair_dist))

    def neg_calc(self, neg_pair_dist, margin):
        # how much bigger is the margin than the neg_pair_dist
        return torch.nn.functional.relu(self.distance.pos_neg_margin(neg_pair_dist, margin))

    def get_default_reducer(self):
        return AvgNonZeroReducer()

    def sub_loss_names(self):
        return ["pos_loss", "neg_loss"]