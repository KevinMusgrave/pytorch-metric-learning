#! /usr/bin/env python3

from .base_miner import BasePostGradientMiner
from ..utils import loss_and_miner_utils as lmu
import torch


class TripletMarginMiner(BasePostGradientMiner):
    """
    Returns triplets that violate the margin
    """
    def __init__(self, margin, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.pos_pair_dist = 0
        self.neg_pair_dist = 0 
        self.avg_triplet_margin = 0
        self.record_these += ["avg_triplet_margin", "pos_pair_dist", "neg_pair_dist"]

    def mine(self, embeddings, labels):
        anchor_idx, positive_idx, negative_idx = lmu.get_all_triplets_indices(labels)
        anchors, positives, negatives = embeddings[anchor_idx], embeddings[positive_idx], embeddings[negative_idx]
        ap_dist = torch.nn.functional.pairwise_distance(anchors, positives, 2)
        an_dist = torch.nn.functional.pairwise_distance(anchors, negatives, 2)
        triplet_margin = ap_dist - an_dist
        self.pos_pair_dist = torch.mean(ap_dist).item()
        self.neg_pair_dist = torch.mean(an_dist).item()
        self.avg_triplet_margin = torch.mean(-triplet_margin).item()
        threshold_condition = triplet_margin > -self.margin
        return anchor_idx[threshold_condition], positive_idx[threshold_condition], negative_idx[threshold_condition]