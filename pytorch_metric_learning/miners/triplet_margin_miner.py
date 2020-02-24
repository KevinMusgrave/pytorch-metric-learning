#! /usr/bin/env python3

from .base_miner import BasePostGradientMiner
from ..utils import loss_and_miner_utils as lmu
import torch


class TripletMarginMiner(BasePostGradientMiner):
    """
    Returns triplets that violate the margin
    Args:
    	margin
    	type_of_triplets: options are "all", "hard", or "semihard".
    		"all" means all triplets that violate the margin
    		"hard" is a subset of "all", but the negative is closer to the anchor than the positive
    		"semihard" is a subset of "all", but the negative is further from the anchor than the positive
    """
    def __init__(self, margin, type_of_triplets="all", **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.add_to_recordable_attributes(list_of_names=["avg_triplet_margin", "pos_pair_dist", "neg_pair_dist"])
        self.type_of_triplets = type_of_triplets
        self.idx_type = "triplet"

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        anchor_idx, positive_idx, negative_idx  = lmu.get_all_triplets_indices(labels, ref_labels)
        anchors, positives, negatives = embeddings[anchor_idx], ref_emb[positive_idx], ref_emb[negative_idx]
        ap_dist = torch.nn.functional.pairwise_distance(anchors, positives, 2)
        an_dist = torch.nn.functional.pairwise_distance(anchors, negatives, 2)
        triplet_margin = ap_dist - an_dist
        self.pos_pair_dist = torch.mean(ap_dist).item()
        self.neg_pair_dist = torch.mean(an_dist).item()
        self.avg_triplet_margin = torch.mean(-triplet_margin).item()
        threshold_condition = triplet_margin > -self.margin
        if self.type_of_triplets == "hard":
        	threshold_condition &= an_dist < ap_dist
        elif self.type_of_triplets == "semihard":
        	threshold_condition &= an_dist > ap_dist
        return anchor_idx[threshold_condition], positive_idx[threshold_condition], negative_idx[threshold_condition]