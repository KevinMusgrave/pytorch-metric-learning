#! /usr/bin/env python3

from .base_miner import BaseTupleMiner
from ..utils import loss_and_miner_utils as lmu
import torch


class TripletMarginMiner(BaseTupleMiner):
    """
    Returns triplets that violate the margin
    Args:
    	margin
    	type_of_triplets: options are "all", "hard", or "semihard".
    		"all" means all triplets that violate the margin
    		"hard" is a subset of "all", but the negative is closer to the anchor than the positive
    		"semihard" is a subset of "all", but the negative is further from the anchor than the positive
            "easy" is all triplets that are not in "all"
    """
    def __init__(self, margin, type_of_triplets="all", tol=0, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.add_to_recordable_attributes(name="margin", is_stat=False)
        self.add_to_recordable_attributes(list_of_names=["avg_triplet_margin", "pos_pair_dist", "neg_pair_dist"], is_stat=True)
        self.type_of_triplets = type_of_triplets
        self.tol = tol

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        anchor_idx, positive_idx, negative_idx  = lmu.get_all_triplets_indices(labels, ref_labels)
        mat = self.distance(embeddings, ref_emb)
        ap_dist = mat[anchor_idx, positive_idx]
        an_dist = mat[anchor_idx, negative_idx]
        triplet_margin = an_dist - ap_dist

        self.pos_pair_dist = torch.mean(ap_dist).item()
        self.neg_pair_dist = torch.mean(an_dist).item()
        self.avg_triplet_margin = torch.mean(triplet_margin).item()

        triplet_margin[torch.abs(triplet_margin) < self.tol] = 0

        if self.type_of_triplets == "easy":
            threshold_condition = self.distance.x_greater_than_y(triplet_margin, self.margin, or_equal=False)
        else:
            threshold_condition = self.distance.x_less_than_y(triplet_margin, self.margin, or_equal=True)
            if self.type_of_triplets == "hard":
                threshold_condition &= self.distance.x_less_than_y(triplet_margin, 0, or_equal=True)
            elif self.type_of_triplets == "semihard":
                threshold_condition &= self.distance.x_greater_than_y(triplet_margin, 0, or_equal=False)
        return anchor_idx[threshold_condition], positive_idx[threshold_condition], negative_idx[threshold_condition]