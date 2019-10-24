#! /usr/bin/env python3

import torch

from . import generic_pair_loss as gpl


class ContrastiveLoss(gpl.GenericPairLoss):
    """
    Contrastive loss using either distance or similarity.
    Args:
        pos_margin: The distance (or similarity) over (under) which positive pairs will contribute to the loss.
        neg_margin: The distance (or similarity) under (over) which negative pairs will contribute to the loss.  
        use_similarity: If True, will use dot product between vectors instead of euclidean distance
        power: Each pair's loss will be raised to this power.
        avg_non_zero_only: Only pairs that contribute non-zero loss will be used in the final loss. 
    """
    def __init__(
        self,
        pos_margin=0,
        neg_margin=1,
        use_similarity=False,
        power=1,
        avg_non_zero_only=True,
        **kwargs
    ):
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.avg_non_zero_only = avg_non_zero_only
        self.num_non_zero_pos_pairs = 0
        self.num_non_zero_neg_pairs = 0
        self.record_these = ["num_non_zero_pos_pairs", "num_non_zero_neg_pairs"]
        self.power = power
        super().__init__(use_similarity=use_similarity, iterate_through_loss=False, **kwargs)

    def pair_based_loss(
        self,
        pos_pair_dist,
        neg_pair_dist,
        pos_pair_anchor_labels,
        neg_pair_anchor_labels,
    ):
        pos_loss, neg_loss = 0, 0
        self.num_non_zero_pos_pairs, self.num_non_zero_neg_pairs = 0, 0
        if len(pos_pair_dist) > 0:
            pos_loss, self.num_non_zero_pos_pairs = self.mask_margin_and_calculate_loss(
                pos_pair_dist, pos_pair_anchor_labels, "pos"
            )
        if len(neg_pair_dist) > 0:
            neg_loss, self.num_non_zero_neg_pairs = self.mask_margin_and_calculate_loss(
                neg_pair_dist, neg_pair_anchor_labels, "neg"
            )
        return pos_loss + neg_loss

    def mask_margin_and_calculate_loss(self, pair_dists, labels, pos_or_neg):
        loss_calc_func = self.pos_calc if pos_or_neg == "pos" else self.neg_calc
        input_margin = self.pos_margin if pos_or_neg == "pos" else self.neg_margin
        margin = self.maybe_mask_param(input_margin, labels)
        per_pair_loss = loss_calc_func(pair_dists, margin) ** self.power
        num_non_zero_pairs = (per_pair_loss > 0).nonzero().size(0)
        if self.avg_non_zero_only:
            loss = torch.sum(per_pair_loss) / (num_non_zero_pairs + 1e-16)
        else:
            loss = torch.mean(per_pair_loss)
        return loss, num_non_zero_pairs

    def pos_calc(self, pos_pair_dist, margin):
        return (
            torch.nn.functional.relu(margin - pos_pair_dist)
            if self.use_similarity
            else torch.nn.functional.relu(pos_pair_dist - margin)
        )

    def neg_calc(self, neg_pair_dist, margin):
        return (
            torch.nn.functional.relu(neg_pair_dist - margin)
            if self.use_similarity
            else torch.nn.functional.relu(margin - neg_pair_dist)
        )
