#! /usr/bin/env python3

from .base_metric_loss_function import BaseMetricLossFunction
import torch
import torch.nn.functional as F
from ..utils import loss_and_miner_utils as lmu


class TripletMarginLoss(BaseMetricLossFunction):
    """
    Args:
        margin: The desired difference between the anchor-positive distance and the
                anchor-negative distance.
        distance_norm: The norm used when calculating distance between embeddings
        power: Each pair's loss will be raised to this power.
        swap: Use the positive-negative distance instead of anchor-negative distance,
              if it violates the margin more.
        smooth_loss: Use the log-exp version of the triplet loss
        avg_non_zero_only: Only pairs that contribute non-zero loss will be used in the final loss.
    """
    def __init__(
        self,
        margin=0.05,
        distance_norm=2,
        power=1,
        swap=False,
        smooth_loss=False,
        avg_non_zero_only=True,
        triplets_per_anchor="all",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.margin = margin
        self.distance_norm = distance_norm
        self.power = power
        self.swap = swap
        self.smooth_loss = smooth_loss
        self.avg_non_zero_only = avg_non_zero_only
        self.triplets_per_anchor = triplets_per_anchor
        self.add_to_recordable_attributes(name="num_non_zero_triplets")
        
    def compute_loss(self, embeddings, labels, indices_tuple):
        indices_tuple = lmu.convert_to_triplets(indices_tuple, labels, t_per_anchor=self.triplets_per_anchor)
        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            self.num_non_zero_triplets = 0
            return 0
        anchors, positives, negatives = embeddings[anchor_idx], embeddings[positive_idx], embeddings[negative_idx]
        a_p_dist = F.pairwise_distance(anchors, positives, self.distance_norm)
        a_n_dist = F.pairwise_distance(anchors, negatives, self.distance_norm)
        if self.swap:
            p_n_dist = F.pairwise_distance(positives, negatives, self.distance_norm)
            a_n_dist = torch.min(a_n_dist, p_n_dist)
        a_p_dist = a_p_dist ** self.power
        a_n_dist = a_n_dist ** self.power
        if self.smooth_loss:
            inside_exp = a_p_dist - a_n_dist
            inside_exp = self.maybe_modify_loss(inside_exp)
            return torch.mean(torch.log(1 + torch.exp(inside_exp)))
        else:
            dist = a_p_dist - a_n_dist
            loss_modified = self.maybe_modify_loss(dist + self.margin)
            relued = torch.nn.functional.relu(loss_modified)
            self.num_non_zero_triplets = (relued > 0).nonzero().size(0)
            if self.avg_non_zero_only:
                return torch.sum(relued) / (self.num_non_zero_triplets + 1e-16)
            return torch.mean(relued)

    def maybe_modify_loss(self, x):
        return x

