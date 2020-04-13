#! /usr/bin/env python3

from .base_metric_loss_function import BaseMetricLossFunction
import torch 
import torch.nn.functional as F 
from ..utils import loss_and_miner_utils as lmu 


class CircleLoss(BaseMetricLossFunction):
    """
    Circle loss for pairwise labels only. Support for class-level labels will be added 
    in the future.
    
    Args:
    m:  The relaxation factor that controls the radious of the decision boundary.
    gamma: The scale factor that determines the largest scale of each similarity score.

    According to the paper, the suggested default values of m and gamma are:

    Face Recognition: m = 0.25, gamma = 256
    Person Reidentification: m = 0.25, gamma = 256
    Fine-grained Image Retrieval: m = 0.4, gamma = 80

    By default, we set m = 0.4 and gamma = 80
    """
    def __init__(
        self, 
        m=0.4,
        gamma=80,
        triplets_per_anchor='all',
        **kwargs
    ):
        super(CircleLoss, self).__init__(**kwargs)
        self.m = m 
        self.gamma = gamma 
        self.triplets_per_anchor = triplets_per_anchor
        self.add_to_recordable_attributes(list_of_names=["num_unique_anchors", "num_triplets"])
        self.soft_plus = torch.nn.Softplus(beta=1)

        assert self.normalize_embeddings, "Embeddings must be normalized in circle loss!"

    def compute_loss(self, embeddings, labels, indices_tuple):
        indices_tuple = lmu.convert_to_triplets(indices_tuple, labels, t_per_anchor=self.triplets_per_anchor)
        anchor_idx, positive_idx, negative_idx = indices_tuple 
        self.num_triplets = len(anchor_idx)
        if self.num_triplets == 0:
            self.num_unique_anchors = 0
            self.num_triplets = 0
            return 0
        anchors, positives, negatives = embeddings[anchor_idx], embeddings[positive_idx], embeddings[negative_idx]
        
        # compute cosine similarities
        # since embeddings are normalized, we only need to compute dot product 
        sp = torch.sum(anchors * positives, dim=1)
        sn = torch.sum(anchors * negatives, dim=1)
        
        # compute some constants
        loss = 0.
        op = 1 + self.m 
        on = -self.m
        delta_p = 1-self.m 
        delta_n = self.m 

        # find unique anchor index 
        # for each unique anchor index, we have (sp1, sp2, ..., spK) (sn1, sn2, ..., snL)
        unique_anchor_idx = torch.unique(anchor_idx)
        self.num_unique_anchors = len(unique_anchor_idx)

        for anchor in unique_anchor_idx:
            mask = anchor_idx == anchor 
            sp_for_this_anchor = sp[mask]
            sn_for_this_anchor = sn[mask]
            alpha_p = torch.clamp(op - sp_for_this_anchor.detach(), min=0.)
            alpha_n = torch.clamp(sn_for_this_anchor.detach() - on, min=0.)

            logit_p = -self.gamma * alpha_p * (sp_for_this_anchor - delta_p)
            logit_n = self.gamma * alpha_n * (sn_for_this_anchor - delta_n)

            loss_for_this_anchor = self.soft_plus(
                torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
            loss += loss_for_this_anchor
        
        loss /= len(unique_anchor_idx)
        return loss






