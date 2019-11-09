#! /usr/bin/env python3

import torch

from .generic_pair_loss import GenericPairLoss
from ..utils import common_functions as c_f

class MultiSimilarityLoss(GenericPairLoss):
    """
    modified from https://github.com/MalongTech/research-ms-loss/
    Args:
        alpha: The exponential weight for positive pairs
        beta: The exponential weight for negative pairs
        base: The shift in the exponent applied to both positive and negative pairs
    """
    def __init__(self, alpha, beta, base=0.5, **kwargs):
        self.alpha = alpha
        self.beta = beta
        self.base = base
        super().__init__(use_similarity=True, iterate_through_loss=True, **kwargs)

    def pair_based_loss(
        self, pos_pairs, neg_pairs, pos_pair_anchor_labels, neg_pair_anchor_labels
    ):
        pos_loss, neg_loss = 0, 0
        if len(pos_pairs) > 0:
            alpha = self.maybe_mask_param(self.alpha, pos_pair_anchor_labels)
            pos_loss = self.exp_loss(pos_pairs, -alpha, 1.0/alpha)
        if len(neg_pairs) > 0:
            beta = self.maybe_mask_param(self.beta, neg_pair_anchor_labels)
            neg_loss = self.exp_loss(neg_pairs, beta, 1.0/beta)
        return pos_loss + neg_loss

    def exp_loss(self, pair, exp_weight, scaler):
        scaler = c_f.try_torch_operation(torch.mean, scaler)
        inside_exp = exp_weight * (pair - self.base)
        return scaler * torch.log(1 + torch.sum(torch.exp(inside_exp)))
