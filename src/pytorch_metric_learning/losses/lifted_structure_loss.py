#! /usr/bin/env python3

import torch

from .generic_pair_loss import GenericPairLoss
from ..utils import loss_and_miner_utils as lmu, common_functions as c_f


class LiftedStructureLoss(GenericPairLoss):
    def __init__(self, neg_margin, pos_margin=0, **kwargs):
        super().__init__(**kwargs, use_similarity=False, mat_based_loss=False)
        self.neg_margin = neg_margin
        self.pos_margin = pos_margin

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, p, a2, _ = indices_tuple

        if len(a1) > 0 and len(a2) > 0:
            pos_pairs = pos_pairs.unsqueeze(1)
            n_per_p = ((a2.unsqueeze(0) == a1.unsqueeze(1)) | (a2.unsqueeze(0) == p.unsqueeze(1))).float()
            neg_pairs = neg_pairs*n_per_p
            keep_mask = (~(n_per_p==0)).float()

            neg_pairs_loss = lmu.logsumexp(self.neg_margin-neg_pairs, keep_mask=keep_mask, add_one=False, dim=1)
            loss_per_pos_pair = neg_pairs_loss + (pos_pairs - self.pos_margin)
            loss_per_pos_pair = torch.relu(loss_per_pos_pair)**2
            loss_per_pos_pair /= 2 # divide by 2 since each positive pair will be counted twice
            return {"loss": {"losses": loss_per_pos_pair, "indices": (a1, p), "reduction_type": "pos_pair"}}
        return self.zero_losses()



class GeneralizedLiftedStructureLoss(GenericPairLoss):
    # The 'generalized' lifted structure loss shown on page 4
    # of the "in defense of triplet loss" paper
    # https://arxiv.org/pdf/1703.07737.pdf
    def __init__(self, neg_margin, pos_margin=0, **kwargs):
        super().__init__(use_similarity=False, mat_based_loss=True, **kwargs)
        self.neg_margin = neg_margin
        self.pos_margin = pos_margin

    def _compute_loss(self, mat, pos_mask, neg_mask):
        pos_loss = lmu.logsumexp(mat - self.pos_margin, keep_mask=pos_mask, add_one=False)
        neg_loss = lmu.logsumexp(self.neg_margin - mat, keep_mask=neg_mask, add_one=False)
        return {"loss": {"losses": torch.relu(pos_loss+neg_loss), "indices": c_f.torch_arange_from_size(mat), "reduction_type": "element"}}
