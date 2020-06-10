#! /usr/bin/env python3

import torch

from .generic_pair_loss import GenericPairLoss
from ..utils import loss_and_miner_utils as lmu, common_functions as c_f


class GeneralizedLiftedStructureLoss(GenericPairLoss):
    # The 'generalized' lifted structure loss shown on page 4
    # of the "in defense of triplet loss" paper
    # https://arxiv.org/pdf/1703.07737.pdf
    def __init__(self, neg_margin, **kwargs):
        super().__init__(use_similarity=False, mat_based_loss=True, **kwargs)
        self.neg_margin = neg_margin

    def _compute_loss(self, mat, pos_mask, neg_mask):
        pos_loss = lmu.logsumexp(mat, keep_mask=pos_mask, add_one=False)
        neg_loss = lmu.logsumexp(self.neg_margin - mat, keep_mask=neg_mask, add_one=False)
        return {"loss": {"losses": torch.relu(pos_loss+neg_loss), "indices": c_f.torch_arange_from_size(mat), "reduction_type": "element"}}
