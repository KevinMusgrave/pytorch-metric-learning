from typing import Unpack

import torch
from torch import Tensor

from ..distances import BaseDistance, CosineSimilarity
from ..types import LossDict, LossOptions, PairIndices
from ..utils import _common
from .generic_pair_loss import GenericPairLoss


class MultiSimilarityLoss(GenericPairLoss):
    decomposes_by_anchor = True

    def __init__(
        self,
        alpha: float = 2,
        beta: float = 50,
        base: float = 0.5,
        **options: Unpack[LossOptions],
    ) -> None:
        super().__init__(**options)
        self.alpha = alpha
        self.beta = beta
        self.base = base

    def pair_loss(self, mat: Tensor, pairs: PairIndices) -> LossDict:
        pos_mask, neg_mask = self.pair_masks(mat, pairs)
        pos_exp = self.distance.margin(mat, self.base)
        neg_exp = self.distance.margin(self.base, mat)
        pos_loss = (1.0 / self.alpha) * _common.logsumexp(
            self.alpha * pos_exp, keep_mask=pos_mask, add_one=True
        )
        neg_loss = (1.0 / self.beta) * _common.logsumexp(
            self.beta * neg_exp, keep_mask=neg_mask, add_one=True
        )
        losses = (pos_loss + neg_loss).squeeze(1)
        rows = torch.where(pos_mask.any(dim=1) | neg_mask.any(dim=1))[0]
        return {"loss": {"losses": losses[rows], "indices": rows}}

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()
