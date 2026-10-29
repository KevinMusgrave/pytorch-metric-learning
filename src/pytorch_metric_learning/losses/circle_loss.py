from typing import Unpack

import torch
from torch import Tensor

from ..distances import BaseDistance, CosineSimilarity
from ..reducers import AvgNonZeroReducer, BaseReducer
from ..types import LossDict, LossOptions, PairIndices
from ..utils import _common
from .generic_pair_loss import GenericPairLoss


class CircleLoss(GenericPairLoss):
    def __init__(
        self, margin: float = 0.4, gamma: float = 80, **options: Unpack[LossOptions]
    ) -> None:
        super().__init__(**options)
        _common.check_distance_type(self, CosineSimilarity)
        self.margin = margin
        self.gamma = gamma

    def pair_loss(self, mat: Tensor, pairs: PairIndices) -> LossDict:
        pos_mask, neg_mask = self.pair_masks(mat, pairs)
        m = self.margin
        alpha_p = torch.relu(1 + m - mat.detach())
        alpha_n = torch.relu(mat.detach() + m)
        pos_logits = -self.gamma * alpha_p * (mat - (1 - m))
        neg_logits = self.gamma * alpha_n * (mat - m)
        logsumexp_pos = _common.logsumexp(
            pos_logits, keep_mask=pos_mask, add_one=False, dim=1
        )
        logsumexp_neg = _common.logsumexp(
            neg_logits, keep_mask=neg_mask, add_one=False, dim=1
        )
        losses = torch.nn.functional.softplus(logsumexp_pos + logsumexp_neg).squeeze(1)
        active = pos_mask.any(dim=1) & neg_mask.any(dim=1)
        return {
            "loss": {"losses": losses * active, "indices": _common.arange_like(mat)}
        }

    def default_reducer(self) -> BaseReducer:
        return AvgNonZeroReducer()

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()
