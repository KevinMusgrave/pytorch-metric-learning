from typing import Unpack

import torch
from torch import Tensor

from ..reducers import AvgNonZeroReducer, BaseReducer
from ..types import LossDict, LossOptions, PairIndices
from .generic_pair_loss import GenericPairLoss


class ContrastiveLoss(GenericPairLoss):
    decomposes_by_anchor = True

    def __init__(
        self,
        pos_margin: float = 0.0,
        neg_margin: float = 1.0,
        **options: Unpack[LossOptions],
    ) -> None:
        super().__init__(**options)
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin

    def pair_loss(self, mat: Tensor, pairs: PairIndices) -> LossDict:
        pos_pairs, neg_pairs = self.pair_values(mat, pairs)
        a1, p, a2, n = pairs
        return {
            "pos_loss": {
                "losses": torch.relu(self.distance.margin(pos_pairs, self.pos_margin)),
                "indices": (a1, p),
            },
            "neg_loss": {
                "losses": torch.relu(self.distance.margin(self.neg_margin, neg_pairs)),
                "indices": (a2, n),
            },
        }

    def default_reducer(self) -> BaseReducer:
        return AvgNonZeroReducer()
