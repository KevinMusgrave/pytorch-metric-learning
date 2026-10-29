from typing import Unpack

import torch
from torch import Tensor

from ..distances import BaseDistance, CosineSimilarity
from ..types import LossDict, LossOptions, PairIndices
from .generic_pair_loss import GenericPairLoss


class IntraPairVarianceLoss(GenericPairLoss):
    def __init__(
        self,
        pos_eps: float = 0.01,
        neg_eps: float = 0.01,
        **options: Unpack[LossOptions],
    ) -> None:
        super().__init__(**options)
        self.pos_eps = pos_eps
        self.neg_eps = neg_eps

    def pair_loss(self, mat: Tensor, pairs: PairIndices) -> LossDict:
        pos_pairs, neg_pairs = self.pair_values(mat, pairs)
        a1, p, a2, n = pairs
        inverted = self.distance.is_inverted
        pos_var = self.variance_with_eps(pos_pairs, self.pos_eps, inverted)
        neg_var = self.variance_with_eps(neg_pairs, self.neg_eps, not inverted)
        return {
            "pos_loss": {
                "losses": torch.relu(pos_var) ** 2,
                "indices": (a1, p),
            },
            "neg_loss": {
                "losses": torch.relu(neg_var) ** 2,
                "indices": (a2, n),
            },
        }

    @staticmethod
    def variance_with_eps(
        pairs: Tensor, eps: float, incentivize_increase: bool
    ) -> Tensor:
        if len(pairs) == 0:
            return pairs
        mean = torch.mean(pairs)
        if incentivize_increase:
            return (1 - eps) * mean - pairs
        return pairs - (1 + eps) * mean

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()
