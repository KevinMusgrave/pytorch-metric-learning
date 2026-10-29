from typing import Unpack

import torch
from torch import Tensor

from ..distances import BaseDistance, CosineSimilarity
from ..types import LossDict, LossOptions, PairIndices
from ..utils import _common
from .generic_pair_loss import GenericPairLoss


class NTXentLoss(GenericPairLoss):
    def __init__(
        self, temperature: float = 0.07, **options: Unpack[LossOptions]
    ) -> None:
        super().__init__(**options)
        self.temperature = temperature

    def pair_loss(self, mat: Tensor, pairs: PairIndices) -> LossDict:
        pos_pairs, neg_pairs = self.pair_values(mat, pairs)
        a1, p, a2, _ = pairs
        if len(a1) == 0 or len(a2) == 0:
            return self.zero_losses(mat)
        dtype = neg_pairs.dtype
        if not self.distance.is_inverted:
            pos_pairs = -pos_pairs
            neg_pairs = -neg_pairs
        pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
        neg_pairs = neg_pairs / self.temperature
        n_per_p = a2.unsqueeze(0) == a1.unsqueeze(1)
        neg_pairs = torch.where(n_per_p, neg_pairs, _common.neg_inf(dtype))
        max_val = torch.max(
            pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0]
        ).detach()
        numerator = torch.exp(pos_pairs - max_val).squeeze(1)
        denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
        log_exp = torch.log((numerator / denominator) + _common.small_val(dtype))
        return {
            "loss": {
                "losses": -log_exp,
                "indices": (a1, p),
                "reduction_type": "pos_pair",
            }
        }

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()
