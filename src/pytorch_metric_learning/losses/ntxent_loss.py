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
        a1, p, a2, _ = pairs
        pos_pairs, neg_pairs = self.pair_values(mat, pairs)
        dtype = _common.accumulation_dtype(mat.dtype)
        sign = 1 if self.distance.is_inverted else -1
        pos_logits = sign * pos_pairs.to(dtype) / self.temperature
        neg_logits = sign * neg_pairs.to(dtype) / self.temperature
        neg_lse = _common.segment_logsumexp(neg_logits, a2, mat.shape[0])
        losses = torch.logaddexp(pos_logits, neg_lse[a1]) - pos_logits
        return {"loss": {"losses": losses.to(mat.dtype), "indices": (a1, p)}}

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()
