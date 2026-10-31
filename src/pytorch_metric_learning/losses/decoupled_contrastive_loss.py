from typing import Unpack

import torch
from torch import Tensor

from ..distances import BaseDistance, CosineSimilarity
from ..types import LossDict, LossOptions, PairIndices
from ..utils import _common
from .generic_pair_loss import GenericPairLoss


class DecoupledContrastiveLoss(GenericPairLoss):
    decomposes_by_anchor = True

    def __init__(
        self,
        temperature: float = 0.1,
        sigma: float | None = None,
        **options: Unpack[LossOptions],
    ) -> None:
        super().__init__(**options)
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if sigma is not None and sigma <= 0:
            raise ValueError("sigma must be positive")
        self.temperature = temperature
        self.sigma = sigma
        self.decomposes_by_anchor = sigma is None

    def pair_loss(self, mat: Tensor, pairs: PairIndices) -> LossDict:
        a1, p, a2, _ = pairs
        pos_pairs, neg_pairs = self.pair_values(mat, pairs)
        dtype = _common.accumulation_dtype(mat.dtype)
        sign = 1 if self.distance.is_inverted else -1
        pos_scores = sign * pos_pairs.to(dtype)
        neg_scores = sign * neg_pairs.to(dtype)
        neg_lse = _common.segment_logsumexp(
            neg_scores / self.temperature, a2, mat.shape[0]
        )[a1]
        keep = torch.isfinite(neg_lse)
        pos_scores = pos_scores[keep]
        attraction = pos_scores / self.temperature
        if self.sigma is not None:
            weights = torch.softmax(pos_scores / self.sigma, dim=0) * len(pos_scores)
            attraction = (2 - weights) * attraction
        losses = neg_lse[keep] - attraction
        return {
            "loss": {"losses": losses.to(mat.dtype), "indices": (a1[keep], p[keep])}
        }

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()
