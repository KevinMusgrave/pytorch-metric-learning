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
        shift = neg_scores.new_full((mat.shape[0],), -torch.inf).scatter_reduce(
            0, a2, neg_scores.detach(), reduce="amax"
        )
        keep = torch.isfinite(shift[a1])
        neg_lse = _common.segment_logsumexp(
            _common.shifted_division(neg_scores, shift[a2], self.temperature),
            a2,
            mat.shape[0],
        )[a1[keep]]
        pos_scores = pos_scores[keep]
        losses = neg_lse - _common.shifted_division(
            pos_scores, shift[a1[keep]], self.temperature
        )
        if self.sigma is not None and len(pos_scores):
            weights = torch.softmax(
                _common.shifted_division(pos_scores, pos_scores.max(), self.sigma),
                dim=0,
            ) * len(pos_scores)
            losses = losses + (weights - 1) * pos_scores / self.temperature
        return {
            "loss": {"losses": losses.to(mat.dtype), "indices": (a1[keep], p[keep])}
        }

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()
