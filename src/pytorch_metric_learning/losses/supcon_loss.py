from typing import Unpack

from torch import Tensor

from ..distances import BaseDistance, CosineSimilarity
from ..reducers import AvgNonZeroReducer, BaseReducer
from ..types import LossDict, LossOptions, PairIndices
from ..utils import _common
from .generic_pair_loss import GenericPairLoss


class SupConLoss(GenericPairLoss):
    def __init__(
        self, temperature: float = 0.1, **options: Unpack[LossOptions]
    ) -> None:
        super().__init__(**options)
        self.temperature = temperature

    def pair_loss(self, mat: Tensor, pairs: PairIndices) -> LossDict:
        pos_mask, neg_mask = self.pair_masks(mat, pairs)
        if len(pairs[0]) == 0 or len(pairs[2]) == 0:
            return self.zero_losses(mat)
        if not self.distance.is_inverted:
            mat = -mat
        mat = mat / self.temperature
        mat_max, _ = mat.max(dim=1, keepdim=True)
        mat = mat - mat_max.detach()
        denominator = _common.logsumexp(
            mat, keep_mask=pos_mask | neg_mask, add_one=False, dim=1
        )
        log_prob = mat - denominator
        positives = pos_mask.sum(dim=1).clamp_min(1)
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / positives
        return {
            "loss": {
                "losses": -mean_log_prob_pos,
                "indices": _common.arange_like(mat),
                "reduction_type": "element",
            }
        }

    def default_reducer(self) -> BaseReducer:
        return AvgNonZeroReducer()

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()
