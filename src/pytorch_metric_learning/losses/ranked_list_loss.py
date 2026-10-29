from typing import Unpack

import torch
from torch import Tensor

from ..distances import BaseDistance, LpDistance
from ..types import IndicesTuple, LossDict, LossOptions
from ..utils import _common, indexing
from .base_metric_loss_function import BaseMetricLossFunction


class RankedListLoss(BaseMetricLossFunction):
    def __init__(
        self,
        margin: float,
        neg_temperature: float,
        imbalance: float = 0.5,
        alpha: float | None = None,
        pos_temperature: float = 0,
        **options: Unpack[LossOptions],
    ) -> None:
        super().__init__(**options)
        if not 0 <= imbalance <= 1:
            raise ValueError("Imbalance must be between 0 and 1")
        self.margin = margin
        self.imbalance = imbalance
        self.alpha = alpha if alpha is not None else 1 + margin / 2
        self.pos_temperature = pos_temperature
        self.neg_temperature = neg_temperature

    def compute_loss(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor | None,
        ref_labels: Tensor | None,
    ) -> LossDict:
        labels = _common.labels_required(labels)
        _common.ref_not_supported(ref_emb)
        _common.indices_tuple_not_supported(indices_tuple)
        mat = self.distance(embeddings)
        matches, diffs = indexing._get_matches_and_diffs(labels)
        boundary = self.alpha - self.margin
        positive = matches & (mat > boundary)
        negative = diffs & (mat < self.alpha)
        loss_p = _weighted_mean(mat - boundary, positive, self.pos_temperature)
        loss_n = _weighted_mean(self.alpha - mat, negative, self.neg_temperature)
        losses = (1 - self.imbalance) * loss_p + self.imbalance * loss_n
        return {"loss": {"losses": losses, "indices": _common.arange_like(losses)}}

    def default_distance(self) -> BaseDistance:
        return LpDistance()


def _weighted_mean(violations: Tensor, mask: Tensor, temperature: float) -> Tensor:
    logits = (temperature * violations).masked_fill(
        ~mask, _common.neg_inf(violations.dtype)
    )
    weights = torch.softmax(logits, dim=1) * mask
    return torch.sum(weights * violations, dim=1)
