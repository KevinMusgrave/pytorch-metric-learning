from typing import Unpack

import torch
from torch import Tensor

from ..distances import BaseDistance, CosineSimilarity
from ..types import IndicesTuple, LossDict, LossOptions
from ..utils import _common, indexing
from .base_metric_loss_function import BaseMetricLossFunction


class SelectivelyContrastiveTripletLoss(BaseMetricLossFunction):
    supports_mat = True
    decomposes_by_anchor = True

    def __init__(
        self,
        temperature: float = 0.1,
        hard_weight: float = 1.0,
        hard_threshold: float | None = None,
        **options: Unpack[LossOptions],
    ) -> None:
        super().__init__(**options)
        if not self.distance.is_inverted:
            raise ValueError(
                "SelectivelyContrastiveTripletLoss penalizes similarity directly, so "
                "it requires a similarity such as CosineSimilarity"
            )
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.hard_weight = hard_weight
        self.hard_threshold = hard_threshold

    def compute_loss(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor | None,
        ref_labels: Tensor | None,
        *,
        mat: Tensor | None = None,
    ) -> LossDict:
        _common.labels_or_indices_tuple_required(labels, indices_tuple)
        triplets = indexing.convert_to_triplets(indices_tuple, labels, ref_labels)
        anchor_idx, positive_idx, negative_idx = triplets
        if len(anchor_idx) == 0:
            return {}
        mat = _common.resolve_matrix(self.distance, embeddings, ref_emb, mat)
        s_ap = mat[anchor_idx, positive_idx]
        s_an = mat[anchor_idx, negative_idx]
        hard = s_an > s_ap
        if self.hard_threshold is not None:
            hard = hard | (s_an > self.hard_threshold)
        easy_loss = torch.nn.functional.softplus((s_an - s_ap) / self.temperature)
        loss = torch.where(hard, self.hard_weight * s_an, easy_loss)
        return {"loss": {"losses": loss, "indices": triplets}}

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()
