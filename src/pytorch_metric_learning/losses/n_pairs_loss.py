from typing import Unpack

import torch
from torch import Tensor

from ..distances import BaseDistance, DotProductSimilarity
from ..types import IndicesTuple, LossDict, LossOptions
from ..utils import _common, indexing
from .base_metric_loss_function import BaseMetricLossFunction


class NPairsLoss(BaseMetricLossFunction):
    def __init__(self, **options: Unpack[LossOptions]) -> None:
        super().__init__(**options)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

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
        anchor_idx, positive_idx = indexing._convert_to_pos_pairs_with_unique_labels(
            indices_tuple, labels
        )
        num_pairs = len(anchor_idx)
        if num_pairs == 0:
            return {}
        anchors, positives = embeddings[anchor_idx], embeddings[positive_idx]
        targets = torch.arange(num_pairs, device=embeddings.device)
        sim_mat = self.distance(anchors, positives)
        if not self.distance.is_inverted:
            sim_mat = -sim_mat
        return {
            "loss": {
                "losses": self.cross_entropy(sim_mat, targets),
                "indices": anchor_idx,
            }
        }

    def default_distance(self) -> BaseDistance:
        return DotProductSimilarity()
