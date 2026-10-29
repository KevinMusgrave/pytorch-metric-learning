from typing import Unpack

import torch
import torch.nn.functional as F
from torch import Tensor

from ..distances import BaseDistance, CosineSimilarity
from ..types import IndicesTuple, LossDict, LossOptions
from ..utils import _common, indexing
from .base_metric_loss_function import BaseMetricLossFunction


class SmoothAPLoss(BaseMetricLossFunction):
    def __init__(
        self, temperature: float = 0.01, **options: Unpack[LossOptions]
    ) -> None:
        super().__init__(**options)
        _common.check_distance_type(self, CosineSimilarity)
        self.temperature = temperature

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()

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
        sims = self.distance(embeddings)
        others = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        same = (labels.unsqueeze(0) == labels.unsqueeze(1)).to(sims.dtype)
        differences = sims.unsqueeze(1) - sims.unsqueeze(2)
        above = F.sigmoid(differences / self.temperature) * others
        all_ranks = 1 + above.sum(dim=-1)
        positive_ranks = 1 + (above * same.unsqueeze(1)).sum(dim=-1)
        ap = (positive_ranks / all_ranks * same).sum(dim=1) / same.sum(dim=1)
        miner_weights = indexing.convert_to_weights(
            indices_tuple, labels, dtype=ap.dtype
        )
        loss = (1 - ap) * miner_weights
        return {
            "loss": {
                "losses": loss,
                "indices": _common.arange_like(loss),
            }
        }
