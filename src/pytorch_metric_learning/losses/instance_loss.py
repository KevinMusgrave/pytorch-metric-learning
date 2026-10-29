from typing import Unpack

import torch
import torch.nn.functional as F
from torch import Tensor

from ..distances import BaseDistance, CosineSimilarity
from ..types import IndicesTuple, LossDict, LossOptions
from ..utils import _common
from .base_metric_loss_function import BaseMetricLossFunction


class InstanceLoss(BaseMetricLossFunction):
    def __init__(self, gamma: float = 64, **options: Unpack[LossOptions]) -> None:
        super().__init__(**options)
        _common.check_distance_type(self, CosineSimilarity)
        self.gamma = gamma

    def compute_loss(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor | None,
        ref_labels: Tensor | None,
    ) -> LossDict:
        labels = _common.labels_required(labels)
        _common.indices_tuple_not_supported(indices_tuple)
        _common.ref_not_supported(ref_emb)
        sim1 = self.distance(embeddings) * self.gamma
        _, sim_label = torch.unique(labels, return_inverse=True)
        loss = F.cross_entropy(sim1, sim_label, reduction="none")
        return {
            "loss": {
                "losses": loss,
                "indices": _common.arange_like(embeddings),
            }
        }

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()
