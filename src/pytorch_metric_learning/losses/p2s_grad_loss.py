from typing import Unpack

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..types import IndicesTuple, LossDict, LossOptions
from ..utils import _common
from .base_metric_loss_function import BaseMetricLossFunction


class P2SGradLoss(BaseMetricLossFunction):
    def __init__(
        self, num_classes: int, embedding_size: int, **options: Unpack[LossOptions]
    ) -> None:
        super().__init__(**options)
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        weight = torch.empty(num_classes, embedding_size).uniform_(-1, 1)
        self.weight = nn.Parameter(F.normalize(weight, dim=1))

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
        _common.check_parameter_placement(self, self.weight, embeddings)
        with torch.no_grad():
            self.weight.copy_(F.normalize(self.weight, dim=1))
        weight = self.weight.clone()
        cos_theta = F.normalize(embeddings, dim=1) @ weight.t()
        cos_theta = cos_theta.clamp(-1, 1)
        target = F.one_hot(labels, self.num_classes).to(cos_theta.dtype)
        return {
            "loss": {
                "losses": F.mse_loss(cos_theta, target),
                "indices": None,
            }
        }
