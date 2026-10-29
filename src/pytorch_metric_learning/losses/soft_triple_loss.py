import math
from typing import Unpack

import torch
import torch.nn.functional as F
from torch import Tensor

from ..distances import BaseDistance, CosineSimilarity
from ..types import IndicesTuple, LossDict, ProxyLossOptions
from ..utils import _common, indexing
from .base_proxy_loss import BaseProxyLoss


class SoftTripleLoss(BaseProxyLoss):
    def __init__(
        self,
        num_classes: int,
        embedding_size: int,
        centers_per_class: int = 10,
        la: float = 20,
        gamma: float = 0.1,
        margin: float = 0.01,
        **options: Unpack[ProxyLossOptions],
    ) -> None:
        super().__init__(num_classes * centers_per_class, embedding_size, **options)
        if not self.distance.is_inverted:
            raise TypeError("distance must be a similarity with is_inverted=True")
        self.la = la
        self.gamma = gamma
        self.margin = margin
        self.num_classes = num_classes
        self.centers_per_class = centers_per_class

    def default_weight_init(self, weight: Tensor) -> None:
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5), mode="fan_out")

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
        _common.check_parameter_placement(self, self.weight, embeddings)
        sim_to_classes = self.get_logits(embeddings)
        dtype = sim_to_classes.dtype
        weights = indexing.convert_to_weights(indices_tuple, labels, dtype=dtype)
        margin = F.one_hot(labels, self.num_classes).to(dtype) * self.margin
        loss = F.cross_entropy(
            self.la * (sim_to_classes - margin), labels, reduction="none"
        )
        return {
            "loss": {
                "losses": loss * weights,
                "indices": _common.arange_like(embeddings),
            }
        }

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()

    def get_logits(self, embeddings: Tensor) -> Tensor:
        sim_to_centers = self.distance(embeddings, self.weight)
        sim_to_centers = sim_to_centers.view(
            -1, self.num_classes, self.centers_per_class
        )
        prob = F.softmax(sim_to_centers / self.gamma, dim=2)
        return torch.sum(prob * sim_to_centers, dim=2)
