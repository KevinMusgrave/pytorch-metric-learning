import math
from collections.abc import Callable
from typing import Unpack

import torch
import torch.nn.functional as F
from torch import Tensor

from ..distances import BaseDistance, CosineSimilarity
from ..regularizers import BaseRegularizer
from ..types import IndicesTuple, LossDict, LossOptions
from ..utils import _common, indexing
from .base_metric_loss_function import BaseMetricLossFunction, regularization_term


def default_center_init(centers: Tensor) -> None:
    torch.nn.init.kaiming_uniform_(centers, a=math.sqrt(5), mode="fan_out")


class SoftTripleLoss(BaseMetricLossFunction):
    def __init__(
        self,
        num_classes: int,
        embedding_size: int,
        centers_per_class: int = 10,
        la: float = 20,
        gamma: float = 0.1,
        margin: float = 0.01,
        *,
        weight_regularizer: BaseRegularizer | None = None,
        weight_reg_weight: float = 1.0,
        weight_init_func: Callable[[Tensor], object] | None = None,
        **options: Unpack[LossOptions],
    ) -> None:
        super().__init__(**options)
        if not self.distance.is_inverted:
            raise ValueError("distance must be a similarity with is_inverted=True")
        self.weight_regularizer = weight_regularizer
        self.weight_reg_weight = weight_reg_weight
        self.la = la
        self.gamma = 1.0 / gamma
        self.margin = margin
        self.num_classes = num_classes
        self.centers_per_class = centers_per_class
        self.weight = torch.nn.Parameter(
            torch.empty(num_classes * centers_per_class, embedding_size)
        )
        (weight_init_func or default_center_init)(self.weight)

    def compute_loss(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor,
        ref_labels: Tensor | None,
    ) -> LossDict:
        labels = _common.labels_required(labels)
        _common.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        _common.check_parameter_placement(self, self.weight, embeddings)
        dtype = embeddings.dtype
        miner_weights = indexing.convert_to_weights(indices_tuple, labels, dtype=dtype)
        sim_to_classes = self.get_logits(embeddings)
        margin = F.one_hot(labels, self.num_classes).to(dtype) * self.margin
        loss = F.cross_entropy(
            self.la * (sim_to_classes - margin), labels, reduction="none"
        )
        return {
            "loss": {
                "losses": loss * miner_weights,
                "indices": _common.arange_like(embeddings),
                "reduction_type": "element",
            }
        }

    def regularization_terms(self, embeddings: Tensor) -> LossDict:
        terms = super().regularization_terms(embeddings)
        if self.weight_regularizer is not None:
            terms["weight_reg_loss"] = regularization_term(
                self.weight_regularizer, self.weight_reg_weight, self.weight
            )
        return terms

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()

    def get_logits(self, embeddings: Tensor) -> Tensor:
        sim_to_centers = self.distance(embeddings, self.weight)
        sim_to_centers = sim_to_centers.view(
            -1, self.num_classes, self.centers_per_class
        )
        prob = F.softmax(sim_to_centers * self.gamma, dim=2)
        return torch.sum(prob * sim_to_centers, dim=2)
