import math
from typing import Unpack

import torch
from torch import Tensor

from ..distances import BaseDistance, CosineSimilarity
from ..types import GE2EVariant, IndicesTuple, LossDict, LossOptions
from ..utils import _common
from .base_metric_loss_function import BaseMetricLossFunction


class GE2ELoss(BaseMetricLossFunction):
    def __init__(
        self,
        variant: GE2EVariant = "softmax",
        init_w: float = 10.0,
        init_b: float = -5.0,
        **options: Unpack[LossOptions],
    ) -> None:
        super().__init__(**options)
        if variant not in ("softmax", "contrast"):
            raise ValueError('variant must be "softmax" or "contrast"')
        if not self.distance.is_inverted:
            raise ValueError("GE2ELoss requires a similarity such as CosineSimilarity")
        if init_w <= 0:
            raise ValueError("init_w must be positive")
        self.variant = variant
        raw_w = init_w + math.log(-math.expm1(-init_w))
        self.raw_w = torch.nn.Parameter(torch.tensor(raw_w))
        self.b = torch.nn.Parameter(torch.tensor(float(init_b)))

    @property
    def w(self) -> Tensor:
        dtype = _common.accumulation_dtype(self.raw_w.dtype)
        return torch.nn.functional.softplus(self.raw_w.to(dtype))

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
        _common.check_parameter_placement(self, self.raw_w, embeddings)
        vectors = self.distance.maybe_normalize(embeddings)
        centroids = _common.class_centroids(vectors, labels)
        rows = torch.where(centroids.has_others)[0]
        if len(rows) == 0 or len(centroids.means) < 2:
            return {}
        dtype = _common.accumulation_dtype(embeddings.dtype)
        w = self.w.to(dtype)
        b = self.b.to(dtype)
        queries = vectors[rows]
        own_class = centroids.class_index[rows].unsqueeze(1)
        own = self.distance.pairwise(queries, centroids.leave_one_out[rows])
        own_scores = w * own.to(dtype) + b
        scores = w * self.distance(queries, centroids.means).to(dtype) + b
        if self.variant == "softmax":
            scores = scores.scatter(1, own_class, own_scores.unsqueeze(1))
            loss = torch.logsumexp(scores, dim=1) - own_scores
        else:
            others = scores.scatter(1, own_class, -torch.inf)
            hardest = others.amax(dim=1)
            loss = 1 - torch.sigmoid(own_scores) + torch.sigmoid(hardest)
        return {"loss": {"losses": loss.to(embeddings.dtype), "indices": rows}}

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()
