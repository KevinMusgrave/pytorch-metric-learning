import math
from typing import Unpack

import torch
from torch import Tensor

from ..distances import LpDistance
from ..types import IndicesTuple, LossDict, LossOptions
from ..utils import _common, indexing
from .base_metric_loss_function import BaseMetricLossFunction


class DynamicSoftMarginLoss(BaseMetricLossFunction):
    hist_: Tensor

    def __init__(
        self,
        min_val: float = -2.0,
        num_bins: int = 10,
        momentum: float = 0.01,
        **options: Unpack[LossOptions],
    ) -> None:
        super().__init__(**options)
        _common.check_distance_type(
            self, LpDistance, normalize_embeddings=True, p=2, power=1
        )
        if not math.isfinite(min_val) or min_val >= 0:
            raise ValueError("min_val must be finite and negative")
        if num_bins < 2:
            raise ValueError("num_bins must be at least 2")
        if not 0 <= momentum <= 1:
            raise ValueError("momentum must be between 0 and 1")
        self.min_val = min_val
        self.num_bins = num_bins
        self.delta = 2 * abs(min_val) / (num_bins - 1)
        self.momentum = momentum
        self.register_buffer("hist_", torch.zeros(num_bins))

    def compute_loss(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor | None,
        ref_labels: Tensor | None,
    ) -> LossDict:
        _common.check_parameter_placement(self, self.hist_, embeddings)
        if labels is None and indices_tuple is None:
            differences = self.compute_loss_without_labels(embeddings, ref_emb)
        else:
            differences = self.compute_loss_with_labels(
                embeddings, labels, indices_tuple, ref_emb, ref_labels
            )
        if len(differences) == 0:
            return {}
        self.update_histogram(differences)
        return {
            "loss": {
                "losses": self.weigh_loss(differences).mean(),
                "indices": None,
            }
        }

    def compute_loss_without_labels(
        self, embeddings: Tensor, ref_emb: Tensor | None
    ) -> Tensor:
        mat = self.distance(embeddings, ref_emb)
        rows, columns = mat.shape
        paired_count = min(rows, columns)
        if paired_count == 0 or rows == columns == 1:
            return mat.reshape(-1)[:0]
        negative = mat.masked_fill(
            torch.eye(rows, columns, dtype=torch.bool, device=mat.device), torch.inf
        )
        row_min = negative.min(dim=1).values[:paired_count]
        column_min = negative.min(dim=0).values[:paired_count]
        return mat.diagonal() - torch.minimum(row_min, column_min)

    def compute_loss_with_labels(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor | None,
        ref_labels: Tensor | None,
    ) -> Tensor:
        anchor, positive, negative = indexing.convert_to_triplets(
            indices_tuple, labels, ref_labels, t_per_anchor="all"
        )
        mat = self.distance(embeddings, ref_emb)
        return mat[anchor, positive] - mat[anchor, negative]

    def bin_positions(self, data: Tensor) -> Tensor:
        data = data.to(dtype=self.hist_.dtype)
        return ((data - self.min_val) / self.delta).clamp(0, self.num_bins - 1)

    @torch.no_grad()
    def update_histogram(self, data: Tensor) -> None:
        positions = self.bin_positions(data)
        lower = positions.long()
        upper = (lower + 1).clamp_max(self.num_bins - 1)
        fraction = positions - lower
        histogram = torch.zeros_like(self.hist_)
        histogram.scatter_add_(0, lower, 1 - fraction)
        histogram.scatter_add_(0, upper, fraction)
        histogram /= data.numel()
        momentum = torch.where(self.hist_.sum() > 0, self.momentum, 1.0)
        self.hist_.lerp_(histogram, momentum)

    def weigh_loss(self, data: Tensor) -> Tensor:
        positions = self.bin_positions(data).long()
        return self.hist_.cumsum(0)[positions] * data
