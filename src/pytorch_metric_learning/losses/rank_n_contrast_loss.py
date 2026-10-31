from collections.abc import Callable
from typing import Unpack

import torch
from torch import Tensor

from ..distances import BaseDistance, LpDistance
from ..types import IndicesTuple, LossDict, LossOptions
from ..utils import _common
from .base_metric_loss_function import BaseMetricLossFunction


def _check_targets(targets: Tensor, name: str) -> None:
    if not targets.is_floating_point():
        raise TypeError(f"{name} must be floating-point targets; use .float()")
    if not torch.isfinite(targets).all():
        raise ValueError(f"{name} must be finite")


class RankNContrastLoss(BaseMetricLossFunction):
    supports_mat = True

    def __init__(
        self,
        temperature: float = 2.0,
        label_ndim: int = 1,
        label_distance: Callable[[Tensor, Tensor], Tensor] | None = None,
        **options: Unpack[LossOptions],
    ) -> None:
        super().__init__(**options)
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if label_ndim not in (1, 2):
            raise ValueError("label_ndim must be 1 or 2")
        self.temperature = temperature
        self.label_ndim = label_ndim
        self.label_distance = label_distance

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
        labels = _common.labels_required(labels)
        _common.indices_tuple_not_supported(indices_tuple)
        references = labels if ref_emb is None else _common.labels_required(ref_labels)
        _check_targets(labels, "labels")
        _check_targets(references, "ref_labels")
        rows = len(labels)
        columns = len(references) - int(ref_emb is None)
        if rows == 0 or columns <= 0:
            return {}
        mat = _common.resolve_matrix(self.distance, embeddings, ref_emb, mat)
        sign = 1 if self.distance.is_inverted else -1
        scores = sign * mat.to(_common.accumulation_dtype(mat.dtype))
        separation = self.label_distances(labels, references)
        if ref_emb is None:
            others = ~torch.eye(rows, dtype=torch.bool, device=labels.device)
            scores = scores[others].view(rows, columns)
            separation = separation[others].view(rows, columns)
        separation, order = torch.sort(separation, dim=1)
        scores = scores.gather(1, order) / self.temperature
        tail = torch.logcumsumexp(scores.flip(1), dim=1).flip(1)
        first_tied = torch.searchsorted(separation, separation, side="left")
        losses = (tail.gather(1, first_tied) - scores).mean(dim=1)
        return {
            "loss": {
                "losses": losses.to(mat.dtype),
                "indices": _common.arange_like(losses),
            }
        }

    def label_distances(self, labels: Tensor, references: Tensor) -> Tensor:
        if self.label_distance is not None:
            separation = self.label_distance(labels, references)
            if separation.shape != (len(labels), len(references)):
                raise ValueError(
                    "label_distance must return shape (len(labels), len(references))"
                )
            if not torch.isfinite(separation).all():
                raise ValueError("label_distance must return finite values")
            return separation
        separation = torch.abs(labels.unsqueeze(1) - references.unsqueeze(0))
        return separation if self.label_ndim == 1 else separation.sum(dim=2)

    def default_distance(self) -> BaseDistance:
        return LpDistance(normalize_embeddings=False)
