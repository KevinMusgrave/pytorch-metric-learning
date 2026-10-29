import warnings
from typing import Unpack

import torch
from torch import Tensor

from ..distances import BaseDistance, LpDistance
from ..types import IndicesTuple, LossDict, LossOptions
from ..utils import _common
from .base_metric_loss_function import BaseMetricLossFunction


class RankedListLoss(BaseMetricLossFunction):
    def __init__(
        self,
        margin: float,
        Tn: float,
        imbalance: float = 0.5,
        alpha: float | None = None,
        Tp: float = 0,
        **options: Unpack[LossOptions],
    ) -> None:
        super().__init__(**options)
        if not 0 <= imbalance <= 1:
            raise ValueError("Imbalance must be between 0 and 1")
        if Tp > 5 or Tn > 5:
            warnings.warn(
                "Values of Tp or Tn are too high and may overflow.", stacklevel=2
            )
        self.margin = margin
        self.imbalance = imbalance
        self.alpha = alpha if alpha is not None else 1 + margin / 2
        self.Tp = Tp
        self.Tn = Tn

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
        _common.indices_tuple_not_supported(indices_tuple)

        mat = self.distance(embeddings)
        eye = torch.eye(len(mat), dtype=torch.bool, device=mat.device)
        mat = mat.masked_fill(eye, 0)
        same = labels.unsqueeze(1) == labels.unsqueeze(0)

        negative = ~same & (mat < self.alpha)
        positive = same & ~eye & (mat > (self.alpha - self.margin))
        N_star = torch.where(negative, mat, 0.0)
        P_star = torch.where(positive, mat, 0.0)
        w_p = torch.where(
            P_star > 0, torch.exp(self.Tp * (P_star - (self.alpha - self.margin))), 0.0
        )
        w_n = torch.where(N_star > 0, torch.exp(self.Tn * (self.alpha - N_star)), 0.0)

        loss_P = torch.sum(
            w_p * (P_star - (self.alpha - self.margin)), dim=1
        ) / torch.sum(w_p + 1e-5, dim=1)
        loss_N = torch.sum(w_n * (self.alpha - N_star), dim=1) / torch.sum(
            w_n + 1e-5, dim=1
        )
        loss_RLL = (1 - self.imbalance) * loss_P + self.imbalance * loss_N
        return {
            "loss": {
                "losses": loss_RLL,
                "indices": _common.arange_like(loss_RLL),
                "reduction_type": "element",
            }
        }

    def default_distance(self) -> BaseDistance:
        return LpDistance()
