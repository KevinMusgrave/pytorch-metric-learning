import math
from typing import Unpack

import torch
from torch import Tensor

from ..distances import LpDistance
from ..types import IndicesTuple, LossDict, LossOptions
from ..utils import _common, indexing
from .base_metric_loss_function import BaseMetricLossFunction


class AngularLoss(BaseMetricLossFunction):
    def __init__(self, alpha: float = 40, **options: Unpack[LossOptions]) -> None:
        super().__init__(**options)
        _common.check_distance_type(
            self, LpDistance, p=2, power=1, normalize_embeddings=True
        )
        self.alpha = alpha

    def compute_loss(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor,
        ref_labels: Tensor | None,
    ) -> LossDict:
        labels = _common.labels_required(labels)
        if ref_labels is None:
            raise ValueError("ref_labels are required with ref_emb")
        a1, p, a2, _ = indexing.convert_to_pairs(indices_tuple, labels, ref_labels)
        if len(a1) == 0 or len(a2) == 0:
            return self.zero_losses(embeddings)
        anchors = self.distance.normalize(embeddings[a1])
        positives = self.distance.normalize(ref_emb[p])
        keep_mask = labels[a1].unsqueeze(1) != ref_labels.unsqueeze(0)
        sq_tan_alpha = math.tan(math.radians(self.alpha)) ** 2
        ap_dot = torch.sum(anchors * positives, dim=1, keepdim=True)
        negatives = self.distance.normalize(ref_emb)
        ap_matmul_embeddings = (anchors + positives) @ negatives.t()
        final_form = (4 * sq_tan_alpha * ap_matmul_embeddings) - (
            2 * (1 + sq_tan_alpha) * ap_dot
        )
        losses = _common.logsumexp(final_form, keep_mask=keep_mask, add_one=True)
        return {
            "loss": {
                "losses": losses.squeeze(1),
                "indices": (a1, p),
                "reduction_type": "pos_pair",
            }
        }
