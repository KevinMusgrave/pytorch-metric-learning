from typing import Unpack

import torch
from torch import Tensor

from ..distances import BaseDistance, LpDistance
from ..types import IndicesTuple, LossDict, LossOptions
from ..utils import _common, indexing
from .base_metric_loss_function import BaseMetricLossFunction


class NCALoss(BaseMetricLossFunction):
    def __init__(
        self, softmax_scale: float = 1, **options: Unpack[LossOptions]
    ) -> None:
        super().__init__(**options)
        self.softmax_scale = softmax_scale

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
        if len(embeddings) <= 1:
            return self.zero_losses(embeddings)
        return self.nca_computation(
            embeddings, ref_emb, labels, ref_labels, indices_tuple
        )

    def nca_computation(
        self,
        query: Tensor,
        reference: Tensor,
        query_labels: Tensor,
        reference_labels: Tensor,
        indices_tuple: IndicesTuple | None,
    ) -> LossDict:
        dtype = query.dtype
        miner_weights = indexing.convert_to_weights(
            indices_tuple, query_labels, dtype=dtype, using_ref=query is not reference
        )
        mat = self.distance(query, reference)
        if not self.distance.is_inverted:
            mat = -mat
        if query is reference:
            mat.fill_diagonal_(_common.neg_inf(dtype))
        same_labels = (query_labels.unsqueeze(1) == reference_labels.unsqueeze(0)).to(
            dtype=dtype
        )
        exp = torch.nn.functional.softmax(self.softmax_scale * mat, dim=1)
        exp = torch.sum(exp * same_labels, dim=1)
        non_zero = exp != 0
        loss = -torch.log(exp[non_zero]) * miner_weights[non_zero]
        return {
            "loss": {
                "losses": loss,
                "indices": _common.arange_like(query)[non_zero],
                "reduction_type": "element",
            }
        }

    def default_distance(self) -> BaseDistance:
        return LpDistance(power=2)
