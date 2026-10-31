from typing import Unpack

import torch
from torch import Tensor

from ..reducers import AvgNonZeroReducer, BaseReducer
from ..types import IndicesTuple, LossDict, LossOptions, TripletsPerAnchor
from ..utils import _common, indexing
from .base_metric_loss_function import BaseMetricLossFunction


class TripletMarginLoss(BaseMetricLossFunction):
    supports_mat = True

    def __init__(
        self,
        margin: float = 0.2,
        swap: bool = False,
        smooth_loss: bool = False,
        triplets_per_anchor: TripletsPerAnchor = "all",
        **options: Unpack[LossOptions],
    ) -> None:
        super().__init__(**options)
        self.margin = margin
        self.swap = swap
        self.smooth_loss = smooth_loss
        self.triplets_per_anchor = triplets_per_anchor

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
        _common.labels_or_indices_tuple_required(labels, indices_tuple)
        triplets = indexing.convert_to_triplets(
            indices_tuple, labels, ref_labels, t_per_anchor=self.triplets_per_anchor
        )
        anchor_idx, positive_idx, negative_idx = triplets
        if self.swap and mat is not None:
            raise ValueError(
                "swap reads reference-to-reference distances, which mat does not hold"
            )
        if len(anchor_idx) == 0:
            return {}
        mat = _common.resolve_matrix(self.distance, embeddings, ref_emb, mat)
        ap_dists = mat[anchor_idx, positive_idx]
        an_dists = mat[anchor_idx, negative_idx]
        if self.swap:
            reference, _ = _common.resolve_reference(
                embeddings, labels, ref_emb, ref_labels
            )
            pn_dists = self.distance.pairwise(
                reference[positive_idx], reference[negative_idx]
            )
            closest = torch.maximum if self.distance.is_inverted else torch.minimum
            an_dists = closest(an_dists, pn_dists)
        violation = self.distance.margin(ap_dists, an_dists) + self.margin
        if self.smooth_loss:
            loss = torch.nn.functional.softplus(violation)
        else:
            loss = torch.relu(violation)
        return {"loss": {"losses": loss, "indices": triplets}}

    def default_reducer(self) -> BaseReducer:
        return AvgNonZeroReducer()
