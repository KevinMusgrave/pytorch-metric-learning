from typing import Unpack

import torch
from torch import Tensor

from ..distances import BaseDistance, LpDistance
from ..reducers import AvgNonZeroReducer, BaseReducer
from ..types import IndicesTuple, LossDict, LossOptions
from ..utils import _common
from .base_metric_loss_function import BaseMetricLossFunction


class CentroidTripletLoss(BaseMetricLossFunction):
    def __init__(
        self,
        margin: float = 0.05,
        smooth_loss: bool = False,
        **options: Unpack[LossOptions],
    ) -> None:
        super().__init__(**options)
        self.margin = margin
        self.smooth_loss = smooth_loss

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
        vectors = embeddings.to(_common.accumulation_dtype(embeddings.dtype))
        centroids = _common.class_centroids(vectors, labels)
        anchors = torch.where(centroids.has_others)[0]
        if len(anchors) == 0:
            return {}
        queries = vectors[anchors]
        positive = self.distance.pairwise(queries, centroids.leave_one_out[anchors])
        mat = self.distance(queries, centroids.means)
        rows, classes = torch.where(mat.new_ones(mat.shape, dtype=torch.bool))
        negative = classes != centroids.class_index[anchors][rows]
        rows, classes = rows[negative], classes[negative]
        violation = self.distance.margin(positive[rows], mat[rows, classes])
        violation = violation + self.margin
        if self.smooth_loss:
            loss = torch.nn.functional.softplus(violation)
        else:
            loss = torch.relu(violation)
        return {"loss": {"losses": loss, "indices": anchors[rows]}}

    def default_distance(self) -> BaseDistance:
        return LpDistance(normalize_embeddings=False, power=2)

    def default_reducer(self) -> BaseReducer:
        return AvgNonZeroReducer()
