from typing import ClassVar, Unpack

import torch
from torch import Tensor

from ..distances import BaseDistance, CosineSimilarity
from ..reducers import AvgNonZeroReducer, BaseReducer
from ..types import IndicesTuple, LossDict, LossOptions, PairIndices
from ..utils import _common, indexing
from .base_metric_loss_function import BaseMetricLossFunction


def _check_label_sets(labels: Tensor, ref_labels: Tensor | None) -> None:
    reference = labels if ref_labels is None else ref_labels
    if labels.dtype != torch.bool or reference.dtype != torch.bool:
        raise TypeError("MultiSupConLoss labels must be boolean; use labels.bool()")
    if labels.ndim != 2 or reference.ndim != 2:
        raise ValueError("MultiSupConLoss labels must have shape (batch, classes)")
    if labels.shape[1] != reference.shape[1]:
        raise ValueError("labels and ref_labels must have the same number of classes")


def _jaccard(labels: Tensor, ref_labels: Tensor | None, dtype: torch.dtype) -> Tensor:
    _check_label_sets(labels, ref_labels)
    query = labels.to(dtype)
    reference = query if ref_labels is None else ref_labels.to(dtype)
    intersection = query @ reference.T
    union = query.sum(dim=1, keepdim=True) + reference.sum(dim=1) - intersection
    return torch.where(union > 0, intersection / union, 1)


class MultiSupConLoss(BaseMetricLossFunction):
    supports_mat = True

    label_ndim: ClassVar[int] = 2

    def __init__(
        self,
        temperature: float = 0.1,
        threshold: float = 0.5,
        **options: Unpack[LossOptions],
    ) -> None:
        super().__init__(**options)
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        self.temperature = temperature
        self.threshold = threshold

    def pair_indices(
        self, labels: Tensor, ref_labels: Tensor | None = None
    ) -> PairIndices:
        similarity = _jaccard(labels, ref_labels, torch.float32)
        pos_mask, neg_mask = self._threshold_masks(similarity, ref_labels is None)
        return PairIndices(*torch.where(pos_mask), *torch.where(neg_mask))

    def _threshold_masks(
        self, similarity: Tensor, exclude_diagonal: bool
    ) -> tuple[Tensor, Tensor]:
        pos_mask = similarity >= self.threshold
        neg_mask = ~pos_mask
        if exclude_diagonal:
            pos_mask.fill_diagonal_(False)
        return pos_mask, neg_mask

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
        dtype = _common.accumulation_dtype(embeddings.dtype)
        similarity = _jaccard(labels, ref_labels, dtype)
        if indices_tuple is None:
            pos_mask, neg_mask = self._threshold_masks(similarity, ref_labels is None)
        else:
            pos_mask, neg_mask = indexing._pair_masks(
                indexing.convert_to_pairs(indices_tuple, labels),
                (similarity.shape[0], similarity.shape[1]),
                embeddings.device,
            )
        mat = _common.resolve_matrix(self.distance, embeddings, ref_emb, mat).to(dtype)
        if not self.distance.is_inverted:
            mat = -mat
        mat = mat / self.temperature
        denominator = _common.logsumexp(
            mat, keep_mask=pos_mask | neg_mask, add_one=False, dim=1
        )
        log_prob = mat - denominator
        positives = pos_mask.sum(dim=1)
        rows = torch.where(positives > 0)[0]
        weighted = (similarity * pos_mask * log_prob).sum(dim=1)[rows] / positives[rows]
        return {"loss": {"losses": (-weighted).to(embeddings.dtype), "indices": rows}}

    def default_reducer(self) -> BaseReducer:
        return AvgNonZeroReducer()

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()
