from typing import Unpack

import torch
from torch import Tensor

from ..distances import BaseDistance, LpDistance
from ..types import IndicesTuple, LossDict, LossOptions
from ..utils import _common, indexing
from .base_metric_loss_function import BaseMetricLossFunction


def _nca_terms(
    mat: Tensor,
    labels: Tensor,
    reference_labels: Tensor,
    weights: Tensor,
    scale: float,
    is_inverted: bool,
    exclude_self: bool,
) -> LossDict:
    dtype = _common.accumulation_dtype(mat.dtype)
    logits = mat.to(dtype) * (1 if is_inverted else -1)
    candidates = torch.ones_like(logits, dtype=torch.bool)
    if exclude_self:
        candidates.fill_diagonal_(False)
    same = (labels.unsqueeze(1) == reference_labels.unsqueeze(0)) & candidates
    valid = torch.where(same.any(dim=1))[0]
    if len(valid) == 0:
        return {}
    logits, candidates, same = logits[valid], candidates[valid], same[valid]
    scores = logits if scale >= 0 else -logits
    offset = scores.masked_fill(~candidates, -torch.inf).amax(dim=1, keepdim=True)
    difference = scores - offset
    logits = difference * abs(scale)
    if abs(scale) < 1:
        logits = torch.where(
            torch.isfinite(difference),
            logits,
            scores * abs(scale) - offset * abs(scale),
        )
    all_lse = torch.logsumexp(logits.masked_fill(~candidates, -torch.inf), dim=1)
    pos_lse = torch.logsumexp(logits.masked_fill(~same, -torch.inf), dim=1)
    losses = (all_lse - pos_lse) * weights[valid]
    return {"loss": {"losses": losses.to(mat.dtype), "indices": valid}}


class NCALoss(BaseMetricLossFunction):
    supports_mat = True

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
        ref_emb: Tensor | None,
        ref_labels: Tensor | None,
        *,
        mat: Tensor | None = None,
    ) -> LossDict:
        labels = _common.labels_required(labels)
        _, reference_labels = _common.resolve_reference(
            embeddings, labels, ref_emb, ref_labels
        )
        weights = indexing.convert_to_weights(
            indices_tuple,
            labels,
            dtype=_common.accumulation_dtype(embeddings.dtype),
            using_ref=ref_emb is not None,
        )
        return _nca_terms(
            _common.resolve_matrix(self.distance, embeddings, ref_emb, mat),
            labels,
            _common.labels_required(reference_labels),
            weights,
            self.softmax_scale,
            self.distance.is_inverted,
            exclude_self=ref_emb is None,
        )

    def default_distance(self) -> BaseDistance:
        return LpDistance(power=2)
