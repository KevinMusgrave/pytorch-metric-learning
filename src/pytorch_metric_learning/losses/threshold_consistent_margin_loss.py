from typing import Unpack

import torch.nn.functional as F
from torch import Tensor

from ..distances import BaseDistance, CosineSimilarity
from ..types import IndicesTuple, LossDict, LossOptions
from ..utils import _common, indexing
from .base_metric_loss_function import BaseMetricLossFunction


class ThresholdConsistentMarginLoss(BaseMetricLossFunction):
    def __init__(
        self,
        lambda_plus: float = 1.0,
        lambda_minus: float = 1.0,
        margin_plus: float = 0.9,
        margin_minus: float = 0.5,
        **options: Unpack[LossOptions],
    ) -> None:
        super().__init__(**options)
        _common.check_distance_type(self, CosineSimilarity)
        self.lambda_plus = lambda_plus
        self.lambda_minus = lambda_minus
        self.margin_plus = margin_plus
        self.margin_minus = margin_minus

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()

    def compute_loss(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor | None,
        ref_labels: Tensor | None,
    ) -> LossDict:
        _common.labels_or_indices_tuple_required(labels, indices_tuple)
        ap, p, an, n = indexing.convert_to_pairs(indices_tuple, labels, ref_labels)
        reference, _ = _common.resolve_reference(
            embeddings, labels, ref_emb, ref_labels
        )
        pos_sims = F.cosine_similarity(embeddings[ap], reference[p])
        neg_sims = F.cosine_similarity(embeddings[an], reference[n])

        s_lte_m = pos_sims <= self.margin_plus
        tcm_pos_num = ((self.margin_plus - pos_sims) * s_lte_m).sum()
        pos_tcm = tcm_pos_num / s_lte_m.sum().clamp_min(1)

        s_gte_m = neg_sims >= self.margin_minus
        tcm_neg_num = ((neg_sims - self.margin_minus) * s_gte_m).sum()
        neg_tcm = tcm_neg_num / s_gte_m.sum().clamp_min(1)

        tcm_loss = self.lambda_plus * pos_tcm + self.lambda_minus * neg_tcm
        return {
            "loss": {
                "losses": tcm_loss,
                "indices": None,
            }
        }
