from torch import Tensor

from ..distances import BaseDistance, CosineSimilarity
from ..regularizers import BaseRegularizer
from ..types import IndicesTuple, LossDict
from ..utils import _common, indexing
from .base_metric_loss_function import BaseMetricLossFunction


class ThresholdConsistentMarginLoss(BaseMetricLossFunction):
    def __init__(
        self,
        lambda_plus: float = 1.0,
        lambda_minus: float = 1.0,
        margin_plus: float = 0.9,
        margin_minus: float = 0.5,
        *,
        distance: BaseDistance | None = None,
        embedding_regularizer: BaseRegularizer | None = None,
        embedding_reg_weight: float = 1.0,
    ) -> None:
        super().__init__(
            distance=distance,
            embedding_regularizer=embedding_regularizer,
            embedding_reg_weight=embedding_reg_weight,
        )
        if not self.distance.is_inverted:
            raise TypeError(
                "ThresholdConsistentMarginLoss requires a similarity with "
                "is_inverted=True"
            )
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
        dtype = _common.accumulation_dtype(embeddings.dtype)
        pos_sims = self.distance.pairwise(embeddings[ap], reference[p]).to(dtype)
        neg_sims = self.distance.pairwise(embeddings[an], reference[n]).to(dtype)

        s_lte_m = pos_sims <= self.margin_plus
        tcm_pos_num = ((self.margin_plus - pos_sims) * s_lte_m).sum()
        pos_tcm = tcm_pos_num / s_lte_m.sum().clamp_min(1)

        s_gte_m = neg_sims >= self.margin_minus
        tcm_neg_num = ((neg_sims - self.margin_minus) * s_gte_m).sum()
        neg_tcm = tcm_neg_num / s_gte_m.sum().clamp_min(1)

        tcm_loss = self.lambda_plus * pos_tcm + self.lambda_minus * neg_tcm
        return {
            "loss": {
                "losses": tcm_loss.to(embeddings.dtype),
                "indices": None,
            }
        }
