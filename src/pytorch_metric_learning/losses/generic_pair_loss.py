from torch import Tensor

from ..types import IndicesTuple, LossDict, PairIndices
from ..utils import _common, indexing
from .base_metric_loss_function import BaseMetricLossFunction


class GenericPairLoss(BaseMetricLossFunction):
    supports_mat = True

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
        pairs = indexing.convert_to_pairs(indices_tuple, labels, ref_labels)
        if all(len(x) == 0 for x in pairs):
            return {}
        matrix = _common.resolve_matrix(self.distance, embeddings, ref_emb, mat)
        return self.pair_loss(matrix, pairs)

    def pair_loss(self, mat: Tensor, pairs: PairIndices) -> LossDict:
        raise NotImplementedError

    @staticmethod
    def pair_values(mat: Tensor, pairs: PairIndices) -> tuple[Tensor, Tensor]:
        a1, p, a2, n = pairs
        return mat[a1, p], mat[a2, n]

    @staticmethod
    def pair_masks(mat: Tensor, pairs: PairIndices) -> tuple[Tensor, Tensor]:
        return indexing._pair_masks(pairs, (mat.shape[0], mat.shape[1]), mat.device)
