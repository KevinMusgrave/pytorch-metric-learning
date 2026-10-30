import torch
from torch import Tensor

from ..distances import BaseDistance, CosineSimilarity
from ..regularizers import BaseRegularizer
from ..types import IndicesTuple, LossDict
from ..utils import _common, indexing
from .base_metric_loss_function import BaseMetricLossFunction


def _unique_pairs(*tensors: Tensor) -> tuple[Tensor, ...]:
    t = torch.stack(tensors)
    t, _ = torch.sort(t, dim=0)
    t = torch.unique(t, dim=1)
    return t.unbind(0)


class HistogramLoss(BaseMetricLossFunction):
    def __init__(
        self,
        num_bins: int = 100,
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
        if not isinstance(num_bins, int):
            raise TypeError("num_bins must be an integer")
        if num_bins < 1:
            raise ValueError("num_bins must be positive")
        self.num_bins = num_bins
        self.delta = 2 / num_bins

    def compute_loss(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor | None,
        ref_labels: Tensor | None,
    ) -> LossDict:
        _common.labels_or_indices_tuple_required(labels, indices_tuple)
        _common.ref_not_supported(ref_emb)
        anchor_idx, positive_idx, negative_idx = indexing.convert_to_triplets(
            indices_tuple, labels, ref_labels, t_per_anchor="all"
        )
        if len(anchor_idx) == 0:
            return {}
        mat = self.distance(embeddings)
        ap_dists = mat[_unique_pairs(anchor_idx, positive_idx)]
        an_dists = mat[_unique_pairs(anchor_idx, negative_idx)]
        p_pos = self.compute_density(ap_dists)
        phi = torch.cumsum(p_pos, dim=0)
        p_neg = self.compute_density(an_dists)
        return {"loss": {"losses": torch.sum(p_neg * phi), "indices": None}}

    def compute_density(self, distances: Tensor) -> Tensor:
        size = distances.size(0)
        distances = distances.clamp(-1, 1)
        r_star = torch.floor((distances.float() + 1) / self.delta).to(dtype=torch.long)
        r_star = r_star.clamp(0, self.num_bins - 1)
        delta_ijr_a = (distances + 1 - r_star * self.delta) / self.delta
        delta_ijr_b = ((r_star + 1) * self.delta - 1 - distances) / self.delta
        density = distances.new_zeros(self.num_bins + 1)
        density.scatter_add_(0, r_star + 1, delta_ijr_a.to(dtype=distances.dtype))
        density.scatter_add_(0, r_star, delta_ijr_b.to(dtype=distances.dtype))
        return density / size

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()
