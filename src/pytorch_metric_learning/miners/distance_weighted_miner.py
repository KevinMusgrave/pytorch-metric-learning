import torch
from torch import Tensor

from ..distances import BaseDistance, LpDistance
from ..types import IndicesTuple
from ..utils import _common, indexing
from .base_miner import BaseMiner


class DistanceWeightedMiner(BaseMiner):
    def __init__(
        self,
        cutoff: float = 0.5,
        nonzero_loss_cutoff: float = 1.4,
        *,
        distance: BaseDistance | None = None,
    ) -> None:
        super().__init__(distance=distance)
        _common.check_distance_type(
            self, LpDistance, p=2, power=1, normalize_embeddings=True
        )
        self.cutoff = float(cutoff)
        self.nonzero_loss_cutoff = float(nonzero_loss_cutoff)

    def mine(
        self, embeddings: Tensor, labels: Tensor, ref_emb: Tensor, ref_labels: Tensor
    ) -> IndicesTuple:
        d = float(embeddings.size(1))
        mat = torch.clamp(self.distance(embeddings, ref_emb), min=self.cutoff)
        log_weights = (2.0 - d) * torch.log(mat) - ((d - 3) / 2) * torch.log(
            1.0 - 0.25 * (mat**2.0)
        )
        finite = torch.isfinite(log_weights)
        different_class = labels.unsqueeze(1) != ref_labels.unsqueeze(0)
        log_weights = torch.where(finite, log_weights, _common.neg_inf(mat.dtype))
        weights = torch.exp(log_weights - log_weights.max())
        weights = weights * different_class * (mat < self.nonzero_loss_cutoff)
        weights = weights / torch.sum(weights, dim=1, keepdim=True).clamp_min(
            _common.small_val(weights.dtype)
        )
        return indexing.get_random_triplet_indices(
            labels, ref_labels=ref_labels, weights=weights
        )
