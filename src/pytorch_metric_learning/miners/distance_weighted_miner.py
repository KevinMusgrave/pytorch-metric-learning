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
        self,
        embeddings: Tensor,
        labels: Tensor,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor | None,
        ref_labels: Tensor | None,
    ) -> IndicesTuple:
        d = float(embeddings.size(1))
        mat = torch.clamp(self.distance(embeddings, ref_emb), min=self.cutoff)
        log_weights = (2.0 - d) * torch.log(mat) - ((d - 3) / 2) * torch.log(
            1.0 - 0.25 * (mat**2.0)
        )
        pairs = indexing.convert_to_pairs(indices_tuple, labels, ref_labels)
        pos_mask, neg_mask = indexing.pair_masks(
            pairs, (mat.shape[0], mat.shape[1]), mat.device
        )
        eligible = (
            torch.isfinite(log_weights) & neg_mask & (mat < self.nonzero_loss_cutoff)
        )
        log_weights = torch.where(
            eligible, log_weights, _common.neg_inf(log_weights.dtype)
        )
        row_max = log_weights.max(dim=1, keepdim=True).values
        weights = torch.exp(log_weights - row_max) * eligible
        return indexing.sample_triplets(pos_mask, eligible, weights=weights)
