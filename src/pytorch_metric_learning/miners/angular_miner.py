import math

import torch
from torch import Tensor

from ..distances import BaseDistance, LpDistance
from ..types import IndicesTuple, TripletIndices
from ..utils import _common, indexing
from .base_miner import BaseMiner


class AngularMiner(BaseMiner):
    def __init__(
        self, alpha: float = 20, *, distance: BaseDistance | None = None
    ) -> None:
        super().__init__(distance=distance)
        _common.check_distance_type(
            self, LpDistance, p=2, power=1, normalize_embeddings=True
        )
        self.alpha = alpha

    def mine(
        self,
        embeddings: Tensor,
        labels: Tensor,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor | None,
        ref_labels: Tensor | None,
    ) -> IndicesTuple:
        anchor_idx, positive_idx, negative_idx = indexing.convert_to_triplets(
            indices_tuple, labels, ref_labels
        )
        reference, _ = _common.resolve_reference(
            embeddings, labels, ref_emb, ref_labels
        )
        anchors = self.distance.normalize(embeddings[anchor_idx])
        positives = self.distance.normalize(reference[positive_idx])
        negatives = self.distance.normalize(reference[negative_idx])
        centers = (anchors + positives) / 2
        ap_dist = torch.linalg.vector_norm(anchors - positives, dim=1)
        nc_dist = torch.linalg.vector_norm(negatives - centers, dim=1)
        angles = torch.atan(ap_dist / (2 * nc_dist))
        keep = angles > math.radians(self.alpha)
        return TripletIndices(anchor_idx[keep], positive_idx[keep], negative_idx[keep])
