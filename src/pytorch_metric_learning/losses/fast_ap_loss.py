from typing import Unpack

import torch
from torch import Tensor

from ..distances import BaseDistance, LpDistance
from ..types import IndicesTuple, LossDict, LossOptions
from ..utils import _common, indexing
from .base_metric_loss_function import BaseMetricLossFunction


class FastAPLoss(BaseMetricLossFunction):
    def __init__(self, num_bins: int = 10, **options: Unpack[LossOptions]) -> None:
        super().__init__(**options)
        _common.check_distance_type(self, LpDistance, normalize_embeddings=True, p=2)
        self.num_bins = int(num_bins)
        self.num_edges = self.num_bins + 1

    def compute_loss(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor | None,
        ref_labels: Tensor | None,
    ) -> LossDict:
        labels = _common.labels_required(labels)
        _common.ref_not_supported(ref_emb)
        dtype, device = embeddings.dtype, embeddings.device
        miner_weights = indexing.convert_to_weights(indices_tuple, labels, dtype=dtype)
        matches, diffs = indexing.get_matches_and_diffs(labels)
        I_pos = matches.to(dtype)
        I_neg = diffs.to(dtype)
        N_pos = torch.sum(I_pos, dim=1)
        safe_N = N_pos > 0
        dist_mat = self.distance(embeddings)

        histogram_max = 2**self.distance.power
        histogram_delta = histogram_max / self.num_bins
        mid_points = torch.linspace(
            0.0, histogram_max, steps=self.num_edges, device=device, dtype=dtype
        ).view(-1, 1, 1)
        pulse = torch.relu(1 - torch.abs(dist_mat - mid_points) / histogram_delta)
        pos_hist = torch.t(torch.sum(pulse * I_pos, dim=2))
        neg_hist = torch.t(torch.sum(pulse * I_neg, dim=2))

        total_pos_hist = torch.cumsum(pos_hist, dim=1)
        total_hist = torch.cumsum(pos_hist + neg_hist, dim=1)

        h_pos_product = pos_hist * total_pos_hist
        safe_H = (h_pos_product > 0) & (total_hist > 0)
        safe_total = torch.where(safe_H, total_hist, torch.ones_like(total_hist))
        fast_ap = torch.where(safe_H, h_pos_product / safe_total, 0.0).sum(dim=1)
        fast_ap = fast_ap / torch.where(safe_N, N_pos, torch.ones_like(N_pos))
        loss = (1 - fast_ap) * miner_weights * safe_N
        return {
            "loss": {
                "losses": loss[safe_N],
                "indices": torch.where(safe_N)[0],
            }
        }

    def default_distance(self) -> BaseDistance:
        return LpDistance(power=2)
