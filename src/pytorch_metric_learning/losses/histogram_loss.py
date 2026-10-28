import torch

from ..distances import CosineSimilarity
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction


def filter_pairs(*tensors: torch.Tensor):
    t = torch.stack(tensors)
    t, _ = torch.sort(t, dim=0)
    t = torch.unique(t, dim=1)
    return t.unbind(0)


class HistogramLoss(BaseMetricLossFunction):
    def __init__(self, n_bins: int | None = None, delta: float | None = None, **kwargs):
        super().__init__(**kwargs)
        if delta is not None and n_bins is not None:
            assert delta == 2 / n_bins, (
                "delta and n_bins must satisfy delta = 2 / n_bins; "
                f"got delta={delta} and n_bins={n_bins}"
            )

        if delta is None:
            delta = 2 / (100 if n_bins is None else n_bins)
        self.delta = delta

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.labels_or_indices_tuple_required(labels, indices_tuple)
        c_f.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        indices_tuple = lmu.convert_to_triplets(
            indices_tuple, labels, ref_labels, t_per_anchor="all"
        )
        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            return self.zero_losses()
        mat = self.distance(embeddings, ref_emb)

        anchor_positive_idx = filter_pairs(anchor_idx, positive_idx)
        anchor_negative_idx = filter_pairs(anchor_idx, negative_idx)
        ap_dists = mat[anchor_positive_idx]
        an_dists = mat[anchor_negative_idx]

        p_pos = self.compute_density(ap_dists)
        phi = torch.cumsum(p_pos, dim=0)

        p_neg = self.compute_density(an_dists)
        return {
            "loss": {
                "losses": torch.sum(p_neg * phi),
                "indices": None,
                "reduction_type": "already_reduced",
            }
        }

    def compute_density(self, distances):
        size = distances.size(0)
        r_star = torch.floor((distances.float() + 1) / self.delta)
        r_star = r_star.to(dtype=torch.long)

        delta_ijr_a = (distances + 1 - r_star * self.delta) / self.delta
        delta_ijr_b = ((r_star + 1) * self.delta - 1 - distances) / self.delta
        delta_ijr_a = delta_ijr_a.to(dtype=distances.dtype)
        delta_ijr_b = delta_ijr_b.to(dtype=distances.dtype)

        density = distances.new_zeros(round(1 + 2 / self.delta))

        density.scatter_add_(0, r_star + 1, delta_ijr_a)
        density.scatter_add_(0, r_star, delta_ijr_b)
        return density / size

    def get_default_distance(self):
        return CosineSimilarity()
