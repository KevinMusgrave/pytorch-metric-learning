import torch

from ..distances import LpDistance
from ..utils import common_functions as c_f
from .base_metric_loss_function import BaseMetricLossFunction


class RankedListLoss(BaseMetricLossFunction):
    r"""Ranked List Loss described in https://arxiv.org/abs/1903.03238
       Default parameters correspond to RLL-Simpler, preferred for exploratory analysis.

    Args:
        * margin (float): margin between positive and negative set
        * imbalance (float): tradeoff between positive and negative sets. As the name suggests this takes into account
                            the imbalance between positive and negative samples in the dataset
        * alpha (float): smallest distance between negative points
        * Tp & Tn (float): temperatures for, respectively, positive and negative pairs weighting
    """

    def __init__(self, margin, Tn, imbalance=0.5, alpha=None, Tp=0, **kwargs):
        super().__init__(**kwargs)

        self.margin = margin

        assert 0 <= imbalance <= 1, "Imbalance must be between 0 and 1"
        self.imbalance = imbalance

        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = 1 + margin / 2

        self.Tp = Tp
        self.Tn = Tn
        self.add_to_recordable_attributes(
            list_of_names=["imbalance", "alpha", "margin", "Tp", "Tn"], is_stat=True
        )

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.labels_required(labels)
        c_f.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        c_f.indices_tuple_not_supported(indices_tuple)

        mat = self.distance(embeddings, embeddings)
        mat.fill_diagonal_(0)
        y = labels.unsqueeze(1) == labels.unsqueeze(0)

        P_star = torch.zeros_like(mat)
        N_star = torch.zeros_like(mat)
        w_p = torch.zeros_like(mat)
        w_n = torch.zeros_like(mat)

        N_star[(~y) * (mat < self.alpha)] = mat[(~y) * (mat < self.alpha)]
        y.fill_diagonal_(False)
        P_star[y * (mat > (self.alpha - self.margin))] = mat[
            y * (mat > (self.alpha - self.margin))
        ]

        w_p[P_star > self.alpha - self.margin] = torch.exp(
            self.Tp
            * (P_star[P_star > self.alpha - self.margin] - (self.alpha - self.margin))
        )
        w_n[0 < N_star] = torch.exp(self.Tn * (self.alpha - N_star[0 < N_star]))

        loss_P = torch.sum(
            w_p * (P_star - (self.alpha - self.margin)), dim=1
        ) / torch.sum(w_p, dim=1)
        loss_N = torch.sum(w_n * (self.alpha - N_star), dim=1) / torch.sum(w_n, dim=1)
        loss_N[loss_N.isnan()] = 0
        loss_RLL = (1 - self.imbalance) * loss_P + self.imbalance * loss_N

        return {
            "loss": {
                "losses": loss_RLL,
                "indices": c_f.torch_arange_from_size(loss_RLL),
                "reduction_type": "element",
            }
        }

    def get_default_distance(self):
        return LpDistance()
