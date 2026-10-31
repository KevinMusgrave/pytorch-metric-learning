from typing import Unpack

import torch
from torch import Tensor

from ..reducers import BaseReducer, DivisorReducer
from ..types import IndicesTuple, LossDict, LossOptions, TripletsPerAnchor
from ..utils import _common, indexing
from .base_metric_loss_function import BaseMetricLossFunction


class MarginLoss(BaseMetricLossFunction):
    supports_mat = True

    beta: Tensor

    def __init__(
        self,
        margin: float = 0.2,
        nu: float = 0,
        beta: float = 1.2,
        triplets_per_anchor: TripletsPerAnchor = "all",
        learn_beta: bool = False,
        num_classes: int | None = None,
        **options: Unpack[LossOptions],
    ) -> None:
        super().__init__(**options)
        self.margin = margin
        self.nu = nu
        self.learn_beta = learn_beta
        self.triplets_per_anchor = triplets_per_anchor
        initial = torch.full((num_classes or 1,), float(beta))
        if learn_beta:
            self.beta = torch.nn.Parameter(initial)
        else:
            self.register_buffer("beta", initial)

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
        triplets = indexing.convert_to_triplets(
            indices_tuple, labels, ref_labels, self.triplets_per_anchor
        )
        anchor_idx, positive_idx, negative_idx = triplets
        if len(anchor_idx) == 0:
            return {}
        _common.check_parameter_placement(self, self.beta, embeddings)
        if len(self.beta) == 1:
            beta = self.beta
        else:
            labels = _common.labels_required(labels)
            beta = self.beta[labels[anchor_idx]]
        mat = _common.resolve_matrix(self.distance, embeddings, ref_emb, mat)
        d_ap = mat[anchor_idx, positive_idx]
        d_an = mat[anchor_idx, negative_idx]
        pos_loss = torch.relu(self.distance.margin(d_ap, beta) + self.margin)
        neg_loss = torch.relu(self.distance.margin(beta, d_an) + self.margin)
        divisor = torch.sum(pos_loss > 0.0) + torch.sum(neg_loss > 0.0)
        terms: LossDict = {
            "loss": {
                "losses": pos_loss + neg_loss,
                "indices": triplets,
                "divisor": divisor,
            }
        }
        if self.learn_beta and len(self.beta) == 1:
            terms["beta_reg_loss"] = {"losses": beta * self.nu, "indices": None}
        elif self.learn_beta:
            terms["beta_reg_loss"] = {
                "losses": beta * self.nu,
                "indices": anchor_idx,
                "divisor": divisor,
            }
        return terms

    def default_reducer(self) -> BaseReducer:
        return DivisorReducer()
