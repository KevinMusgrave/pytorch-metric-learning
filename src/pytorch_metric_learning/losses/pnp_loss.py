from typing import Literal, Unpack

import torch
from torch import Tensor

from ..distances import BaseDistance, CosineSimilarity
from ..types import IndicesTuple, LossDict, LossOptions
from ..utils import _common, indexing
from .base_metric_loss_function import BaseMetricLossFunction

type PNPVariant = Literal["Ds", "Dq", "Iu", "Ib", "O"]
VARIANTS: tuple[PNPVariant, ...] = ("Ds", "Dq", "Iu", "Ib", "O")


class PNPLoss(BaseMetricLossFunction):
    VARIANTS = VARIANTS

    def __init__(
        self,
        b: float = 2,
        alpha: float = 1,
        anneal: float = 0.01,
        variant: PNPVariant = "O",
        **options: Unpack[LossOptions],
    ) -> None:
        super().__init__(**options)
        _common.check_distance_type(self, CosineSimilarity)
        if variant not in VARIANTS:
            raise ValueError(f"variant={variant} but must be one of {VARIANTS}")
        self.b = b
        self.alpha = alpha
        self.anneal = anneal
        self.variant = variant

    def compute_loss(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor,
        ref_labels: Tensor | None,
    ) -> LossDict:
        _common.indices_tuple_not_supported(indices_tuple)
        labels = _common.labels_required(labels)
        _common.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        dtype = embeddings.dtype
        matches, diffs = indexing.get_matches_and_diffs(labels)
        I_pos = matches.to(dtype)
        I_pos.fill_diagonal_(1)
        I_neg = diffs.to(dtype)
        N_pos = torch.sum(I_pos, dim=1)
        sim_all = self.distance(embeddings)

        sim_diff = sim_all.unsqueeze(1) - sim_all.unsqueeze(2)
        sim_sg = self.sigmoid(sim_diff, temp=self.anneal) * I_neg.unsqueeze(0)
        sim_all_rk = torch.sum(sim_sg, dim=-1)

        if self.variant == "Ds":
            sim_all_rk = torch.log(1 + sim_all_rk)
        elif self.variant == "Dq":
            sim_all_rk = 1 / (1 + sim_all_rk) ** (self.alpha)
        elif self.variant == "Iu":
            sim_all_rk = (1 + sim_all_rk) * torch.log(1 + sim_all_rk)
        elif self.variant == "Ib":
            b = self.b
            sim_all_rk = 1 / b**2 * (b * sim_all_rk - torch.log(1 + b * sim_all_rk))

        loss = torch.sum(sim_all_rk * I_pos, dim=-1) / N_pos
        loss = loss.mean()
        if self.variant == "Dq":
            loss = 1 - loss
        return {
            "loss": {
                "losses": loss,
                "indices": None,
                "reduction_type": "already_reduced",
            }
        }

    @staticmethod
    def sigmoid(tensor: Tensor, temp: float = 1.0) -> Tensor:
        exponent = torch.clamp(-tensor / temp, min=-50, max=50)
        return 1.0 / (1.0 + torch.exp(exponent))

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()
