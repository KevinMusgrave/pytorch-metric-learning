import torch
from torch import Tensor

from ..distances import BaseDistance, CosineSimilarity
from ..regularizers import BaseRegularizer
from ..types import IndicesTuple, LossDict, PNPVariant
from ..utils import _common, indexing
from .base_metric_loss_function import BaseMetricLossFunction

_VARIANTS: tuple[PNPVariant, ...] = ("Ds", "Dq", "Iu", "Ib", "O")


class PNPLoss(BaseMetricLossFunction):
    supports_mat = True

    def __init__(
        self,
        b: float = 2,
        alpha: float = 1,
        anneal: float = 0.01,
        variant: PNPVariant = "O",
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
        _common.check_distance_type(self, CosineSimilarity)
        if variant not in _VARIANTS:
            raise ValueError(f"variant={variant} but must be one of {_VARIANTS}")
        self.b = b
        self.alpha = alpha
        self.anneal = anneal
        self.variant = variant

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
        _common.indices_tuple_not_supported(indices_tuple)
        labels = _common.labels_required(labels)
        _common.ref_not_supported(ref_emb)
        dtype = embeddings.dtype
        matches, diffs = indexing._get_matches_and_diffs(labels)
        rows = matches.any(dim=1)
        if not rows.any():
            return {}
        I_pos = matches.to(dtype)
        I_pos.fill_diagonal_(1)
        I_neg = diffs.to(dtype)
        N_pos = torch.sum(I_pos, dim=1)
        sim_all = _common.resolve_matrix(self.distance, embeddings, None, mat)

        output_dtype = sim_all.dtype
        sim_all = sim_all.to(_common.accumulation_dtype(output_dtype))
        sim_diff = sim_all.unsqueeze(1) - sim_all.unsqueeze(2)
        sim_sg = torch.sigmoid(sim_diff / self.anneal) * I_neg.unsqueeze(0)
        sim_all_rk = torch.sum(sim_sg, dim=-1)

        if self.variant == "Ds":
            sim_all_rk = torch.log1p(sim_all_rk)
        elif self.variant == "Dq":
            sim_all_rk = -torch.expm1(-self.alpha * torch.log1p(sim_all_rk))
        elif self.variant == "Iu":
            sim_all_rk = (1 + sim_all_rk) * torch.log1p(sim_all_rk)
        elif self.variant == "Ib":
            b = self.b
            sim_all_rk = 1 / b**2 * _x_minus_log1p(b * sim_all_rk)

        loss = (torch.sum(sim_all_rk * I_pos, dim=-1) / N_pos)[rows].mean()
        loss = loss.to(output_dtype)
        return {
            "loss": {
                "losses": loss,
                "indices": None,
            }
        }

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()


def _x_minus_log1p(value: Tensor) -> Tensor:
    small = value.abs() < 1e-3
    x = torch.where(small, value, 0)
    series = x.square() * (0.5 + x * (-1 / 3 + x * (0.25 + x * (-0.2 + x / 6))))
    return torch.where(small, series, value - torch.log1p(value))
