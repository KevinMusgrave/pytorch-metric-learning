from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor

from ..distances import BaseDistance
from ..types import IndicesTuple
from ..utils import _common, indexing
from .base_miner import BaseMiner

type MiningStrategy = Literal["hard", "semihard", "easy", "all"]
STRATEGIES = ("hard", "semihard", "easy", "all")


class BatchEasyHardMiner(BaseMiner):
    def __init__(
        self,
        pos_strategy: MiningStrategy = "easy",
        neg_strategy: MiningStrategy = "semihard",
        allowed_pos_range: Sequence[float] | None = None,
        allowed_neg_range: Sequence[float] | None = None,
        *,
        distance: BaseDistance | None = None,
    ) -> None:
        super().__init__(distance=distance)
        if pos_strategy not in STRATEGIES or neg_strategy not in STRATEGIES:
            choices = '" or "'.join(STRATEGIES)
            raise ValueError(
                f'pos_strategy and neg_strategy must each be one of "{choices}"'
            )
        if pos_strategy == neg_strategy == "semihard":
            raise ValueError('pos_strategy and neg_strategy cannot both be "semihard"')
        if "semihard" in (pos_strategy, neg_strategy) and "all" in (
            pos_strategy,
            neg_strategy,
        ):
            raise ValueError('"semihard" cannot be used in combination with "all"')
        self.pos_strategy = pos_strategy
        self.neg_strategy = neg_strategy
        self.allowed_pos_range = allowed_pos_range
        self.allowed_neg_range = allowed_neg_range

    def mine(
        self, embeddings: Tensor, labels: Tensor, ref_emb: Tensor, ref_labels: Tensor
    ) -> IndicesTuple:
        mat = self.distance(embeddings, ref_emb)
        a1_idx, p_idx, a2_idx, n_idx = indexing.get_all_pairs_indices(
            labels, ref_labels
        )
        a = torch.arange(mat.size(0), device=mat.device)
        pos_all, neg_all = self.pos_strategy == "all", self.neg_strategy == "all"

        if self.pos_strategy == "semihard":
            (negative_dists, negative_indices), a2n_keep = self.get_negatives(
                mat, a2_idx, n_idx
            )
            (positive_dists, positive_indices), a1p_keep = self.get_positives(
                mat, a1_idx, p_idx, negative_dists
            )
        elif self.neg_strategy == "semihard":
            (positive_dists, positive_indices), a1p_keep = self.get_positives(
                mat, a1_idx, p_idx
            )
            (negative_dists, negative_indices), a2n_keep = self.get_negatives(
                mat, a2_idx, n_idx, positive_dists
            )
        else:
            if not pos_all:
                (positive_dists, positive_indices), a1p_keep = self.get_positives(
                    mat, a1_idx, p_idx
                )
            if not neg_all:
                (negative_dists, negative_indices), a2n_keep = self.get_negatives(
                    mat, a2_idx, n_idx
                )

        if not pos_all and not neg_all:
            keep = a1p_keep & a2n_keep
            return a[keep], positive_indices[keep], a[keep], negative_indices[keep]
        if pos_all and not neg_all:
            return a1_idx, p_idx, a[a2n_keep], negative_indices[a2n_keep]
        if not pos_all and neg_all:
            return a[a1p_keep], positive_indices[a1p_keep], a2_idx, n_idx
        return a1_idx, p_idx, a2_idx, n_idx

    def get_positives(
        self,
        mat: Tensor,
        a1_idx: Tensor,
        p_idx: Tensor,
        negative_dists: Tensor | None = None,
    ) -> tuple[tuple[Tensor, Tensor], Tensor]:
        hardest = self.pos_strategy in ("hard", "semihard")
        return self.select_per_row(
            mat, a1_idx, p_idx, hardest, self.allowed_pos_range, negative_dists
        )

    def get_negatives(
        self,
        mat: Tensor,
        a2_idx: Tensor,
        n_idx: Tensor,
        positive_dists: Tensor | None = None,
    ) -> tuple[tuple[Tensor, Tensor], Tensor]:
        hardest = self.neg_strategy == "easy"
        return self.select_per_row(
            mat, a2_idx, n_idx, hardest, self.allowed_neg_range, positive_dists
        )

    def select_per_row(
        self,
        mat: Tensor,
        anchor_idx: Tensor,
        other_idx: Tensor,
        farthest: bool,
        val_range: Sequence[float] | None,
        semihard_thresholds: Tensor | None,
    ) -> tuple[tuple[Tensor, Tensor], Tensor]:
        use_max = farthest != self.distance.is_inverted
        inf = _common.neg_inf(mat.dtype) if use_max else _common.pos_inf(mat.dtype)
        candidate = torch.zeros_like(mat, dtype=torch.bool)
        candidate[anchor_idx, other_idx] = True
        if semihard_thresholds is not None:
            threshold = semihard_thresholds.unsqueeze(1)
            candidate &= mat < threshold if use_max else mat > threshold
        if val_range is not None:
            candidate &= (mat >= val_range[0]) & (mat <= val_range[1])
        non_inf_rows = torch.any(candidate, dim=1)
        masked = torch.where(candidate, mat, inf)
        selected = torch.max(masked, dim=1) if use_max else torch.min(masked, dim=1)
        return (selected.values, selected.indices), non_inf_rows
