import torch
from torch import Tensor

from ..types import IndicesTuple

from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_miner import BaseMiner


class BatchEasyHardMiner(BaseMiner):
    HARD = "hard"
    SEMIHARD = "semihard"
    EASY = "easy"
    ALL = "all"
    all_batch_mining_strategies = (HARD, SEMIHARD, EASY, ALL)

    def __init__(
        self,
        pos_strategy=EASY,
        neg_strategy=SEMIHARD,
        allowed_pos_range=None,
        allowed_neg_range=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if (
            pos_strategy not in self.all_batch_mining_strategies
            or neg_strategy not in self.all_batch_mining_strategies
        ):
            choices = '" or "'.join(self.all_batch_mining_strategies)
            raise ValueError(
                f'pos_strategy and neg_strategy must each be one of "{choices}"'
            )
        if pos_strategy == neg_strategy == self.SEMIHARD:
            raise ValueError(
                "pos_strategy and neg_strategy cannot both be \"semihard\""
            )

        if (pos_strategy == self.ALL and neg_strategy == self.SEMIHARD) or (
            neg_strategy == self.ALL and pos_strategy == self.SEMIHARD
        ):
            raise ValueError("\"semihard\" cannot be used in combination with \"all\"")

        self.pos_strategy = pos_strategy
        self.neg_strategy = neg_strategy
        self.allowed_pos_range = allowed_pos_range
        self.allowed_neg_range = allowed_neg_range

    def mine(
        self,
        embeddings: Tensor,
        labels: Tensor,
        ref_emb: Tensor,
        ref_labels: Tensor,
    ) -> IndicesTuple:
        mat = self.distance(embeddings, ref_emb)
        a1_idx, p_idx, a2_idx, n_idx = lmu.get_all_pairs_indices(labels, ref_labels)
        a = torch.arange(mat.size(0), device=mat.device)

        if self.pos_strategy == self.SEMIHARD and self.neg_strategy != self.ALL:
            (negative_dists, negative_indices), a2n_keep = self.get_negatives(
                mat, a2_idx, n_idx
            )
            (positive_dists, positive_indices), a1p_keep = self.get_positives(
                mat, a1_idx, p_idx, negative_dists
            )
        elif self.neg_strategy == self.SEMIHARD and self.pos_strategy != self.ALL:
            (positive_dists, positive_indices), a1p_keep = self.get_positives(
                mat, a1_idx, p_idx
            )
            (negative_dists, negative_indices), a2n_keep = self.get_negatives(
                mat, a2_idx, n_idx, positive_dists
            )
        else:
            if self.pos_strategy != self.ALL:
                (positive_dists, positive_indices), a1p_keep = self.get_positives(
                    mat, a1_idx, p_idx
                )
            if self.neg_strategy != self.ALL:
                (negative_dists, negative_indices), a2n_keep = self.get_negatives(
                    mat, a2_idx, n_idx
                )

        if self.ALL not in [self.pos_strategy, self.neg_strategy]:
            a_keep_idx = torch.where(a1p_keep & a2n_keep)
            a = a[a_keep_idx]
            p = positive_indices[a_keep_idx]
            n = negative_indices[a_keep_idx]
            return a, p, a, n
        elif self.pos_strategy == self.ALL and self.neg_strategy != self.ALL:
            a2 = a[a2n_keep]
            n = negative_indices[a2n_keep]
            return a1_idx, p_idx, a2, n
        elif self.pos_strategy != self.ALL and self.neg_strategy == self.ALL:
            a1 = a[a1p_keep]
            p = positive_indices[a1p_keep]
            return a1, p, a2_idx, n_idx
        else:
            return a1_idx, p_idx, a2_idx, n_idx

    def get_positives(self, mat, a1_idx, p_idx, negative_dists=None):
        pos_func = self.get_mine_function(self.pos_strategy)
        return pos_func(mat, a1_idx, p_idx, self.allowed_pos_range, negative_dists)

    def get_negatives(self, mat, a2_idx, n_idx, positive_dists=None):
        neg_func = self.get_mine_function(
            self.EASY if self.neg_strategy in [self.HARD, self.SEMIHARD] else self.HARD
        )
        return neg_func(mat, a2_idx, n_idx, self.allowed_neg_range, positive_dists)

    def get_mine_function(self, strategy):
        if strategy in [self.HARD, self.SEMIHARD]:
            mine_func = (
                self.get_min_per_row
                if self.distance.is_inverted
                else self.get_max_per_row
            )
        elif strategy == self.EASY:
            mine_func = (
                self.get_max_per_row
                if self.distance.is_inverted
                else self.get_min_per_row
            )
        else:
            raise NotImplementedError

        return mine_func

    def get_max_per_row(self, *args, **kwargs):
        return self.get_x_per_row("max", *args, **kwargs)

    def get_min_per_row(self, *args, **kwargs):
        return self.get_x_per_row("min", *args, **kwargs)

    def get_x_per_row(
        self,
        xtype,
        mat,
        anchor_idx,
        other_idx,
        val_range=None,
        semihard_thresholds=None,
    ):
        if xtype not in ['min', 'max']:
            raise ValueError("xtype must be \"min\" or \"max\"")
        inf = c_f.pos_inf(mat.dtype) if xtype == "min" else c_f.neg_inf(mat.dtype)
        mask = torch.ones_like(mat) * inf
        mask[anchor_idx, other_idx] = 1
        if semihard_thresholds is not None:
            if xtype == "min":
                condition = mat <= semihard_thresholds.unsqueeze(1)
            else:
                condition = mat >= semihard_thresholds.unsqueeze(1)
            mask[condition] = inf
        if val_range is not None:
            mask[(mat > val_range[1]) | (mat < val_range[0])] = inf

        non_inf_rows = torch.any(mask != inf, dim=1)
        mat = mat.clone()
        mat[mask == inf] = inf
        dist_fn = torch.min if xtype == "min" else torch.max
        return dist_fn(mat, dim=1), non_inf_rows
