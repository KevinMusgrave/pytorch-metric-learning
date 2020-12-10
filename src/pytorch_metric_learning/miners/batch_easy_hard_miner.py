from .base_miner import BaseTupleMiner
import torch
from ..utils import loss_and_miner_utils as lmu, common_functions as c_f


# https://github.com/littleredxh/EasyPositiveHardNegative


class BatchEasyHardMiner(BaseTupleMiner):
    HARD = "hard"
    EASY = "easy"
    all_batch_mining_strategies = [HARD, EASY]

    def __init__(
        self,
        pos_strategy=EASY,
        neg_strategy=HARD,
        allowed_pos_range=None,
        allowed_neg_range=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        if not (
            pos_strategy in self.all_batch_mining_strategies
            and neg_strategy in self.all_batch_mining_strategies
        ):
            raise NotImplementedError(
                "pos_strategy and neg_strategy must be in {}".format(
                    self.all_batch_mining_strategies
                )
            )

        self.pos_strategy = pos_strategy
        self.neg_strategy = neg_strategy
        self.allowed_pos_range = allowed_pos_range
        self.allowed_neg_range = allowed_neg_range

        self.add_to_recordable_attributes(
            list_of_names=[
                "triplet_dist",
                "pos_pair_dist",
                "neg_pair_dist",
            ],
            is_stat=True,
        )

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        mat = self.distance(embeddings, ref_emb)
        a1_idx, p_idx, a2_idx, n_idx = lmu.get_all_pairs_indices(labels, ref_labels)
        pos_func = self.get_mine_function(self.pos_strategy)
        neg_func = self.get_mine_function(
            self.EASY if self.neg_strategy == self.HARD else self.HARD
        )

        (positive_dists, positive_indices), a1p_keep = pos_func(
            mat, a1_idx, p_idx, self.allowed_pos_range
        )
        (negative_dists, negative_indices), a2n_keep = neg_func(
            mat, a2_idx, n_idx, self.allowed_neg_range
        )

        a_keep_idx = torch.where(a1p_keep & a2n_keep)
        self.set_stats(positive_dists[a_keep_idx], negative_dists[a_keep_idx])
        a = torch.arange(mat.size(0)).to(positive_indices.device)[a_keep_idx]
        p = positive_indices[a_keep_idx]
        n = negative_indices[a_keep_idx]
        return a, p, n

    def get_mine_function(self, strategy):
        if strategy == self.HARD:
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

    def get_max_per_row(self, mat, anchor_idx, other_idx, val_range=None):
        mask = torch.zeros_like(mat)
        mask[anchor_idx, other_idx] = 1
        if val_range is not None:
            mask[(mat > val_range[1]) | (mat < val_range[0])] = 0
        mat_masked = mat * mask
        non_zero_rows = torch.any(mask != 0, dim=1)
        return torch.max(mat_masked, dim=1), non_zero_rows

    def get_min_per_row(self, mat, anchor_idx, other_idx, val_range=None):
        pos_inf = c_f.pos_inf(mat.dtype)
        mask = torch.ones_like(mat) * pos_inf
        mask[anchor_idx, other_idx] = 1

        if val_range is not None:
            mask[(mat > val_range[1]) | (mat < val_range[0])] = pos_inf

        non_inf_rows = torch.any(mask != pos_inf, dim=1)
        mat = mat.clone()
        mat[mask == pos_inf] = pos_inf
        return torch.min(mat, dim=1), non_inf_rows

    def set_stats(self, positive_dists, negative_dists):
        if self.collect_stats:
            with torch.no_grad():
                triplet_func = self.get_func_for_stats(True)
                pos_func = self.get_func_for_stats(self.pos_strategy == self.HARD)
                neg_func = self.get_func_for_stats(self.neg_strategy == self.EASY)

                len_pd = len(positive_dists)
                len_pn = len(negative_dists)
                if len_pd > 0 and len_pn > 0:
                    self.triplet_dist = triplet_func(
                        positive_dists - negative_dists
                    ).item()
                if len_pd > 0:
                    self.pos_pair_dist = pos_func(positive_dists).item()
                if len_pn > 0:
                    self.neg_pair_dist = neg_func(negative_dists).item()

    def get_func_for_stats(self, min_if_inverted):
        if min_if_inverted:
            return torch.min if self.distance.is_inverted else torch.max
        return torch.max if self.distance.is_inverted else torch.min