from .base_miner import BaseTupleMiner
import torch
from ..utils import loss_and_miner_utils as lmu, common_functions as c_f


class BatchEasyHardMiner(BaseTupleMiner):
    HARD = "hard"
    SEMIHARD = "semihard"
    EASY = "easy"
    ALL = "all"
    all_batch_mining_strategies = [HARD, SEMIHARD, EASY, ALL]

    def __init__(
        self,
        pos_strategy=EASY,
        neg_strategy=SEMIHARD,
        allowed_pos_range=None,
        allowed_neg_range=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        if not (
            pos_strategy in self.all_batch_mining_strategies
            and neg_strategy in self.all_batch_mining_strategies
        ):
            raise ValueError(
                '\npos_strategy must be one of "{0}"\nneg_strategy must be one of "{0}"'.format(
                    '" or "'.join(self.all_batch_mining_strategies)
                )
            )
        if pos_strategy == neg_strategy == self.SEMIHARD:
            raise ValueError('pos_strategy and neg_strategy cannot both be "semihard"')

        if (pos_strategy == self.ALL and neg_strategy == self.SEMIHARD) or (
            neg_strategy == self.ALL and pos_strategy == self.SEMIHARD
        ):
            raise ValueError('"semihard" cannot be used in combination with "all"')

        self.pos_strategy = pos_strategy
        self.neg_strategy = neg_strategy
        self.allowed_pos_range = allowed_pos_range
        self.allowed_neg_range = allowed_neg_range

        self.add_to_recordable_attributes(
            list_of_names=[
                "easiest_triplet",
                "hardest_triplet",
                "easiest_pos_pair",
                "hardest_pos_pair",
                "easiest_neg_pair",
                "hardest_neg_pair",
            ],
            is_stat=True,
        )

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        mat = self.distance(embeddings, ref_emb)
        a1_idx, p_idx, a2_idx, n_idx = lmu.get_all_pairs_indices(labels, ref_labels)
        a = torch.arange(mat.size(0)).to(mat.device)

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
            self.set_stats(positive_dists[a_keep_idx], negative_dists[a_keep_idx])
            a = a[a_keep_idx]
            p = positive_indices[a_keep_idx]
            n = negative_indices[a_keep_idx]
            return a, p, a, n
        elif self.pos_strategy == self.ALL and self.neg_strategy != self.ALL:
            self.set_stats(mat[a1_idx, p_idx], negative_dists[a2n_keep])
            a2 = a[a2n_keep]
            n = negative_indices[a2n_keep]
            return a1_idx, p_idx, a2, n
        elif self.pos_strategy != self.ALL and self.neg_strategy == self.ALL:
            self.set_stats(positive_dists[a1p_keep], mat[a2_idx, n_idx])
            a1 = a[a1p_keep]
            p = positive_indices[a1p_keep]
            return a1, p, a2_idx, n_idx
        else:
            self.set_stats(mat[a1_idx, p_idx], mat[a2_idx, n_idx])
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

    def get_max_per_row(
        self, mat, anchor_idx, other_idx, val_range=None, semihard_thresholds=None
    ):
        mask = torch.zeros_like(mat)
        mask[anchor_idx, other_idx] = 1
        if semihard_thresholds is not None:
            mask[mat >= semihard_thresholds.unsqueeze(1)] = 0
        if val_range is not None:
            mask[(mat > val_range[1]) | (mat < val_range[0])] = 0
        mat_masked = mat * mask
        non_zero_rows = torch.any(mask != 0, dim=1)
        return torch.max(mat_masked, dim=1), non_zero_rows

    def get_min_per_row(
        self, mat, anchor_idx, other_idx, val_range=None, semihard_thresholds=None
    ):
        pos_inf = c_f.pos_inf(mat.dtype)
        mask = torch.ones_like(mat) * pos_inf
        mask[anchor_idx, other_idx] = 1

        if semihard_thresholds is not None:
            mask[mat <= semihard_thresholds.unsqueeze(1)] = pos_inf
        if val_range is not None:
            mask[(mat > val_range[1]) | (mat < val_range[0])] = pos_inf

        non_inf_rows = torch.any(mask != pos_inf, dim=1)
        mat = mat.clone()
        mat[mask == pos_inf] = pos_inf
        return torch.min(mat, dim=1), non_inf_rows

    def set_stats(self, positive_dists, negative_dists):
        if self.collect_stats:
            with torch.no_grad():
                len_pd = len(positive_dists)
                len_pn = len(negative_dists)
                if (
                    len_pd > 0
                    and len_pn > 0
                    and self.ALL not in [self.pos_strategy, self.neg_strategy]
                ):
                    easiest_triplet_func = self.get_func_for_stats(False)
                    hardest_triplet_func = self.get_func_for_stats(True)
                    self.easiest_triplet = easiest_triplet_func(
                        positive_dists - negative_dists
                    ).item()
                    self.hardest_triplet = hardest_triplet_func(
                        positive_dists - negative_dists
                    ).item()
                if len_pd > 0:
                    easy_pos_func = self.get_func_for_stats(False)
                    hard_pos_func = self.get_func_for_stats(True)
                    self.easiest_pos_pair = easy_pos_func(positive_dists).item()
                    self.hardest_pos_pair = hard_pos_func(positive_dists).item()
                if len_pn > 0:
                    easy_neg_func = self.get_func_for_stats(True)
                    hard_neg_func = self.get_func_for_stats(False)
                    self.easiest_neg_pair = easy_neg_func(negative_dists).item()
                    self.hardest_neg_pair = hard_neg_func(negative_dists).item()

    def get_func_for_stats(self, min_if_inverted):
        if min_if_inverted:
            return torch.min if self.distance.is_inverted else torch.max
        return torch.max if self.distance.is_inverted else torch.min