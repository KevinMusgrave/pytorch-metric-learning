#! /usr/bin/env python3
from .base_miner import BasePostGradientMiner
import torch
from ..utils import loss_and_miner_utils as lmu


# mining method used in Hard Aware Deeply Cascaded Embeddings
# https://arxiv.org/abs/1611.05720
class HDCMiner(BasePostGradientMiner):
    def __init__(self, filter_amounts, use_sim_mat=False, **kwargs):
        super().__init__(**kwargs)
        self.num_pos_pairs_per_round = torch.zeros(len(filter_amounts))
        self.num_neg_pairs_per_round = torch.zeros(len(filter_amounts))
        self.record_these = ["num_pos_pairs_per_round", "num_neg_pairs_per_round"]
        self.filter_amounts = filter_amounts
        self.use_sim_mat = use_sim_mat
        self.i = 0
        self.reset_prev_idx()

    def mine(self, embeddings, labels):
        if self.i == 0:
            self.num_pos_pairs_per_round *= 0
            self.num_neg_pairs_per_round *= 0
            if self.use_sim_mat:
                self.sim_mat = lmu.sim_mat(embeddings)
            else:
                self.dist_mat = lmu.dist_mat(embeddings, squared=False)
            self.a1_idx, self.p_idx, self.a2_idx, self.n_idx = lmu.get_all_pairs_indices(
                labels
            )
            self.reset_prev_idx()

        self.maybe_set_to_prev()
        curr_filter = self.filter_amounts[self.i]
        if curr_filter != 1:
            mat = self.sim_mat if self.use_sim_mat else self.dist_mat
            pos_pair_ = mat[self.a1_idx, self.p_idx]
            neg_pair_ = mat[self.a2_idx, self.n_idx]

            a1, p, a2, n = [], [], [], []

            for name, v in {"pos": pos_pair_, "neg": neg_pair_}.items():
                num_pairs = len(v)
                k = int(curr_filter * num_pairs)
                largest = self.should_select_largest(name)
                _, idx = torch.topk(v, k=k, largest=largest)
                self.append_original_indices(name, idx, a1, p, a2, n)

            self.a1_idx = torch.cat(a1)
            self.p_idx = torch.cat(p)
            self.a2_idx = torch.cat(a2)
            self.n_idx = torch.cat(n)

        self.num_pos_pairs_per_round[self.i] = len(self.a1_idx)
        self.num_neg_pairs_per_round[self.i] = len(self.a2_idx)
        self.set_prev_idx()
        self.i = (self.i + 1) % len(self.filter_amounts)
        return self.a1_idx, self.p_idx, self.a2_idx, self.n_idx

    def should_select_largest(self, name):
        if self.use_sim_mat:
            return False if name == "pos" else True
        return True if name == "pos" else False

    def append_original_indices(self, name, idx, a1, p, a2, n):
        if name == "pos":
            a1.append(self.a1_idx[idx])
            p.append(self.p_idx[idx])
        else:
            a2.append(self.a2_idx[idx])
            n.append(self.n_idx[idx])

    def maybe_set_to_prev(self):
        if self.prev_a1 is not None:
            self.a1_idx = self.prev_a1
            self.p_idx = self.prev_p
            self.a2_idx = self.prev_a2
            self.n_idx = self.prev_n

    def reset_prev_idx(self):
        self.prev_a1 = None
        self.prev_p = None
        self.prev_a2 = None
        self.prev_n = None

    def set_prev_idx(self, reset=False):
        self.prev_a1 = self.a1_idx.clone()
        self.prev_p = self.p_idx.clone()
        self.prev_a2 = self.a2_idx.clone()
        self.prev_n = self.n_idx.clone()
