#! /usr/bin/env python3
from .base_miner import BaseTupleMiner
import torch
from ..utils import loss_and_miner_utils as lmu
import math

# mining method used in Hard Aware Deeply Cascaded Embeddings
# https://arxiv.org/abs/1611.05720
class HDCMiner(BaseTupleMiner):
    def __init__(self, filter_percentage, **kwargs):
        super().__init__(**kwargs)
        self.filter_percentage = filter_percentage
        self.reset_idx()

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        mat = self.distance(embeddings, ref_emb)
        self.set_idx(labels, ref_labels)

        for name, (anchor, other) in {"pos": (self.a1, self.p), "neg": (self.a2, self.n)}.items():
            if len(anchor) > 0:
                pairs = mat[anchor, other]
                num_pairs = len(pairs)
                k = int(math.ceil(self.filter_percentage * num_pairs))
                largest = self.should_select_largest(name)
                _, idx = torch.topk(pairs, k=k, largest=largest)
                self.filter_original_indices(name, idx)

        return self.a1, self.p, self.a2, self.n

    def should_select_largest(self, name):
        if self.distance.is_inverted:
            return False if name == "pos" else True
        return True if name == "pos" else False

    def set_idx(self, labels, ref_labels):
        if not self.was_set_externally:
            self.a1, self.p, self.a2, self.n = lmu.get_all_pairs_indices(labels, ref_labels)

    def set_idx_externally(self, external_indices_tuple, labels):
        self.a1, self.p, self.a2, self.n = lmu.convert_to_pairs(external_indices_tuple, labels)
        self.was_set_externally = True

    def reset_idx(self):
        self.a1, self.p, self.a2, self.n = None, None, None, None
        self.was_set_externally = False

    def filter_original_indices(self, name, idx):
        if name == "pos":
            self.a1 = self.a1[idx]
            self.p = self.p[idx]
        else:
            self.a2 = self.a2[idx]
            self.n = self.n[idx]