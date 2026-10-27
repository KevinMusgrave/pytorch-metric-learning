import math

import torch

from ..utils import loss_and_miner_utils as lmu
from .base_miner import BaseMiner


class HDCMiner(BaseMiner):
    def __init__(self, filter_percentage=0.5, **kwargs):
        super().__init__(**kwargs)
        self.filter_percentage = filter_percentage
        self.reset_idx()

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        mat = self.distance(embeddings, ref_emb)
        if self.external_indices_tuple is None:
            a1, p, a2, n = lmu.get_all_pairs_indices(labels, ref_labels)
        else:
            a1, p, a2, n = self.external_indices_tuple
        a1, p = self.hardest(mat, a1, p, largest=not self.distance.is_inverted)
        a2, n = self.hardest(mat, a2, n, largest=self.distance.is_inverted)
        return a1, p, a2, n

    def hardest(self, mat, anchors, others, largest):
        if len(anchors) == 0:
            return anchors, others
        k = math.ceil(self.filter_percentage * len(anchors))
        _, idx = torch.topk(mat[anchors, others], k=k, largest=largest)
        return anchors[idx], others[idx]

    def set_idx_externally(self, external_indices_tuple, labels):
        self.external_indices_tuple = lmu.convert_to_pairs(
            external_indices_tuple, labels
        )

    def reset_idx(self):
        self.external_indices_tuple = None
