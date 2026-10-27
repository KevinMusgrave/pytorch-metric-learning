import math

import torch

from ..utils import loss_and_miner_utils as lmu
from .base_miner import BaseMiner


class HDCMiner(BaseMiner):
    def __init__(self, filter_percentage=0.5, **kwargs):
        super().__init__(**kwargs)
        self.filter_percentage = filter_percentage

    def forward(
        self, embeddings, labels, ref_emb=None, ref_labels=None, candidates=None
    ):
        with torch.no_grad():
            labels, ref_emb, ref_labels = self.prepare_inputs(
                embeddings, labels, ref_emb, ref_labels
            )
            pairs = lmu.convert_to_pairs(candidates, labels, ref_labels)
            mining_output = self.hardest_pairs(embeddings, ref_emb, pairs)
        self.output_assertion(mining_output)
        return mining_output

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        pairs = lmu.get_all_pairs_indices(labels, ref_labels)
        return self.hardest_pairs(embeddings, ref_emb, pairs)

    def hardest_pairs(self, embeddings, ref_emb, pairs):
        mat = self.distance(embeddings, ref_emb)
        a1, p, a2, n = pairs
        a1, p = self.hardest(mat, a1, p, largest=not self.distance.is_inverted)
        a2, n = self.hardest(mat, a2, n, largest=self.distance.is_inverted)
        return a1, p, a2, n

    def hardest(self, mat, anchors, others, largest):
        if len(anchors) == 0:
            return anchors, others
        k = math.ceil(self.filter_percentage * len(anchors))
        _, idx = torch.topk(mat[anchors, others], k=k, largest=largest)
        return anchors[idx], others[idx]
