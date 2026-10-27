from ..utils import loss_and_miner_utils as lmu
from .base_miner import BaseMiner


class PairMarginMiner(BaseMiner):
    def __init__(self, pos_margin=0.2, neg_margin=0.8, **kwargs):
        super().__init__(**kwargs)
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        mat = self.distance(embeddings, ref_emb)
        a1, p, a2, n = lmu.get_all_pairs_indices(labels, ref_labels)
        pos_pair = mat[a1, p]
        neg_pair = mat[a2, n]
        pos_mask = (
            pos_pair < self.pos_margin
            if self.distance.is_inverted
            else pos_pair > self.pos_margin
        )
        neg_mask = (
            neg_pair > self.neg_margin
            if self.distance.is_inverted
            else neg_pair < self.neg_margin
        )
        return a1[pos_mask], p[pos_mask], a2[neg_mask], n[neg_mask]
