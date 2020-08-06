from .base_miner import BaseTupleMiner
from ..utils import loss_and_miner_utils as lmu, common_functions as c_f
import torch
from ..distances import CosineSimilarity


class MultiSimilarityMiner(BaseTupleMiner):
    def __init__(self, epsilon, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.add_to_recordable_attributes(name="epsilon", is_stat=False)

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        dtype = embeddings.dtype
        mat = self.distance(embeddings, ref_emb)
        a1, p, a2, n = lmu.get_all_pairs_indices(labels, ref_labels)

        if len(a1) == 0 or len(a2) == 0:
            empty = torch.LongTensor([]).to(labels.device)
            return empty.clone(), empty.clone(), empty.clone(), empty.clone()

        mat_neg_sorting = mat
        mat_pos_sorting = mat.clone()

        pos_ignore = c_f.pos_inf(dtype) if self.distance.is_inverted else c_f.neg_inf(dtype)
        neg_ignore = c_f.neg_inf(dtype) if self.distance.is_inverted else c_f.pos_inf(dtype)

        mat_pos_sorting[a2, n] = pos_ignore
        mat_neg_sorting[a1, p] = neg_ignore
        if embeddings is ref_emb:
            mat_pos_sorting.fill_diagonal_(pos_ignore)
            mat_neg_sorting.fill_diagonal_(neg_ignore)

        pos_sorted, pos_sorted_idx = torch.sort(mat_pos_sorting, dim=1)
        neg_sorted, neg_sorted_idx = torch.sort(mat_neg_sorting, dim=1)

        if self.distance.is_inverted:
            hard_pos_idx = torch.where(pos_sorted - self.epsilon < neg_sorted[:, -1].unsqueeze(1))
            hard_neg_idx = torch.where(neg_sorted + self.epsilon > pos_sorted[:, 0].unsqueeze(1))
        else:
            hard_pos_idx = torch.where(pos_sorted + self.epsilon > neg_sorted[:, 0].unsqueeze(1))
            hard_neg_idx = torch.where(neg_sorted - self.epsilon < pos_sorted[:, -1].unsqueeze(1))     

        a1 = hard_pos_idx[0]
        p = pos_sorted_idx[a1, hard_pos_idx[1]]
        a2 = hard_neg_idx[0]
        n = neg_sorted_idx[a2, hard_neg_idx[1]]
        
        return a1, p, a2, n

    def get_default_distance(self):
        return CosineSimilarity()