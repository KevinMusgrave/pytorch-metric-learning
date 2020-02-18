import torch
from .base_metric_loss_function import BaseMetricLossFunction
from ..utils import loss_and_miner_utils as lmu

class NTXentLoss(BaseMetricLossFunction):

    def __init__(self, temperature, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def compute_loss(self, embeddings, labels, indices_tuple):
        cosine_similarity = lmu.sim_mat(embeddings)
        if not self.normalize_embeddings:
            embedding_norms_mat = self.embedding_norms.unsqueeze(0)*self.embedding_norms.unsqueeze(1)
            cosine_similarity = cosine_similarity / (embedding_norms_mat)
        cosine_similarity = cosine_similarity / self.temperature

        a1, p, a2, n = lmu.convert_to_pairs(indices_tuple, labels)

        if len(a1) > 0 and len(a2) > 0:
            pos_pairs = cosine_similarity[a1, p].unsqueeze(1)
            neg_pairs = cosine_similarity[a2, n]
            n_per_p = (a2.unsqueeze(0) == a1.unsqueeze(1)).float()
            neg_pairs = neg_pairs*n_per_p
            neg_pairs[n_per_p==0] = float('-inf')

            max_val = torch.max(pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0])
            numerator = torch.exp(pos_pairs - max_val).squeeze(1)
            denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
            log_exp = torch.log((numerator/denominator) + 1e-20)
            return torch.mean(-log_exp)
        return 0