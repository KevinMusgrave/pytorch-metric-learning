import torch
from .generic_pair_loss import GenericPairLoss
from ..utils import common_functions as c_f

class NTXentLoss(GenericPairLoss):

    def __init__(self, temperature, **kwargs):
        super().__init__(use_similarity=True, mat_based_loss=False, **kwargs)
        self.temperature = temperature

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, p, a2, _ = indices_tuple
        dtype = neg_pairs.dtype

        if len(a1) > 0 and len(a2) > 0:
            pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
            neg_pairs = neg_pairs / self.temperature
            n_per_p = (a2.unsqueeze(0) == a1.unsqueeze(1)).type(dtype)
            neg_pairs = neg_pairs*n_per_p
            neg_pairs[n_per_p==0] = c_f.neg_inf(dtype)

            max_val = torch.max(pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0])
            numerator = torch.exp(pos_pairs - max_val).squeeze(1)
            denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
            log_exp = torch.log((numerator/denominator) + c_f.small_val(dtype))
            return {"loss": {"losses": -log_exp, "indices": (a1, p), "reduction_type": "pos_pair"}}
        return self.zero_losses()



