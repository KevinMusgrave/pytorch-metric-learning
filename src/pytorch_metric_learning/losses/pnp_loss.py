import torch

from ..distances import LpDistance
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction


class PNPLoss(BaseMetricLossFunction):
    def __init__(self, b, alpha, anneal, variant, **kwargs):
        super().__init__(**kwargs)
        self.b = b
        self.alpha = alpha
        self.anneal = anneal
        self.variant = variant

        """
        Adapted from https://github.com/interestingzhuo/PNPloss
        """

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.labels_required(labels)
        c_f.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        dtype, device = embeddings.dtype, embeddings.device

        N = labels.size(0)
        a1_idx, p_idx, a2_idx, n_idx = lmu.get_all_pairs_indices(labels)
        I_pos = torch.zeros(N, N, dtype=dtype, device=device)
        I_neg = torch.zeros(N, N, dtype=dtype, device=device)
        I_pos[a1_idx, p_idx] = 1
        I_neg[a2_idx, n_idx] = 1
        N_pos = torch.sum(I_pos, dim=1)
        safe_N = N_pos > 0
        if torch.sum(safe_N) == 0:
            return self.zero_losses()
        sim_all = self.compute_aff(embeddings)

        
        mask = I_neg.unsqueeze(dim=1).repeat(1, N, 1)
        
        sim_all_repeat = sim_all.unsqueeze(dim=1).repeat(1, N, 1)
        # compute the difference matrix
        sim_diff = sim_all_repeat - sim_all_repeat.permute(0, 2, 1)
        # pass through the sigmoid and ignores the relevance score of the query to itself
        sim_sg = self.sigmoid(sim_diff, temp=self.anneal) * mask
        # compute the number of negatives before 
        sim_all_rk = torch.sum(sim_sg, dim=-1)
        
        if self.variant == "PNP-D_s":
            sim_all_rk = torch.log(1 + sim_all_rk)
        elif self.variant == "PNP-D_q":
            sim_all_rk = 1 / (1 + sim_all_rk) ** (self.alpha)

        elif self.variant == "PNP-I_u":
            sim_all_rk = (1 + sim_all_rk) * torch.log(1 + sim_all_rk)

        elif self.variant == "PNP-I_b":
            b = self.b
            sim_all_rk = 1 / b**2 * (b * sim_all_rk - torch.log(1 + b * sim_all_rk))
        elif self.variant == "PNP-O":
            pass
        else:
            raise Exception("variantation <{}> not available!".format(self.variant))
        
        loss = (sim_all_rk * I_pos) / N_pos.reshape(-1, 1)
        loss = torch.sum(loss) / N
        if self.variant == "PNP-D_q":
            loss = 1 - loss

        return {
            "loss": {
                "losses": loss,
                "indices": torch.where(safe_N)[0],
                "reduction_type": "already_reduced",
            }
        }

    def sigmoid(self, tensor, temp=1.0):
        """temperature controlled sigmoid
        takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
        """
        exponent = -tensor / temp
        # clamp the input tensor for stability
        exponent = torch.clamp(exponent, min=-50, max=50)
        y = 1.0 / (1.0 + torch.exp(exponent))
        return y

    def compute_aff(self, x):
        """computes the affinity matrix between an input vector and itself"""
        x = torch.nn.functional.normalize(x, dim=-1)
        return torch.mm(x, x.t())
