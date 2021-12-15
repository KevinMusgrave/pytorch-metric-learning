import warnings

import torch
import torch.nn.functional as F

from .base_metric_loss_function import BaseMetricLossFunction
from ..utils import common_functions as c_f


class VICRegLoss(BaseMetricLossFunction):
    def __init__(
        self,
        invariance_lambda=25,
        variance_mu=25,
        covariance_v=1,
        eps=1e-4,
        **kwargs
    ):
        super().__init__(**kwargs)
        """
        The overall loss function is a weighted average of the invariance, variance and covariance terms:
            L(Z, Z') = λs(Z, Z') + µ[v(Z) + v(Z')] + ν[c(Z) + c(Z')],
        where λ, µ and ν are hyper-parameters controlling the importance of each term in the loss.
        """
        self.invariance_lambda = invariance_lambda
        self.variance_mu = variance_mu
        self.covariance_v = covariance_v
        self.eps = eps

    def forward(self, embeddings, ref_emb):
        """
        x should have shape (N, embedding_size)
        """
        self.reset_stats()
        loss_dict = self.compute_loss(embeddings, ref_emb)
        return self.reducer(loss_dict, embeddings, c_f.torch_arange_from_size(embeddings))


    def compute_loss(
        self, embeddings, ref_emb
    ):

        invariance_loss = self.invariance_loss(embeddings, ref_emb)
        variance_loss = self.variance_loss(embeddings, ref_emb)
        covariance_loss = self.covariance_loss(embeddings, ref_emb)

        loss = self.invariance_lambda * invariance_loss + self.variance_mu * variance_loss + self.covariance_v * covariance_loss
        return {
            "loss": {
                "losses": loss,
                "indices": None,
                "reduction_type": "already_reduced",
            }
        }

    def invariance_loss(self, emb, ref_emb):
        return F.mse_loss(emb, ref_emb)

    def variance_loss(self, emb, ref_emb):
        std_emb = torch.sqrt(emb.var(dim=0) + self.eps)
        std_ref_emb = torch.sqrt(ref_emb.var(dim=0) + self.eps)
        var_loss = torch.mean(F.relu(1 - std_emb)) + torch.mean(F.relu(1 - std_ref_emb))
        return var_loss

    def covariance_loss(self, emb, ref_emb):
        _, D = emb.size()
        cov_emb = torch.cov(emb.T)
        cov_ref_emb = torch.cov(ref_emb.T)

        diag = torch.eye(D, device=cov_emb.device)
        cov_loss = cov_emb[~diag.bool()].pow_(2).sum() / D + cov_ref_emb[~diag.bool()].pow_(2).sum() / D
        return cov_loss
        

