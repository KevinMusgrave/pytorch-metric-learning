import torch
import torch.nn.functional as F

from ..utils import common_functions as c_f
from .base_metric_loss_function import BaseMetricLossFunction


class VICRegLoss(BaseMetricLossFunction):
    def __init__(
        self, invariance_lambda=25, variance_mu=25, covariance_v=1, eps=1e-4, **kwargs
    ):
        if "distance" in kwargs:
            raise ValueError("VICRegLoss cannot use a distance function")
        if "embedding_regularizer" in kwargs:
            raise ValueError("VICRegLoss cannot use a regularizer")
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
        return self.reducer(
            loss_dict, embeddings, c_f.torch_arange_from_size(embeddings)
        )

    def compute_loss(self, embeddings, ref_emb):
        invariance_loss = self.invariance_lambda * self.invariance_loss(
            embeddings, ref_emb
        )
        variance_loss1, variance_loss2 = self.variance_loss(embeddings, ref_emb)
        covariance_loss = self.covariance_v * self.covariance_loss(embeddings, ref_emb)
        var_loss_size = c_f.torch_arange_from_size(variance_loss1)
        return {
            "invariance_loss": {
                "losses": invariance_loss,
                "indices": c_f.torch_arange_from_size(invariance_loss),
                "reduction_type": "element",
            },
            "variance_loss1": {
                "losses": self.variance_mu * variance_loss1,
                "indices": var_loss_size,
                "reduction_type": "element",
            },
            "variance_loss2": {
                "losses": self.variance_mu * variance_loss2,
                "indices": var_loss_size,
                "reduction_type": "element",
            },
            "covariance_loss": {
                "losses": covariance_loss,
                "indices": None,
                "reduction_type": "already_reduced",
            },
        }

    def invariance_loss(self, emb, ref_emb):
        return torch.mean((emb - ref_emb) ** 2, dim=1)

    def variance_loss(self, emb, ref_emb):
        std_emb = torch.sqrt(emb.var(dim=0) + self.eps)
        std_ref_emb = torch.sqrt(ref_emb.var(dim=0) + self.eps)
        return F.relu(1 - std_emb) / 2, F.relu(1 - std_ref_emb) / 2 # / 2 for averaging

    def covariance_loss(self, emb, ref_emb):
        N, D = emb.size()
        emb = emb - emb.mean(dim=0)
        ref_emb = ref_emb - ref_emb.mean(dim=0)
        cov_emb = (emb.T @ emb) / (N - 1)
        cov_ref_emb = (ref_emb.T @ ref_emb) / (N - 1)

        diag = torch.eye(D, device=cov_emb.device)
        cov_loss = (
            cov_emb[~diag.bool()].pow_(2).sum() / D
            + cov_ref_emb[~diag.bool()].pow_(2).sum() / D
        )
        return cov_loss

    def _sub_loss_names(self):
        return [
            "invariance_loss",
            "variance_loss1",
            "variance_loss2",
            "covariance_loss",
        ]
