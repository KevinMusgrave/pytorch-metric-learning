import torch
import torch.nn.functional as F
from torch import Tensor


class VICRegLoss(torch.nn.Module):
    def __init__(
        self,
        invariance_lambda: float = 25,
        variance_mu: float = 25,
        covariance_v: float = 1,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.invariance_lambda = invariance_lambda
        self.variance_mu = variance_mu
        self.covariance_v = covariance_v
        self.eps = eps

    def forward(self, embeddings: Tensor, ref_emb: Tensor) -> Tensor:
        if embeddings.shape != ref_emb.shape or embeddings.ndim != 2:
            raise ValueError("embeddings and ref_emb must be 2D tensors of equal shape")
        if len(embeddings) < 2:
            raise ValueError(
                "VICRegLoss needs at least 2 rows to estimate variance and covariance"
            )
        return (
            self.invariance_lambda * self.invariance_loss(embeddings, ref_emb)
            + self.variance_mu * self.variance_loss(embeddings, ref_emb)
            + self.covariance_v * self.covariance_loss(embeddings, ref_emb)
        )

    @staticmethod
    def invariance_loss(emb: Tensor, ref_emb: Tensor) -> Tensor:
        return F.mse_loss(emb, ref_emb)

    def variance_loss(self, emb: Tensor, ref_emb: Tensor) -> Tensor:
        std_emb = torch.sqrt(emb.var(dim=0) + self.eps)
        std_ref_emb = torch.sqrt(ref_emb.var(dim=0) + self.eps)
        return (
            torch.mean(F.relu(1 - std_emb)) + torch.mean(F.relu(1 - std_ref_emb))
        ) / 2

    @staticmethod
    def covariance_loss(emb: Tensor, ref_emb: Tensor) -> Tensor:
        N, D = emb.shape
        off_diagonal = ~torch.eye(D, dtype=torch.bool, device=emb.device)
        total = emb.new_zeros(())
        for x in (emb, ref_emb):
            centered = x - x.mean(dim=0)
            cov = (centered.T @ centered) / (N - 1)
            total = total + cov[off_diagonal].pow(2).sum() / D
        return total
