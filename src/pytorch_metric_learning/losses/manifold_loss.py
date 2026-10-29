import math
from typing import Unpack

import torch
from torch import Tensor, nn

from ..distances import BaseDistance, CosineSimilarity
from ..types import IndicesTuple, LossDict, LossOptions
from ..utils import _common
from .base_metric_loss_function import BaseMetricLossFunction


class ManifoldLoss(BaseMetricLossFunction):
    def __init__(
        self,
        embedding_size: int,
        num_proxies: int = 50,
        lambda_c: float = 1.0,
        alpha: float = 0.8,
        margin: float = 5e-4,
        **options: Unpack[LossOptions],
    ) -> None:
        super().__init__(**options)
        if lambda_c < 0:
            raise ValueError(f"lambda_c must be non-negative; got {lambda_c}")
        self.num_proxies = num_proxies
        self.embedding_size = embedding_size
        self.proxies = nn.Parameter(torch.randn(num_proxies, embedding_size))
        self.lambda_c = lambda_c
        self.alpha = alpha
        self.margin = margin

    def compute_loss(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor | None,
        ref_labels: Tensor | None,
    ) -> LossDict:
        _common.ref_not_supported(ref_emb)
        _common.indices_tuple_not_supported(indices_tuple)
        if embeddings.shape[1] != self.embedding_size:
            raise ValueError("Embedding dimension must match embedding_size")
        _common.check_parameter_placement(self, self.proxies, embeddings)

        N = len(embeddings)
        device = embeddings.device
        if labels is not None:
            meta_classes = labels
        elif self.num_proxies > N:
            raise ValueError(
                "Without labels, the batch must contain at least num_proxies rows"
            )
        else:
            extra = torch.randint(
                0, self.num_proxies, (N - self.num_proxies,), device=device
            )
            meta_classes = torch.cat(
                (torch.arange(self.num_proxies, device=device), extra)
            )
            meta_classes = meta_classes[torch.randperm(N, device=device)]
        rows = torch.arange(N, device=device)

        embs_and_proxies = torch.cat([embeddings, self.proxies], dim=0)
        S = self.distance(embs_and_proxies, embs_and_proxies).clamp(0, math.inf)
        S = torch.exp(S / 0.5)
        Y = torch.eye(N + self.num_proxies, device=S.device, dtype=S.dtype)
        S = S - S * Y
        D_inv_half = torch.pow(torch.sum(S, dim=1), -1 / 2).diag()
        S_bar = D_inv_half.mm(S).mm(D_inv_half)
        L = torch.inverse(Y.float() - self.alpha * S_bar.float()).to(S_bar.dtype)
        F = (1 - self.alpha) * L
        F_p = F[N:, :]
        F_e = F[:N, :]

        loss_ctx = torch.nn.functional.cosine_similarity(
            F_e, F_p.unsqueeze(1), dim=-1
        ).t()
        loss_ctx = self.ranking_term(loss_ctx, rows, meta_classes)
        if self.lambda_c == math.inf:
            return {"loss": {"losses": loss_ctx, "indices": None}}
        loss_int = self.ranking_term(F[:N, N:], rows, meta_classes)
        return {
            "loss": {"losses": loss_int + self.lambda_c * loss_ctx, "indices": None}
        }

    def ranking_term(
        self, scores: Tensor, rows: Tensor, meta_classes: Tensor
    ) -> Tensor:
        violations = scores - scores[rows, meta_classes].view(-1, 1) + self.margin
        excluded = torch.zeros_like(violations, dtype=torch.bool)
        excluded[rows, meta_classes] = True
        violations = violations.masked_fill(excluded | (violations < 0), -math.inf)
        return torch.log(1 + torch.sum(torch.exp(violations), dim=1)).mean()

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()
