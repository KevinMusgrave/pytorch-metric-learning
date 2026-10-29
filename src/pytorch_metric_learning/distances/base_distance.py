from typing import cast

import torch
from torch import Tensor

from ..utils._common import accumulation_dtype, normalization_eps


class BaseDistance(torch.nn.Module):
    is_inverted: bool = False

    def __init__(
        self,
        *,
        normalize_embeddings: bool = True,
        p: float = 2,
        power: float = 1,
    ) -> None:
        super().__init__()
        self.normalize_embeddings = normalize_embeddings
        self.p = p
        self.power = power

    def forward(self, query_emb: Tensor, ref_emb: Tensor | None = None) -> Tensor:
        self.check_shapes(query_emb, ref_emb)
        query_emb_normalized = self.maybe_normalize(query_emb)
        if ref_emb is None:
            ref_emb = query_emb
            ref_emb_normalized = query_emb_normalized
        else:
            ref_emb_normalized = self.maybe_normalize(ref_emb)
        mat = self.compute_mat(query_emb_normalized, ref_emb_normalized)
        if self.power != 1:
            mat = mat**self.power
        if mat.size() != torch.Size((query_emb.size(0), ref_emb.size(0))):
            raise ValueError(
                "compute_mat must return shape (num_queries, num_references)"
            )
        return mat

    def pairwise(self, query_emb: Tensor, ref_emb: Tensor) -> Tensor:
        self.check_shapes(query_emb, ref_emb)
        if query_emb.shape != ref_emb.shape:
            raise ValueError("pairwise inputs must have the same shape")
        values = self.pairwise_distance(
            self.maybe_normalize(query_emb), self.maybe_normalize(ref_emb)
        )
        if self.power != 1:
            values = values**self.power
        return values

    def compute_mat(self, query_emb: Tensor, ref_emb: Tensor) -> Tensor:
        raise NotImplementedError

    def pairwise_distance(self, query_emb: Tensor, ref_emb: Tensor) -> Tensor:
        raise NotImplementedError

    def margin(self, x: Tensor | float, y: Tensor | float) -> Tensor:
        return cast(Tensor, y - x if self.is_inverted else x - y)

    def normalize(self, embeddings: Tensor, dim: int = 1) -> Tensor:
        work = embeddings.to(accumulation_dtype(embeddings.dtype))
        eps = normalization_eps(embeddings.dtype)
        normalized = torch.nn.functional.normalize(work, p=self.p, dim=dim, eps=eps)
        return normalized.to(embeddings.dtype)

    def maybe_normalize(self, embeddings: Tensor, dim: int = 1) -> Tensor:
        if self.normalize_embeddings:
            return self.normalize(embeddings, dim=dim)
        return embeddings

    def get_norm(self, embeddings: Tensor, dim: int = 1) -> Tensor:
        return torch.norm(embeddings, p=self.p, dim=dim)

    def check_shapes(self, query_emb: Tensor, ref_emb: Tensor | None) -> None:
        if query_emb.ndim != 2 or (ref_emb is not None and ref_emb.ndim != 2):
            raise ValueError(
                "embeddings must be a 2D tensor of shape (batch_size, embedding_size)"
            )
