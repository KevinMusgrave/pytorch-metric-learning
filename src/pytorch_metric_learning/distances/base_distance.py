from contextlib import AbstractContextManager, nullcontext
from typing import cast

import torch
from torch import Tensor

from ..utils._common import accumulation_dtype, normalize


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
        query_input = self.prepare_input(query_emb)
        if ref_emb is None:
            ref_emb = query_emb
            ref_input = query_input
        else:
            ref_input = self.prepare_input(ref_emb)
        with self.value_precision(query_input):
            mat = self.compute_mat(query_input, ref_input)
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
        query_input = self.prepare_input(query_emb)
        ref_input = self.prepare_input(ref_emb)
        with self.value_precision(query_input):
            values = self.pairwise_distance(query_input, ref_input)
            if self.power != 1:
                values = values**self.power
        return values

    def prepare_input(self, embeddings: Tensor) -> Tensor:
        if self.normalize_embeddings:
            return self.normalize(embeddings)
        return embeddings.to(accumulation_dtype(embeddings.dtype))

    def value_precision(self, prepared: Tensor) -> AbstractContextManager:
        if (
            self.normalize_embeddings
            or not torch.amp.autocast_mode.is_autocast_available(prepared.device.type)
        ):
            return nullcontext()
        return torch.autocast(prepared.device.type, enabled=False)

    def compute_mat(self, query_emb: Tensor, ref_emb: Tensor) -> Tensor:
        raise NotImplementedError

    def pairwise_distance(self, query_emb: Tensor, ref_emb: Tensor) -> Tensor:
        raise NotImplementedError

    def margin(self, x: Tensor | float, y: Tensor | float) -> Tensor:
        return cast(Tensor, y - x if self.is_inverted else x - y)

    def normalize(self, embeddings: Tensor, dim: int = 1) -> Tensor:
        return normalize(embeddings, p=self.p, dim=dim)

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
