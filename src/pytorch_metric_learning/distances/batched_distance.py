from collections.abc import Iterator

import torch
from torch import Tensor

from .base_distance import BaseDistance


class BatchedDistance(torch.nn.Module):
    def __init__(self, distance: BaseDistance, batch_size: int = 32) -> None:
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.distance = distance
        self.batch_size = batch_size

    def forward(
        self, query_emb: Tensor, ref_emb: Tensor | None = None
    ) -> Iterator[tuple[Tensor, int, int]]:
        ref_emb = query_emb if ref_emb is None else ref_emb
        for start in range(0, query_emb.shape[0], self.batch_size):
            end = min(start + self.batch_size, query_emb.shape[0])
            yield self.distance(query_emb[start:end], ref_emb), start, end

    @property
    def is_inverted(self) -> bool:
        return self.distance.is_inverted
