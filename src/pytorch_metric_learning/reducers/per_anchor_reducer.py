from typing import Literal

import torch
from torch import Tensor

from ..utils._common import accumulation_dtype
from .base_reducer import BaseReducer
from .mean_reducer import MeanReducer

type Aggregation = Literal["mean", "sum"]


class PerAnchorReducer(BaseReducer):
    def __init__(
        self, reducer: BaseReducer | None = None, aggregation: Aggregation = "mean"
    ) -> None:
        super().__init__()
        if aggregation not in ("mean", "sum"):
            raise ValueError('aggregation must be "mean" or "sum"')
        self.reducer = reducer if reducer is not None else MeanReducer()
        self.aggregation = aggregation

    def element_reduction(
        self,
        losses: Tensor,
        anchors: Tensor,
        embeddings: Tensor,
        labels: Tensor | None,
    ) -> Tensor:
        size = embeddings.shape[0]
        dtype = accumulation_dtype(losses.dtype)
        totals = losses.new_zeros(size, dtype=dtype).index_add(
            0, anchors, losses.to(dtype)
        )
        counts = torch.bincount(anchors, minlength=size)
        if self.aggregation == "mean":
            totals = totals / counts.clamp_min(1)
        present = torch.where(counts > 0)[0]
        per_anchor = totals[present].to(losses.dtype)
        return self.reducer.reduce(
            {"losses": per_anchor, "indices": present}, embeddings, labels
        )
