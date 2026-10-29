from collections.abc import Mapping
from typing import cast

import torch
from torch import Tensor

from ..types import LossDict
from ..utils._common import zero_loss
from .base_reducer import BaseReducer
from .mean_reducer import MeanReducer


class MultipleReducers(BaseReducer):
    def __init__(
        self,
        reducers: Mapping[str, BaseReducer],
        default_reducer: BaseReducer | None = None,
    ) -> None:
        super().__init__()
        if not all(isinstance(r, BaseReducer) for r in reducers.values()):
            raise TypeError("reducers must map loss term names to BaseReducer objects")
        self.reducers = torch.nn.ModuleDict(reducers)
        self.default_reducer = (
            MeanReducer() if default_reducer is None else default_reducer
        )

    def forward(
        self, loss_dict: LossDict, embeddings: Tensor, labels: Tensor | None
    ) -> Tensor:
        reduced = (
            self.reducer_for(name).reduce(term, embeddings, labels)
            for name, term in loss_dict.items()
        )
        return sum(reduced, start=zero_loss(embeddings))

    def element_reduction(
        self,
        losses: Tensor,
        anchors: Tensor,
        embeddings: Tensor,
        labels: Tensor | None,
    ) -> Tensor:
        raise TypeError(
            "MultipleReducers selects reducers by term name and cannot reduce an "
            "unnamed term; nest the per-term reducers instead"
        )

    def reducer_for(self, name: str) -> BaseReducer:
        if name in self.reducers:
            return cast(BaseReducer, self.reducers[name])
        return self.default_reducer
