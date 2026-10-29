import functools
import operator
from collections.abc import Mapping

import torch
from torch import Tensor

from ..types import LossDict
from .base_reducer import BaseReducer
from .mean_reducer import MeanReducer


class MultipleReducers(BaseReducer):
    def __init__(
        self,
        reducers: Mapping[str, BaseReducer],
        default_reducer: BaseReducer | None = None,
    ) -> None:
        super().__init__()
        self.reducers = torch.nn.ModuleDict(reducers)
        self.default_reducer = (
            MeanReducer() if default_reducer is None else default_reducer
        )

    def forward(
        self, loss_dict: LossDict, embeddings: Tensor, labels: Tensor | None
    ) -> Tensor:
        reduced = [
            self.reducer_for(name).reduce(term, embeddings, labels)
            for name, term in loss_dict.items()
        ]
        return functools.reduce(operator.add, reduced)

    def reducer_for(self, name: str) -> BaseReducer:
        if name in self.reducers:
            reducer = self.reducers[name]
            assert isinstance(reducer, BaseReducer)
            return reducer
        return self.default_reducer
