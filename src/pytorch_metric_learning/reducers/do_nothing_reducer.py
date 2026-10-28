from torch import Tensor

from ..types import LossDict
from .base_reducer import BaseReducer


class DoNothingReducer(BaseReducer):
    def forward(
        self,
        loss_dict: LossDict,
        embeddings: Tensor,
        labels: Tensor | None,
    ) -> LossDict:
        return loss_dict
