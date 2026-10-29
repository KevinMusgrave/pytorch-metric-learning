from torch import Tensor, nn

from ..distances import BaseDistance, LpDistance
from ..reducers import BaseReducer, MeanReducer
from ..types import LossDict
from ..utils._common import arange_like


class BaseRegularizer(nn.Module):
    def __init__(
        self,
        *,
        distance: BaseDistance | None = None,
        reducer: BaseReducer | None = None,
    ) -> None:
        super().__init__()
        self.distance = distance if distance is not None else self.default_distance()
        self.reducer = reducer if reducer is not None else self.default_reducer()

    def default_distance(self) -> BaseDistance:
        return LpDistance()

    def default_reducer(self) -> BaseReducer:
        return MeanReducer()

    def compute_loss(self, x: Tensor) -> LossDict:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        return self.reducer(self.compute_loss(x), x, arange_like(x))
