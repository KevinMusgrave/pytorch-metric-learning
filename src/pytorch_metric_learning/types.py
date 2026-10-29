from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, NamedTuple, NotRequired, TypedDict

from torch import Tensor

if TYPE_CHECKING:
    from .distances import BaseDistance
    from .losses import BaseMetricLossFunction, MultipleLosses
    from .reducers import BaseReducer
    from .regularizers import BaseRegularizer


class PairIndices(NamedTuple):
    pos_anchors: Tensor
    positives: Tensor
    neg_anchors: Tensor
    negatives: Tensor


class TripletIndices(NamedTuple):
    anchors: Tensor
    positives: Tensor
    negatives: Tensor


type IndicesTuple = PairIndices | TripletIndices
type TripletsPerAnchor = int | Literal["all"]
type MiningStrategy = Literal["hard", "semihard", "easy", "all"]


class LossTerm(TypedDict):
    losses: Tensor
    indices: Tensor | tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor] | None
    divisor: NotRequired[Tensor | float | int]


type LossDict = dict[str, LossTerm]
type MetricLoss = BaseMetricLossFunction | MultipleLosses


class LossOptions(TypedDict, total=False):
    distance: BaseDistance | None
    reducer: BaseReducer | None
    embedding_regularizer: BaseRegularizer | None
    embedding_reg_weight: float


class ProxyLossOptions(LossOptions, total=False):
    weight_regularizer: BaseRegularizer | None
    weight_reg_weight: float
    weight_init_func: Callable[[Tensor], object] | None


class RegularizerOptions(TypedDict, total=False):
    distance: BaseDistance | None
    reducer: BaseReducer | None


type KNNFunction = Callable[[Tensor, int, Tensor, bool], tuple[Tensor, Tensor]]
type KMeansFunction = Callable[[Tensor, int], Tensor]
type LabelComparison = Callable[[Tensor, Tensor], Tensor]
