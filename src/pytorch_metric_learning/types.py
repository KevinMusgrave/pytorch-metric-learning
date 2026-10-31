from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from typing import TYPE_CHECKING, Literal, NamedTuple, NotRequired, TypedDict

from torch import Tensor

if TYPE_CHECKING:
    import numpy as np

    from .distances import BaseDistance
    from .losses import BaseMetricLossFunction, MultipleLosses
    from .reducers import BaseReducer
    from .regularizers import BaseRegularizer

__all__ = [
    "AnchorAggregation",
    "ArrayLike",
    "DatasetLabels",
    "DistributedObjective",
    "GE2EVariant",
    "GradedRelevance",
    "IndicesTuple",
    "KMeansFunction",
    "KNNFunction",
    "LabelComparison",
    "LossDict",
    "LossOptions",
    "LossTerm",
    "MetricLoss",
    "MetricNeighbors",
    "MiningStrategy",
    "NeighborLimit",
    "PNPVariant",
    "PairFunction",
    "PairIndices",
    "ProxyLossOptions",
    "RegularizerOptions",
    "SubCenterOutliers",
    "TripletIndices",
    "TripletsPerAnchor",
]


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
type PairFunction = Callable[[Tensor, Tensor], PairIndices]
type TripletsPerAnchor = int | Literal["all"]
type MiningStrategy = Literal["hard", "semihard", "easy", "all"]
type PNPVariant = Literal["Ds", "Dq", "Iu", "Ib", "O"]
type GE2EVariant = Literal["softmax", "contrast"]
type AnchorAggregation = Literal["mean", "sum"]
type DistributedObjective = Literal["global", "local_anchor"]


class SubCenterOutliers(NamedTuple):
    outliers: Tensor
    dominant_centers: Tensor


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
type GradedRelevance = Callable[[Tensor, Tensor], Tensor]
type ArrayLike = Tensor | np.ndarray
type NeighborLimit = int | Literal["max_bin_count"] | None
type MetricNeighbors = int | Literal["relevant", "all"]
type DatasetLabels = Tensor | np.ndarray | Iterable[Hashable]
