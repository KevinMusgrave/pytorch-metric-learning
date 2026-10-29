from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

from torch import Tensor

if TYPE_CHECKING:
    from .distances import BaseDistance
    from .reducers import BaseReducer
    from .regularizers import BaseRegularizer

type PairIndices = tuple[Tensor, Tensor, Tensor, Tensor]
type TripletIndices = tuple[Tensor, Tensor, Tensor]
type IndicesTuple = PairIndices | TripletIndices
type ReductionType = Literal[
    "already_reduced", "element", "pos_pair", "neg_pair", "triplet"
]
type TripletsPerAnchor = int | Literal["all"]


class LossTerm(TypedDict):
    losses: Tensor
    indices: Tensor | tuple[Tensor, Tensor] | TripletIndices | None
    reduction_type: ReductionType
    divisor: NotRequired[Tensor | float | int]


type LossDict = dict[str, LossTerm]


class LossOptions(TypedDict, total=False):
    distance: BaseDistance | None
    reducer: BaseReducer | None
    embedding_regularizer: BaseRegularizer | None
    embedding_reg_weight: float


class RegularizerOptions(TypedDict, total=False):
    distance: BaseDistance | None
    reducer: BaseReducer | None


type KNNFunction = Callable[[Tensor, int, Tensor, bool], tuple[Tensor, Tensor]]
type KMeansFunction = Callable[[Tensor, int], Tensor]
type LabelComparison = Callable[[Tensor, Tensor], Tensor]
