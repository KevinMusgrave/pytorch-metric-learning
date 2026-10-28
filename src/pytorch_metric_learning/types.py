from typing import Literal, NotRequired, TypedDict

from torch import Tensor

type PairIndices = tuple[Tensor, Tensor, Tensor, Tensor]
type TripletIndices = tuple[Tensor, Tensor, Tensor]
type IndicesTuple = PairIndices | TripletIndices
type ReductionType = Literal[
    "already_reduced", "element", "pos_pair", "neg_pair", "triplet"
]


class LossTerm(TypedDict):
    losses: Tensor | Literal[0]
    indices: Tensor | tuple[Tensor, Tensor] | TripletIndices | list[Tensor] | None
    reduction_type: ReductionType
    divisor: NotRequired[Tensor | float | int]


type LossDict = dict[str, LossTerm]
