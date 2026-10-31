from collections.abc import Callable, Sequence
from typing import ClassVar, cast

from torch import Tensor, nn

from ..distances import BaseDistance, LpDistance
from ..reducers import BaseReducer, MeanReducer
from ..regularizers import BaseRegularizer
from ..types import IndicesTuple, LossDict, LossTerm
from ..utils import _common


def _regularization_term(
    regularizer: BaseRegularizer, weight: float, x: Tensor
) -> LossTerm:
    return {"losses": regularizer(x) * weight, "indices": None}


class BaseMetricLossFunction(nn.Module):
    label_ndim: ClassVar[int] = 1
    supports_mat: ClassVar[bool] = False

    def __init__(
        self,
        *,
        distance: BaseDistance | None = None,
        reducer: BaseReducer | None = None,
        embedding_regularizer: BaseRegularizer | None = None,
        embedding_reg_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.distance = distance if distance is not None else self.default_distance()
        self.reducer = reducer if reducer is not None else self.default_reducer()
        self.embedding_regularizer = embedding_regularizer
        self.embedding_reg_weight = embedding_reg_weight

    def default_distance(self) -> BaseDistance:
        return LpDistance()

    def default_reducer(self) -> BaseReducer:
        return MeanReducer()

    def compute_loss(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor | None,
        ref_labels: Tensor | None,
    ) -> LossDict:
        raise NotImplementedError

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor | None = None,
        indices_tuple: Sequence[Tensor] | None = None,
        ref_emb: Tensor | None = None,
        ref_labels: Tensor | None = None,
        *,
        mat: Tensor | None = None,
    ) -> Tensor:
        terms = self.objective_terms(
            embeddings, labels, indices_tuple, ref_emb, ref_labels, mat
        )
        penalties = _distinct_penalties(self, terms, embeddings)
        loss = self.reducer(terms, embeddings, labels)
        for term in penalties.values():
            loss = loss + term["losses"]
        return loss

    def loss_terms(
        self,
        embeddings: Tensor,
        labels: Tensor | None = None,
        indices_tuple: Sequence[Tensor] | None = None,
        ref_emb: Tensor | None = None,
        ref_labels: Tensor | None = None,
        *,
        mat: Tensor | None = None,
    ) -> LossDict:
        terms = self.objective_terms(
            embeddings, labels, indices_tuple, ref_emb, ref_labels, mat
        )
        return {**terms, **_distinct_penalties(self, terms, embeddings)}

    def objective_terms(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: Sequence[Tensor] | None,
        ref_emb: Tensor | None,
        ref_labels: Tensor | None,
        mat: Tensor | None = None,
    ) -> LossDict:
        _common.prepare_inputs(embeddings, labels, ref_emb, ref_labels, self.label_ndim)
        indices = _common.as_indices_tuple(indices_tuple, embeddings.device)
        if mat is None:
            return self.compute_loss(embeddings, labels, indices, ref_emb, ref_labels)
        _common.check_matrix(self, mat, embeddings, ref_emb)
        _common.check_matrix_gradient(mat, embeddings, ref_emb)
        compute = cast(Callable[..., LossDict], self.compute_loss)
        return compute(embeddings, labels, indices, ref_emb, ref_labels, mat=mat)

    def regularization_terms(self, embeddings: Tensor) -> LossDict:
        if self.embedding_regularizer is None:
            return {}
        return {
            "embedding_reg_loss": _regularization_term(
                self.embedding_regularizer, self.embedding_reg_weight, embeddings
            )
        }


def _distinct_penalties(
    loss: BaseMetricLossFunction, objective: LossDict, embeddings: Tensor
) -> LossDict:
    penalties = loss.regularization_terms(embeddings)
    collisions = sorted(objective.keys() & penalties.keys())
    if collisions:
        raise ValueError(
            f"{type(loss).__name__} objective terms {collisions} collide with "
            "regularization term names; rename the objective terms"
        )
    return penalties
