from torch import Tensor, nn

from ..distances import BaseDistance, LpDistance
from ..reducers import BaseReducer, MeanReducer
from ..regularizers import BaseRegularizer
from ..types import IndicesTuple, LossDict, LossTerm
from ..utils import _common


def regularization_term(regularizer: nn.Module, weight: float, x: Tensor) -> LossTerm:
    return {
        "losses": regularizer(x) * weight,
        "indices": None,
        "reduction_type": "already_reduced",
    }


class BaseMetricLossFunction(nn.Module):
    sub_loss_names: tuple[str, ...] = ("loss",)

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
        ref_emb: Tensor,
        ref_labels: Tensor | None,
    ) -> LossDict:
        raise NotImplementedError

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor | None = None,
        indices_tuple: IndicesTuple | None = None,
        ref_emb: Tensor | None = None,
        ref_labels: Tensor | None = None,
    ) -> Tensor:
        embeddings, labels, ref_emb, ref_labels = _common.prepare_inputs(
            embeddings, labels, ref_emb, ref_labels
        )
        terms = self.compute_loss(
            embeddings, labels, indices_tuple, ref_emb, ref_labels
        )
        loss = self.reducer(terms, embeddings, labels)
        for term in self.regularization_terms(embeddings).values():
            loss = loss + term["losses"]
        return loss

    def loss_terms(
        self,
        embeddings: Tensor,
        labels: Tensor | None = None,
        indices_tuple: IndicesTuple | None = None,
        ref_emb: Tensor | None = None,
        ref_labels: Tensor | None = None,
    ) -> LossDict:
        embeddings, labels, ref_emb, ref_labels = _common.prepare_inputs(
            embeddings, labels, ref_emb, ref_labels
        )
        terms = self.compute_loss(
            embeddings, labels, indices_tuple, ref_emb, ref_labels
        )
        return {**terms, **self.regularization_terms(embeddings)}

    def regularization_terms(self, embeddings: Tensor) -> LossDict:
        if self.embedding_regularizer is None:
            return {}
        return {
            "embedding_reg_loss": regularization_term(
                self.embedding_regularizer, self.embedding_reg_weight, embeddings
            )
        }

    def zero_losses(self, embeddings: Tensor) -> LossDict:
        return {name: zero_term(embeddings) for name in self.sub_loss_names}


def zero_term(embeddings: Tensor) -> LossTerm:
    return {
        "losses": _common.zero_loss(embeddings),
        "indices": None,
        "reduction_type": "already_reduced",
    }
