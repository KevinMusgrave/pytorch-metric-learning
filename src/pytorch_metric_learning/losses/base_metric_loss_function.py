from torch import Tensor, nn

from ..types import IndicesTuple, LossDict, LossTerm
from ..utils import common_functions as c_f
from ..utils.module_with_distance_and_reducer import ModuleWithReducerAndDistance


def regularization_term(regularizer: nn.Module, weight: float, x: Tensor) -> LossTerm:
    return {
        "losses": regularizer(x) * weight,
        "indices": None,
        "reduction_type": "already_reduced",
    }


class BaseMetricLossFunction(ModuleWithReducerAndDistance):
    def __init__(
        self,
        embedding_regularizer: nn.Module | None = None,
        embedding_reg_weight: float = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.embedding_regularizer = embedding_regularizer
        self.embedding_reg_weight = embedding_reg_weight

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
    ) -> Tensor | LossDict:
        c_f.check_shapes(embeddings, labels)
        if labels is not None:
            labels = labels.to(device=embeddings.device)
        ref_emb, ref_labels = c_f.set_ref_emb(embeddings, labels, ref_emb, ref_labels)
        loss_dict = self.compute_loss(
            embeddings, labels, indices_tuple, ref_emb, ref_labels
        )
        output = self.reducer(loss_dict, embeddings, labels)
        regularization = self.regularization_terms(embeddings)
        if not regularization:
            return output
        if isinstance(output, dict):
            return {**output, **regularization}
        return output + sum(term["losses"] for term in regularization.values())

    def regularization_terms(self, embeddings: Tensor) -> LossDict:
        if self.embedding_regularizer is None:
            return {}
        return {
            "embedding_reg_loss": regularization_term(
                self.embedding_regularizer, self.embedding_reg_weight, embeddings
            )
        }

    def zero_loss(self) -> LossTerm:
        return {"losses": 0, "indices": None, "reduction_type": "already_reduced"}

    def zero_losses(self) -> LossDict:
        return {loss_name: self.zero_loss() for loss_name in self.sub_loss_names()}
