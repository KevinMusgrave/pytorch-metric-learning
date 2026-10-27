from ..utils import common_functions as c_f
from ..utils.module_with_records_and_reducer import ModuleWithRecordsReducerAndDistance


def regularization_term(regularizer, weight, x):
    return {
        "losses": regularizer(x) * weight,
        "indices": None,
        "reduction_type": "already_reduced",
    }


class BaseMetricLossFunction(ModuleWithRecordsReducerAndDistance):
    def __init__(self, embedding_regularizer=None, embedding_reg_weight=1, **kwargs):
        super().__init__(**kwargs)
        self.embedding_regularizer = embedding_regularizer
        self.embedding_reg_weight = embedding_reg_weight

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        raise NotImplementedError

    def forward(
        self, embeddings, labels=None, indices_tuple=None, ref_emb=None, ref_labels=None
    ):
        self.reset_stats()
        c_f.check_shapes(embeddings, labels)
        if labels is not None:
            labels = c_f.to_device(labels, embeddings)
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

    def regularization_terms(self, embeddings):
        if self.embedding_regularizer is None:
            return {}
        return {
            "embedding_reg_loss": regularization_term(
                self.embedding_regularizer, self.embedding_reg_weight, embeddings
            )
        }

    def zero_loss(self):
        return {"losses": 0, "indices": None, "reduction_type": "already_reduced"}

    def zero_losses(self):
        return {loss_name: self.zero_loss() for loss_name in self.sub_loss_names()}
