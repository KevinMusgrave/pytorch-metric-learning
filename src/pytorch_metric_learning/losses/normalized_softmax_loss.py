import torch

from ..distances import DotProductSimilarity
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction, regularization_term


class NormalizedSoftmaxLoss(BaseMetricLossFunction):
    def __init__(
        self,
        num_classes,
        embedding_size,
        temperature=0.05,
        *,
        weight_regularizer=None,
        weight_reg_weight=1,
        weight_init_func=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.weight_regularizer = weight_regularizer
        self.weight_reg_weight = weight_reg_weight
        self.W = torch.nn.Parameter(torch.empty(embedding_size, num_classes))
        (weight_init_func or torch.nn.init.normal_)(self.W)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")
        self.add_to_recordable_attributes(
            list_of_names=["embedding_size", "num_classes", "temperature"],
            is_stat=False,
        )

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.labels_required(labels)
        c_f.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        c_f.check_parameter_placement(self, self.W, embeddings)
        dtype = embeddings.dtype
        miner_weights = lmu.convert_to_weights(indices_tuple, labels, dtype=dtype)
        logits = self.get_logits(embeddings)
        unweighted_loss = self.cross_entropy(logits, labels)
        miner_weighted_loss = unweighted_loss * miner_weights
        return {
            "loss": {
                "losses": miner_weighted_loss,
                "indices": c_f.torch_arange_from_size(embeddings),
                "reduction_type": "element",
            }
        }

    def regularization_terms(self, embeddings):
        terms = super().regularization_terms(embeddings)
        if self.weight_regularizer is not None:
            terms["weight_reg_loss"] = regularization_term(
                self.weight_regularizer, self.weight_reg_weight, self.W.t()
            )
        return terms

    def get_default_distance(self):
        return DotProductSimilarity()

    def get_logits(self, embeddings):
        normalized_W = self.distance.normalize(self.W, dim=0)
        logits = self.distance(embeddings, normalized_W.t()) / self.temperature
        if not self.distance.is_inverted:
            logits = -logits
        return logits
