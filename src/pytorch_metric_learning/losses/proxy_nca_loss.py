import torch
from torch import Tensor

from ..types import IndicesTuple, LossDict
from ..utils import common_functions as c_f
from .base_metric_loss_function import regularization_term
from .nca_loss import NCALoss


class ProxyNCALoss(NCALoss):
    def __init__(
        self,
        num_classes,
        embedding_size,
        *,
        weight_regularizer=None,
        weight_reg_weight=1,
        weight_init_func=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.weight_regularizer = weight_regularizer
        self.weight_reg_weight = weight_reg_weight
        self.proxies = torch.nn.Parameter(torch.empty(num_classes, embedding_size))
        (weight_init_func or torch.nn.init.normal_)(self.proxies)
        self.register_buffer(
            "proxy_labels", torch.arange(num_classes), persistent=False
        )

    def compute_loss(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor,
        ref_labels: Tensor | None,
    ) -> LossDict:
        labels = c_f.labels_required(labels)
        c_f.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        c_f.check_parameter_placement(self, self.proxies, embeddings)
        embeddings = embeddings.to(dtype=self.proxies.dtype)
        return self.nca_computation(
            embeddings, self.proxies, labels, self.proxy_labels, indices_tuple
        )

    def regularization_terms(self, embeddings):
        terms = super().regularization_terms(embeddings)
        if self.weight_regularizer is not None:
            terms["weight_reg_loss"] = regularization_term(
                self.weight_regularizer, self.weight_reg_weight, self.proxies
            )
        return terms

    def get_logits(self, embeddings):
        c_f.check_parameter_placement(self, self.proxies, embeddings)
        embeddings = embeddings.to(dtype=self.proxies.dtype)
        return self.distance(embeddings, self.proxies)
