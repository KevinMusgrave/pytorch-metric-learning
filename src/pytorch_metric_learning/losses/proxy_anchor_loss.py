import torch
from torch import Tensor

from ..types import IndicesTuple, LossDict

from ..distances import CosineSimilarity
from ..reducers import DivisorReducer
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction, regularization_term


def default_proxy_init(proxies):
    torch.nn.init.kaiming_normal_(proxies, mode="fan_out")


class ProxyAnchorLoss(BaseMetricLossFunction):
    def __init__(
        self,
        num_classes,
        embedding_size,
        margin=0.1,
        alpha=32,
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
        (weight_init_func or default_proxy_init)(self.proxies)
        self.num_classes = num_classes
        self.margin = margin
        self.alpha = alpha

    def compute_loss(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor,
        ref_labels: Tensor | None,
    ) -> LossDict:
        c_f.labels_required(labels)
        c_f.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        c_f.check_parameter_placement(self, self.proxies, embeddings)
        dtype = embeddings.dtype
        miner_weights = lmu.convert_to_weights(
            indices_tuple, labels, dtype=dtype
        ).unsqueeze(1)
        miner_weights = miner_weights - 1

        cos = self.get_logits(embeddings)

        pos_mask = torch.nn.functional.one_hot(labels, self.num_classes)
        neg_mask = 1 - pos_mask

        with_pos_proxies = torch.where(torch.sum(pos_mask, dim=0) != 0)[0]

        pos_exp = self.distance.margin(cos, self.margin)
        neg_exp = self.distance.margin(-self.margin, cos)

        pos_term = lmu.logsumexp(
            (self.alpha * pos_exp) + miner_weights,
            keep_mask=pos_mask.bool(),
            add_one=True,
            dim=0,
        )
        neg_term = lmu.logsumexp(
            (self.alpha * neg_exp) + miner_weights,
            keep_mask=neg_mask.bool(),
            add_one=True,
            dim=0,
        )

        loss_indices = c_f.torch_arange_from_size(self.proxies)

        return {
            "pos_loss": {
                "losses": pos_term.squeeze(0),
                "indices": loss_indices,
                "reduction_type": "element",
                "divisor": len(with_pos_proxies),
            },
            "neg_loss": {
                "losses": neg_term.squeeze(0),
                "indices": loss_indices,
                "reduction_type": "element",
                "divisor": self.num_classes,
            },
        }

    def regularization_terms(self, embeddings):
        terms = super().regularization_terms(embeddings)
        if self.weight_regularizer is not None:
            terms["weight_reg_loss"] = regularization_term(
                self.weight_regularizer, self.weight_reg_weight, self.proxies
            )
        return terms

    def get_default_reducer(self):
        return DivisorReducer()

    def get_default_distance(self):
        return CosineSimilarity()

    def sub_loss_names(self):
        return ["pos_loss", "neg_loss"]

    def get_logits(self, embeddings):
        return self.distance(embeddings, self.proxies)
