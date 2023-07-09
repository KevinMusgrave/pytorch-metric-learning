from typing import Dict

import torch

from ..utils import common_functions as c_f

SUPPORTED_REGULARIZATION_TYPES = ["custom", "weight", "embedding"]


class RegularizerMixin:
    """Base class for regularization losses.
    regularizer: function-like object or `nn.Module` that transforms input data into single number or single-element `torch.Tensor`
    """

    def __init__(self, regularizer=None, reg_weight=1, type="custom", **kwargs):
        self.check_type(type)
        self.regularizer = regularizer if regularizer is not None else (lambda data: 0)
        self.reg_weight = reg_weight
        if regularizer is not None:
            self.add_to_recordable_attributes(
                list_of_names=[f"{type}_reg_weight"], is_stat=False
            )

    def regularization_loss(self, data):
        loss = self.regularizer(data) * self.reg_weight
        return loss

    def add_regularization_to_loss_dict(self, loss_dict: Dict[str, Dict], data):
        loss_dict[self.reg_loss_type] = {
            "losses": self.regularization_loss(data),
            "indices": None,
            "reduction_type": "already_reduced",
        }

    def check_type(self, type: str):
        if type not in SUPPORTED_REGULARIZATION_TYPES:
            raise ValueError(
                f"Type provided not supported. Supported types are {', '.join(SUPPORTED_REGULARIZATION_TYPES)}, given type is {type}."
            )
        self.reg_loss_type = f"{type}_reg_loss"


def get_default_weight_init_func():
    return c_f.TorchInitWrapper(torch.nn.init.normal_)


class WeightRegularizerMixin(RegularizerMixin):
    def __init__(self, weight_init_func=None, **kwargs):
        kwargs["type"] = "weight"
        super().__init__(**kwargs)
        self.weight_init_func = (
            weight_init_func
            if weight_init_func is not None
            else get_default_weight_init_func()
        )

    def add_weight_regularization_to_loss_dict(self, loss_dict, weights):
        self.add_regularization_to_loss_dict(loss_dict, weights)


class EmbeddingRegularizerMixin(RegularizerMixin):
    def __init__(self, **kwargs):
        kwargs["type"] = "embedding"
        super().__init__(**kwargs)

    def add_embedding_regularization_to_loss_dict(self, loss_dict, embeddings):
        self.add_regularization_to_loss_dict(loss_dict, embeddings)
