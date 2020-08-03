from ..utils import common_functions as c_f
import torch

class WeightMixin:
    def __init__(self, weight_init_func=None, **kwargs):
        super().__init__(**kwargs)
        self.weight_init_func = weight_init_func
        if self.weight_init_func is None:
            self.weight_init_func = self.get_default_weight_init_func()
            
    def get_default_weight_init_func(self):
        return c_f.TorchInitWrapper(torch.nn.init.normal_)


class WeightRegularizerMixin:
    def __init__(self, weight_regularizer=None, weight_reg_weight=1, **kwargs):
        super().__init__(**kwargs)
        self.weight_regularizer = weight_regularizer
        self.weight_reg_weight = weight_reg_weight
        self.add_to_recordable_attributes(list_of_names=["weight_reg_weight"], is_stat=False)

    def weight_regularization_loss(self, weights):
        if self.weight_regularizer is None:
            loss = 0
        else:
            loss = self.weight_regularizer(weights) * self.weight_reg_weight
        return {"losses": loss, "indices": None, "reduction_type": "already_reduced"}

    def add_weight_regularization_to_loss_dict(self, loss_dict, weights):
        loss_dict["weight_reg_loss"] = self.weight_regularization_loss(weights)

    def regularization_loss_names(self):
        return ["weight_reg_loss"]



class EmbeddingRegularizerMixin:
    def __init__(self, embedding_regularizer=None, embedding_reg_weight=1, **kwargs):
        super().__init__(**kwargs)
        self.embedding_regularizer = embedding_regularizer
        self.embedding_reg_weight = embedding_reg_weight
        self.add_to_recordable_attributes(list_of_names=["embedding_reg_weight"], is_stat=False)

    def embedding_regularization_loss(self, embeddings):
        if self.embedding_regularizer is None:
            loss = 0
        else:
            loss = self.embedding_regularizer(embeddings) * self.embedding_reg_weight
        return {"losses": loss, "indices": None, "reduction_type": "already_reduced"}

    def add_embedding_regularization_to_loss_dict(self, loss_dict, embeddings):
        loss_dict["embedding_reg_loss"] = self.embedding_regularization_loss(embeddings)

    def regularization_loss_names(self):
        return ["embedding_reg_loss"]
