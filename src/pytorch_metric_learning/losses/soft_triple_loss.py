import math

import torch
import torch.nn.functional as F

from ..distances import CosineSimilarity
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction, regularization_term


def default_center_init(centers):
    torch.nn.init.kaiming_uniform_(centers, a=math.sqrt(5))


class SoftTripleLoss(BaseMetricLossFunction):
    def __init__(
        self,
        num_classes,
        embedding_size,
        centers_per_class=10,
        la=20,
        gamma=0.1,
        margin=0.01,
        *,
        weight_regularizer=None,
        weight_reg_weight=1,
        weight_init_func=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.weight_regularizer = weight_regularizer
        self.weight_reg_weight = weight_reg_weight
        assert self.distance.is_inverted
        self.la = la
        self.gamma = 1.0 / gamma
        self.margin = margin
        self.num_classes = num_classes
        self.centers_per_class = centers_per_class
        self.fc = torch.nn.Parameter(
            torch.empty(embedding_size, num_classes * centers_per_class)
        )
        (weight_init_func or default_center_init)(self.fc)
        self.add_to_recordable_attributes(
            list_of_names=[
                "la",
                "gamma",
                "margin",
                "centers_per_class",
                "num_classes",
                "embedding_size",
            ],
            is_stat=False,
        )

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.labels_required(labels)
        c_f.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        c_f.check_parameter_placement(self, self.fc, embeddings)
        dtype = embeddings.dtype
        miner_weights = lmu.convert_to_weights(indices_tuple, labels, dtype=dtype)
        sim_to_classes = self.get_logits(embeddings)
        margin = torch.zeros(
            sim_to_classes.shape, dtype=dtype, device=embeddings.device
        )
        margin[torch.arange(0, margin.shape[0]), labels] = self.margin
        loss = F.cross_entropy(
            self.la * (sim_to_classes - margin), labels, reduction="none"
        )
        loss = loss * miner_weights
        return {
            "loss": {
                "losses": loss,
                "indices": c_f.torch_arange_from_size(embeddings),
                "reduction_type": "element",
            }
        }

    def regularization_terms(self, embeddings):
        terms = super().regularization_terms(embeddings)
        if self.weight_regularizer is not None:
            terms["weight_reg_loss"] = regularization_term(
                self.weight_regularizer, self.weight_reg_weight, self.fc.t()
            )
        return terms

    def get_default_distance(self):
        return CosineSimilarity()

    def get_logits(self, embeddings):
        sim_to_centers = self.distance(embeddings, self.fc.t())
        sim_to_centers = sim_to_centers.view(
            -1, self.num_classes, self.centers_per_class
        )
        prob = F.softmax(sim_to_centers * self.gamma, dim=2)
        return torch.sum(prob * sim_to_centers, dim=2)
