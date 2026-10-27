import math

import torch

from ..distances import CosineSimilarity
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction, regularization_term


class LargeMarginSoftmaxLoss(BaseMetricLossFunction):
    n_range: torch.Tensor
    margin_choose_n: torch.Tensor
    cos_powers: torch.Tensor
    alternating: torch.Tensor

    def __init__(
        self,
        num_classes,
        embedding_size,
        margin=4,
        scale=1,
        *,
        weight_regularizer=None,
        weight_reg_weight=1,
        weight_init_func=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        c_f.assert_distance_type(self, CosineSimilarity)
        self.weight_regularizer = weight_regularizer
        self.weight_reg_weight = weight_reg_weight
        self.margin = margin
        self.num_classes = num_classes
        self.scale = scale
        self.register_stats("avg_angle")
        self.init_margin()
        self.W = torch.nn.Parameter(torch.empty(embedding_size, num_classes))
        (weight_init_func or torch.nn.init.normal_)(self.W)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

    def init_margin(self):
        self.margin = int(self.margin)
        self.max_n = self.margin // 2
        n_range = torch.arange(self.max_n + 1)
        coefficients = [float(math.comb(self.margin, 2 * n)) for n in n_range]
        self.register_buffer("n_range", n_range, persistent=False)
        self.register_buffer(
            "margin_choose_n", torch.tensor(coefficients), persistent=False
        )
        self.register_buffer("cos_powers", self.margin - 2 * n_range, persistent=False)
        self.register_buffer("alternating", (-1) ** n_range, persistent=False)

    def get_cos_with_margin(self, cosine):
        cosine = cosine.unsqueeze(1)
        cos_powered = cosine**self.cos_powers
        sin_powered = (1 - cosine**2) ** self.n_range
        terms = self.alternating * self.margin_choose_n * cos_powered * sin_powered
        return torch.sum(terms, dim=1)

    def get_cosine(self, embeddings):
        return self.distance(embeddings, self.W.t())

    def get_angles(self, cosine_of_target_classes):
        angles = torch.acos(torch.clamp(cosine_of_target_classes, -1, 1))
        if self.collect_stats:
            with torch.no_grad():
                self.avg_angle = math.degrees(torch.mean(angles).item())
        return angles

    def get_target_mask(self, embeddings, labels):
        batch_size = labels.size(0)
        mask = torch.zeros(
            batch_size,
            self.num_classes,
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        mask[torch.arange(batch_size), labels] = 1
        return mask

    def modify_cosine_of_target_classes(self, cosine_of_target_classes):
        cos_with_margin = self.get_cos_with_margin(cosine_of_target_classes)
        angles = self.get_angles(cosine_of_target_classes)
        with torch.no_grad():
            k = (angles / (math.pi / self.margin)).floor()
        return ((-1) ** k) * cos_with_margin - (2 * k)

    def scale_logits(self, logits, embeddings):
        embedding_norms = self.distance.get_norm(embeddings)
        weight_norms = self.distance.get_norm(self.W, dim=0)
        product_of_magnitudes = weight_norms.unsqueeze(0) * embedding_norms.unsqueeze(1)
        return logits * product_of_magnitudes * self.scale

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.labels_required(labels)
        c_f.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        c_f.check_parameter_placement(self, self.W, embeddings)
        dtype = embeddings.dtype
        miner_weights = lmu.convert_to_weights(indices_tuple, labels, dtype=dtype)
        mask = self.get_target_mask(embeddings, labels)
        cosine = self.get_cosine(embeddings)
        cosine_of_target_classes = cosine[mask == 1]
        modified_cosine_of_target_classes = self.modify_cosine_of_target_classes(
            cosine_of_target_classes
        )
        diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(
            1
        )
        logits = cosine + (mask * diff)
        logits = self.scale_logits(logits, embeddings)
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
        return CosineSimilarity()

    def get_logits(self, embeddings):
        logits = self.get_cosine(embeddings)
        return self.scale_logits(logits, embeddings)
