import math

import numpy as np
import scipy.special
import torch

from ..distances import CosineSimilarity
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction
from .mixins import WeightRegularizerMixin


class LargeMarginSoftmaxLoss(WeightRegularizerMixin, BaseMetricLossFunction):
    """
    Implementation of https://arxiv.org/pdf/1612.02295.pdf
    """

    def __init__(self, num_classes, embedding_size, margin=4, scale=1, **kwargs):
        super().__init__(**kwargs)
        c_f.assert_distance_type(self, CosineSimilarity)
        self.margin = margin
        self.num_classes = num_classes
        self.scale = scale
        self.add_to_recordable_attributes(
            list_of_names=["num_classes", "margin", "scale"], is_stat=False
        )
        self.add_to_recordable_attributes(name="avg_angle", is_stat=True)
        self.init_margin()
        self.W = torch.nn.Parameter(torch.Tensor(embedding_size, num_classes))
        self.weight_init_func(self.W)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

    def init_margin(self):
        self.margin = int(self.margin)
        self.max_n = self.margin // 2
        ## For the trigonometric multiple-angle formula ##
        self.n_range = torch.tensor([n for n in range(0, self.max_n + 1)])
        self.margin_choose_n = torch.tensor(
            [scipy.special.binom(self.margin, 2 * n) for n in self.n_range]
        )
        self.cos_powers = torch.tensor([self.margin - (2 * n) for n in self.n_range])
        self.alternating = torch.tensor([(-1) ** n for n in self.n_range])

    def get_cos_with_margin(self, cosine):
        cosine = cosine.unsqueeze(1)
        for attr in ["n_range", "margin_choose_n", "cos_powers", "alternating"]:
            setattr(self, attr, c_f.to_device(getattr(self, attr), cosine))
        cos_powered = cosine**self.cos_powers
        sin_powered = (1 - cosine**2) ** self.n_range
        terms = (
            self.alternating * self.margin_choose_n * cos_powered * sin_powered
        )  # Equation 7 in the paper
        return torch.sum(terms, dim=1)

    def get_cosine(self, embeddings):
        return self.distance(embeddings, self.W.t())

    def get_angles(self, cosine_of_target_classes):
        angles = torch.acos(torch.clamp(cosine_of_target_classes, -1, 1))
        if self.collect_stats:
            with torch.no_grad():
                self.avg_angle = np.degrees(torch.mean(angles).item())
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
            k = (
                angles / (math.pi / self.margin)
            ).floor()  # Equation 6: angles needs to be between [k*pi/m and (k+1)*pi/m]
        return ((-1) ** k) * cos_with_margin - (2 * k)

    def scale_logits(self, logits, embeddings):
        embedding_norms = self.distance.get_norm(embeddings)
        weight_norms = self.distance.get_norm(self.W, dim=0)
        product_of_magnitudes = weight_norms.unsqueeze(0) * embedding_norms.unsqueeze(1)
        return logits * product_of_magnitudes * self.scale

    def cast_types(self, dtype, device):
        self.W.data = c_f.to_device(self.W.data, device=device, dtype=dtype)
        self.n_range = c_f.to_device(self.n_range, device=device, dtype=dtype)
        self.margin_choose_n = c_f.to_device(
            self.margin_choose_n, device=device, dtype=dtype
        )
        self.cos_powers = c_f.to_device(self.cos_powers, device=device, dtype=dtype)
        self.alternating = c_f.to_device(self.alternating, device=device, dtype=dtype)

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.labels_required(labels)
        c_f.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        dtype, device = embeddings.dtype, embeddings.device
        self.cast_types(dtype, device)
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
        loss_dict = {
            "loss": {
                "losses": miner_weighted_loss,
                "indices": c_f.torch_arange_from_size(embeddings),
                "reduction_type": "element",
            }
        }
        self.add_weight_regularization_to_loss_dict(loss_dict, self.W.t())
        return loss_dict

    def get_default_distance(self):
        return CosineSimilarity()

    def get_logits(self, embeddings):
        logits = self.get_cosine(embeddings)
        return self.scale_logits(logits, embeddings)
