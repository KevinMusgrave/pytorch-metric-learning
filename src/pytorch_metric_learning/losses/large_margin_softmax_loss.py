#! /usr/bin/env python3

from .weight_regularizer_mixin import WeightRegularizerMixin
from .base_metric_loss_function import BaseMetricLossFunction
from ..utils import loss_and_miner_utils as lmu, common_functions as c_f
import scipy.special
import torch
import math
import numpy as np

class LargeMarginSoftmaxLoss(WeightRegularizerMixin, BaseMetricLossFunction):
    """
    Implementation of https://arxiv.org/pdf/1612.02295.pdf
    """
    def __init__(self, margin, num_classes, embedding_size, scale=1, normalize_weights=False, scale_logits_by_magnitudes=True, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.num_classes = num_classes
        self.scale = scale
        self.normalize_weights = normalize_weights
        self.scale_logits_by_magnitudes = scale_logits_by_magnitudes
        self.add_to_recordable_attributes(name="avg_angle", is_stat=True)
        self.init_margin()
        self.W = torch.nn.Parameter(torch.randn(embedding_size, num_classes))
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def init_margin(self):
        self.margin = int(self.margin)
        self.max_n = (self.margin // 2)
        ## For the trigonometric multiple-angle formula ##
        self.n_range = torch.tensor([n for n in range(0, self.max_n+1)])
        self.margin_choose_n = torch.tensor([scipy.special.binom(self.margin, 2*n) for n in self.n_range])
        self.cos_powers = torch.tensor([self.margin-(2*n) for n in self.n_range])
        self.alternating = torch.tensor([(-1)**n for n in self.n_range])

    def get_cos_with_margin(self, cosine):
        cosine = cosine.unsqueeze(1)
        for attr in ["n_range", "margin_choose_n", "cos_powers", "alternating"]:
            setattr(self, attr, getattr(self, attr).to(cosine.device))
        cos_powered = cosine**self.cos_powers
        sin_powered = (1-cosine**2)**self.n_range
        terms = self.alternating*self.margin_choose_n*cos_powered*sin_powered # Equation 7 in the paper
        return torch.sum(terms, dim=1)

    def get_weights(self):
        if self.normalize_weights:
            return torch.nn.functional.normalize(self.W, p=2, dim=0)
        return self.W

    def get_cosine(self, embeddings):
        weights = self.get_weights()
        self.weight_norms = torch.norm(weights, p=2, dim=0)
        # self.embedding_norms is computed in BaseMetricLossFunction
        self.product_of_magnitudes = self.weight_norms.unsqueeze(0)*self.embedding_norms.unsqueeze(1)
        return torch.matmul(embeddings, weights) / self.product_of_magnitudes

    def get_angles(self, cosine_of_target_classes):
        angles = torch.acos(torch.clamp(cosine_of_target_classes, -1 + 1e-7, 1 - 1e-7))
        self.avg_angle = np.degrees(torch.mean(angles).item())
        return angles

    def get_target_mask(self, embeddings, labels):
        batch_size = labels.size(0)
        mask = torch.zeros(batch_size, self.num_classes, dtype=embeddings.dtype).to(embeddings.device)
        mask[torch.arange(batch_size), labels] = 1
        return mask

    def modify_cosine_of_target_classes(self, cosine_of_target_classes, *args):
        _, _, labels, _ = args
        cos_with_margin = self.get_cos_with_margin(cosine_of_target_classes)
        angles = self.get_angles(cosine_of_target_classes)
        with torch.no_grad():
            k = (angles / (math.pi / self.margin)).floor() # Equation 6: angles needs to be between [k*pi/m and (k+1)*pi/m]
        return ((-1)**k)*cos_with_margin - (2*k)

    def cast_types(self, dtype, device):
        self.W.data = self.W.data.to(device).type(dtype)
        self.n_range = self.n_range.to(device).type(dtype)
        self.margin_choose_n = self.margin_choose_n.to(device).type(dtype)
        self.cos_powers = self.cos_powers.to(device).type(dtype)
        self.alternating = self.alternating.to(device).type(dtype)

    def compute_loss(self, embeddings, labels, indices_tuple):
        dtype, device = embeddings.dtype, embeddings.device
        self.cast_types(dtype, device)
        miner_weights = lmu.convert_to_weights(indices_tuple, labels, dtype=dtype)
        mask = self.get_target_mask(embeddings, labels)
        cosine = self.get_cosine(embeddings)
        cosine_of_target_classes = cosine[mask == 1]
        modified_cosine_of_target_classes = self.modify_cosine_of_target_classes(cosine_of_target_classes, cosine, embeddings, labels, mask)
        diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(1)
        logits = cosine + (mask*diff)
        if self.scale_logits_by_magnitudes:
            logits = logits * self.product_of_magnitudes
        unweighted_loss = self.cross_entropy(logits * self.scale, labels)
        miner_weighted_loss = unweighted_loss*miner_weights
        loss_dict = {"loss": {"losses": miner_weighted_loss, "indices": c_f.torch_arange_from_size(embeddings), "reduction_type": "element"}}
        loss_dict["reg_loss"] = self.regularization_loss(self.W.t())
        return loss_dict


    def sub_loss_names(self):
        return ["loss", "reg_loss"]