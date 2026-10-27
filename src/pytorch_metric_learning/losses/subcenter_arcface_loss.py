import math

import torch

from ..utils import common_functions as c_f
from .arcface_loss import ArcFaceLoss


class SubCenterArcFaceLoss(ArcFaceLoss):
    def __init__(
        self,
        num_classes,
        embedding_size,
        margin=28.6,
        scale=64,
        sub_centers=3,
        *,
        weight_regularizer=None,
        weight_reg_weight=1,
        weight_init_func=None,
        **kwargs,
    ):
        super().__init__(
            num_classes * sub_centers,
            embedding_size,
            margin=margin,
            scale=scale,
            weight_regularizer=weight_regularizer,
            weight_reg_weight=weight_reg_weight,
            weight_init_func=weight_init_func,
            **kwargs,
        )
        self.sub_centers = sub_centers
        self.num_classes = num_classes

    def get_cosine(self, embeddings):
        cosine = self.distance(embeddings, self.W.t())
        cosine = cosine.view(-1, self.num_classes, self.sub_centers)
        cosine, _ = cosine.max(axis=2)
        return cosine

    def get_outliers(
        self, embeddings, labels, threshold=75, return_dominant_centers=True
    ):
        c_f.check_shapes(embeddings, labels)
        c_f.check_parameter_placement(self, self.W, embeddings)
        dtype, device = embeddings.dtype, embeddings.device
        cos_threshold = math.cos(math.radians(threshold))
        outliers = []
        dominant_centers = torch.Tensor(self.W.shape[0], self.num_classes).to(
            dtype=dtype, device=device
        )
        with torch.no_grad():
            for label in range(self.num_classes):
                target_samples = labels == label
                if not target_samples.any():
                    continue
                target_indices = target_samples.nonzero()
                target_embeddings = embeddings[target_samples]

                sub_centers = self.W[
                    :, label * self.sub_centers : (label + 1) * self.sub_centers
                ]
                distances = self.distance(target_embeddings, sub_centers.t())
                max_sub_center_idxs = torch.argmax(distances, axis=1)
                max_sub_center_count = torch.bincount(max_sub_center_idxs)
                dominant_idx = torch.argmax(max_sub_center_count)
                dominant_centers[:, label] = sub_centers[:, dominant_idx]

                dominant_dist = distances[:, dominant_idx]
                drop_dists = dominant_dist < cos_threshold
                drop_idxs = target_indices[drop_dists]
                outliers.extend(drop_idxs.detach().tolist())
        outliers = torch.tensor(outliers, device=device).flatten()
        if return_dominant_centers:
            return outliers, dominant_centers
        return outliers
