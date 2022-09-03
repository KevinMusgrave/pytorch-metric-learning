import math

import numpy as np
import torch

from ..utils import common_functions as c_f
from .arcface_loss import ArcFaceLoss


class SubCenterArcFaceLoss(ArcFaceLoss):
    """
    Implementation of https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf
    """

    def __init__(self, *args, margin=28.6, scale=64, sub_centers=3, **kwargs):
        num_classes, embedding_size = kwargs["num_classes"], kwargs["embedding_size"]
        super().__init__(
            num_classes * sub_centers, embedding_size, margin=margin, scale=scale
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
        dtype, device = embeddings.dtype, embeddings.device
        self.cast_types(dtype, device)
        cos_threshold = math.cos(np.radians(threshold))
        outliers = []
        dominant_centers = torch.Tensor(self.W.shape[0], self.num_classes).to(
            dtype=dtype, device=device
        )
        with torch.no_grad():
            for label in range(self.num_classes):
                target_samples = labels == label
                if (target_samples is False).all():
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
                # "distances" are actually cosine similarities
                drop_dists = dominant_dist < cos_threshold
                drop_idxs = target_indices[drop_dists]
                outliers.extend(drop_idxs.detach().tolist())
        outliers = torch.tensor(outliers, device=device).flatten()
        return outliers if not return_dominant_centers else outliers, dominant_centers
