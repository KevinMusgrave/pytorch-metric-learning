#! /usr/bin/env python3

from .base_metric_loss_function import BaseMetricLossFunction
import numpy as np
import torch
from ..utils import loss_and_miner_utils as lmu

class ArcFaceLoss(BaseMetricLossFunction):
    """
    Implementation of https://arxiv.org/pdf/1801.07698.pdf
    """
    def __init__(self, margin, num_labels, embedding_size, scale=64, **kwargs):
        self.margin = np.radians(margin)
        self.scale = scale
        self.num_labels = num_labels
        self.avg_angle = 0
        self.min_cos = -1 + 1e-7
        self.max_cos = 1 - 1e-7
        self.record_these = ["avg_angle"]
        super().__init__(**kwargs)
        self.W = torch.nn.Parameter(torch.randn(embedding_size, num_labels))
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        
    def compute_loss(self, embeddings, labels, indices_tuple):
        miner_weights = lmu.convert_to_weights(indices_tuple, labels)
        batch_size = labels.size(0)
        mask = torch.zeros(batch_size, self.num_labels).to(embeddings.device)
        mask[torch.arange(batch_size), labels] = 1
        if self.normalize_embeddings:
            curr_W = torch.nn.functional.normalize(self.W, p=2, dim=1)
        cosine = torch.matmul(embeddings, curr_W)
        cosine_of_target_classes = cosine[mask == 1]
        angle = torch.acos(torch.clamp(cosine_of_target_classes, self.min_cos, self.max_cos))
        self.avg_angle = np.degrees(torch.mean(angle).item())
        margin = self.maybe_mask_param(self.margin, labels)
        diff = (torch.cos(angle + margin) - cosine_of_target_classes).unsqueeze(1)
        cosine = cosine + (mask*diff)
        unweighted_loss = self.cross_entropy(cosine * self.scale, labels)
        return torch.mean(unweighted_loss*miner_weights)

