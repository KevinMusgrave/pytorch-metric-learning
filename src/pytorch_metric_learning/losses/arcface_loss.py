#! /usr/bin/env python3

from .large_margin_softmax_loss import LargeMarginSoftmaxLoss
import numpy as np
import torch

class ArcFaceLoss(LargeMarginSoftmaxLoss):
    """
    Implementation of https://arxiv.org/pdf/1801.07698.pdf
    """
    def init_margin(self):
        self.margin = np.radians(self.margin)
        
    def cast_types(self, dtype, device):
        self.W.data = self.W.data.to(device).type(dtype)

    def modify_cosine_of_target_classes(self, cosine_of_target_classes, *args):
        angles = self.get_angles(cosine_of_target_classes)
        return torch.cos(angles + self.margin)

    def scale_logits(self, logits, *_):
        return logits * self.scale